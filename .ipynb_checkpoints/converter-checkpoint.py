# Main conversion logic: single raw tif file to many nrrd files, each corresponding to a single time point

from pathlib import Path
import tifffile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import jax.numpy as jnp

from .config import THREAD_POOL_SIZE
from .processors import process_batch_jax
from .utils import parallel_save_outputs, calculate_optimal_chunk_size

def convert_tif_to_nrrd(
    input_path, output_dir, 
    n_z=84, spacing_lat=0.54, spacing_axi=0.54, 
    generate_mip=True, z_range=(3,80),
    channels=[1,2], 
    noise_path=None, chunk_size=None
):
    """Optimized memory-efficient conversion from TIF to NRRD
    
    Processes volumes with the following steps:
    1. Input: volumes of 966x630x84
    2. Select z-slices 3:80 to get 966x630x77
    3. Perform background subtraction
    4. 3x3 binning to get final volumes of 322x210x77
    """
    
    # Create output directories
    nrrd_dir = Path(output_dir) / "NRRD"
    mip_dir = Path(output_dir) / "MIP" if generate_mip else None
    nrrd_dir.mkdir(parents=True, exist_ok=True)
    if generate_mip:
        mip_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare noise data
    noise_data = None
    if noise_path:
        noise_data = jnp.array(tifffile.imread(str(noise_path)), dtype=jnp.uint16)
        print(f"Loaded noise data shape: {noise_data.shape}")

    # Get TIF info and calculate optimal chunk size
    with tifffile.TiffFile(str(input_path)) as tif:
        n_pages = len(tif.pages)
        shape = tif.pages[0].shape
        dtype = tif.pages[0].dtype
        
        if chunk_size is None:
            chunk_size = calculate_optimal_chunk_size(shape + (n_z,), dtype)

    # Calculate processing parameters
    t_size = n_pages // n_z
    n_chunks = (t_size + chunk_size - 1) // chunk_size
    
    # Initialize thread pool for parallel I/O
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        # Process chunks
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            t_start = chunk_idx * chunk_size
            t_end = min(t_start + chunk_size, t_size)
            current_chunk_size = t_end - t_start
            
            # Load chunk data in parallel
            futures = []
            for t_offset in range(current_chunk_size):
                t = t_start + t_offset
                start_page = t * n_z
                end_page = start_page + n_z
                futures.append(executor.submit(
                    lambda idx: tifffile.imread(str(input_path), key=idx),
                    range(start_page, end_page)
                ))
            
            # Gather results and prepare batch
            chunk_data = np.stack([f.result() for f in futures])
            print(f"Loaded chunk data shape: {chunk_data.shape}")
            chunk_jax = jnp.array(chunk_data)
            print(f"Converted to JAX array shape: {chunk_jax.shape}")
            
            # Process batch using JAX
            processed_data = process_batch_jax(chunk_jax, noise_data)
            
            # Prepare CUDA binning
            threadsperblock = (8, 8, 8)
            blockspergrid_x = (processed_data.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
            blockspergrid_y = (processed_data.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
            blockspergrid_z = (processed_data.shape[2] + threadsperblock[2] - 1) // threadsperblock[2]
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
            
            # Process results and save in parallel
            save_tasks = []
            for t_offset in range(current_chunk_size):
                t = t_start + t_offset
                
                for c in channels:
                    # Prepare paths
                    output_path_str = str(output_dir)
                    prefix = output_path_str[output_path_str.rindex('/')+1:output_path_str.index('output')].rstrip('_')
                    basename = f"{prefix}_t{t+1:04d}_ch{c}"  # Added +1 here to start from t0001
                    nrrd_path = nrrd_dir / f"{basename}.nrrd"
                    mip_path = mip_dir / f"{basename}.png" if generate_mip else None
                    
                    # Extract and process volume
                    vol_data = processed_data[t_offset, :, :, z_range[0]:z_range[1]]
                    
                    # Add to save tasks
                    save_tasks.append((
                        nrrd_path,
                        mip_path,
                        vol_data,
                        [spacing_lat, spacing_lat, spacing_axi]
                    ))
            
            # Execute save tasks in parallel
            list(executor.map(parallel_save_outputs, save_tasks))