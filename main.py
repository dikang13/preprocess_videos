import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import jax.numpy as jnp
import tifffile

from fileio import get_total_pages, load_nd2_chunk, load_tif_chunk, parallel_save_outputs
from transform import background_subtract, bin_and_transpose
from noise import get_avg_noise
from utils import parse_slice, parse_list, parse_float_list, THREAD_POOL_SIZE
import re

def preprocess(
    input_path, output_dir, noise_path, blank_dir,
    chunk_size,
    n_z, x_range, y_range, z_range, channels,
    bitdepth,
    binsize,
    save_as,
    spacing
):
    # Get input metadata
    ext = Path(input_path).suffix
    assert ext in ['.nd2', '.tif', '.tiff'], "IMPORTANT: You can only feed in the path to a .tif or .nd2 file in input_path!!!"
        
    # Load or prepare noise data
    noise_data = None
    if noise_path:
        noise_data = jnp.array(tifffile.imread(str(noise_path)), dtype=jnp.int32)
        print(f"Loaded noise data of shape {noise_data.shape} from {noise_path}")
    elif blank_dir:
        noise_data, savepath = get_avg_noise(blank_dir, x_range, y_range)
        print(f"Computed noise data of shape {noise_data.shape} by averaging all frames in {blank_dir}. "
              f"In the future, you can reuse this noise data by setting noise_path to {savepath}")
    else:
        print(f"IMPORTANT: Neither noise_path nor blank_dir is provided. Background subtraction is skipped")
        
    # Create output directories
    if save_as == "nrrd":
        nrrd_dir = Path(output_dir) / "NRRD"
        mip_dir = Path(output_dir) / "MIP"
        nrrd_dir.mkdir(parents=True, exist_ok=True)
        mip_dir.mkdir(parents=True, exist_ok=True)
    
    n_pages = get_total_pages(input_path, ext)
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
    
                if ext == '.nd2':
                    futures.append(executor.submit(
                        load_nd2_chunk,
                        input_path,
                        start_page, end_page,
                        x_range, y_range
                    ))            
                else: # must be .tif or .tiff then
                    futures.append(executor.submit(
                       load_tif_chunk,
                       input_path,
                       start_page, end_page,
                       x_range, y_range
                    ))
            
            # Gather results and prepare batch
            chunk_data = jnp.array([f.result() for f in futures])
            print(f"Loaded a chunk as JAX array of shape {chunk_data.shape}")
    
            for c in range(len(channels)): # per color channel
                out = chunk_data[:, :, c, :, :]
                
                # Background subtract only if noise data is not None
                if noise_data is not None:
                    out = background_subtract(out, noise_data[c, :, :], x_range, y_range, bitdepth)
        
                # Bin across (x,y) only if binsize is not None
                if binsize > 0:
                    out = bin_and_transpose(out, binsize)
                else:
                    print(f"IMPORTANT: Binsize is set to 0. Post-hoc binning is skipped")
                
                # Export output file(s)
                if save_as == 'nrrd': 
                    processed_data = out.astype(jnp.uint16)
                    save_tasks = []
                    for t_offset in range(current_chunk_size):
                        t = t_start + t_offset
                        
                        # Prepare paths
                        output_path_str = str(output_dir)
                        prefix = re.findall(r'202\d{1}-\d{2}-\d{2}-\d{2}', output_path_str)[-1] if re.findall(r'202\d{1}-\d{2}-\d{2}-\d{2}', output_path_str) else ''
                        basename = f"{prefix}_t{t+1:04d}_ch{c+1}"
                        nrrd_path = nrrd_dir / f"{basename}.nrrd"
                        mip_path = mip_dir / f"{basename}.png"
                        
                        # Extract each z-stack, add to save tasks
                        vol_data = processed_data[t_offset, :, :, z_range]
                        save_tasks.append((
                            nrrd_path,
                            mip_path,
                            vol_data,
                            spacing
                        ))
                            
                    # Execute save tasks in parallel
                    list(executor.map(parallel_save_outputs, save_tasks))
                    
                elif save_as == 'tif':
                    processed_data = out.astype(jnp.int16)
                    output_path_str = os.path.join(output_dir, f"processed_video_ch{c}.tif")
                    tifffile.imwrite(output_path_str, processed_data, bigtiff=True) 
                    print(f"Processed video is saved as one tif per channel here: {output_path_str}")
                    
                else:
                    raise ValueError(f"IMPORTANT: You need to save the output either as tif or nrrd files under {output_dir}!!!")


def main():
    parser = argparse.ArgumentParser(description='Process microscopy image data with background subtraction and binning.')
    
    # Input/Output paths
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input file path, eg /path/to/data.nd2')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory, eg /path/to/output')
    parser.add_argument('--noise_path', type=str, default=None,
                        help='Path to noise reference file')
    parser.add_argument('--blank_dir', type=str, default=None,
                        help='Folder path to multiple blank images acquired when camera is capped')
    
    # Processing parameters
    parser.add_argument('--save_as', type=str, default='nrrd', choices=['nrrd', 'tif'],
                        help='File extension for outputs')
    parser.add_argument('--binsize', type=int, default=3,
                        help='Number of pixels in x or y that will be binned. E.g. binsize=3 means every 9 pixels becomes a superpixel. Setting binsize=0 skips binning.')
    parser.add_argument('--bitdepth', type=int, default=12, choices=[8, 11, 12, 16],
                        help='Bit depth of input images')
    
    # Image dimensions
    parser.add_argument('--n_z', type=int, default=80,
                        help='Number of z-slices per volume')
    parser.add_argument('--x_range', type=parse_slice, default='0,966',
                        help='Range of pixels along x dimension to include as "start,end"')
    parser.add_argument('--y_range', type=parse_slice, default='0,630',
                        help='Range of pixels along y dimension to include as "start,end"')
    parser.add_argument('--z_range', type=parse_slice, default='3,80',
                        help='Range of z-slices to include as "start,end"')
    parser.add_argument('--channels', type=parse_list, default='1,2',
                        help='Channels to process as "1,2" or "[1,2]"')
    parser.add_argument('--spacing', type=parse_float_list, default='0.54,0.54,0.54',
                        help='Post-binning voxel dimension. For unbinned images, voxel dim = 0.18,0.18,0.18')
    
    # Performance settings
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='Number of timepoints to process at once')
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU device number to use')

    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        noise_path=args.noise_path,
        blank_dir=args.blank_dir,
        chunk_size=args.chunk_size,
        n_z=args.n_z,
        x_range=args.x_range,
        y_range=args.y_range,
        z_range=args.z_range,
        channels=args.channels,
        bitdepth=args.bitdepth,
        binsize=args.binsize,
        save_as=args.save_as,
        spacing=args.spacing
    )

if __name__ == '__main__':
    main()