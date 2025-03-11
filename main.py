import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import nd2
import tifffile
import re
import dask
import dask.array as da
from fileio import batch_save_tif, batch_save_nrrd
from process import bin_and_subtract, bin_xy
from noise import get_noise_data, compute_uniform_noise_data
from utils import parse_slice, parse_list, parse_float_list
import gc
import psutil
import time
from datetime import datetime
import jax
jax.clear_caches()
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', False)  # Use float32/int32 by default
import jax.numpy as jnp

# Create JIT-compiled version for JAX operations
bin_xy_jit = jax.jit(bin_xy, static_argnames=['binsize'])
bin_and_subtract_jit = jax.jit(bin_and_subtract, static_argnames=['binsize', 'bitdepth'])

def print_mem_usage(gpu):
    process = psutil.Process(os.getpid())
    print(f"Process Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    print(f"GPU memory stats: {gpu.memory_stats()}")

def preprocess(
    input_path, output_dir, 
    noise_path, blank_dir, bg_percentile,
    global_t_start, global_t_end, 
    chunk_size, num_workers,
    n_z, z_range,
    bitdepth,
    binsize,
    save_as,
    spacing
):
    print(f"===Time when process starts: {datetime.now().strftime("%H:%M:%S")}===")
    
    # Get the first GPU device (if available)
    gpu = jax.devices("gpu")[0]

    # Create output directories
    Path(output_dir, "NRRD").mkdir(parents=True, exist_ok=True)
    base_nrrd_path = Path(output_dir) / "NRRD"

    # Load ND2 data as a lazy Dask array
    with nd2.ND2File(input_path) as nd2_file:
        all_frames = nd2_file.to_dask() # 4D array (T*Z, C, X, Y)
    n_pages, n_channels, n_x, n_y = all_frames.shape
    # t_size = n_pages // n_z

    # Load noise data if a .tif file is provided, or compute noise by averaging frames in blank_dir
    noise_data = None
    noise_path = Path(base_nrrd_path) / "noise.tif"
    
    if noise_path.is_file():
        noise_data = jnp.array(tifffile.imread(str(noise_path)))
        print(f"Loaded noise data of shape {noise_data.shape} from {noise_path}")
    else:
        print("⚠️⚠️⚠️ WARNING: No blank frames provided. Assuming uniform background for all pixels.⚠️⚠️⚠️")
        noise_data = all_frames[:400].compute()
        noise_data = noise_data.reshape(5, n_z, n_channels, n_x, n_y)
        noise_data = noise_data.transpose(0, 2, 1, 3, 4)  # 5D array (T, C, Z, X, Y)
        noise_data = compute_uniform_noise_data(noise_data, bg_percentile) # compute noise per channel from the first 5 time points
        noise_data = bin_xy(noise_data, binsize) # bin noise data to superpixels
        noise_data = jnp.floor(noise_data / 100) * 100  # Round down to nearest hundred
        print(f"Uniform background of shape {noise_data.shape} and value {noise_data[:,0,0]} is subtracted from each pixel in the respective channels.")
        tifffile.imwrite(noise_path, noise_data)

    # Load binned noise to GPU
    noise_data = jax.device_put(noise_data.astype(jnp.int16))
        
    # Perform jax array computations in series, and file saving in parallel   
    t_size = global_t_end - global_t_start
    n_chunks = (t_size + chunk_size - 1) // chunk_size
    print(f"Input dimensions: t={t_size}, z={n_z}, c={n_channels}, x={n_x}, y={n_y}")

    prefix = re.findall(r'202\d{1}-\d{2}-\d{2}-\d{2}', str(output_dir))
    prefix = prefix[-1] if prefix else ''
    
    for chunk_idx in tqdm(range(n_chunks)):           
        t_start = chunk_idx * chunk_size + global_t_start
        t_end = min(t_start + chunk_size, t_size + global_t_start)
        current_chunk_size = t_end - t_start
        page_start = t_start * n_z
        page_end = t_end * n_z
        print(f"Starting chunk {chunk_idx+1}/{n_chunks}")
        
        # Bring a small chunk of the Dask array to RAM
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            cpu_data = all_frames[page_start:page_end].compute() 
        cpu_data = cpu_data.reshape(current_chunk_size, n_z, n_channels, n_x, n_y) # 5D array (T, Z, C, X, Y)
        cpu_data = cpu_data[:, z_range, :, :, :] # ignore the z-slices with lots of motion artifacts
        print(f"===Time when input is sent to GPU: {datetime.now().strftime("%H:%M:%S")}===")
        
        with jax.default_device(gpu):  # Explicitly manage device context
            data = jax.device_put(jnp.asarray(cpu_data).astype(jnp.int16))
            output = bin_and_subtract_jit(data, noise_data, binsize, bitdepth)
            output.block_until_ready()
            cpu_output = jax.device_get(output)
            del data 
            del output
            
        # Parallel NRRD file saving
        print(f"===Time when output is taken down to CPU: {datetime.now().strftime("%H:%M:%S")}===")
        tasks = [
            (
                base_nrrd_path / f"{prefix}_t{t_start + t_offset + 1:04d}_ch{c+1}.nrrd",
                cpu_output[t_offset, c],  # Extract per timepoint per channel
                spacing
            )
            for t_offset in range(current_chunk_size)
            for c in range(n_channels)
        ]
        batch_save_nrrd(tasks, num_workers)  # Adjust workers based on CPU
        gc.collect()
        jax.device_get(jnp.ones(1))  # Force sync with device
        time.sleep(1)
        print(f"===Time when iteration finishes: {datetime.now().strftime("%H:%M:%S")}===")
            
    print(f"===Time when process finishes: {datetime.now().strftime("%H:%M:%S")}===")
        
def main():
    parser = argparse.ArgumentParser(description='Process microscopy image data with background subtraction and binning.')
    
    # Input/Output paths
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input file path, e.g. /store1/shared/panneuralGFP_SWF1212/data_raw/2025-02-03/2025-02-03-18.nd2')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory, e.g. /store1/shared/panneuralGFP_SWF1212/data_processed/2025-02-03-13_output/neuropal/2025-02-03-18')
    parser.add_argument('--noise_path', type=str, default=None,
                        help='Path to noise reference file, e.g. /storage/fs/store1/shared/confocal_debugging/avg_noise.tif')
    parser.add_argument('--blank_dir', type=str, default=None,
                        help='Folder path to blank images with no samples, e.g. /storage/fs/store1/shared/confocal_debugging/data_raw/2025-02-blank-frames')
    parser.add_argument('--bg_percentile', type=int, default=20,
                        help='The percentile of pixel intensity in the first few frames below which all values are subtracted as background')                        
    # Processing parameters
    parser.add_argument('--save_as', type=str, default='nrrd', choices=['nrrd', 'tif'],
                        help='File extension for outputs')
    parser.add_argument('--binsize', type=int, default=3,
                        help='Binning factor. binsize=3 means 3x3 spatial binning => 9 raw pixels -> 1 superpixel.')
    parser.add_argument('--bitdepth', type=int, default=12, choices=[8, 11, 12, 16],
                        help='Bit depth of input images')
    
    # Image dimensions
    parser.add_argument('--n_z', type=int, default=80,
                        help='Number of z-slices per volume/timepoint.')
    parser.add_argument('--z_range', type=parse_slice, default='3,80',
                        help='Start,end range of z-slices to be included.')
    parser.add_argument('--spacing', type=parse_float_list, default=[[0.54, 0, 0], [0, 0.54, 0], [0, 0, 0.54]],
                        help='Final voxel dimension for saving (e.g. NRRD).')
    
    # Performance settings
    parser.add_argument('--global_t_start', type=int, default=0,
                        help='Timepoints (volumes) to process per chunk.')
    parser.add_argument('--global_t_end', type=int, default=1600,
                        help='Timepoints (volumes) to process per chunk.')    
    parser.add_argument('--chunk_size', type=int, default=32,
                        help='Timepoints (volumes) to process per chunk.')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of workers for parallel file IO.')    
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU device number to use.')

    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Using device:", jax.devices())
    
    preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        noise_path=args.noise_path,
        blank_dir=args.blank_dir,
        bg_percentile=args.bg_percentile,
        global_t_start=args.global_t_start, 
        global_t_end=args.global_t_end, 
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        n_z=args.n_z,
        z_range=args.z_range,
        bitdepth=args.bitdepth,
        binsize=args.binsize,
        save_as=args.save_as,
        spacing=args.spacing
    )

if __name__ == '__main__':
    main()