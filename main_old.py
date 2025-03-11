import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

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
from noise import get_noise_data, compute_uniform_noise_data
from utils import parse_slice, parse_list, parse_float_list
import gc
import psutil
import time
from datetime import datetime
from functools import partial
import jax
from jax import lax, jit
import jax.numpy as jnp

jax.clear_caches()
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', False)  # Use float32/int32 by default

@partial(jit, static_argnums=(1,))
def bin_xy(data, binsize):
    """
    Optimized bin_xy using JAX's lax.reduce_window for better performance on RTX A4000.
    
    Parameters:
    - data: jnp.array of shape (..., X, Y) → Input image
    - binsize: int → Factor by which to downsample (bin) the image
    
    Returns:
    - Binned array of shape (..., X//binsize, Y//binsize)
    """
    # Get dimensions
    h, w = data.shape[-2], data.shape[-1]
    
    # Calculate output dimensions
    h_bins, w_bins = h // binsize, w // binsize
    
    # Crop to exact multiple of binsize
    data = data[..., :h_bins * binsize, :w_bins * binsize]
    
    # Define window dimensions: sliding windows of size (binsize x binsize)
    # with strides of the same size for non-overlapping windows
    window_dims = (1,) * (data.ndim - 2) + (binsize, binsize)
    strides = (1,) * (data.ndim - 2) + (binsize, binsize)
    
    # Use reduce_window with sum reduction for efficient binning on GPU
    result = lax.reduce_window(
        data,
        init_value=0,
        computation=lax.add,
        window_dimensions=window_dims,
        window_strides=strides,
        padding='VALID'
    )
    return result # (..., X//binsize, Y//binsize)

@partial(jit, static_argnums=(2, 3))
def bin_and_subtract(chunk_data, noise_binned, binsize, bitdepth):
    """
    Optimized version of bin_and_subtract specifically for RTX A4000 GPU.
    Tailored for batch shape (32, 77, 2, 630, 966).
    
    Parameters:
    - chunk_data: jnp.array of shape (T=32, Z=77, C=2, X=630, Y=966) → Input image batch
    - noise_data: jnp.array of shape (C=2, X=630, Y=966) → Background noise for each channel
    - binsize: int → Factor by which to downsample (bin) the image
    - bitdepth: int → Bit depth for clipping values
    
    Returns:
    - chunk data: jnp.array of shape (T=32, Z=77, C=2, X//binsize, Y//binsize)
    """
    # Apply binning with optimized function
    chunk_binned = bin_xy(chunk_data, binsize)
    
    # Expand dimensions for broadcasting - more memory efficient
    noise_expanded = jnp.expand_dims(noise_binned, axis=(0, 1))
    
    # Subtract and clip in one operation to minimize intermediate allocations
    result = jnp.clip(chunk_binned - noise_expanded, 0, 2 ** bitdepth) # (T, Z, C, X, Y)
    
    # Transpose and convert to int16
    return jnp.transpose(result, axes=(0, 2, 4, 3, 1)) # (T, C, Y, X, Z)


def print_mem_usage():
    process = psutil.Process(os.getpid())
    print(f"Process Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


class OptimizedND2Loader:
    """
    Memory-efficient ND2 file loader using a single Dask array with optimizations
    to prevent memory bloat across iterations.
    """
    def __init__(self, filepath, n_z):
        self.filepath = filepath
        self.n_z = n_z  # Number of z-slices per timepoint
        
        # Create a single Dask array with optimal chunking
        with nd2.ND2File(filepath) as nd2_file:
            # Initialize Dask array with rechunking for better performance
            self.dask_array = nd2_file.to_dask()
            
            # Get metadata
            self.shape = self.dask_array.shape
            self.dtype = self.dask_array.dtype
            self.n_pages, self.n_channels, self.n_x, self.n_y = self.shape
            self.t_size = self.n_pages // self.n_z

    def get_chunk(self, t_start, t_end, z_range=None):
        """
        Get a chunk of data using optimized Dask loading with minimal memory footprint.
        
        Parameters:
        - t_start: Starting timepoint index
        - t_end: Ending timepoint index (exclusive)
        - z_range: slice object for z-range selection
        
        Returns:
        - Numpy array of shape (T, Z, C, X, Y)
        """
        current_chunk_size = t_end - t_start
        page_start = t_start * self.n_z
        page_end = t_end * self.n_z
        
        print(f"Loading chunk t={t_start} to t={t_end} using optimized Dask")
        data = self.dask_array[page_start:page_end].compute()
        # Reshape to expected dimensions
        data = data.reshape(current_chunk_size, self.n_z, self.n_channels, self.n_x, self.n_y)
        data = data[:, z_range, :, :, :].astype('int16')
        _ = gc.collect()
        return data


def preprocess(
    input_path, output_dir, 
    noise_path, blank_dir, bg_percentile,
    chunk_size, num_workers,
    n_z, z_range,
    bitdepth,
    binsize,
    save_as,
    spacing
):
    print(f"===entire process starts at {datetime.now().strftime('%H:%M:%S')}===")
    gpu = jax.devices("gpu")[0]
    
    # Load noise data if a .tif file is provided, or compute noise by averaging frames in blank_dir
    noise_data = None

    if blank_dir:
        noise_path = get_noise_data(blank_dir)
        print(f"Averaged blank frames from {blank_dir} is saved under {noise_path}.")
    
    if noise_path:
        noise_data = jnp.array(tifffile.imread(str(noise_path)))
        print(f"Loaded noise data of shape {noise_data.shape} from {noise_path}")
    else:
        print("⚠️⚠️⚠️ WARNING: No blank frames provided. Assuming uniform background for all pixels.⚠️⚠️⚠️")

    # Create output directories
    Path(output_dir, "NRRD").mkdir(parents=True, exist_ok=True)
    prefix = re.findall(r'202\d{1}-\d{2}-\d{2}-\d{2}', str(output_dir))
    prefix = prefix[-1] if prefix else ''
    base_nrrd_path = Path(output_dir) / "NRRD"
    
    # Initialize optimized ND2 loader
    nd2_loader = OptimizedND2Loader(input_path, n_z)
    n_channels, n_x, n_y = nd2_loader.n_channels, nd2_loader.n_x, nd2_loader.n_y
    t_size = nd2_loader.t_size
    
    # If blank frames are not passed in, a uniform background assumed for subtraction
    if noise_data is None:
        # Get first few frames for noise estimation
        first_frames = nd2_loader.get_chunk(0, 5)
        noise_data = first_frames.reshape(5, n_z, n_channels, n_x, n_y)
        noise_data = noise_data.transpose(0, 2, 1, 3, 4)  # 5D array (T, C, Z, X, Y)
        noise_data = compute_uniform_noise_data(noise_data, bg_percentile) # compute noise per channel from the first 5 time points
        noise_data = bin_xy(noise_data, binsize) # bin noise data to superpixels
        noise_data = jnp.floor(noise_data / 100) * 100  # Round down to nearest hundred
        print(f"Uniform background of shape {noise_data.shape} and value {noise_data[:,0,0]} is subtracted from each pixel in the respective channels.")
        # Clean up the first frames
        del first_frames
        gc.collect()
    else:
        noise_data = bin_xy(noise_data, binsize) # bin noise data to superpixels

    # Load binned noise to GPU
    noise_data = jax.device_put(noise_data.astype(jnp.int16))
    
    # Perform jax array computations in series, and file saving in parallel
    n_chunks = (t_size + chunk_size - 1) // chunk_size
    print(f"Input of dimensions t={t_size}, z={n_z}, c={n_channels}, x={n_x}, y={n_y} is divided into {n_chunks} chunks:")
    print_mem_usage()
    
    # Process each chunk
    for chunk_idx in tqdm(range(n_chunks)):
        t_start = chunk_idx * chunk_size
        t_end = min(t_start + chunk_size, t_size)
        current_chunk_size = t_end - t_start
        print(f"Starting chunk {chunk_idx+1}/{n_chunks}")
        print_mem_usage()
        
        # Get chunk data using optimized Dask loader
        print(f"===retrieving input via Dask at {datetime.now().strftime('%H:%M:%S')}===")
        cpu_data = nd2_loader.get_chunk(t_start, t_end, z_range)
        
        print(f"===input is available for processing at {datetime.now().strftime('%H:%M:%S')}===")
        print_mem_usage()
        
        # Send to GPU and process
        print(f"===input is sent to GPU at {datetime.now().strftime('%H:%M:%S')}===")
        with jax.default_device(gpu):  # Explicitly manage device context
            gpu_data = jax.device_put(jnp.asarray(cpu_data))
            gpu_output = bin_and_subtract(gpu_data, noise_data, binsize, bitdepth)
            gpu_output.block_until_ready()
            cpu_output = jax.device_get(gpu_output)
            jax.device_get(jnp.ones(1))  # Force sync with device
            del gpu_data 
            del gpu_output
            
        # Parallel NRRD file saving
        print(f"===output is taken down to CPU at {datetime.now().strftime('%H:%M:%S')}===")
        
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
        
        # Immediate cleanup after processing
        del cpu_data
        del cpu_output
        time.sleep(1)
        
        print_mem_usage()
        print(f"===chunk processing finishes at {datetime.now().strftime('%H:%M:%S')}===")
    
    print(f"===entire process finishes at {datetime.now().strftime('%H:%M:%S')}===")
        
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
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='Timepoints (volumes) to process per chunk.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for parallel file IO.')    
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU device number to use.')

    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Using device:", jax.devices())  

    # precompile kernels
    dummy_chunk = jnp.ones((32, 77, 2, 630, 966)).astype(jnp.int16)  # Slightly smaller to avoid edge effects
    dummy_noise = jnp.ones((2, 210, 322)).astype(jnp.int16)
    _ = bin_and_subtract(dummy_chunk, dummy_noise, 3, 12)
    print("Kernels compiled and ready!")
    
    preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        noise_path=args.noise_path,
        blank_dir=args.blank_dir,
        bg_percentile=args.bg_percentile,
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