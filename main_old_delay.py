import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

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
# from process import bin_and_subtract, bin_xy
from noise import get_noise_data, compute_uniform_noise_data
from utils import parse_slice, parse_list, parse_float_list
import gc
import psutil
import time
from datetime import datetime
from functools import partial
import jax
from jax import lax, jit
jax.clear_caches()
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', False)  # Use float32/int32 by default
import jax.numpy as jnp

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
    
    # Quick dimension check
    # print("Binned chunk shape: {}", chunk_binned.shape)
    # print("Binned noise shape: {}", noise_binned.shape)
    
    # Expand dimensions for broadcasting - more memory efficient
    noise_expanded = jnp.expand_dims(noise_binned, axis=(0, 1))
    
    # Subtract and clip in one operation to minimize intermediate allocations
    result = jnp.clip(chunk_binned - noise_expanded, 0, 2 ** bitdepth) # (T, Z, C, X, Y)
    # print("Result shape: {}", result.shape)
    
    # Transpose and convert to int16
    return jnp.transpose(result, axes=(0, 2, 4, 3, 1)) # (T, C, Y, X, Z)


def print_mem_usage():
    process = psutil.Process(os.getpid())
    print(f"Process Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

@dask.delayed
def process_chunk_delayed(all_frames, 
                          page_start, page_end, t_start, t_end, 
                          n_z, n_channels, n_x, n_y, z_range, 
                          gpu, noise_data, binsize, bitdepth, 
                          base_nrrd_path, prefix, num_workers):
    cpu_data = all_frames[page_start:page_end].compute()
    cpu_data = cpu_data.reshape(t_end - t_start, n_z, n_channels, n_x, n_y)  # 5D array (T, Z, C, X, Y)
    cpu_data = cpu_data[:, z_range, :, :, :]  # Ignore the z-slices with motion artifacts
    print(f"===input is sent to GPU at {datetime.now().strftime('%H:%M:%S')}===")

    with jax.default_device(gpu):  # Explicitly manage device context
        gpu_data = jax.device_put(jnp.asarray(cpu_data))
        gpu_output = bin_and_subtract(gpu_data, noise_data, binsize, bitdepth)
        gpu_output.block_until_ready()
        cpu_output = jax.device_get(gpu_output)
        
        # Ensure GPU memory is freed
        del gpu_data
        del gpu_output
        jax.device_get(jnp.ones(1))  # Force sync with device

    print(f"===output is taken down to CPU at {datetime.now().strftime('%H:%M:%S')}===")

    # Parallel NRRD file saving
    tasks = [
        (
            base_nrrd_path / f"{prefix}_t{t_start + t_offset + 1:04d}_ch{c+1}.nrrd",
            cpu_output[t_offset, c],  # Extract per timepoint per channel
            spacing
        )
        for t_offset in range(cpu_output.shape[0])  # Match time dimension
        for c in range(n_channels)
    ]
    batch_save_nrrd(tasks, num_workers)  # Adjust workers based on CPU

    # Ensure CPU memory is freed
    del cpu_data
    del cpu_output
    return f"===time points {t_start}-{t_end} processed at {datetime.now().strftime("%H:%M:%S")}==="

# def process_chunk(cpu_data, t_start, t_end, n_z, n_channels, n_x, n_y, z_range, gpu, noise_data, binsize, bitdepth, base_nrrd_path, prefix, num_workers):
#     cpu_data = cpu_data.reshape(t_end-t_start, n_z, n_channels, n_x, n_y) # 5D array (T, Z, C, X, Y)
#     cpu_data = cpu_data[:, z_range, :, :, :] # ignore the z-slices with lots of motion artifacts
#     print(f"=== input is sent to GPU at {datetime.now().strftime("%H:%M:%S")}===")

#     with jax.default_device(gpu):  # Explicitly manage device context
#         gpu_data = jax.device_put(jnp.asarray(cpu_data))
#         gpu_output = bin_and_subtract(gpu_data, noise_data, binsize, bitdepth)
#         gpu_output.block_until_ready()
#         cpu_output = jax.device_get(gpu_output)
#         del gpu_data 
#         del gpu_output
#         jax.device_get(jnp.ones(1))  # Force sync with device
#     print(f"=== output is taken down to CPU at {datetime.now().strftime("%H:%M:%S")}===")

#     # Parallel NRRD file saving
#     tasks = [
#         (
#             base_nrrd_path / f"{prefix}_t{t_start + t_offset + 1:04d}_ch{c+1}.nrrd",
#             cpu_output[t_offset, c],  # Extract per timepoint per channel
#             spacing
#         )
#         for t_offset in range(current_chunk_size)
#         for c in range(n_channels)
#     ]
#     batch_save_nrrd(tasks, num_workers)  # Adjust workers based on CPU
#     del cpu_data
#     del cpu_output

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
    print(f"===entire process starts at {datetime.now().strftime("%H:%M:%S")}===")
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
        
    # Load ND2 data as a lazy Dask array
    with nd2.ND2File(input_path) as nd2_file:
        all_frames = nd2_file.to_dask().astype('int16') # 4D array (T*Z, C, X, Y)
    
    n_pages, n_channels, n_x, n_y = all_frames.shape
    t_size = n_pages // n_z

    # Reshape Dask array for chunking
    # with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    #     all_frames = all_frames.reshape(t_size, n_z, n_channels, n_x, n_y) # 5D array (T, Z, C, X, Y)
    #     all_frames = all_frames[:, z_range, :, :, :].astype('int16') # ignore the z-slices with lots of motion artifacts

    # If blank frames are not passed in, a uniform background assumed for subtraction
    if noise_data is None:
        noise_data = all_frames[:800].compute()
        noise_data = noise_data.reshape(10, n_z, n_channels, n_x, n_y)
        noise_data = noise_data.transpose(0, 2, 1, 3, 4)  # 5D array (T, C, Z, X, Y)
        noise_data = compute_uniform_noise_data(noise_data, bg_percentile) # compute noise per channel from the first 5 time points
        noise_data = bin_xy(noise_data, binsize) # bin noise data to superpixels
        noise_data = jnp.floor(noise_data / 100) * 100  # Round down to nearest hundred
        print(f"Uniform background of shape {noise_data.shape} and value {noise_data[:,0,0]} is subtracted from each pixel in the respective channels.")
    else:
        noise_data = bin_xy(noise_data, binsize) # bin noise data to superpixels

    # Load binned noise to GPU
    # f"{base_nrrd_path}/uniform_noise.tif",
    noise_data = jax.device_put(noise_data.astype(jnp.int16))
    # print(f"Noise_data is {noise_data.dtype}.")
        
    # Perform jax array computations in series, and file saving in parallel   
    n_chunks = (t_size + chunk_size - 1) // chunk_size
    print(f"Input of dimensions t={t_size}, z={n_z}, c={n_channels}, x={n_x}, y={n_y} is divided into {n_chunks} chunks:")
    print_mem_usage()

    params = {
    "n_z": n_z,
    "n_channels": n_channels,
    "n_x": n_x,
    "n_y": n_y,
    "z_range": z_range,
    "gpu": gpu,
    "noise_data": noise_data,  # Placeholder
    "binsize": binsize,
    "bitdepth": bitdepth,
    "base_nrrd_path": base_nrrd_path,
    "prefix": prefix,
    "num_workers": num_workers,
}
    delayed_tasks = []
    
    for chunk_idx in range(n_chunks):           
        t_start = chunk_idx * chunk_size
        t_end = min(t_start + chunk_size, t_size)
        # current_chunk_size = t_end - t_start
        # cpu_data = all_frames[t_start:t_end].compute() # Bring a small chunk of the Dask array to RAM
        # initial_gpu_mem = gpu.memory_stats()["bytes_in_use"]
        page_start = t_start * n_z
        page_end = t_end * n_z
        # print(f"Starting chunk {chunk_idx+1}/{n_chunks}")

        task = process_chunk_delayed(all_frames, page_start, page_end, t_start, t_end, **params)
        delayed_tasks.append(task)
    dask.compute(*delayed_tasks)
    gc.collect()
        
        # with dask.config.set(scheduler='synchronous'):
        #     cpu_data = all_frames[page_start:page_end].compute()
        #     cpu_data = cpu_data.reshape(t_end-t_start, n_z, n_channels, n_x, n_y) # 5D array (T, Z, C, X, Y)
        #     cpu_data = cpu_data[:, z_range, :, :, :] # ignore the z-slices with lots of motion artifacts
        # print(f"=== input is sent to GPU at {datetime.now().strftime("%H:%M:%S")}===")

        # with jax.default_device(gpu):  # Explicitly manage device context
        #     data = jax.device_put(jnp.asarray(cpu_data))
        #     # print(f"data is {data.dtype}.")
        #     output = bin_and_subtract(data, noise_data, binsize, bitdepth)
        #     output.block_until_ready()
        #     cpu_output = jax.device_get(output)
        #     del data 
        #     del output
        #     jax.device_get(jnp.ones(1))  # Force sync with device
            
        # # Parallel NRRD file saving
        # print(f"=== output is taken down to CPU at {datetime.now().strftime("%H:%M:%S")}===")
        # cpu_data = cpu_data.copy()
        # del cpu_data
        # tasks = [
        #     (
        #         base_nrrd_path / f"{prefix}_t{t_start + t_offset + 1:04d}_ch{c+1}.nrrd",
        #         cpu_output[t_offset, c],  # Extract per timepoint per channel
        #         spacing
        #     )
        #     for t_offset in range(current_chunk_size)
        #     for c in range(n_channels)
        # ]
        # batch_save_nrrd(tasks, num_workers)  # Adjust workers based on CPU
        # del cpu_output
        # gc.collect()
        # print_mem_usage()
        # final_gpu_mem = gpu.memory_stats()["bytes_in_use"]
        # print(f"Memory difference: {(final_gpu_mem - initial_gpu_mem) / 1024**2:.2f} MB")
    print(f"===entire process finishes at {datetime.now().strftime("%H:%M:%S")}===")
        
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