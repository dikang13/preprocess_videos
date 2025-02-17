import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import nd2
import jax
import jax.numpy as jnp
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

# Create JIT-compiled version for JAX operations
bin_xy_jit = jax.jit(bin_xy, static_argnames=['binsize'])
# subtract_bg_jit = jax.jit(subtract_bg, static_argnames=['bitdepth'])
bin_and_subtract_jit = jax.jit(bin_and_subtract, static_argnames=['binsize', 'bitdepth'])

def print_mem_usage(label):
    process = psutil.Process(os.getpid())
    print(f"[{label}] Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

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

    # Load noise data if a .tif file is provided, or compute noise by averaging frames in blank_dir
    noise_data = None
    
    if noise_path:
        noise_data = jnp.array(tifffile.imread(str(noise_path)))
        noise_data = bin_xy_jit(noise_data, binsize) # bin to superpixel
        print(f"Loaded noise data of shape {noise_data.shape} from {noise_path}")
    elif blank_dir:
        noise_data, savepath = get_noise_data(blank_dir)
        noise_data = bin_xy_jit(noise_data, binsize) # bin to superpixel
        print(f"Computed noise data of shape {noise_data.shape} by averaging frames from {blank_dir}.")
    else:
        print("⚠️⚠️⚠️ WARNING: No blank frames provided. Assuming uniform background for all pixels.⚠️⚠️⚠️")

    # Create output directories
    if save_as == "nrrd":
        Path(output_dir, "NRRD").mkdir(parents=True, exist_ok=True)
        # Path(output_dir, "MIP").mkdir(parents=True, exist_ok=True)
        prefix = re.findall(r'202\d{1}-\d{2}-\d{2}-\d{2}', str(output_dir))
        prefix = prefix[-1] if prefix else ''
        base_nrrd_path = Path(output_dir) / "NRRD"
        # base_mip_path = Path(output_dir) / "MIP"
    else: # .tif or .tiff
        base_tif_path = Path(output_dir)
        
    # Load ND2 data as a lazy Dask array
    with nd2.ND2File(input_path) as nd2_file:
        all_frames = nd2_file.to_dask() # 4D array (T*Z, C, X, Y)
    
    n_pages, n_channels, n_x, n_y = all_frames.shape
    t_size = n_pages // n_z
    print(f"Input dimensions: t={t_size}, z={n_z}, c={n_channels}, x={n_x}, y={n_y}")

    # Reshape Dask array for chunking
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        all_frames = all_frames.reshape(t_size, n_z, n_channels, n_x, n_y) # 5D array (T, Z, C, X, Y)
        all_frames = all_frames.transpose(0, 2, 1, 3, 4)  # 5D array (T, C, Z, X, Y)
        all_frames = all_frames[:, :, z_range, :, :] # ignore the z-slices with lots of motion artifacts
    
    n_chunks = (t_size + chunk_size - 1) // chunk_size
    dask.config.set(**{'array.slicing.split_large_chunks': True})

    # Perform jax array computations in series, and file saving in parallel
    for chunk_idx in tqdm(range(n_chunks)):
        t_start = chunk_idx * chunk_size
        t_end = min(t_start + chunk_size, t_size)
        current_chunk_size = t_end - t_start

        # Bring a small chunk of the Dask array to RAM
        temp = all_frames[t_start:t_end].compute()
        chunk_data = jnp.array(temp)
        del temp
        # print_mem_usage("After chunk from dask to mem")

        # If blank frames are not passed in, a uniform background assumed for subtraction
        if noise_data is None:
            noise_data = compute_uniform_noise_data(chunk_data, bg_percentile) # compute noise per channel from the first chunk
            noise_data = bin_xy_jit(noise_data, binsize) # bin noise data to superpixels
            noise_data = jnp.floor(noise_data / 100) * 100  # Round down to nearest hundred
            print(f"Uniform background of {noise_data[:,0,0]} will be subtracted from binned superpixels in the respective channels.")
        
        # Apply transformations
        chunk_data = bin_and_subtract_jit(chunk_data, noise_data, binsize, bitdepth) # Bin to superpixel
        chunk_data.block_until_ready()
            
        # Parallel NRRD saving
        if save_as == 'nrrd':
            tasks = [
                (
                    base_nrrd_path / f"{prefix}_t{t_start + t_offset + 1:04d}_ch{c+1}.nrrd",
                    # base_mip_path / f"{prefix}_t{t_start + t_offset + 1:04d}_ch{c+1}.png",
                    chunk_data[t_offset, c, ...],  # Extract per timepoint per channel
                    spacing
                )
                for t_offset in range(current_chunk_size)
                for c in range(n_channels)
            ]
            batch_save_nrrd(tasks, num_workers)  # Adjust workers based on CPU
        
        # Parallel TIF saving
        elif save_as == 'tif':
            tasks = [
                (
                    base_tif_path / f"{prefix}_t{t_start + 1:04d}_to_t{t_offset + 1:04d}_ch{c+1}.tif",
                    chunk_data[:, c, ...]  # Extract per-channel time series
                )
                for c in range(n_channels)
            ]            
            batch_save_tif(tasks, num_workers)
        else:
            raise ValueError("You must save the output as either 'tif' or 'nrrd'!") 
            
        print(f"Finished processing chunk {chunk_idx+1}/{n_chunks}")
        gc.collect()
        print_mem_usage("After garbage collection")

        
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