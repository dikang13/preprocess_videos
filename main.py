import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import nd2
import jax
import jax.numpy as jnp
import tifffile
import re
import dask
from fileio import parallel_save_outputs
from transform import background_subtract_channels, bin_and_transpose_channels
from noise import get_avg_noise
from utils import parse_slice, parse_list, parse_float_list, THREAD_POOL_SIZE

def preprocess(
    input_path, output_dir, noise_path, blank_dir,
    chunk_size,
    n_z, x_range, y_range, z_range, channels,
    bitdepth,
    binsize,
    save_as,
    spacing
):
        
    # Load or prepare noise data
    noise_data = None
    subtract_800 = False
    if noise_path:
        noise_data = jnp.array(tifffile.imread(str(noise_path)), dtype=jnp.uint16)
        print(f"Loaded noise data of shape {noise_data.shape} from {noise_path}")
    elif blank_dir:
        noise_data, savepath = get_avg_noise(blank_dir, x_range, y_range)
        print(f"Computed noise data of shape {noise_data.shape} by averaging all frames in {blank_dir}. "
              f"In the future, you can reuse this noise data by setting noise_path to {savepath}")
    else:
        # if background subtraction is not performed, the lowest superpixel value will be ~900, which needs to be subtracted 
        subtract_800 = True
        print("IMPORTANT: Neither noise_path nor blank_dir is provided. Background is assumed to be 800 for all binned superpixels.")
        
    # Create output directories if saving as NRRD
    if save_as == "nrrd":
        nrrd_dir = Path(output_dir) / "NRRD"
        mip_dir = Path(output_dir) / "MIP"
        nrrd_dir.mkdir(parents=True, exist_ok=True)
        mip_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load all frames into CPU memory at once as dask array   
    with nd2.ND2File(input_path) as nd2_file:
        all_frames = nd2_file.to_dask()
    nd2_file.close()
    n_pages = all_frames.shape[0]

    # Reshape into 5D array
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        all_frames = all_frames.reshape(all_frames.shape[0] // n_z, n_z, *all_frames.shape[1:])  # [T, Z, C, H, W]
        all_frames = all_frames.transpose(0,2,1,3,4)   # [T, C, Z, H, W]
    t_size = all_frames.shape[0]

    # Step 2: Process frames in JAX-sized chunks
    n_chunks = (t_size + chunk_size - 1) // chunk_size

    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as pool:
        for chunk_idx in tqdm(range(n_chunks)):
            t_start = chunk_idx * chunk_size
            t_end = min(t_start + chunk_size, t_size)
            current_chunk_size = t_end - t_start

            # Extract chunk from preloaded data
            chunk_data = all_frames[t_start:t_end].compute() # bring dask chunk into memory
            print(f"Processing chunk {chunk_idx+1}/{n_chunks}, shape: {chunk_data.shape}")

            chunk_data = jnp.array(chunk_data, dtype=jnp.int16)

            # Apply transformations
            chunk_data = background_subtract_channels(
                batch_data_5d=chunk_data,
                noise_data_3d=noise_data,
                x_range=x_range,
                y_range=y_range,
                bitdepth=bitdepth
            )

            chunk_data = bin_and_transpose_channels(
                batch_data_5d=chunk_data,
                binsize=binsize,
                bitdepth=bitdepth,
                subtract_800=subtract_800
            )

            # Step 4: Parallelized file output
            n_channels = chunk_data.shape[1]
            save_tasks = []

            for c in range(n_channels):
                out = chunk_data[:, c, ...]  # shape [T, Z, H, W]

                if save_as == 'nrrd':
                    processed_data = out.astype(jnp.uint16)
                    for t_offset in range(current_chunk_size):
                        t = t_start + t_offset

                        # Generate filenames
                        output_path_str = str(output_dir)
                        prefix_matches = re.findall(r'202\d{1}-\d{2}-\d{2}-\d{2}', output_path_str)
                        prefix = prefix_matches[-1] if prefix_matches else ''
                        basename = f"{prefix}_t{t+1:04d}_ch{c+1}"

                        nrrd_path = Path(output_dir) / "NRRD" / f"{basename}.nrrd"
                        mip_path  = Path(output_dir) / "MIP"  / f"{basename}.png"

                        # Extract volume and reorder
                        vol_data = processed_data[t_offset, :, :, :]
                        vol_data = jnp.transpose(vol_data, axes=(1, 0, 2))

                        save_tasks.append((nrrd_path, mip_path, vol_data, spacing))

                elif save_as == 'tif':
                    processed_data = out.astype(jnp.int16)
                    out_path = os.path.join(output_dir, f"processed_video_ch{c+1}.tif")
                    tifffile.imwrite(out_path, processed_data, bigtiff=True)
                    print(f"Processed video is saved as one tif per channel here: {out_path}")

                else:
                    raise ValueError("You must save the output as either 'tif' or 'nrrd'!")

                del out, processed_data  # Free memory

            # Execute saving in parallel
            list(pool.map(parallel_save_outputs, save_tasks))

            del chunk_data  # Free memory
    del all_frames  # Clear major chunk from memory


def main():
    parser = argparse.ArgumentParser(description='Process microscopy image data with background subtraction and binning.')
    
    # Input/Output paths
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input file path, e.g., /path/to/data.nd2')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory, e.g., /path/to/output')
    parser.add_argument('--noise_path', type=str, default=None,
                        help='Path to noise reference file')
    parser.add_argument('--blank_dir', type=str, default=None,
                        help='Folder path to multiple blank images acquired when camera is capped')
    
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
    parser.add_argument('--x_range', type=parse_slice, default='0,966',
                        help='Start,end range along x dimension.')
    parser.add_argument('--y_range', type=parse_slice, default='0,630',
                        help='Start,end range along y dimension.')
    parser.add_argument('--z_range', type=parse_slice, default='3,80',
                        help='Start,end range of z-slices to include.')
    parser.add_argument('--channels', type=parse_list, default='1,2',
                        help='Channels to process as "1,2" or "[1,2]"')
    parser.add_argument('--spacing', type=parse_float_list, default='0.54,0.54,0.54',
                        help='Final voxel dimension for saving (e.g. NRRD).')
    
    # Performance settings
    parser.add_argument('--chunk_size', type=int, default=50,
                        help='Timepoints (volumes) to process per chunk.')
    parser.add_argument('--gpu', type=int, default=2,
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