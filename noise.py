import jax
import jax.numpy as jnp
from pathlib import Path
import tifffile
import os
from tqdm import tqdm
import nd2

def get_noise_data_nd2(blank_files: [str], savepath: str):
    """JAX-accelerated computation of averaged noise array from multiple .nd2 files"""
    # Use the first file in the blank directory to initialize the noise array
    filepath = blank_files[0]
    with nd2.ND2File(filepath) as nd2_file:
        all_frames = nd2_file.to_dask() # [T*Z, C, X, Y]
    n_pages, n_channels, n_x, n_y = all_frames.shape

    # Initialize separate noise data arrays for each channel
    noise_data = None
    
    # Process files with progress bar
    for filepath in tqdm(blank_files, desc="Processing noise files"):
        with nd2.ND2File(filepath) as nd2_file:
            all_frames = nd2_file.to_dask() # [T*Z, C, X, Y]
        assert all_frames.shape[-2] == n_x and all_frames.shape[-1] == n_y, "All blank frames should have the same XY dimension!!!"
        
        n_pages, n_channels, n_x, n_y = all_frames.shape
        to_ram = min(n_pages, 2000) # take the first 2000 frames if the blank files are longer than that
        chunk_data = jnp.array(all_frames[:to_ram].compute())
        chunk_avg = jnp.mean(chunk_data, axis=0) # [C, X, Y]
        if noise_data is None:
            noise_data = chunk_avg
        else:
            noise_data = (noise_data + chunk_avg) /2

    noise_data = jnp.rint(noise_data).astype('int16')  # round to the nearest integer
    print(f"Averaged noise is of dimension {noise_data.shape}")
    tifffile.imwrite(savepath, noise_data) # export tif file for next time
    return savepath


def get_noise_data(blank_dir: str):
    """
    Compute average noise from blank images

    Returns:
    - noise_data: jnp.array of shape (C, X, Y)
    """
    savepath = Path(blank_dir) / 'avg_noise.tif'
    if savepath.is_file():
        os.remove(savepath) # recompute avg_noise.tif from blank files only

    blank_files = [str(p) for p in Path(blank_dir).glob('*.nd2')]
        
    if not blank_files:
        raise ValueError(f"No .nd2 files found in directory: {blank_dir}")
        
    return get_noise_data_nd2(blank_files, savepath)


def compute_uniform_noise_data(chunk_data, bg_percentile):
    """
    Compute a uniform background value for each channel based on a given percentile.

    Parameters:
    - chunk_data: jnp.array of shape (T, C, Z, X, Y)
    - bg_percentile: float, percentile value to compute background.

    Returns:
    - noise_data: jnp.array of shape (C, X, Y)
    """
    n_t, n_channels, n_z, n_x, n_y = chunk_data.shape  # [T, C, Z, X, Y]
    frame_data = jnp.mean(chunk_data, axis=(0,1))  # [C, X, Y]

    # Vectorized function to compute percentile per channel
    def compute_uniform_bg(c):
        uniform_bg = jnp.percentile(frame_data[c, ...], bg_percentile)  # Returns scalar
        uniform_bg = jnp.rint(uniform_bg).astype(jnp.int16)
        return uniform_bg

    noise_values = jax.vmap(compute_uniform_bg)(jnp.arange(n_channels))  # Shape: (C,)
    noise_data = jnp.broadcast_to(noise_values[:, None, None], (n_channels, n_x, n_y))

    return noise_data.astype(jnp.int16) # [C, X, Y]
