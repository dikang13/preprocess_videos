import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from nd2reader import ND2Reader
import tifffile
import os
from tqdm import tqdm
from fileio import nd2dim

def validate_file_consistency(filepath: str, reference_shape: tuple, file_type: str):
    """Validate that a file matches the reference dimensions in the last two dimensions"""
    if file_type == 'nd2':
        with ND2Reader(filepath) as images:
            current_shape = images[0].shape
    else:  # tif
        first_img = tifffile.imread(filepath)
        current_shape = first_img.shape[-2:]  # Get last two dimensions
    
    if current_shape[-2:] < reference_shape[-2:]:
       raise ValueError(
           f"File {filepath} has last two dimensions {current_shape[-2:]} which are smaller than the "
           f"reference shape's last two dimensions {reference_shape[-2:]}. Files must be at least as large as reference."
       )

def process_chunk(frames, prev_mean, total_frames_before):
    """Process a chunk of frames using JAX acceleration"""
    mean = prev_mean
    for i in range(frames.shape[0]):
        curr_frame = i + total_frames_before + 1
        mean += (frames[i, ...] - mean) / curr_frame
    return mean
    

def get_avg_noise_nd2(blank_files: [str], savepath: str, chunk_size: int = 160):
    """JAX-accelerated computation of averaged 2D noise array from .nd2 files"""
    first_file_dim = nd2dim(blank_files[0])
    n_channels = 2 if first_file_dim[-1] == 2 else 1
    
    # Get dimensions dynamically
    X, Y = first_file_dim[1], first_file_dim[0]

    # Initialize separate noise data arrays for each channel
    noise_data = np.zeros((n_channels, X, Y), dtype=np.float32)
    total_frames_per_channel = np.zeros(n_channels, dtype=int)
    
    # Process files with progress bar
    for filepath in tqdm(blank_files, desc="Processing noise files"):
        frames_by_channel = [[] for _ in range(n_channels)]
        
        with ND2Reader(filepath) as images:
            for frame_idx in range(len(images)):
                for ch in range(n_channels):
                    frame = images.get_frame_2D(c=ch, t=frame_idx, z=0)
                    frames_by_channel[ch].append(frame)
        
        # Process each channel independently
        for ch in range(n_channels):
            frames = np.array(frames_by_channel[ch])

            # Process in chunks for this channel
            for i in range(0, len(frames), chunk_size):
                chunk = jnp.array(frames[i:i + chunk_size])
                channel_data = np.array(process_chunk(
                    chunk,
                    jnp.array(noise_data[ch:ch+1]),
                    total_frames_per_channel[ch]
                ))
                noise_data[ch] = channel_data[0]
                total_frames_per_channel[ch] += len(chunk)

    noise_data = jnp.rint(noise_data).astype('uint16')  # round to the nearest integer
    tifffile.imwrite(savepath, noise_data)
    return noise_data


def get_avg_noise_tif(blank_files: [str], savepath: str, chunk_size: int = 160):
    """JAX-accelerated computation of averaged 2D noise array from .tif files"""
    first_img = tifffile.imread(blank_files[0])
    
    # Determine reference shape dynamically
    if first_img.ndim == 2:
        n_channels = 1
        X, Y = first_img.shape
    elif first_img.ndim == 3:
        if first_img.shape[-1] in [1, 2]:
            n_channels = first_img.shape[-1]
            X, Y = first_img.shape[:-1]
        else:
            n_channels = 1
            X, Y = first_img.shape
    elif first_img.ndim == 4:
        n_channels = first_img.shape[-1]
        X, Y = first_img.shape[1:-1]

    # Initialize noise data array
    noise_data = np.zeros((n_channels, X, Y), dtype=np.float32)
    total_frames = 0

    for filepath in tqdm(blank_files, desc="Processing noise files"):
        img = tifffile.imread(filepath)

        # Standardize dimensions
        if img.ndim == 2:
            img = img[np.newaxis, np.newaxis, ...]
        elif img.ndim == 3:
            if img.shape[-1] in [1, 2]:
                img = img[np.newaxis, ...]
                img = np.transpose(img, (0, 3, 1, 2))
            else:
                img = img[..., np.newaxis]
                img = np.transpose(img, (0, 3, 1, 2))
        elif img.ndim == 4:
            img = np.transpose(img, (0, 3, 1, 2))
        
        # Process in chunks
        for i in range(0, len(img), chunk_size):
            chunk = jnp.array(img[i:i + chunk_size])
            noise_data = jnp.array(process_chunk(chunk, jnp.array(noise_data), total_frames))
            total_frames += len(chunk)

    noise_data = jnp.rint(noise_data).astype('uint16')  # round to the nearest integer
    tifffile.imwrite(savepath, noise_data)
    return noise_data


def get_avg_noise(blank_dir: str, chunk_size: int = 160):
    """Compute average noise from blank images"""
    savepath = Path(blank_dir) / 'avg_noise.tif'
    if savepath.is_file():
        os.remove(savepath)

    blank_files_tif = [str(p) for p in Path(blank_dir).glob('*.tif*')]
    blank_files_nd2 = [str(p) for p in Path(blank_dir).glob('*.nd2')]
    assert min([len(blank_files_tif), len(blank_files_nd2)]) == 0, "You can only have one file type in this directory."

    if len(blank_files_tif) > 0:
        blank_files = blank_files_tif
        ext = 'tif'
    else:
        blank_files = blank_files_nd2
        ext = 'nd2'
        
    if not blank_files:
        raise ValueError(f"No .nd2 or .tif files found in directory: {blank_dir}")

    # Validate all files have same dimensions
    reference_shape = tifffile.imread(blank_files[0]).shape[-2:]
    for filepath in blank_files:
        validate_file_consistency(filepath, reference_shape, ext)

    if ext == 'tif':
        noise_data = get_avg_noise_tif(blank_files, savepath, chunk_size)
    else:
        noise_data = get_avg_noise_nd2(blank_files, savepath, chunk_size)
        
    return noise_data, savepath


def compute_uniform_noise_data(chunk_data, bg_percentile):
    """
    Compute a uniform background value for each channel based on a given percentile.

    Parameters:
    - chunk_data: jnp.array of shape (C, Z, X, Y)
    - bg_percentile: float, percentile value to compute background.

    Returns:
    - noise_data: jnp.array of shape (C, X, Y)
    """
    n_channels, n_z, n_x, n_y = chunk_data.shape  # [C, Z, X, Y]

    # Vectorized function to compute percentile per channel
    def compute_uniform_bg(c):
        uniform_bg = jnp.percentile(chunk_data[c, ...], bg_percentile)  # Returns scalar
        uniform_bg = jnp.rint(uniform_bg).astype(jnp.uint16)
        return uniform_bg

    noise_values = jax.vmap(compute_uniform_bg)(jnp.arange(n_channels))  # Shape: (C,)
    noise_data = jnp.broadcast_to(noise_values[:, None, None], (n_channels, n_x, n_y))

    return noise_data
