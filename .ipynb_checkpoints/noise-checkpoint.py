import numpy as np
import jax.numpy as jnp
import jax
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
        # current_shape = tifffile.imread(filepath).shape
        first_img = tifffile.imread(filepath)
    
        # Determine reference shape and channels
        if first_img.ndim == 2:
            current_shape = first_img.shape
        elif first_img.ndim == 3:
            if first_img.shape[-1] in [1, 2]:
                current_shape = (first_img.shape[-1], *first_img.shape[:-1])
            else:
                current_shape = first_img[0].shape
        elif first_img.ndim == 4:
            current_shape = (first_img.shape[-1], *first_img.shape[1:-1])
        
    if current_shape[-2:][0] < reference_shape[-2:][0] or current_shape[-2:][1] < reference_shape[-2:][1]:
       raise ValueError(
           f"File {filepath} has last two dimensions {current_shape[-2:]} which are smaller than the "
           f"reference shape's last two dimensions {reference_shape[-2:]}. Files must be at least as large as reference."
       )

def process_chunk(frames, prev_mean, total_frames_before, y_range, x_range):
    """Process a chunk of frames using JAX acceleration"""
    mean = prev_mean
    for i in range(frames.shape[0]):
        curr_frame = i + total_frames_before + 1
        if frames.ndim == 4:  # Multi-channel
            mean += (frames[i, :, y_range, x_range] - mean) / curr_frame
        else:  # Single channel
            mean += (frames[i, y_range, x_range] - mean) / curr_frame
    return mean

# Create JIT-compiled version
process_chunk_jit = jax.jit(process_chunk, static_argnames=['y_range', 'x_range'])


def get_avg_noise_nd2(blank_files: [str], savepath: str, x_range: slice, y_range: slice, chunk_size: int = 100):
    """JAX-accelerated computation of averaged 2D noise array from .nd2 files"""
    # Get reference shape from first file
    first_file_dim = nd2dim(blank_files[0])
    
    # Determine reference shape and number of channels
    n_channels = 2 if first_file_dim[-1] == 2 else 1
    
    # Initialize separate noise data arrays for each channel
    noise_data = np.zeros((n_channels, first_file_dim[1], first_file_dim[0]), dtype=np.float32)
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
            
            # Add necessary dimensions if needed
            if frames.ndim == 2:
                frames = frames[np.newaxis, np.newaxis, ...]
            elif frames.ndim == 3:
                frames = frames[:, np.newaxis, ...]
            
            # Process in chunks for this channel
            for i in range(0, len(frames), chunk_size):
                chunk = jnp.array(frames[i:i + chunk_size])
                channel_data = np.array(process_chunk_jit(
                    chunk,
                    jnp.array(noise_data[ch:ch+1]),
                    total_frames_per_channel[ch],
                    y_range,
                    x_range
                ))
                noise_data[ch] = channel_data[0]
                total_frames_per_channel[ch] += len(chunk)
    
    noise_data = np.swapaxes(noise_data, -2, -1)
    tifffile.imwrite(savepath, noise_data)
    return noise_data


def get_avg_noise_tif(blank_files: [str], savepath: str, x_range: slice, y_range: slice, chunk_size: int = 100):
    """JAX-accelerated computation of averaged 2D noise array from .tif files"""
    reference_shape = (y_range.stop, x_range.stop)
    first_img = tifffile.imread(blank_files[0])
    
    # Determine reference shape and channels
    if first_img.ndim == 2:
        n_channels = 1
    elif first_img.ndim == 3:
        if first_img.shape[-1] in [1, 2]:
            n_channels = first_img.shape[-1]
        else:
            n_channels = 1
    elif first_img.ndim == 4:
        n_channels = first_img.shape[-1]
    
    # Initialize noise data array
    if n_channels == 1:
        noise_data = np.zeros(reference_shape[-2:] if len(reference_shape) == 2 
                            else reference_shape[-2:], dtype=np.float32)
    else:
        noise_data = np.zeros((n_channels, reference_shape[0], reference_shape[1]), dtype=np.float32)
    
    total_frames = 0

    for filepath in tqdm(blank_files, desc="Processing noise files"):
        img = tifffile.imread(filepath)
        
        # Standardize dimensions
        if img.ndim == 2:
            img = img[np.newaxis, np.newaxis, ..., ::-1] if img.shape[1] > img.shape[0] else img[np.newaxis, np.newaxis, ...]
        elif img.ndim == 3:
            if img.shape[-1] in [1, 2]:
                img = img[np.newaxis, ...]
                img = np.transpose(img, (0, 3, 1, 2))
                if img.shape[2] > img.shape[3]:
                    img = np.transpose(img, (0, 1, 3, 2))
            else:
                img = img[..., np.newaxis]
                img = np.transpose(img, (0, 3, 1, 2))
                if img.shape[2] > img.shape[3]:
                    img = np.transpose(img, (0, 1, 3, 2))
        elif img.ndim == 4:
            img = np.transpose(img, (0, 3, 1, 2))
        
        # Process in chunks
        for i in range(0, len(img), chunk_size):
            chunk = jnp.array(img[i:i + chunk_size])
            noise_data = np.array(process_chunk(chunk, jnp.array(noise_data), 
                                              total_frames, y_range, x_range))
            total_frames += len(chunk)

    noise_data = np.swapaxes(noise_data, -2, -1)
    tifffile.imwrite(savepath, noise_data)
    return noise_data


def get_avg_noise(blank_dir: str, x_range: slice, y_range: slice, chunk_size: int = 100):
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
    for filepath in blank_files:
        validate_file_consistency(filepath, (y_range.stop, x_range.stop), ext)

    if ext == 'tif':
        noise_data = get_avg_noise_tif(blank_files, savepath, x_range, y_range, chunk_size)
    else:
        noise_data = get_avg_noise_nd2(blank_files, savepath, x_range, y_range, chunk_size)
        
    return noise_data, savepath