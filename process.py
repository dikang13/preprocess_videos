import jax.numpy as jnp

# # JAX configurations
# jax.config.update('jax_platform_name', 'gpu')
# jax.config.update('jax_disable_jit', True)  # only disable jit if you would like to enter debugging mode


# def background_subtract(batch_data, noise_data, x_range, y_range, bitdepth):
#     """Subtract background from a batch of images with cropping and clipping"""
#     # Crop to given xy dimensions
#     batch_data = batch_data[:, :, x_range, y_range]
    
#     # Expand and tile noise
#     noise_expanded = jnp.expand_dims(noise_data[:x_range, y_range], axis=(0, 2))  # (1, 1, x, y)
#     noise_tiled = jnp.tile(noise_expanded, (batch_data.shape[0], batch_data.shape[1], 1, 1))
    
#     # Subtract and clip
#     assert bitdepth in [8,11,12,16], "IMPORTANT: only 8, 11, 12 or 16-bit images are supported!!!"
#     return jnp.clip(batch_data - noise_tiled, 0, 2 ** bitdepth)


# def bin_and_transpose(batch_data, binsize, bitdepth):
#     """Bin images and transpose to final orientation"""
#     batch_size, n_slices, h, w = batch_data.shape
#     h_bins, w_bins = h // binsize, w // binsize
#     binned = jnp.einsum('bsijkl->bsik', 
#                       batch_data.reshape(batch_size, n_slices, h_bins, binsize, w_bins, binsize))
#     binned = jnp.transpose(binned, (0, 3, 2, 1))
#     return jnp.clip(binned, 0, 2 ** bitdepth)


# def background_subtract(batch_data, noise_data, bitdepth):
#     """
#     Subtract background from a batch of multi-channel images with cropping and clipping.

#     Parameters:
#     - batch_data: jnp.array of shape (T, C, Z, X, Y)  → Input image batch
#     - noise_data: jnp.array of shape (C, X, Y)  → Background noise for each channel
#     - bitdepth: int (8, 11, 12, 16)  → Bit depth for clipping

#     Returns:
#     - Subtracted batch data of shape (T, C, Z, X, Y) with noise removed
#     """
#     # Assert equal X and Y dimensions
#     assert (batch_data.shape[-2] == noise_data.shape[-2] and 
#             batch_data.shape[-1] == noise_data.shape[-1]), \
#         "Real data and noise data should have identical XY dimensions"
    
#     # Tile Noise to Match Batch Shape
#     # (1, C, 1, X, Y) → (T, C, Z, X, Y)
#     noise_expanded = noise_data[:, None, :, :]  # (C, 1, X, Y)
#     noise_expanded = jnp.expand_dims(noise_expanded, axis=0)  # (1, C, 1, X, Y)
#     noise_tiled = jnp.broadcast_to(noise_expanded, batch_data.shape)  # (T, C, Z, X, Y)

#     # Subtract background noise & Clip Values
#     batch_data = jnp.clip(batch_data - noise_tiled, 0, 2 ** bitdepth)


def bin_xy(data, binsize):
    """
    Bin image into superpixels in XY dimensions.

    Parameters:
    - data: jnp.array of shape (..., X, Y) → Input image
    - binsize: int → Factor by which to downsample (bin) the image
    - bitdepth: int (8, 11, 12, 16) → Bit depth for clipping

    Returns:
    - Binned and transposed array of shape (..., X//binsize, Y//binsize)
    """
    h, w = data.shape[-2], data.shape[-1]
    h_bins, w_bins = h // binsize, w // binsize

    # Reshape to group pixels into bins
    data = data[..., :h_bins * binsize, :w_bins * binsize]  # Crop excess rows and columns
    data_binned = data.reshape(*data.shape[:-2], h_bins, binsize, w_bins, binsize)

    # Sum over bins
    data_binned = data_binned.sum(axis=(-3, -1))  # Summing in XY only
    return data_binned


def subtract_bg(chunk_data, noise_data):
    """
    Subtract background jnp array from real data jnp array, where (C, X, Y) dimensions are matching in size

    Parameters:
    - chunk_data: jnp.array of shape  (T, Z, C, X, Y) → Input image batch
    - noise_data: jnp.array of shape (C, X, Y) → Background noise for each channel

    Returns:
    - chunk data: jnp.array of shape  (T, Z, C, X, Y)
    """
    # assert chunk_data.shape[-2:] == noise_data.shape[-2:], "Real data and noise data should have identical XY dimensions"   
    
    noise_expanded = jnp.expand_dims(noise_data, axis=(0, 2)) # (C, X, Y) -> (1, C, 1, X, Y)
    
    return chunk_data - noise_expanded # matched to (T, C, Z, X, Y)
    