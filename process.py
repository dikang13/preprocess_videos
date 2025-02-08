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


# def bin_and_transpose(batch_data, binsize, bitdepth):
#     """
#     Bin images and transpose to final orientation.

#     Parameters:
#     - batch_data: jnp.array of shape (T, C, Z, X, Y)  → Input batch
#     - binsize: int → Factor by which to downsample (bin) the image
#     - bitdepth: int (8, 11, 12, 16) → Bit depth for clipping

#     Returns:
#     - Binned and transposed array of shape (T, C, Z, X//binsize, Y//binsize)
#     """
#     batch_size, n_channels, n_slices, h, w = batch_data.shape
#     h_bins, w_bins = h // binsize, w // binsize

#     # Perform binning using einsum (batch-friendly)
#     binned = jnp.einsum('btcijkl->btcik', 
#                          batch_data.reshape(batch_size, n_channels, n_slices, h_bins, binsize, w_bins, binsize))

#     # Transpose dimensions for proper orientation
#     binned = jnp.transpose(binned, (0, 1, 3, 2, 4))

#     return jnp.clip(binned, 0, 2 ** bitdepth)

def process_images(batch_data, noise_data, bitdepth, binsize):
    """
    Applies background subtraction and binning sequentially on multi-channel image data.

    Parameters:
    - batch_data: jnp.array of shape (T, C, Z, X, Y) → Input image batch
    - noise_data: jnp.array of shape (C, X, Y) → Background noise for each channel
    - bitdepth: int (8, 11, 12, 16) → Bit depth for clipping
    - binsize: int → Factor for downsampling (binning)

    Returns:
    - Processed image batch of shape (T, C, Z, X//binsize, Y//binsize)
    """
    # Step 1: Background Subtraction
    assert batch_data.shape[-2:] == noise_data.shape[-2:], \
        "Real data and noise data should have identical XY dimensions"
    
    noise_expanded = jnp.expand_dims(noise_data, axis=(0, 2))  # Shape: (1, C, 1, X, Y)
    noise_tiled = jnp.broadcast_to(noise_expanded, batch_data.shape)  # Shape: (T, C, Z, X, Y)
    
    batch_data = jnp.clip(batch_data - noise_tiled, 0, 2 ** bitdepth)

    # Step 2: Binning
    batch_size, n_channels, n_slices, h, w = batch_data.shape
    h_bins, w_bins = h // binsize, w // binsize

    binned = jnp.einsum('btcijkl->btcik', 
                         batch_data.reshape(batch_size, n_channels, n_slices, h_bins, binsize, w_bins, binsize))

    # Step 3: Clip to fit in the range for bit depth
    return jnp.clip(binned, 0, 2 ** bitdepth)
