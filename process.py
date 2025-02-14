import jax.numpy as jnp

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
    

def bin_and_subtract(chunk_data, noise_data, binsize, bitdepth):

    # Sum over bins
    chunk_data_binned = bin_xy(chunk_data, binsize)

    """
    Subtract background jnp array from real data jnp array, where (C, X, Y) dimensions are matching in size

    Parameters:
    - chunk_data: jnp.array of shape  (T, Z, C, X, Y) → Input image batch
    - noise_data: jnp.array of shape (C, X, Y) → Background noise for each channel

    Returns:
    - chunk data: jnp.array of shape  (T, Z, C, X, Y)
    """

    assert chunk_data_binned.shape[-2:] == noise_data.shape[-2:], "Real data and noise data should have identical XY dimensions"   
    noise_expanded = jnp.expand_dims(noise_data, axis=(0, 2)) # (C, X, Y) -> (1, C, 1, X, Y)
    noise_subtracted = chunk_data_binned - noise_expanded # matched to (T, C, Z, X, Y)
    clipped = jnp.clip(noise_subtracted, 0, 2 ** bitdepth) # Clip values to fit in bit range
    
    return jnp.transpose(clipped, axes=(0,1,4,3,2)) # Rearrange axes
    