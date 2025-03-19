import jax
import jax.numpy as jnp
# from functools import partial
# from jax import lax, jit

# @partial(jit, static_argnums=(1,))
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
    data_binned = data.reshape(*data.shape[:-2], h_bins, binsize, w_bins, binsize)
    data_binned = data_binned.sum(axis=(-3, -1))  # Summing over bins in XY only
    
    return data_binned

# @partial(jit, static_argnums=(2, 3))
def bin_and_subtract(chunk_data, noise_binned, binsize, bitdepth):
    """
    Optimized version of bin_and_subtract specifically for RTX A4000 GPU.
    Tailored for batch shape (16, 77, 2, 630, 966).
    
    Parameters:
    - chunk_data: jnp.array of shape (T=16, Z=77, C=2, X=630, Y=966) → Input image batch
    - noise_data: jnp.array of shape (C=2, X=630, Y=966) → Background noise for each channel
    - binsize: int → Factor by which to downsample (bin) the image
    - bitdepth: int → Bit depth for clipping values
    
    Returns:
    - chunk data: jnp.array of shape (T=16, Z=77, C=2, X//binsize, Y//binsize)
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
    result = jnp.transpose(result.astype('int16'), axes=(0, 2, 4, 3, 1)) # (T, C, Y, X, Z)
    print(f"Result shape {result.shape} and {result.dtype}")
    
    # Transpose and convert to int16
    return result
