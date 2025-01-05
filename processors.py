# Core processing function

import jax
import jax.numpy as jnp
from numba import cuda

@jax.jit
def process_batch_jax(batch_data, noise_data=None):
    """Process a batch of images using JAX with automatic batching"""
    # First crop 84 z-slices to include only slices 3:80
    batch_data = batch_data[:, 3:80, :630, :966]  # Shape: (batch, 77, 630, 966)

    # Prepare averaged noise data - a single frame of shape (630, 966)
    if noise_data is not None:
        # Add batch and z-slice dims to noise
        noise_expanded = jnp.expand_dims(noise_data, axis=(0, 1))  # Shape: (1, 1, 630, 966)
        # Tile noise to match batch dimensions (batch, 77, 630, 966)
        noise_tiled = jnp.tile(noise_expanded, (batch_data.shape[0], batch_data.shape[1], 1, 1))
        
        # Perform noise subtraction to get rid of fixed pattern noise from raw video
        batch_data = jnp.maximum(batch_data, noise_tiled) - noise_tiled
    
    # Clip values for 12-bit video
    batch_data = jnp.clip(batch_data, 0, 4096)
    
    # Reshape for 3x3 binning
    batch_size, n_slices, h, w = batch_data.shape  # n_slices=77, h=630, w=966
    h_bins, w_bins = h // 3, w // 3  # 210, 322
    reshaped = batch_data.reshape(batch_size, n_slices, h_bins, 3, w_bins, 3)
    
    # Sum over 3x3 bins
    binned = jnp.sum(reshaped, axis=(3, 5))  # Shape: (batch, 77, 210, 322)
    
    # Transpose to get (batch, 322, 210, 77)
    binned = jnp.transpose(binned, (0, 3, 2, 1))
    
    return binned.astype(jnp.uint16)