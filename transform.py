import jax
import jax.numpy as jnp

# JAX configurations
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_disable_jit', True)  # only disable jit if you would like to enter debugging mode

def background_subtract(batch_data, noise_data, x_range, y_range, bitdepth):
    """Subtract background from a batch of images with cropping and clipping"""
    # Crop
    batch_data = batch_data[:, :, x_range, y_range]
    
    # Expand and tile noise
    noise_expanded = jnp.expand_dims(noise_data[x_range, y_range], axis=(0, 1))  # (1, 1, x, y)
    noise_tiled = jnp.tile(noise_expanded, (batch_data.shape[0], batch_data.shape[1], 1, 1))
    
    # Subtract and clip
    assert bitdepth in [8,11,12,16], "IMPORTANT: only 8, 11, 12 or 16-bit images are supported!!!"
    return jnp.clip(batch_data - noise_tiled, 0, 2 ** bitdepth)

# Create JIT-compiled version with static slice arguments
background_subtract_jit = jax.jit(background_subtract, static_argnames=['x_range', 'y_range', 'bitdepth'])

def bin_and_transpose(batch_data, binsize):
    """Bin images and transpose to final orientation"""
    batch_size, n_slices, h, w = batch_data.shape
    h_bins, w_bins = h // binsize, w // binsize
    binned = jnp.einsum('bsijkl->bsik', 
                      batch_data.reshape(batch_size, n_slices, h_bins, binsize, w_bins, binsize))
    return jnp.transpose(binned, (0, 3, 2, 1))

# Create JIT-compiled version with static slice arguments
bin_and_transpose_jit = jax.jit(bin_and_transpose, static_argnames=['binsize'])