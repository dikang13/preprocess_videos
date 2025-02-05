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
background_subtract_jit = jax.jit(
    background_subtract, 
    static_argnames=['x_range', 'y_range', 'bitdepth']
)

def background_subtract_channels(batch_data_5d, noise_data_3d, x_range, y_range, bitdepth):
    """
    Vectorized background subtraction across channels using vmap.
    
    batch_data_5d: [B, C, Z, H, W]
    noise_data_3d: [C, H, W] or None
    x_range, y_range: slices (static)
    bitdepth: int
    """
    if noise_data_3d is None:
        # If background subtraction is skipped entirely, optionally crop here if desired:
        # return batch_data_5d[..., x_range, y_range]
        return batch_data_5d

    def subtract_single_channel(chan_data_4d, noise_2d):
        # shape of chan_data_4d -> [B, Z, H, W]
        # shape of noise_2d      -> [H, W]
        return background_subtract_jit(
            chan_data_4d,
            noise_2d,
            x_range=x_range,
            y_range=y_range,
            bitdepth=bitdepth
        )

    # We map over channel dimension: in_axes = (1, 0) means
    # - from batch_data_5d, take axis=1 as input to subtract_single_channel
    # - from noise_data_3d, take axis=0
    # out_axes=1 to keep the same channel dimension layout
    return jax.vmap(subtract_single_channel, in_axes=(1, 0), out_axes=1)(
        batch_data_5d, noise_data_3d
    )


def bin_and_transpose(batch_data, binsize, bitdepth, subtract_800):
    """Bin images and transpose to final orientation"""
    batch_size, n_slices, h, w = batch_data.shape
    h_bins, w_bins = h // binsize, w // binsize
    binned = jnp.einsum('bsijkl->bsik', 
                      batch_data.reshape(batch_size, n_slices, h_bins, binsize, w_bins, binsize))
    binned = jnp.transpose(binned, (0, 3, 2, 1))
    
    if subtract_800:
        binned = jnp.subtract(binned, 800)
    return jnp.clip(binned, 0, 2 ** bitdepth)

# Create JIT-compiled version with static slice arguments
bin_and_transpose_jit = jax.jit(
    bin_and_transpose, 
    static_argnames=['binsize', 'bitdepth']
)


def bin_and_transpose_channels(batch_data_5d, binsize, bitdepth, subtract_800):
    """
    Vectorized bin_and_transpose across channels using vmap.
    
    batch_data_5d: [B, C, Z, ..., ...]
    """
    if binsize <= 0:
        # Skip binning
        return batch_data_5d

    def bin_single_channel(chan_data_4d):
        # shape [B, Z, H, W]
        return bin_and_transpose_jit(chan_data_4d, binsize, bitdepth, subtract_800)

    # vmap over channel dimension again
    return jax.vmap(bin_single_channel, in_axes=1, out_axes=1)(batch_data_5d)