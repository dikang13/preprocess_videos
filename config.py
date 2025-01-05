import multiprocessing

# Constants and configurations
THREAD_POOL_SIZE = max(1, multiprocessing.cpu_count() - 1)
BATCH_SIZE = 32  # Optimal batch size for GPU processing
MAX_MEMORY_GB = 32  # Maximum memory usage in GB

# JAX configurations
def configure_jax():
    import jax
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_disable_jit', False)