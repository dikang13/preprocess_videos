from .converter import convert_tif_to_nrrd
from .config import configure_jax

# Configure JAX when the package is imported
configure_jax()