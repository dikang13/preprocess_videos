# Utility functions to optimize parallelization depending on GPU capacity

import numpy as np
from pathlib import Path
import nrrd
import tifffile

def calculate_optimal_chunk_size(shape, dtype):
    """Calculate optimal chunk size based on available memory"""
    element_size = np.dtype(dtype).itemsize
    total_elements = np.prod(shape)
    memory_usage = total_elements * element_size
    max_memory = MAX_MEMORY_GB * 1024**3  # Convert GB to bytes
    return max(1, int(max_memory / memory_usage))


def parallel_save_outputs(save_args):
    """Helper function for parallel saving of outputs"""
    nrrd_path, mip_path, vol_data, spacing = save_args
    
    # Prepare NRRD header attributes
    header = {
        'type': 'uint16',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': [vol_data.shape[0], vol_data.shape[1], vol_data.shape[2]],
        'space directions': [[spacing[0], 0, 0], [0, spacing[1], 0], [0, 0, spacing[2]]],
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': [0, 0, 0]
    }
    
    # Save NRRD, overwriting existing files
    nrrd_path = Path(nrrd_path)
    if nrrd_path.exists():
        nrrd_path.unlink()  # Delete existing file
    nrrd.write(str(nrrd_path), vol_data, header)
    
    # Save MIP (optional), overwriting existing files
    if mip_path:
        mip_path = Path(mip_path)
        if mip_path.exists():
            mip_path.unlink()  # Delete existing file
        mip = np.max(vol_data, axis=2).astype(np.uint16)
        tifffile.imwrite(str(mip_path), mip)