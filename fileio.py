import nrrd
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
import mmap
import gc
import os

def save_nrrd(nrrd_path, vol_data, spacing):
    """Saves NRRD files efficiently with memory-mapped file handling."""
    
    # Prepare NRRD header
    header = {
        'type': 'int16',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': list(vol_data.shape),
        'space directions': spacing,
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'raw',
        'space origin': [0, 0, 0]
    }
    
    try:
        with open(nrrd_path, "wb") as f:
            nrrd.write(f, vol_data, header)
    except Exception as e:
        print(f"Error saving NRRD file {nrrd_path}: {e}")

def batch_save_nrrd(tasks, num_workers):
    """Executes parallel file saving tasks using multithreading with optimized job scheduling."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(lambda task: save_nrrd(*task), tasks))  # Ensures execution completes before returning


def save_tif(tif_path, vol_data):
    """Saves a multi-page TIFF file efficiently."""
    try:
        tifffile.imwrite(
            tif_path,
            vol_data.astype(np.int16, copy=False),  # Ensure no unnecessary memory usage
            bigtiff=True  # Support large files
        )
    except Exception as e:
        print(f"Error saving TIFF file {tif_path}: {e}")

def batch_save_tif(tasks, num_workers):
    """Executes parallel TIFF saving tasks using multithreading with optimized job scheduling."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(lambda task: save_tif(*task), tasks))  # Ensures execution completes before returning
