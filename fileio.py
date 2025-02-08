from pathlib import Path
import nrrd
import numpy as np
import tifffile
from concurrent.futures import ProcessPoolExecutor

def save_nrrd(nrrd_path, mip_path, vol_data, spacing):
    """Saves NRRD and MIP files in parallel."""
    
    # Prepare NRRD header
    header = {
        'type': 'int16',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': list(vol_data.shape),
        'space directions': spacing,
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': [0, 0, 0]
    }
    
    # Save NRRD (overwrite if exists)
    nrrd_path = Path(nrrd_path)
    nrrd.write(str(nrrd_path), vol_data, header)

    # Save MIP (overwrite if exists)
    mip = np.max(vol_data, axis=2).astype(np.int16)
    tifffile.imwrite(str(mip_path), mip) # MIP exported as PNGs instead of NRRDs


def batch_save_nrrd(tasks, num_workers=8):
    """Executes parallel file saving tasks using multithreading."""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda task: save_nrrd(*task), tasks)


def save_tif(tif_path, vol_data):
    """Saves a multi-page TIFF file."""
    tifffile.imwrite(
        tif_path,
        vol_data.astype(np.int16),  # Ensure int16 format
        bigtiff=True  # Support large files
    )

def batch_save_tif(tasks, num_workers=8):
    """Executes parallel TIFF saving tasks using multithreading."""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda task: save_tif(*task), tasks)


# import dask
# import numpy as np
# import tifffile
# import nrrd
# from pathlib import Path

# def save_nrrd(nrrd_path, mip_path, vol_data, spacing):
#     """Saves NRRD and its MIP."""
#     header = {
#         'type': 'int16',
#         'dimension': 3,
#         'space': 'left-posterior-superior',
#         'sizes': list(vol_data.shape),
#         'space directions': spacing,
#         'kinds': ['domain', 'domain', 'domain'],
#         'endian': 'little',
#         'encoding': 'gzip',  # Compression for efficiency
#         'space origin': [0, 0, 0]
#     }

#     # Save NRRD
#     nrrd.write(str(nrrd_path), vol_data.astype(np.int16), header)

#     # Save MIP (max intensity projection)
#     mip = np.max(vol_data, axis=2).astype(np.int16)
#     tifffile.imwrite(str(mip_path), mip)


# def save_tif(tif_path, vol_data):
#     """Saves a multi-page TIFF file."""
#     tifffile.imwrite(str(tif_path), vol_data.astype(np.int16), bigtiff=True)


# def batch_save_nrrd(batch_tasks):
#     """Batch saves multiple NRRD files to reduce computation graph size."""
#     for nrrd_path, mip_path, vol_data, spacing in batch_tasks:
#         header = {
#             'type': 'int16',
#             'dimension': 3,
#             'space': 'left-posterior-superior',
#             'sizes': list(vol_data.shape),
#             'space directions': spacing,
#             'kinds': ['domain', 'domain', 'domain'],
#             'endian': 'little',
#             'encoding': 'gzip',
#             'space origin': [0, 0, 0]
#         }
#         nrrd.write(str(nrrd_path), vol_data.astype(np.int16), header)
#         mip = np.max(vol_data, axis=2).astype(np.int16)
#         tifffile.imwrite(str(mip_path), mip)
