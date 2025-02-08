import os
import numpy as np
import tifffile
import nrrd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def parallel_save_outputs(nrrd_path, mip_path, vol_data, spacing):
    # Prepare NRRD header attributes
    header = {
        'type': 'int16',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': [vol_data.shape[0], vol_data.shape[1], vol_data.shape[2]],
        'space directions': [[spacing[0], 0, 0], [0, spacing[1], 0], [0, 0, spacing[2]]],
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space origin': [0, 0, 0]
    }
    
    # Save NRRD, overwriting if exists
    nrrd_path = Path(nrrd_path)
    if nrrd_path.exists():
        nrrd_path.unlink()  # Delete existing file
    nrrd.write(str(nrrd_path), vol_data, header)
    
    # Save MIP, overwriting if exists
    if mip_path.exists():
        mip_path.unlink()  # Delete existing file
    mip = np.max(vol_data, axis=2).astype(np.int16)
    tifffile.imwrite(str(mip_path), mip)


def load_nd2_chunk(input_path, start_page, end_page, x_range, y_range):
    """Load specific pages from an ND2 file, handling multiple channels."""
    images = ND2Reader(input_path)
    c_size = images.sizes['c']
    
    t_len = end_page - start_page
    data = np.empty((t_len, c_size, y_range.stop, x_range.stop), dtype=np.int16)

    for t_idx, t in enumerate(range(start_page, end_page)):
        for c in range(c_size):
            data[t_idx, c, y_range, x_range] = images.get_frame_2D(c=c, t=t, z=0)
    
    return data.transpose(0, 1, 3, 2)


def parallel_load_nd2(input_path, total_pages, x_range, y_range, num_workers=4):
    """Parallel load ND2 file by splitting time pages into chunks."""
    chunk_size = total_pages // num_workers
    futures = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_workers - 1 else total_pages
            futures.append(executor.submit(load_nd2_chunk, input_path, start, end, x_range, y_range))
        
        results = [f.result() for f in futures]
    
    return np.concatenate(results, axis=0)


def load_tif_chunk(input_path, start_page, end_page, x_range, y_range):
   """Load and reshape TIF chunk with proper dimensions"""
   img = tifffile.imread(str(input_path), key=range(start_page, end_page))
   
   if img.ndim == 4:  # (frames, height, width, channels)
       img = img.transpose(0, 3, 1, 2)
       img = img.transpose(0, 1, 3, 2)
       return img[:,:,x_range,y_range]
   elif img.ndim == 3:  # (frames, height, width)
       img = img[:, np.newaxis, :, :]
       img = img.transpose(0, 1, 3, 2)
       return img[:,:,x_range,y_range]
   else:
       raise ValueError(f"Unexpected number of dimensions: {img.ndim}")


def get_total_pages(input_path, extension):
    """Get total number of pages in file"""
    if extension == '.nd2':
        x_size, y_size, z_size, t_size, c_size = nd2dim(input_path)
        return t_size
    else:  # tif/tiff
        with tifffile.TiffFile(input_path) as tif:
            return len(tif.pages)


def nd2dim(path_nd2: str, verbose: bool = False) -> tuple:
    """Get dimensions of an ND2 file."""
    if not os.path.isfile(path_nd2):
        raise FileNotFoundError("ND2 file does not exist.")
        
    with ND2Reader(path_nd2) as images:
        # Get first frame and check data type
        first_frame = images.get_frame_2D(c=0, t=0, z=0)
        assert first_frame.dtype in [np.float64, np.uint16], "Unexpected data type"
        
        # Get dimensions with default value of 1 if dimension doesn't exist
        dimensions = {'x': 1, 'y': 1, 'c': 1, 't': 1, 'z': 1}
        for key in dimensions.keys():
            if key in images.sizes:
                dimensions[key] = images.sizes[key]
                
        x_size = dimensions['x']
        y_size = dimensions['y']
        c_size = dimensions['c']
        t_size = dimensions['t']
        z_size = dimensions['z']
        
        if verbose:
            print(f"x:{x_size}, y:{y_size}, c:{c_size}, t:{t_size}, z:{z_size}")
            
        return (x_size, y_size, z_size, t_size, c_size)