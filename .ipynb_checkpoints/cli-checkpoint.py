# Command-line interface

import argparse
import os
from .converter import convert_tif_to_nrrd
from .config import MAX_MEMORY_GB

def main():
    parser = argparse.ArgumentParser(description='Optimized TIF to NRRD conversion')
    parser.add_argument('input_path', type=str, help='Input TIF file path')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('--n-z', type=int, default=84, help='Number of z-slices per volume')
    parser.add_argument('--spacing-lat', type=float, default=0.54, help='Lateral spacing')
    parser.add_argument('--spacing-axi', type=float, default=0.54, help='Axial spacing')
    parser.add_argument('--no-mip', action='store_true', help='Disable MIP generation')
    parser.add_argument('--channels', type=int, nargs='+', default=[1], help='Channels to process')
    parser.add_argument('--noise-path', type=str, help='Path to noise reference TIF file')
    parser.add_argument('--chunk-size', type=int, help='Number of timepoints to process at once')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use')
    parser.add_argument('--max-memory', type=int, default=32, help='Maximum memory usage in GB')
    
    args = parser.parse_args()
    
    # Update global constants
    global MAX_MEMORY_GB
    MAX_MEMORY_GB = args.max_memory
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    convert_tif_to_nrrd(
        input_path=args.input_path,
        output_dir=args.output_dir,
        n_z=args.n_z,
        spacing_lat=args.spacing_lat,
        spacing_axi=args.spacing_axi,
        generate_mip=not args.no_mip,
        channels=args.channels,
        noise_path=args.noise_path,
        chunk_size=args.chunk_size
    )