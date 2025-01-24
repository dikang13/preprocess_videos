import argparse
import subprocess
import multiprocessing

# Constants for optimization
THREAD_POOL_SIZE = max(1, multiprocessing.cpu_count() - 1)

def parse_slice(slice_str):
    """Convert string of format 'start,end' to a slice object"""
    try:
        start, end = map(int, slice_str.split(','))
        return slice(start, end)
    except:
        raise argparse.ArgumentTypeError("Slice must be in format 'start,end'")

def parse_list(list_str):
    """Convert string of format '[x,y,...]' to a list of integers"""
    try:
        # Remove brackets and split by comma
        values = list_str.strip('[]').split(',')
        # Convert to integers
        return [int(x.strip()) for x in values]
    except:
        raise argparse.ArgumentTypeError("List must be in format '[1,2]' or '1,2'")

def parse_float_list(list_str):
    """Convert string of format '[x,y,z]' or 'x,y,z' to a list of floats"""
    try:
        # Remove brackets and split by comma
        values = list_str.strip('[]').split(',')
        # Convert to floats
        return [float(x.strip()) for x in values]
    except:
        raise argparse.ArgumentTypeError("List must be in format '[0.54,0.54,0.54]' or '0.54,0.54,0.54'")