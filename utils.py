import argparse
import ast

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

def parse_float_list(value):
    """Parses a string representing a nested list of floats."""
    try:
        parsed_value = ast.literal_eval(value)  # Safely evaluate the string into a Python list
        if isinstance(parsed_value, list) and all(isinstance(row, list) and all(isinstance(num, (int, float)) for num in row) for row in parsed_value):
            return parsed_value  # Return as List of lists
        else:
            raise ValueError
    except (SyntaxError, ValueError):
        raise argparse.ArgumentTypeError("Invalid format. Expected a list of lists, e.g., '[[0.54, 0, 0], [0, 0.54, 0], [0, 0, 0.54]]'")

def print_mem_usage(label):
    process = psutil.Process(os.getpid())
    print(f"[{label}] Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")