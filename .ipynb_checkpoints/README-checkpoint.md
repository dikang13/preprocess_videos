```markdown
# Video Preprocessing Pipeline

A Python package for efficient preprocessing of large microscopy video data, optimized for GPU acceleration using JAX and CUDA.

## Features
- Efficient processing of large TIF files
- Background noise subtraction
- 3x3 binning optimization
- GPU acceleration with JAX
- Parallel I/O operations
- MIP (Maximum Intensity Projection) generation
- NRRD output format support

## System Requirements
- Python 3.12+
- NVIDIA GPU with CUDA support
- GPU Memory: >= 8GB recommended
- System Memory: >= 32GB recommended

## Installation

### Using conda (recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/preprocess_videos.git
cd preprocess_videos

# Create and activate conda environment
conda env create -f environment.yaml
conda activate video_preprocess
```

### Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
python -m preprocess_videos.cli input_path output_dir [options]

# Example
python -m preprocess_videos.cli /path/to/input.tif /path/to/output_dir --n-z 84 --channels 1 2
```

### Options
- `--n-z`: Number of z-slices per volume (default: 84)
- `--spacing-lat`: Lateral spacing (default: 0.54)
- `--spacing-axi`: Axial spacing (default: 0.54)
- `--no-mip`: Disable MIP generation
- `--channels`: Channels to process (default: [1,2])
- `--noise-path`: Path to noise reference TIF file
- `--chunk-size`: Number of timepoints to process at once
- `--gpu`: GPU device number to use (default: 0)
- `--max-memory`: Maximum memory usage in GB (default: 32)

### Python API
```python
from preprocess_videos import convert_tif_to_nrrd

convert_tif_to_nrrd(
    input_path='path/to/input.tif',
    output_dir='path/to/output',
    n_z=84,
    spacing_lat=0.54,
    spacing_axi=0.54,
    generate_mip=True,
    channels=[1],
    noise_path=None,
    chunk_size=None
)
```

## Project Structure
```
preprocess_videos/
├── __init__.py
├── cli.py           # Command line interface
├── config.py        # Configuration and constants
├── converter.py     # Main conversion logic
├── processors.py    # Core processing functions
└── utils.py         # Utility functions
```

## Input/Output Specifications

### Input
- TIF files with dimensions: 966x630x84 per timepoint
- Optional noise reference TIF file for background subtraction

### Output
- NRRD files: 322x210x77 volumes after processing
- Optional MIP (Maximum Intensity Projection) PNG files
- Output directory structure:
  ```
  output_dir/
  ├── NRRD/
  │   └── prefix_t0001_ch1.nrrd
  └── MIP/
      └── prefix_t0001_ch1.png
  ```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


## Contact
dikang13@gmail.com
```