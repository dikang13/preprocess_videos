# Video Preprocessing Pipeline

A Python package for efficient preprocessing of large microscopy video data, optimized for GPU acceleration using JAX and CUDA.

## Features
This package processes time-lapsed fluorescent microscopy images with:
- Support for ND2/TIF formats and multi-channel 2D/3D images
- Fixed pattern noise computation and subtraction
- Configurable pixel binning for enhanced signal and reduced file size
- Multiple output formats (NRRD, TIF) with MIP generation for 3D images


## Installation
## System Requirements
- Python 3.12+
- NVIDIA GPU with CUDA support
- GPU Memory: >= 8GB recommended
- System Memory: >= 32GB recommended
  
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
# Create and activate virtual environment
python -m venv env
source env/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# For GPU support
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### User-specified parameters

- `--input_path`: Path to input file (.nd2 or .tif)
- `--output_dir`: Directory for processed outputs
- `--noise_path`: Path to precomputed noise reference file (.tif)
- `--blank_dir`: Directory containing noise reference files to be averaged
- `--chunk_size`: Number of frames to process at once
- `--n_z`: Number of Z-slices
- `--x_range`: X dimension range as "start,end"
- `--y_range`: Y dimension range as "start,end"
- `--z_range`: Z dimension range as "start,end"
- `--channels`: Channels to process as "1,2"
- `--bitdepth`: Bit depth of images
- `--binsize`: Binning factor
- `--save_as`: Output format ('nrrd' or 'tif')
- `--gpu`: GPU device number (optional, only if user wants to dedicate a specific GPU to the current run)

### Examples
#### Processing ND2 Files
```bash
python main.py \
    --input_path /store1/data_raw/2025-01-14/2025-01-14-ZylaBackground.nd2 \
    --output_dir /store1/data_processed/2025-01-14_output \
    --noise_path /store1/data_raw/2025-01-14/avg_noise.tif \
    --n_z 84 \
    --x_range 0,966 \
    --y_range 0,636 \
    --z_range 3,80 \
    --channels 1,2 \
    --bitdepth 12 \
    --binsize 3 \
    --save_as nrrd
```

#### Processing TIF Files
```bash
python main.py \
    --input_path /store1/501659/2025-01-23-05_exp7.2ms_fast_ram_16bit.tif \
    --output_dir /store1/501659/2025-01-23-05_output \
    --blank_dir /store1/501659/noise_data \
    --n_z 100 \
    --x_range 0,968 \
    --y_range 0,636 \
    --z_range 3,80 \
    --channels 1,2 \
    --bitdepth 12 \
    --binsize 3 \
    --save_as tif \
```