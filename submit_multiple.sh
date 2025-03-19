#!/bin/bash

# Display usage information
usage() {
    echo "Usage: $0 -i INPUT_BASE -o OUTPUT_BASE [-g GPU_ID]"
    echo "  -i INPUT_BASE   Base directory containing raw data and metadata.txt"
    echo "  -o OUTPUT_BASE  Base directory where processed data will be saved"
    echo "  -g GPU_ID       GPU ID to use for processing (0-3, default: 0)"
    echo "  -h              Display this help message"
    exit 1
}

# Parse command line arguments
GPU_ID=0  # Default GPU ID
while getopts "i:o:g:h" opt; do
    case $opt in
        i) INPUT_BASE="$OPTARG" ;;
        o) OUTPUT_BASE="$OPTARG" ;;
        g) GPU_ID="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$INPUT_BASE" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "Error: Both INPUT_BASE and OUTPUT_BASE must be specified"
    usage
fi

# Validate GPU_ID is between 0-3
if ! [[ "$GPU_ID" =~ ^[0-3]$ ]]; then
    echo "Error: GPU_ID must be a number between 0 and 3"
    usage
fi

# Define metadata file path
METADATA_FILE="$INPUT_BASE/metadata.txt"

# Check if metadata file exists
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found at $METADATA_FILE"
    exit 1
fi

# Check if output directory exists, create if not
if [ ! -d "$OUTPUT_BASE" ]; then
    echo "Creating output directory: $OUTPUT_BASE"
    mkdir -p "$OUTPUT_BASE"
fi

# Set environment variable
export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "Using GPU $GPU_ID for processing"

# Process a primary image and its associated fluorescence images
process_primary_image() {
    local primary=$1
    
    echo "Processing primary image: $primary"
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 0 \
        --global_t_end 400

    sleep 5
    
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 400 \
        --global_t_end 800

    sleep 5
    
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 800 \
        --global_t_end 1200

    sleep 5
    
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 1200 \
        --global_t_end 1600
        
    sleep 5 
    
    # Extract the associated images for this primary from the metadata file
    local in_section=0
    local fluorescence_images=()
    
    while IFS= read -r line || [ -n "$line" ]; do
        # Detect start of a section for this primary image
        if [[ $line == \"$primary\": ]]; then
            in_section=1
            continue
        fi
        
        # Detect end of section
        if [[ $in_section -eq 1 && $line == "}" ]]; then
            in_section=0
            continue
        fi
        
        # Extract fluorescence image IDs while in the correct section
        if [[ $in_section -eq 1 && $line =~ = ]]; then
            # Extract the image ID (removing quotes)
            local fluorescence_id=$(echo "$line" | cut -d'"' -f2)
            fluorescence_images+=("$fluorescence_id")
        fi
    done < "$METADATA_FILE"
    
    # Process each fluorescence image
    for fluorescence_id in "${fluorescence_images[@]}"; do
        echo "Processing fluorescence image: $fluorescence_id"
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_ALLOCATOR=platform \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
        XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT=0 \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        python main.py \
            --input_path "$INPUT_BASE/$fluorescence_id.nd2" \
            --output_dir "$OUTPUT_BASE/${primary}_output/neuropal/$fluorescence_id" \
            --gpu $GPU_ID &
    done
}

# Find all primary images from metadata file
primary_images=()
while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line =~ ^\"([0-9-]+)\": ]]; then
        primary_images+=("${BASH_REMATCH[1]}")
    fi
done < "$METADATA_FILE"

echo "Found ${#primary_images[@]} primary images in metadata file"
echo "All processing will use GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"

# Process each primary image and its associated fluorescence images
for primary in "${primary_images[@]}"; do
    process_primary_image "$primary"
done

echo "All processing commands completed using GPU $GPU_ID"