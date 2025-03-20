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

# For debugging
echo "INPUT_BASE set to: $INPUT_BASE"
echo "OUTPUT_BASE set to: $OUTPUT_BASE"

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
echo "Using GPU $GPU_ID for processing"

# Process a primary image and its associated fluorescence images
process_primary_image() {
    local primary=$1
    
    echo "Processing freely moving video: $INPUT_BASE/$primary.nd2"
    # Verify the file exists
    if [ ! -f "$INPUT_BASE/$primary.nd2" ]; then
        echo "WARNING: Input file $INPUT_BASE/$primary.nd2 not found. Skipping."
        return 1
    fi

    sleep 5
    
    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 0 \
        --global_t_end 0
    
    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 0 \
        --global_t_end 200 &
    
    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 200 \
        --global_t_end 400 &

    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 400 \
        --global_t_end 600 &

    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 600 \
        --global_t_end 800 &
    
    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 800 \
        --global_t_end 1000 &

    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 1000 \
        --global_t_end 1200 &

    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 1200 \
        --global_t_end 1400 & 

    XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
        --input_path "$INPUT_BASE/$primary.nd2" \
        --output_dir "$OUTPUT_BASE/${primary}_output" \
        --gpu $GPU_ID \
        --global_t_start 1400 \
        --global_t_end 1600
    
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
            local fluorescence_id=$(echo "$line" | cut -d\" -f2)
            fluorescence_images+=("$fluorescence_id")
        fi
    done < "$METADATA_FILE"
    
    # Process each fluorescence image
    for fluorescence_id in "${fluorescence_images[@]}"; do
        echo "Processing immobilized video: $INPUT_BASE/$fluorescence_id"
        XLA_PYTHON_CLIENT_ALLOCATOR=platform python main.py \
            --input_path "$INPUT_BASE/$fluorescence_id.nd2" \
            --output_dir "$OUTPUT_BASE/${primary}_output/neuropal/$fluorescence_id" \
            --gpu $GPU_ID &
    done

    sleep 5
}

# Find all primary images from metadata file
primary_images=()
while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line =~ ^\"([0-9-]+)\": ]]; then
        primary_images+=("${BASH_REMATCH[1]}")
        echo "Found freely moving video: ${BASH_REMATCH[1]}"
    fi
done < "$METADATA_FILE"

echo "Found a total of ${#primary_images[@]} freely moving videos in metadata file"
if [ ${#primary_images[@]} -eq 0 ]; then
    echo "WARNING: No freely moving videos found in metadata file. Check the format of $METADATA_FILE"
    # Debug content of file
    echo "First 10 lines of metadata file:"
    head -n 10 "$METADATA_FILE"
fi
echo "All processing will use GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"

# Process each primary image and its associated fluorescence images
for primary in "${primary_images[@]}"; do
    process_primary_image "$primary"
done

echo "All processing commands for $INPUT_BASE completed using GPU $GPU_ID."
echo "You may now exit from this terminal."