#!/bin/bash

# Define defaults (will be overridden by command-line arguments)
date=""
input_base=""
output_base=""
gpu=""

# Function to clear memory caches at user level (no sudo required)
clear_memory() {
    echo "Clearing memory..."
    # Drop caches from current process and children
    sync
    # Try to trigger garbage collection in Python
    python3 -c "import gc; gc.collect()" 2>/dev/null || true
    # Clear any temporary files in /tmp owned by current user
    find /tmp -user $(whoami) -type f -mmin -60 -delete 2>/dev/null || true
}

# Allow overriding via command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --date)
            date="$2"
            shift 2
            ;;
        --input_base)
            input_base="$2"
            shift 2
            ;;
        --output_base)
            output_base="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure required variables are set
if [[ -z "$date" || -z "$input_base" || -z "$output_base" ]]; then
    echo "Error: --date, --input_base, and --output_base must be specified."
    exit 1
fi

# Read metadata file
input_dir="${input_base}/${date}"
metadata_file="${input_dir}/metadata.txt"

if [[ ! -f "$metadata_file" ]]; then
    echo "Error: Metadata file not found at: $metadata_file"
    exit 1
fi

# Arrays to store parent-child relationships
declare -A conditions
declare -A parent_dirs
current_parent=""

# Process metadata file
while IFS= read -r line || [[ -n "$line" ]]; do
    # Remove spaces and quotes
    clean_line=$(echo "$line" | tr -d '[:space:]')
    
    # Skip empty lines
    if [[ -z "$clean_line" ]]; then
        continue
    fi
    
    # Check if line is a parent directory identifier
    if [[ "$clean_line" =~ ^\"([^\"]+)\" ]]; then
        current_parent="${BASH_REMATCH[1]}"
    # Check if line contains an assignment
    elif [[ "$clean_line" =~ ([^=]+)=\"([^\"]+)\" ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        if [[ -n "$value" ]]; then
            conditions["$value"]="$current_parent"
        fi
    fi
done < "$metadata_file"

# Add GPU argument if specified
gpu_arg=""
if [[ -n "$gpu" ]]; then
    gpu_arg="--gpu $gpu"
fi

# Get list of files before processing
files=()
for file in "${input_dir}"/${date}-*.nd2; do
    # Skip if no files match the pattern
    if [[ ! -f "$file" ]]; then
        echo "Warning: No .nd2 files found matching pattern: ${input_dir}/${date}-*.nd2"
        break
    fi
    files+=("$file")
done

# Process each file
for file in "${files[@]}"; do
    filename=$(basename "$file" .nd2)
    
    # Check if this file is mentioned in the metadata as a child
    if [[ -n "${conditions[$filename]}" ]]; then
        # This is a child file
        parent="${conditions[$filename]}"
        output_dir="${output_base}/${parent}_output/neuropal/${filename}"
        
        # Execute command for this child file
        cmd="MALLOC_TRIM_THRESHOLD_=0 python3 main.py \\
    --input_path $file \\
    --output_dir $output_dir $gpu_arg"
    
        echo "Executing: $cmd"
        eval "$cmd"
        
        # Capture exit status
        status=$?
        if [ $status -ne 0 ]; then
            echo "Error: Command failed with exit code $status"
            echo "Failed command: $cmd"
            exit $status
        fi
    else
        # This is a parent file - split into time slices
        output_dir="${output_base}/${filename}_output"
        
        # Time slices for parent files (320 frames per slice)
        time_slices=(
            "0 320"
            "320 640"
            "640 960"
            "960 1280"
            "1280 1600"
        )
        
        # Process each time slice
        for ((i=0; i<${#time_slices[@]}; i++)); do
            # Split the time slice string into start and end
            read -r t_start t_end <<< "${time_slices[$i]}"
            
            # Execute command for this time slice
            cmd="MALLOC_TRIM_THRESHOLD_=0 python3 main.py \\
    --input_path $file \\
    --output_dir $output_dir \\
    --global_t_start $t_start \\
    --global_t_end $t_end $gpu_arg"
            
            echo "Executing: $cmd"
            eval "$cmd"
            
            # Capture exit status
            status=$?
            if [ $status -ne 0 ]; then
                echo "Error: Command failed with exit code $status"
                echo "Failed command: $cmd"
                exit $status
            fi
            
            # Sleep between time slices (except after the last one)
            if [ $i -lt $((${#time_slices[@]} - 1)) ]; then
                echo "Sleeping for 5 seconds between time slices..."
                sleep 5
            fi
        done
    fi
    
    # Clear memory after each file
    clear_memory
    
    # Add a short delay to allow system to stabilize
    sleep 2
done

echo "All processing completed successfully."