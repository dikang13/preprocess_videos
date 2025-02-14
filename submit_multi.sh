#!/bin/bash

# # Define base date (might change)
# date="2025-02-10"

# # Define base paths (might change)
# input_base="/store1/shared/panneuralGFP_SWF1212/data_raw/$date"
# output_base="/store1/shared/panneuralGFP_SWF1212/data_processed"

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
input_base="$input_base/$date"
metadata_file="$input_base/metadata.txt"
declare -A conditions
declare -A parent_dirs
current_parent=""

while IFS= read -r line; do
    line=$(echo "$line" | tr -d '[:space:]')  # Remove spaces
    if [[ "$line" =~ ^\" ]]; then
        current_parent=$(echo "$line" | tr -d '":')_output
        parent_dirs[$current_parent]=""
    elif [[ "$line" =~ "=" ]]; then
        key=$(echo "$line" | cut -d '=' -f1 | tr -d ' ')
        value=$(echo "$line" | cut -d '=' -f2 | tr -d '"')
        if [[ -n "$value" ]]; then  # Ensure value is not empty
            conditions[$value]="$current_parent"
        fi
    fi
done < "$metadata_file"

# Generate command lines
cmd=""

for file in "$input_base"/${date}-*.nd2; do
    filename=$(basename "$file" .nd2)
    parent_dir="${conditions[$filename]}"
    if [[ -z "$parent_dir" ]]; then
        parent_dir="${filename}_output"
        output_dir="${output_base}/${parent_dir}"
    else
        output_dir="${output_base}/${parent_dir}/neuropal/${filename}"
    fi
    
    cmd+="python main.py \\
    --input_path $file \\
    --output_dir $output_dir && \\
"
done

# Remove the trailing ' && \' from the last command
cmd=${cmd%" && \\
"}

# Print the commands
echo -e "$cmd"

# Execute the generated commands
eval "$cmd"