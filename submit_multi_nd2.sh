#!/bin/bash
# Define project directory
prj_dir = "/store1/shared/panneuralGFP_SWF1212"

# Define base date
date="2025-02-06"

# Define input base path
input_base="$prj_dir/data_raw/$date"

# Define identity of each nd2 file as specified in a seperate txt file in input_base

# Define output base path
output_base="$prj_dir/data_processed/"

# Define first output directory based on the first input file
first_output_dir="${output_base}${date}-01_output"

# Generate the first command
cmd="python main.py \
    --input_path $input_base/${date}-01.nd2 \
    --output_dir $first_output_dir && \\
"

# Generate subsequent commands
for i in {02..04}; do
    cmd+="python main.py \
    --input_path $input_base/${date}-${i}.nd2 \
    --output_dir $first_output_dir/neuropal/${date}-${i} && \\
"
done

# Remove the trailing ' && \' from the last command
cmd=${cmd%" && \\
"}

# Print the commands
echo -e "$cmd"
