import subprocess
import shutil
import os

def remove_empty_lines_from_file(input_file):
    # Read the original file and filter out empty lines
    with open(input_file, 'r') as f:
        lines = [line for line in f.readlines() if line.strip()]  # Keep only non-empty lines

    # Write the non-empty lines back to the same file
    with open(input_file, 'w') as f:
        for line in lines:
            f.write(line)


def process_tensor_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:

        # Replace brackets with braces
        line = line.replace("[", "{").replace("]", "}")
        line = line.replace("<", "[").replace(">", "]")
        line = line.replace(".", "_")
        # Strip any leading/trailing whitespace
        processed_lines.append(line.strip())
        
    # Write processed lines to a new file
    with open(output_file, 'w') as f:
        for line in processed_lines:
            f.write(line + "\n")

    


# Remove empty lines
input_file = 'weights.txt'
remove_empty_lines_from_file(input_file)

# Change to readable C code
input_file = 'weights.txt'
output_file = 'gemmini_weights.txt'
process_tensor_file(input_file, output_file)

input_file = 'biases.txt'
output_file = 'gemmini_biases.txt'
process_tensor_file(input_file, output_file)


input_file = 'conv_inputs.txt'
output_file = 'gemmini_inputs.txt'
process_tensor_file(input_file, output_file)


# List of input files
input_files = ["gemmini_inputs.txt", "gemmini_weights.txt", "gemmini_biases.txt"]

# Output file where contents will be combined
output_file = "conv_layer_params.h"
directives = "#include <include/gemmini_params.h>\n#include <stdbool.h>\n\n"
biases = "static acc_t conv_1_b = 0;\nstatic acc_t layer1_0_conv1_b = 0;\nstatic acc_t layer1_0_conv2_b = 0;\nstatic acc_t layer2_0_conv1_b = 0;\nstatic acc_t layer2_0_conv2_b = 0;\nstatic acc_t layer3_0_conv1_b = 0;\nstatic acc_t layer3_0_conv2_b = 0;\n"


# Open the output file in write mode
with open(output_file, "w") as outfile:
    outfile.write(directives)
    # Loop through the input files and append their content to the output file
    for file in input_files:
        with open(file, "r") as infile:
            outfile.write(infile.read())
            outfile.write("\n")  # Add a newline between files to separate their contents
    outfile.write(biases)

print("Successfully created the header file needed for Gemmini")


subprocess.run(["python3", "seperate_output_mat.py"])


# Copy header file to Gemmini directory
source_file = "conv_layer_params.h"
destination_file = "/home/michael/chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/conv_layer_params.h"

shutil.copy(source_file, destination_file)

#Copy verified output of each conv layer to gemmini directory
source_dir = 'verified_outputs'
destination_dir = '/home/michael/chipyard/generators/gemmini/verified_outputs'

# Remove the destination directory if it exists
if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)

# Copy the directory and its contents
shutil.copytree(source_dir, destination_dir)


print("Header file and verified outputs copied to Gemmini directory")