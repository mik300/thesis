import os

# Input and output directory
input_file = "gemmini_outputs.txt"
output_dir = "verified_outputs"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize variables
file_counter = 0
output_lines = []

# Open and read the input file
with open(input_file, "r") as infile:
    for line in infile:
        # Check if the line starts with 'output_mat:'
        if line.startswith("output_mat:"):
            # If we already have content collected, save it to a file
            if output_lines:
                # Create a new file for the previous output_mat section
                file_counter += 1
                output_file = os.path.join(output_dir, f"output_mat_{file_counter}.txt")
                with open(output_file, "w") as outfile:
                    outfile.writelines(output_lines)
                output_lines = []  # Reset for the next section

        # Collect the current line (including the 'output_mat:' header)
        output_lines.append(line)

    # Save the last section if it exists
    if output_lines:
        file_counter += 1
        output_file = os.path.join(output_dir, f"output_mat_{file_counter}.txt")
        with open(output_file, "w") as outfile:
            outfile.writelines(output_lines)


