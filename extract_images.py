import torch
import numpy as np
import warnings
from neural_networks.utils import get_loaders_split
warnings.simplefilter(action='ignore', category=FutureWarning)

min_val=-127 
max_val=127
image_file = "images.txt"
image_header = "images.h"
def quantize_image(image):
    t_max = torch.max(torch.abs(torch.min(image)), torch.abs(torch.max(image))).item()
    scaling_factor = 127/t_max
    quantized_image = torch.clamp(torch.round(scaling_factor * image), min=min_val, max=max_val).to(torch.int8)
    quantized_image = quantized_image.permute(0, 2, 3, 1) #permute the dimensions to match the dimensions of weights in gemmini
    return quantized_image

def log_image(image, layer_name):
    image_clean = image.detach().cpu().numpy()  
    formatted_image = image_clean.tolist()  
    list_string = str(formatted_image).replace('\n', '').replace('  ', ' ')  
    list_string = list_string + ";"
    with open(image_file, 'a') as f:
        f.write(f'{layer_name} row_align(1) = {list_string}\n')


def init_log():
    with open(image_file, 'w') as f:
        f.write("")
    with open(image_header, 'w') as f:
        f.write("")

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

data_dir = "./data/"
batch_size = 4
dataset = "cifar10"
num_workers = 4
split_val = 0.1
disable_aug = True
_, _, test_loader = get_loaders_split(data_dir, batch_size=batch_size, dataset_type=dataset, num_workers=num_workers, split_val=split_val, disable_aug=disable_aug, test_size=batch_size, resize_to_imagenet=True)

init_log()

print(f"Number of images in the test set: {len(test_loader.dataset)}")
for idx, (image, label) in enumerate(test_loader):
    image = quantize_image(image)
    log_image(image, f'static const elem_t alexnet_images<{image.shape[0]}><{image.shape[1]}><{image.shape[2]}><{image.shape[3]}>')
    log_image(label, f'const int labels<{batch_size}>')
    process_tensor_file(image_file, image_header)
    
print(f'The input images and labels for alexnet are saved in {image_header}')