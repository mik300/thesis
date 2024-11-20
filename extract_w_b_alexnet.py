import pickle
import torch
import numpy as np
import warnings
from neural_networks.utils import get_loaders_split
warnings.simplefilter(action='ignore', category=FutureWarning)

parameters_file = 'alexnet_params.txt'

def log_params(params, layer_name):
    # Convert the tensor to a NumPy array for clean output
    params_clean = params.detach().cpu().numpy()  # Move to CPU and convert to NumPy
    # Format the output as a string
    formatted_params = params_clean.tolist()  # Convert to list for better formatting
    list_string = str(formatted_params).replace('\n', '').replace('  ', ' ')  # Replace newlines if any
    list_string = list_string + ";"
    with open(parameters_file, 'a') as f:
        f.write(f'{layer_name} row_align(1) = {list_string}\n')


def init_log():
    with open(parameters_file, 'w') as f:
        f.write("")



min_val=-127 #min_val is chosen to be -127 and not -128 because it should be in accordance with the quantization done in axx_layers.py
max_val=127
def quantize_weights(weights):
    t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
    scaling_factor = 127/t_max
    quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=min_val, max=max_val).to(torch.int8)
    quantized_weights = quantized_weights.permute(0, 2, 3, 1) #permute the dimensions to match the dimensions of weights in gemmini
    return quantized_weights

def quantize_linear_weights(weights):
    t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
    scaling_factor = 127/t_max
    quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=min_val, max=max_val).to(torch.int8)
    quantized_weights = quantized_weights.permute(1, 0) #permute the dimensions to match the dimensions of weights in gemmini
    return quantized_weights

def quantize_biases(biases):
    t_max = torch.max(torch.abs(torch.min(biases)), torch.abs(torch.max(biases))).item()
    scaling_factor = 127/t_max
    quantized_biases = torch.clamp(torch.round(scaling_factor * biases), min=min_val, max=max_val).to(torch.int8)
    return quantized_biases


def flatten_weights(out_channels, kernel_dim, in_channels, weights):
    patch_size = kernel_dim * kernel_dim * in_channels
    flattened_weights = torch.empty((patch_size, out_channels), dtype=torch.int8)
    for outc in range(out_channels):
        for krow in range(kernel_dim):
            for kcol in range(kernel_dim):
                for inc in range(in_channels):
                    wmatrow = krow*kernel_dim*in_channels + kcol*in_channels + inc
                    flattened_weights[wmatrow][outc] = weights[outc][krow][kcol][inc]
    return flattened_weights

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

torch.set_printoptions(threshold=float('inf'), precision=16, sci_mode=False)

model = torch.load("neural_networks/models/alexnet_a8_w8_b32_fake_quant_cifar10_ReLU_epochs83.pth")

init_log()

#quantized_weights.shape[0] = OUT_CHANNELS
# quantized_weights.shape[1] = KERNEL_DIM
# quantized_weights.shape[2] = KERNEL_DIM
# quantized_weights.shape[3] = IN_CHANNELS
print(f'Quantizing and flattening weights...')
weights = model['model_state_dict']['conv1.weight']
quantized_weights = quantize_weights(weights)
flattened_weights = flatten_weights(quantized_weights.shape[0], quantized_weights.shape[1], quantized_weights.shape[3], quantized_weights)
log_params(flattened_weights, f'static const elem_t conv_1_w<{flattened_weights.shape[0]}><{flattened_weights.shape[1]}>')
print("conv1 done")

weights = model['model_state_dict']['conv2.weight']
quantized_weights = quantize_weights(weights)
flattened_weights = flatten_weights(quantized_weights.shape[0], quantized_weights.shape[1], quantized_weights.shape[3], quantized_weights)
log_params(flattened_weights, f'static const elem_t conv_2_w<{flattened_weights.shape[0]}><{flattened_weights.shape[1]}>')
print("conv2 done")

weights = model['model_state_dict']['conv3.weight']
quantized_weights = quantize_weights(weights)
flattened_weights = flatten_weights(quantized_weights.shape[0], quantized_weights.shape[1], quantized_weights.shape[3], quantized_weights)
log_params(flattened_weights, f'static const elem_t conv_3_w<{flattened_weights.shape[0]}><{flattened_weights.shape[1]}>')
print("conv3 done")

weights = model['model_state_dict']['conv4.weight']
quantized_weights = quantize_weights(weights)
flattened_weights = flatten_weights(quantized_weights.shape[0], quantized_weights.shape[1], quantized_weights.shape[3], quantized_weights)
log_params(flattened_weights, f'static const elem_t conv_4_w<{flattened_weights.shape[0]}><{flattened_weights.shape[1]}>')
print("conv4 done")

weights = model['model_state_dict']['conv5.weight']
quantized_weights = quantize_weights(weights)
flattened_weights = flatten_weights(quantized_weights.shape[0], quantized_weights.shape[1], quantized_weights.shape[3], quantized_weights)
log_params(flattened_weights,  f'static const elem_t conv_5_w<{flattened_weights.shape[0]}><{flattened_weights.shape[1]}>')
print("conv5 done")

weights = model['model_state_dict']['fc1.weight']
quantized_weights = quantize_linear_weights(weights)
log_params(quantized_weights, f'static const elem_t fc_6_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}>')
print("fc6 done")

weights = model['model_state_dict']['fc2.weight']
quantized_weights = quantize_linear_weights(weights)
log_params(quantized_weights, f'static const elem_t fc_7_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}>')
print("fc7 done")

weights = model['model_state_dict']['fc3.weight']
quantized_weights = quantize_linear_weights(weights)
log_params(quantized_weights, f'static const elem_t fc_8_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}>')
print("fc8 done")

print(f'Quantizing biases...')
biases = model['model_state_dict']['conv1.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases, f'static elem_t conv_1_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv2.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases, f'static elem_t conv_2_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv3.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases, f'static elem_t conv_3_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv4.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases, f'static elem_t conv_4_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv5.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases,  f'static elem_t conv_5_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['fc1.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases, f'static elem_t fc_6_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['fc2.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases, f'static elem_t fc_7_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['fc3.bias']
quantized_biases = quantize_biases(biases)
log_params(quantized_biases, f'static elem_t fc_8_b<{quantized_biases.shape[0]}>')


print(f"Weights and biases have been saved to {parameters_file}")

output_file = "alexnet_params.h"
process_tensor_file(parameters_file, output_file)
