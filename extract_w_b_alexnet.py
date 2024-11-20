import pickle
import torch
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

weights_file = 'alexnet_weights.txt'
biases_file = 'alexnet_biases.txt'

def log_weights(weights, layer_name):
    # Convert the tensor to a NumPy array for clean output
    weights_clean = weights.detach().cpu().numpy()  # Move to CPU and convert to NumPy
    # Format the output as a string
    formatted_weights = weights_clean.tolist()  # Convert to list for better formatting
    list_string = str(formatted_weights).replace('\n', '').replace('  ', ' ')  # Replace newlines if any
    list_string = list_string + ";"
    with open(weights_file, 'a') as f:
        f.write(f'{layer_name} row_align(1) = {list_string}\n')

def log_biases(biases, layer_name):
    biases_clean = biases.detach().cpu().numpy()
    formatted_biases = biases_clean.tolist()
    list_string = str(formatted_biases).replace('\n', '').replace('  ', ' ')
    list_string = list_string + ";"
    with open(biases_file, 'a') as f:
        f.write(f'{layer_name} row_align(1) = {list_string}\n')

def init_log():
    with open(weights_file, 'w') as f:
        f.write("")
    with open(biases_file, 'w') as f:
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

torch.set_printoptions(threshold=float('inf'), precision=16, sci_mode=False)

model = torch.load("neural_networks/models/alexnet_a8_w8_b32_fake_quant_cifar10_ReLU_epochs83.pth")

init_log()


weights = model['model_state_dict']['conv1.weight']
quantized_weights = quantize_weights(weights)
log_weights(quantized_weights, f'static const elem_t conv_1_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')

weights = model['model_state_dict']['conv2.weight']
quantized_weights = quantize_weights(weights)
log_weights(quantized_weights, f'static const elem_t conv2_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')

weights = model['model_state_dict']['conv3.weight']
quantized_weights = quantize_weights(weights)
log_weights(quantized_weights, f'static const elem_t conv3_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')

weights = model['model_state_dict']['conv4.weight']
quantized_weights = quantize_weights(weights)
log_weights(quantized_weights, f'static const elem_t conv4_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')

weights = model['model_state_dict']['conv5.weight']
quantized_weights = quantize_weights(weights)
log_weights(quantized_weights,  f'static const elem_t conv5_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')

weights = model['model_state_dict']['fc1.weight']
quantized_weights = quantize_linear_weights(weights)
log_weights(quantized_weights, f'static const elem_t fc_6_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}>')

weights = model['model_state_dict']['fc2.weight']
quantized_weights = quantize_linear_weights(weights)
log_weights(quantized_weights, f'static const elem_t fc_7_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}>')

weights = model['model_state_dict']['fc3.weight']
quantized_weights = quantize_linear_weights(weights)
log_weights(quantized_weights, f'static const elem_t fc_8_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}>')


print(f"Weights have been scaled to int8 and saved to {weights_file}")



biases = model['model_state_dict']['conv1.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases, f'static elem_t conv_1_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv2.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases, f'static elem_t conv_2_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv3.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases, f'static elem_t conv_3_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv4.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases, f'static elem_t conv_4_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['conv5.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases,  f'static elem_t conv_5_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['fc1.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases, f'static elem_t fc_6_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['fc2.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases, f'static elem_t fc_7_b<{quantized_biases.shape[0]}>')

biases = model['model_state_dict']['fc3.bias']
quantized_biases = quantize_biases(biases)
log_weights(quantized_biases, f'static elem_t fc_8_b<{quantized_biases.shape[0]}>')


print(f"Biases have been scaled to int8 and saved to {biases_file}")

