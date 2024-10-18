import pickle
import torch
import numpy as np


def log_weights(weights, layer_name):
    
    # Convert the tensor to a NumPy array for clean output
    weights_clean = weights.detach().cpu().numpy()  # Move to CPU and convert to NumPy
    # Format the output as a string
    formatted_weights = weights_clean.tolist()  # Convert to list for better formatting
    list_string = str(formatted_weights).replace('\n', '').replace('  ', ' ')  # Replace newlines if any
    list_string = list_string + ";"
    with open('weights.txt', 'a') as f:
        f.write(f'{layer_name} row_align(1) = {list_string}\n')

def log_biases(biases, layer_name):
    biases_clean = biases.detach().cpu().numpy()
    formatted_biases = biases_clean.tolist()
    list_string = str(formatted_biases).replace('\n', '').replace('  ', ' ')
    list_string = list_string + ";"
    with open('biases.txt', 'a') as f:
        f.write(f'{layer_name} row_align(1) = {list_string}\n')

def init_log():
    with open('weights.txt', 'w') as f:
        f.write("")
    with open('biases.txt', 'w') as f:
        f.write("")

torch.set_printoptions(threshold=float('inf'), precision=16, sci_mode=False)

model = torch.load("neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_calibrated.pth")

init_log()

weights = model['model_state_dict']['conv1.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(0, 2, 3, 1) #permute the dimensions to match the dimensions of weights in gemmini
log_weights(quantized_weights, f'static elem_t conv_1_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')

weights = model['model_state_dict']['layer1.0.conv1.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(0, 2, 3, 1)
log_weights(quantized_weights, f'static elem_t layer1_0_conv_1_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')


weights = model['model_state_dict']['layer1.0.conv2.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(0, 2, 3, 1)
log_weights(quantized_weights, f'static elem_t layer1_0_conv_2_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')


weights = model['model_state_dict']['layer2.0.conv1.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(0, 2, 3, 1)
log_weights(quantized_weights, f'static elem_t layer2_0_conv_1_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')


weights = model['model_state_dict']['layer2.0.conv2.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(0, 2, 3, 1)
log_weights(quantized_weights,  f'static elem_t layer2_0_conv_2_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')


weights = model['model_state_dict']['layer3.0.conv1.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(0, 2, 3, 1)
log_weights(quantized_weights, f'static elem_t layer3_0_conv_1_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')


weights = model['model_state_dict']['layer3.0.conv2.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(0, 2, 3, 1)
log_weights(quantized_weights, f'static elem_t layer3_0_conv_2_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}><{quantized_weights.shape[2]}><{quantized_weights.shape[3]}>')


weights = model['model_state_dict']['linear.weight']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_weights = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_weights = quantized_weights.permute(1, 0)
log_weights(quantized_weights, f'static elem_t linear_w<{quantized_weights.shape[0]}><{quantized_weights.shape[1]}>')


print("Weights have been scaled to int8 and saved to weights.txt")


biases = model['model_state_dict']['linear.bias']
t_max = torch.max(torch.abs(torch.min(weights)), torch.abs(torch.max(weights))).item()
scaling_factor = 127/t_max
quantized_biases = torch.clamp(torch.round(scaling_factor * weights), min=-128, max=127).to(torch.int8)
quantized_biases = quantized_biases.permute(1, 0)
log_biases(quantized_biases, f'static elem_t linear_b<{quantized_biases.shape[0]}><{quantized_biases.shape[1]}>')
print("Biases have been scaled to int8 and saved to biases.txt")



## Comparing the weights in _weights.pkl and ReLU.pth (they're the same)
# compare = 'layer3.0.conv2'
# layer1_0_conv1_weights = model[f'{compare}']
# layer1_0_conv1_weights_from_ReLU = model2['model_state_dict'><f'{compare}.weight']

# #layer1_0_conv1_weights = layer1_0_conv1_weights.to('cpu')
# layer1_0_conv1_weights_from_ReLU = layer1_0_conv1_weights_from_ReLU.to('cpu')
# print(f'layer1_0_conv1_weights shape = {layer1_0_conv1_weights}')
# print(f'layer1_0_conv1_weights_from_ReLU shape = {layer1_0_conv1_weights_from_ReLU}')

# if torch.equal(layer1_0_conv1_weights, layer1_0_conv1_weights_from_ReLU):
#     print('Equal')
# else:
#     print('Different')
