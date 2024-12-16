import torch
import torch.nn.functional as F
import numpy as np
# from transaxx.layers.adapt_convolution_layer import *


torch.set_printoptions(threshold=float('inf'), precision=16, sci_mode=False)
# Define types for input, weights, and bias
elem_t = np.int32
acc_t = np.int32

# Random number generator emulating the C functions
next_val = 1  # Global state for the PRNG


def myrand():
    global next_val
    next_val = next_val * 1103515245 + 12345
    return (next_val // 65536) % 32768


def mysrand(seed):
    global next_val
    next_val = seed


def init_pseudo_random(buf, size):
    for i in range(size):
        buf[i] = elem_t((myrand() % 8) - 8)


def init_pseudo_random_acc(buf, size):
    for i in range(size):
        buf[i] = acc_t((myrand() % 5) - 2)


# Define input dimensions
BATCH_SIZE = 1
IN_CHANNELS = 3  # Number of input channels
OUT_CHANNELS = 3  # Number of output channels
IN_ROW_DIM, IN_COL_DIM = 5, 5  # Input feature map dimensions
KERNEL_DIM = 3  # Convolution kernel size
stride = 2
padding = 1

# Compute output dimensions
OUT_ROW_DIM = (IN_ROW_DIM - KERNEL_DIM + 2 * padding) // stride + 1
OUT_COL_DIM = (IN_COL_DIM - KERNEL_DIM + 2 * padding) // stride + 1

# Initialize inputs, weights, and biases
input_mat = np.zeros((BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS), dtype=elem_t)
weights = np.zeros((OUT_CHANNELS, KERNEL_DIM, KERNEL_DIM, IN_CHANNELS), dtype=elem_t)
bias = np.zeros(OUT_CHANNELS, dtype=acc_t)

mysrand(3)
print("Randomize inputs...")
init_pseudo_random(input_mat.ravel(), input_mat.size)

print("Randomize weights...")
init_pseudo_random(weights.ravel(), weights.size)

print("Randomize bias...")
init_pseudo_random_acc(bias, bias.size)

# Convert to PyTorch tensors and adjust shapes
input_tensor = torch.tensor(input_mat.transpose(0, 3, 1, 2), dtype=torch.float32)  # Shape: [B, C, H, W]
weights_tensor = torch.tensor(weights.transpose(0, 3, 1, 2), dtype=torch.float32)  # Shape: [O, I, KH, KW]
bias_tensor = torch.tensor(bias, dtype=torch.float32)  # Shape: [O]

# Perform convolution
output_tensor = F.conv2d(input_tensor, weights_tensor, bias=bias_tensor, stride=stride, padding=padding)

# adapt_conv = AdaptConv2D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, kernel_size=KERNEL_DIM, stride=stride, padding=padding, bias=False, axx_mult="bw_mult_8_8_255", quant_bits = 8, fake_quant = True)
# adapt_conv.set_axx_kernel()
# adapt_conv.weight.data = weights_tensor.clone()
# transaxx_output_tensor = adapt_conv(input_tensor)

# Print results
print("\nBias:")
print(bias_tensor)

print("\nWeights:")
weights_tensor_permutted = weights_tensor.permute(0, 2, 3, 1)
print(weights_tensor_permutted)

print("Input Tensor:")
input_tensor_permutted = input_tensor.permute(0, 2, 3, 1)
print(input_tensor_permutted)

print("\nOutput Tensor:")
output_tensor_permutted = output_tensor.permute(0, 2, 3, 1)
print(output_tensor_permutted)

# print("\nTransAxx Output Tensor:")
# output_tensor_permutted = transaxx_output_tensor.permute(0, 2, 3, 1)
# print(output_tensor_permutted)
