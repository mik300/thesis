import pickle
import torch

filename = f'neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_scaling_factors.pkl'  
# Open the pickle file in read-binary mode
with open(filename, 'rb') as file:
    data = pickle.load(file)


print(data.keys())
print(f"conv1: {data['conv1']}")
print(f"layer1.0.conv1: {data['layer1.0.conv1']}")
print(f"layer1.0.conv2: {data['layer1.0.conv2']}")
print(f"layer2.0.conv1: {data['layer2.0.conv1']}")
print(f"layer2.0.conv2: {data['layer2.0.conv2']}")
print(f"layer3.0.conv1: {data['layer3.0.conv1']}")
print(f"layer3.0.conv2: {data['layer3.0.conv2']}")


