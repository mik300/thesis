import pickle
import torch

filename = f'neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_weights.pkl'  
# Open the pickle file in read-binary mode
with open(filename, 'rb') as file:
    data = pickle.load(file)

# filename2 = f'neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU.pkl'  
# # Open the pickle file in read-binary mode
# with open(filename2, 'rb') as file:
#     data2 = pickle.load(file)

print(data.keys())
print(f"conv1: {data['conv1'].shape}")
print(f"layer1.0.conv1: {data['layer1.0.conv1'][0,0,0,0].dtype}")
print(f"layer1.0.conv2: {data['layer1.0.conv2'].shape}")
print(f"layer2.0.conv1: {data['layer2.0.conv1'].shape}")
print(f"layer2.0.conv2: {data['layer2.0.conv2'].shape}")
print(f"layer3.0.conv1: {data['layer3.0.conv1'].shape}")
print(f"layer3.0.conv2: {data['layer3.0.conv2'].shape}")


