import pickle
import torch

approx_level = input("Enter the approximation level: ")
filename = f'neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_bw_mult_9_9_{approx_level}_inputs.pkl'  

# Open the pickle file in read-binary mode
with open(filename, 'rb') as file:
    data = pickle.load(file)

#torch.set_printoptions(edgeitems=1000, linewidth=300)
#print(data)
print(data.keys())
print(f"conv1: {data['conv1'].shape}")
print(f"layer1.0.conv1: {data['layer1.0.conv1'].shape}")
print(f"layer1.0.conv2: {data['layer1.0.conv2'].shape}")
print(f"layer2.0.conv1: {data['layer2.0.conv1'].shape}")
print(f"layer2.0.conv2: {data['layer2.0.conv2'].shape}")
print(f"layer3.0.conv1: {data['layer3.0.conv1'].shape}")
print(f"layer3.0.conv2: {data['layer3.0.conv2'].shape}")


