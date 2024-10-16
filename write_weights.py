import pickle
import torch

def log_to_file(weights, layer_name):
    with open('weights.txt', 'a') as f:
        f.write(f'{layer_name} = {weights}\n')

def init_log():
    with open('weights.txt', 'w') as f:
        f.write("")

filename = f'neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_weights.pkl'  
# Open the pickle file in read-binary mode
with open(filename, 'rb') as file:
    data = pickle.load(file)

init_log()

print(data.keys())

log_to_file(data['conv1'].shape, 'conv1')
log_to_file(data['layer1.0.conv1'].shape, 'layer1.0.conv1')
log_to_file(data['layer1.0.conv2'].shape, 'layer1.0.conv2')
log_to_file(data['layer2.0.conv1'].shape, 'layer2.0.conv1')
log_to_file(data['layer2.0.conv2'].shape, 'layer2.0.conv2')
log_to_file(data['layer3.0.conv1'].shape, 'layer3.0.conv1')
log_to_file(data['layer3.0.conv2'].shape, 'layer3.0.conv2')
print("Weights written to weights.txt")
