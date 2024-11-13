import torch

# checkpoint = torch.load("backup/resnet56_a8_w8_b32_fake_quant_cifar10_ReLU.pth")
# print("Epoch:", checkpoint['epoch'])
# print("Train Loss:", checkpoint['train_loss'])
#print("Train Accuracy:", checkpoint['train_acc'])
#print("Test Loss:", checkpoint['test_loss'])
#print("Test Accuracy:", checkpoint['test_acc'])
#print("Device:", checkpoint['device'])

#print("Train Parameters:", checkpoint['train_parameters'])

# resnet = torch.load("neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_calibrated.pth")
# print(resnet.keys())
# print("------")

at_resnet = torch.load("fast_adversarial/AT_models/AT_resnet8_fake_float_cifar10_ReLU_optO2_alpha10_epsilon8_5.pth")
#print(at_resnet.keys())
print("------")
print(f"conv1.weight = {at_resnet['conv1.weight'][0,0,0,0]}")

at_resnet = torch.load("fast_adversarial/AT_models/AT_resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_optO2_alpha10_epsilon8_5.pth")
#print(at_resnet.keys())
print("------")

print(f"conv1.weight = {at_resnet['conv1.weight'][0,0,0,0]}")
# for key in resnet['model_state_dict'].keys():
#     print(f"{key} ({type(key)})")

#print(f"epoch = {resnet['epoch']}")
#print(f"conv1.weight = {resnet['model_state_dict']['conv1.weight'][0,0,0,0]}")
#print(f"layer1.0.bn1.bias = {resnet['model_state_dict']['layer1.0.bn1.bias']}")
#print(f"layer1.0.conv1.weight = {resnet['model_state_dict']['layer1.0.conv1.weight'][0,0,0,0]}")
# print(f"layer3.0.conv2.weight = {resnet['model_state_dict']['layer3.0.conv2.weight'].shape}")
# print(f"train_loss = {resnet['train_loss']}")
# print(f"train_acc = {resnet['train_acc']}")
# print(f"test_loss = {resnet['test_loss']}")
# print(f"test_acc = {resnet['test_acc']}")
# print(f"device = {resnet['device']}")
# print(f"train_parameters = {resnet['train_parameters']}")
#print(f"linear.weight = {resnet['model_state_dict']['linear.weight'].shape}")


# key_to_find = 'weight'
# if key_to_find in resnet:
#     print(f"The key '{key_to_find}' exists with value: {resnet[key_to_find]}")
# else:
#     print(f"The key '{key_to_find}' does not exist.")


#print(f"layer1.0.conv1.weight = {resnet['model_state_dict']['layer1.0.conv1.weight'].shape}")