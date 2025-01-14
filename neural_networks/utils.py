import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from neural_networks.custom_layers import Conv2d, Linear, Act
from neural_networks.adapt.approx_layers.axx_layers import AdaPT_Conv2d
from neural_networks.CIFAR10.resnet import ResidualModule
from torchvision import ops
from torchvision.models import efficientnet as ef
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import random
import warnings

from transaxx.classification.utils import *
from transaxx.layers.adapt_convolution_layer import AdaptConv2D
from transaxx.classification.ptflops import get_model_complexity_info
from transaxx.classification.ptflops.pytorch_ops import conv_flops_counter_hook

def save_activations(model, input_tensor, filename="input_tensors.pkl"):
    input_tensors = {}
    def get_inputs(module_name):
        def forward_hook(module, input, output):
            input_tensors[module_name] = input[0]
        return forward_hook

    hooks = []

    # Recursive function to attach hooks to convolutional layers
    def recursive_hook(module, prefix=""):
        for name, child in module.named_children():
            module_name = prefix + ('.' if prefix else '') + name
            if isinstance(child, AdaPT_Conv2d):
                # Attach a hook to the convolutional layer
                hook = child.register_forward_hook(get_inputs(module_name))
                hooks.append(hook)
            # Recursively apply to child modules
            recursive_hook(child, module_name)

    recursive_hook(model)
    # Perform a forward pass to trigger the hooks
    with torch.no_grad():
        model(input_tensor)

    # Remove all hooks after the forward pass
    for hook in hooks:
        hook.remove()

    with open(filename, 'wb') as f:
        pickle.dump(input_tensors, f)
    print(f"Input tensors saved to {filename}")

def save_weights(model, filename='weight_tensors.pkl'):
    weights = {}

    def process_module(module, path=''):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, (AdaPT_Conv2d)):
                if hasattr(child, 'weight'):
                    weights[child_path] = child.weight
                    # print("DEBUG SAVED SCALING FACTOR:  ", child_path)
                else:
                    print(f"Warning: 'weight' not found in {child_path}")
            process_module(child, child_path)

    process_module(model)

    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Weights saved to {filename}")


def save_scaling_factors(model, filename='scaling_factors.pkl'):
    scaling_factors = {}
    def process_module(module, path=''):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, (Conv2d, Linear, Act)):
                if hasattr(child, 'scaling_factor'):
                    scaling_factors[child_path] = child.scaling_factor
                    # print("DEBUG SAVED SCALING FACTOR:  ", child_path)
                else:
                    print(f"Warning: 'scaling_factor' not found in {child_path}")
            process_module(child, child_path)

    process_module(model)

    with open(filename, 'wb') as f:
        pickle.dump(scaling_factors, f)
    print(f"Scaling factors saved to {filename}")

def load_scaling_factors(model, filename='scaling_factors.pkl', device="cuda"):
    # Load the dictionary using pickle
    with open(filename, 'rb') as f:
        scaling_factors = pickle.load(f)

    def assign_scaling_factor(module, path=''):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if child_path in scaling_factors:
                if hasattr(child, 'scaling_factor'):
                    child.update_act_scale(scaling_factors[child_path].to(device))
                    #print("DEBUG RELOADED SCALING FACTOR:  ", child_path)
                else:
                    print(f"Warning: 'scaling_factor' attribute does not exist in {child_path}. Creating attribute.")
                    setattr(child, 'scaling_factor', scaling_factors[child_path])
            assign_scaling_factor(child, child_path)

    assign_scaling_factor(model)
    #print(f"Scaling factors loaded from {filename} and assigned to the model")

def update_calibrator(in_module, name, calibrate, calibrator_status, override_calibration):
    done = False
    if  isinstance(in_module, Conv2d) or isinstance(in_module, Linear) or isinstance(in_module, Act) or isinstance(in_module, AdaPT_Conv2d):
        done = True
        in_module.calibrate = calibrate
        in_module.calibrator.status = calibrator_status
        if override_calibration:
            in_module.calibrator.calibrate_funct()
        if not calibrate:
            in_module.update_act_scale()
            print(f'Layer {name} calibrated with scaling factor {in_module.scaling_factor}, alpha is {in_module.calibrator.calibrated_value}')

    return done

def update_calibrator_status_resnet(model, calibrate, calibrator_status, override_calibration):
    for name, module in model.named_children():
        done = update_calibrator(module, name, calibrate, calibrator_status, override_calibration)
        if not done:
            if isinstance(module, ResidualModule) or isinstance(module, nn.Sequential):
                update_calibrator_status_resnet(module, calibrate, calibrator_status, override_calibration)


def calibrate_model(model, data_loader, device="cuda", model_type="resnet", data_limit=1e8):
    model.eval()
    # order of calibrator_status ["itminmax", "binedges", "ithists", "computeth", "itxaboveth", "compthvalue"]
    print("computing min e max")
    if model_type == "resnet":
        update_calibrator_status_resnet(model, True, "itminmax", False)
    with torch.no_grad():
        tot = 0
        for i, (X, y) in enumerate(tqdm(data_loader)):
            X, y = X.to(device), y.to(device)
            output = model(X)
            tot += y.size(0)
            if tot >= data_limit:
                break
    print("computing binedges")
    if model_type == "resnet":
        update_calibrator_status_resnet(model, True, "binedges", True)
    print("computing histograms")

    if model_type == "resnet":
        update_calibrator_status_resnet(model, True, "ithists", False)
    with torch.no_grad():
        tot = 0
        for i, (X, y) in enumerate(tqdm(data_loader)):
            X, y = X.to(device), y.to(device)
            output = model(X)
            tot += y.size(0)
            if tot >= data_limit:
                break
    print("computing threshold")

    if model_type == "resnet":
        update_calibrator_status_resnet(model, True, "computeth", True)
    print("computing values above threshold")

    if model_type == "resnet":
        update_calibrator_status_resnet(model, True, "itxaboveth", False)
    with torch.no_grad():
        tot = 0
        for i, (X, y) in enumerate(tqdm(data_loader)):
            X, y = X.to(device), y.to(device)
            output = model(X)
            tot += y.size(0)
            if tot >= data_limit:
                break
    print("computing th value")

    if model_type == "resnet":
        update_calibrator_status_resnet(model, True, "compthvalue", True)
    print("calibration done, updating scaling factors")

    if model_type == "resnet":
        update_calibrator_status_resnet(model, False, None, False)

def evaluate_test_accuracy(test_loader, model, device="cuda"):
    """
    Evaluate the model top-1 accuracy using the dataloader passed as argument
    @param test_loader: dataloader that contains the test images and ground truth, could be either the test or validation split
    @param model: any torch.nn.Module with a final dense layer
    @param device: use "cpu" for adapt with approximate models, "cuda" for GPU for float or quantized baseline models
    @return: return the test loss and test accuracy as floating points values
    """
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader)):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mean_std = {"mean": cifar10_mean, "std": cifar10_std}
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
fashionmnist_mean = 0.286
fasnionmnist_std = 0.353
mnist_mean = 0.1307
mnist_std = 0.3081
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)


def get_loaders_split(dir_, batch_size, dataset_type, num_workers=2, split_val=0.2, disable_aug=False, is_integer=False, test_size=None, resize_to_imagenet=False):
    """
    Generate train and test dataloaders
    @param dir_: string, directory in which the CIFAR-10 dataset is stored
    @param batch_size: int, dimension of the batch
    @param dataset_type: str, dataset type can be "cifar10", "mnist" or "fashionmnist"
    @param num_workers: int, number of workers used for the pre-processing of the dataset, min is 0 max is the thread count of the CPU
    @param split_val: float, number between 0 and 1 used to split the training set in training/validation, the validation set has a dimension of split_val*training_set
    @param disable_aug: disable data augmentation, se to False in order to process the same data for reproducibility
    @param is_integer: normalization done for integer activations, must beset to True only for NEMO processing
    @return: returns the dataloaders
    """
    if is_integer:
        int_mean = -0.5
    else:
        int_mean = 0

    if dataset_type == "cifar10":
        if disable_aug:
            if resize_to_imagenet:
                print(f'Resizing for imagenet')
                train_transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), transforms.Normalize(int_mean, 1)])
            else:
                train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), transforms.Normalize(int_mean, 1)])
        else:
            if resize_to_imagenet:
                print(f'Resizing for imagenet')
                train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.Resize((227, 227)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), transforms.Normalize(int_mean, 1)])
            else:
                train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), transforms.Normalize(int_mean, 1)])
        if resize_to_imagenet:
            print(f'Resizing for imagenet')
            test_transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), transforms.Normalize(int_mean, 1)])
        else:
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std), transforms.Normalize(int_mean, 1)])

        train_dataset = datasets.CIFAR10(dir_, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR10(dir_, train=True, transform=test_transform, download=True)
        test_dataset = datasets.CIFAR10(dir_, train=False, transform=test_transform, download=True)

    elif dataset_type == "cifar100":
        if disable_aug:
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std), transforms.Normalize(int_mean, 1)])
        else:
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std), transforms.Normalize(int_mean, 1)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std),
                                             transforms.Normalize(int_mean, 1)])
        train_dataset = datasets.CIFAR100(dir_, train=True, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR100(dir_, train=True, transform=test_transform, download=True)
        test_dataset = datasets.CIFAR100(dir_, train=False, transform=test_transform, download=True)

    elif dataset_type == "fashionmnist":
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(fashionmnist_mean, fasnionmnist_std), transforms.Normalize(int_mean, 1)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(fashionmnist_mean, fasnionmnist_std), transforms.Normalize(int_mean, 1)])
        train_dataset = datasets.FashionMNIST(dir_, train=True, transform=train_transform, download=True)
        val_dataset = datasets.FashionMNIST(dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.FashionMNIST(dir_, train=False, transform=test_transform, download=True)

    elif dataset_type == "mnist":
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std), transforms.Normalize(int_mean, 1)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std), transforms.Normalize(int_mean, 1)])
        train_dataset = datasets.MNIST(dir_, train=True, transform=train_transform, download=True)
        val_dataset = datasets.MNIST(dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.MNIST(dir_, train=False, transform=test_transform, download=True)

    elif dataset_type == "imagenet":
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

        # Load the ImageNet validation dataset
        train_dataset = datasets.ImageNet(root=dir_, split='train', transform=data_transforms)
        val_dataset = datasets.ImageNet(root=dir_, split='val', transform=data_transforms)
        test_dataset = None

    else:
        exit("unknown dataset type")

    if dataset_type == "imagenet":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        # Evaluation of indices for training and validation splits
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int((split_val * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        # Train and validation samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        if test_size is not None:
            #subset_indices = [random.randint(0, len(test_dataset) - 1) for _ in range(test_size)] 
            #print(f'subset_indices = {subset_indices}')
            subset_indices = list(range(test_size))  # Selecting first 4 images
            test_dataset = Subset(test_dataset, subset_indices)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, sampler=train_sampler, num_workers=num_workers)

        valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, sampler=valid_sampler, num_workers=num_workers)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

class ArgumentWarning(Warning):
    pass

def init_transaxx(model, conv_axx_levels, linear_axx_levels, args, quant_bits=8, device='cuda', fake_quant=True):
    fake_quant = not fake_quant
    #Initialize conv layers
    conv2d_layers = [(name, module) for name, module in model.named_modules() if (isinstance(module, torch.nn.Conv2d) or isinstance(module, AdaptConv2D)) and ("head" not in name and "reduction" not in name)]
    linear_layers = [(name, module) for name, module in model.named_modules() if (isinstance(module, torch.nn.Linear) or isinstance(module, AdaPT_Linear)) and ("head" not in name and "reduction" not in name)]
    print(f'Number of conv layer is {len(conv2d_layers)}')
    print(f'Number of linear layer is {len(linear_layers)}')


    axx_list = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_0', 'axx_power' : 1.0, 'quant_bits' : quant_bits, 'fake_quant' : False}]*len(conv2d_layers)
    axx_list_linear = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_0', 'axx_power' : 1.0, 'quant_bits' : quant_bits, 'fake_quant' : False}]*len(linear_layers)

    print(f'len(conv2d_layers) = {len(conv2d_layers)}')
    print(f'len(conv_axx_levels) = {len(conv_axx_levels)}')
    if len(conv2d_layers) != len(conv_axx_levels): #If number of conv_axx_levels is less than number of conv layers, extend conv_axx_levels with all 0s
        if len(conv2d_layers) < len(conv_axx_levels):
            warnings.warn(f"Warning: The size of axx-level is larger than the number of conv layers, the first {len(conv2d_layers)} will be used", category=ArgumentWarning)
            conv_axx_levels = conv_axx_levels[:len(conv2d_layers)]
        else:
            num_elements_to_add = len(conv2d_layers) - len(conv_axx_levels)
            conv_axx_levels.extend([0] * num_elements_to_add)

    if len(linear_layers) != len(linear_axx_levels):
        if len(linear_layers) < len(linear_axx_levels):
            warnings.warn(f"Warning: The size of linear-axx-level is larger than the number of linear layers, the first {len(linear_layers)} will be used", category=ArgumentWarning)
            linear_axx_levels = linear_axx_levels[:len(linear_layers)]
        else:
            num_elements_to_add = len(linear_layers) - len(linear_axx_levels)
            linear_axx_levels.extend([0] * num_elements_to_add)

    for i in range(len(conv_axx_levels)):
        axx_list[i:i+1] = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_{conv_axx_levels[i]}', 'axx_power' : 0.7082, 'quant_bits' : 8, 'fake_quant' : False}] * 1
    for i in range(len(linear_axx_levels)):
        axx_list_linear[i:i+1] = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_{linear_axx_levels[i]}', 'axx_power' : 0.7082, 'quant_bits' : 8, 'fake_quant' : False}] * 1


    start = time.time()
    replace_conv_layers(model, AdaptConv2D, axx_list, 0, 0, layer_count=[0], returned_power = [0], initialize = True)  
    replace_linear = False
    if replace_linear:
        replace_linear_layers(model,  AdaPT_Linear, axx_list_linear, 0, 0, layer_count=[0], returned_power = [0], initialize = True)  
    print('Time to compile cuda extensions: ', time.time()-start)

    with torch.cuda.device(0):
        total_macs, total_params, layer_specs = get_model_complexity_info(model, (3, 32, 32),as_strings=False, print_per_layer_stat=True,
                                                            custom_modules_hooks={AdaptConv2D : conv_flops_counter_hook}, 
                                                            param_units='M', flops_units='MMac',
                                                            verbose=True)
    print(f'Computational complexity:  {total_macs/1000000:.2f} MMacs')
    print(f'Number of parameters::  {total_params/1000000:.2f} MParams')     

    _, calib_data = cifar10_data_loader(data_path="./data/", batch_size=args.batch_size)
    with torch.no_grad():
        stats = collect_stats(model, calib_data, num_batches=2, device=device)
        amax = compute_amax(model, method="percentile", percentile=99.99, device=device) 

    for i in range(len(conv2d_layers)):
        axx_list[i:i+1] = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_{conv_axx_levels[i]}', 'axx_power' : 0.7082, 'quant_bits' : quant_bits, 'fake_quant' : fake_quant}] * 1
    for i in range(len(linear_layers)):
        axx_list_linear[i:i+1] = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_{linear_axx_levels[i]}', 'axx_power' : 0.7082, 'quant_bits' : quant_bits, 'fake_quant' : fake_quant}] * 1
    
    returned_power = [0]
    print(f"axx_list size = {len(axx_list)}")
    print(f"conv2d_layers size = {len(conv2d_layers)}")
    print(f"axx_list_linear size = {len(axx_list_linear)}")
    print(f"linear_layers size = {len(linear_layers)}")

    i = 0
    for axx in axx_list:
        print(f"{conv2d_layers[i][0]}: {axx['axx_mult']}")
        i += 1
    i = 0
    print(f'len(axx_list_linear) = {len(axx_list_linear)}')
    print(f'len(linear_layers) = {len(linear_layers)}')
    for axx in axx_list_linear:
        print(f"{linear_layers[i][0]}: {axx['axx_mult']}")
        i += 1
    
    replace_conv_layers(model,  AdaptConv2D, axx_list, total_macs, total_params, layer_count=[0], returned_power = returned_power, initialize = False) 
    if replace_linear:
        replace_linear_layers(model, AdaPT_Linear, axx_list_linear, total_macs, total_params, layer_count=[0], returned_power = returned_power, initialize = False) 


def init_transaxx_train(model, conv_axx_levels, args, quant_bits=8, device='cuda', fake_quant=True):
    fake_quant = not fake_quant
    #Initialize conv layers
    conv2d_layers = [(name, module) for name, module in model.named_modules() if (isinstance(module, torch.nn.Conv2d) or isinstance(module, AdaptConv2D)) and ("head" not in name and "reduction" not in name)]
    print(f'Number of conv layer is {len(conv2d_layers)}')
    axx_list = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_0', 'axx_power' : 1.0, 'quant_bits' : quant_bits, 'fake_quant' : False}]*len(conv2d_layers)
    if len(conv2d_layers) != len(conv_axx_levels):
        if len(conv2d_layers) < len(conv_axx_levels):
            warnings.warn(f"The size of linear-axx-level is larger than the number of conv layers, the first {len(conv2d_layers)} will be used")
            conv_axx_levels = conv_axx_levels[:len(conv2d_layers)]
        else:
            num_elements_to_add = len(conv2d_layers) - len(conv_axx_levels)
            conv_axx_levels.extend([0] * num_elements_to_add)

    for i in range(len(conv_axx_levels)):
        axx_list[i:i+1] = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_{conv_axx_levels[i]}', 'axx_power' : 0.7082, 'quant_bits' : 8, 'fake_quant' : False}] * 1

    start = time.time()
    replace_conv_layers(model, AdaptConv2D, axx_list, 0, 0, layer_count=[0], returned_power = [0], initialize = True)  
    print('Time to compile cuda extensions: ', time.time()-start)

    with torch.cuda.device(0):
        total_macs, total_params, layer_specs = get_model_complexity_info(model, (3, 32, 32),as_strings=False, print_per_layer_stat=True,
                                                            custom_modules_hooks={AdaptConv2D : conv_flops_counter_hook}, 
                                                            param_units='M', flops_units='MMac',
                                                            verbose=True)
    print(f'Computational complexity:  {total_macs/1000000:.2f} MMacs')
    print(f'Number of parameters::  {total_params/1000000:.2f} MParams')     

    _, calib_data = cifar10_data_loader(data_path="./data/", batch_size=args.batch_size)
    with torch.no_grad():
        stats = collect_stats(model, calib_data, num_batches=2, device=device)
        amax = compute_amax(model, method="percentile", percentile=99.99, device=device) 

    for i in range(len(conv2d_layers)):
        axx_list[i:i+1] = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_{conv_axx_levels[i]}', 'axx_power' : 0.7082, 'quant_bits' : quant_bits, 'fake_quant' : fake_quant}] * 1
    #axx_list[0:len(conv2d_layers)] = [{'axx_mult' : f'bw_mult_{quant_bits}_{quant_bits}_{appr_level}', 'axx_power' : 0.7082, 'quant_bits' : quant_bits, 'fake_quant' : fake_quant}] * len(conv2d_layers)
    returned_power = [0]
    print(f"axx_list size = {len(axx_list)}")
    print(f"conv2d_layers size = {len(conv2d_layers)}")
    i = 0
    for axx in axx_list:
        print(f"{conv2d_layers[i][0]}: {axx['axx_mult']}")
        i += 1
    replace_conv_layers(model,  AdaptConv2D, axx_list, total_macs, total_params, layer_count=[0], returned_power = returned_power, initialize = False) 


def set_model_axx_levels(model, conv_axx_level_list, conv_axx_level, linear_axx_level_list, linear_axx_level):
    nb_conv2d_layers = [(name, module) for name, module in model.named_modules() if (isinstance(module, torch.nn.Conv2d) or isinstance(module, AdaPT_Conv2d)) and ("head" not in name and "reduction" not in name)]
    nb_linear_layers = [(name, module) for name, module in model.named_modules() if (isinstance(module, torch.nn.Linear) or isinstance(module, AdaPT_Linear)) and ("head" not in name and "reduction" not in name)]

    conv_axx_levels = []
    if conv_axx_level_list is None:
        for i in range(len(nb_conv2d_layers)):
            conv_axx_levels.append(conv_axx_level)
    elif len(nb_conv2d_layers) != len(conv_axx_level_list):
        if len(nb_conv2d_layers) < len(conv_axx_level_list):
            warnings.warn(f"The size of linear-axx-level is larger than the number of conv layers, the first {len(conv_axx_level_list)} will be used")
            conv_axx_levels = conv_axx_level_list[:len(nb_conv2d_layers)]
        else:
            num_elements_to_add = len(nb_conv2d_layers) - len(conv_axx_level_list)
            conv_axx_levels = conv_axx_level_list
            conv_axx_levels.extend([0] * num_elements_to_add)
    else:
        conv_axx_levels = conv_axx_level_list


    if linear_axx_level_list is None:
        linear_axx_levels = []
        for i in range(len(nb_linear_layers)):
            linear_axx_levels.append(linear_axx_level)
    else:
        linear_axx_levels = linear_axx_level_list

    return conv_axx_levels, linear_axx_levels