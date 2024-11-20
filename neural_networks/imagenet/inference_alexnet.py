import argparse
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from neural_networks.imagenet.alexnet import alexnet
from neural_networks.utils import get_loaders_split, evaluate_test_accuracy, calibrate_model, load_scaling_factors, save_scaling_factors
from neural_networks.adapt.approx_layers.axx_layers import AdaPT_Conv2d
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

mult_index = 0
def update_layer(in_module, name, mult_type):
    global mult_index
    new_module = False
    if isinstance(in_module, AdaPT_Conv2d):
        print(f'mult_type = {mult_type}')
        new_module = True
        in_module.axx_mult = mult_type
        in_module.update_kernel()
        mult_index += 1
    return new_module

def update_model(model, mult_base, appr_level_list):
    global mult_index
    for name, module in model.named_children():
        mult_type = mult_base + str(appr_level_list[mult_index])
        new_module = update_layer(module, name, mult_type)
        if not new_module:
            update_model(module, mult_base, appr_level_list)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation-function', default="ReLU", type=str, help="Activation function used for each act layer.")
    parser.add_argument('--batch-size', default=100, type=int, help="Number of images processed during each iteration")
    parser.add_argument('--data-dir', default="./data/", type=str, help="Directory in which the MNIST and FASHIONMNIST dataset are stored or should be downloaded")
    parser.add_argument('--dataset', default="cifar10", type=str, help="")

    parser.add_argument('--num-workers', default=4, type=int, help="Number of threads used during the preprocessing of the dataset")
    parser.add_argument('--threads', default=12, type=int, help="Number of threads used during the inference, used only when neural-network-type is set to adapt")
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducible random initialization")
    parser.add_argument('--split-val', default=0.1, type=float, help="The split-val is used to divide the training set in training and validation with the following dimensions: train=train_images*(1-split_val)  valid=train_images*split_val")
    parser.add_argument('--act-bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight-bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias-bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake-quant', default=True, type=bool, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--neural-network', default="alexnet", type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--execution-type', default='quant', type=str, help="Select type of neural network and precision. Options are: float, quant, adapt. \n float: the neural network is executed with floating point precision.\n quant: the neural network weight, bias and activations are quantized to 8 bit\n adapt: the neural network is quantized to 8 bit and processed with exact/approximate multipliers")
    parser.add_argument('--disable-aug', default=False, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--reload', default=False, type=bool, help="Set to True to reload a pretraind model, set to False to train a new one")
    parser.add_argument('--continue-training', default=False, type=bool, help="Set to True to continue the training for a number of epochs that is the difference between the already trained epochs and the total epochs")
    parser.add_argument('--appr-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--appr-level-list', type=int, nargs=8, help="Exactly 8 integers specifying levels of approximation for each layer")
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.execution_type == 'adapt':
        device = "cpu"
        torch.set_num_threads(args.threads)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device used: {device}')

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif args.dataset == "imagenet":
        num_classes = 1000
    #else:
        #exit("Dataset not supported")

    if args.appr_level_list is None:
        approximation_levels = [args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level]
    else:
        approximation_levels = args.appr_level_list

    if args.execution_type == "quant" or args.execution_type == "adapt":
        execution_type = "quant"
        namebit = "_a"+str(args.act_bit)+"_w"+str(args.weight_bit)+"_b"+str(args.bias_bit)
    else:
        execution_type = "float"
        namebit = ""

    if args.execution_type == "quant" or args.execution_type == "adapt":
        if args.fake_quant:
            namequant = "_fake"
        else:
            namequant = "_int"
    else:
        namequant = ""

    #filename = model_dir + args.neural_network + namebit + namequant + "_" + execution_type + "_" + args.dataset +"_" + args.activation_function + "_epochs83.pth"
    filename = "neural_networks/models/alexnet_float_cifar10_ReLU.pth"
    filename_sc = model_dir + args.neural_network + namebit + namequant + "_" + execution_type + "_" + args.dataset +"_" + args.activation_function + '_scaling_factors.pkl'

    print(filename)
    #print(filename_sc)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    train_loader, valid_loader, test_loader = get_loaders_split(args.data_dir, batch_size=args.batch_size, dataset_type=args.dataset, num_workers=args.num_workers, split_val=args.split_val, disable_aug=args.disable_aug, resize_to_imagenet=True)

    print(f"Number of images in the test set: {len(test_loader.dataset)}")

    mode = {"execution_type":args.execution_type, "act_bit":args.act_bit, "weight_bit":args.weight_bit, "bias_bit":args.bias_bit, "fake_quant":args.fake_quant, "classes":num_classes, "act_type":args.activation_function}

    if args.neural_network == "alexnet":
        model = alexnet(mode).to(device)
    else:
        exit("error unknown CNN model name")

    print(f"model type = {type(model)}")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    #load_scaling_factors(model, filename_sc, device)

    
    base_mult = "bw_mult_9_9_"
    update_model(model, base_mult, approximation_levels)
    test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
    print(f'Mult: {approximation_levels}, test loss:{test_loss}, final test acc:{test_acc}')

if __name__ == "__main__":
    main()
