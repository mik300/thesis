import argparse
import numpy as np
import torch
from pathlib import Path
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
from neural_networks.utils import set_model_axx_levels, init_transaxx, get_loaders_split, evaluate_test_accuracy, calibrate_model, load_scaling_factors, save_scaling_factors, save_weights, save_activations
from neural_networks.adapt.approx_layers.axx_layers import AdaPT_Conv2d
from transaxx.classification.utils import *
from transaxx.layers.adapt_convolution_layer import AdaptConv2D
from transaxx.classification.ptflops import get_model_complexity_info
from transaxx.classification.ptflops.pytorch_ops import conv_flops_counter_hook
from tqdm import tqdm
import torch.nn.functional as F
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
#import matplotlib.pyplot as plt
"""
This python file can be used to test the performance of the ResNet neural networks with different approximate multiplier configurations.
It also saves in a portable python format the weights and input activations of each layer for each multiplier configuration.
The code is organized as follows: 
1- loads the model and dataset
2- sets the requested approximation level to each layer (see functions "update_layer" and "update_model"
3- performs a dummy run with a single random image 
4- evaluates the test accuracy for each approximation level
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_function', default="ReLU", type=str, help="Activation function used for each act layer.")
    parser.add_argument('--batch-size', default=100, type=int, help="Number of images processed during each iteration")
    parser.add_argument('--data-dir', default="./data/", type=str, help="Directory in which the MNIST and FASHIONMNIST dataset are stored or should be downloaded")
    parser.add_argument('--dataset', default="cifar10", type=str, help="Select cifar10 or cifar100")
    parser.add_argument('--epochs', default=128, type=int, help="Number of training epochs")
    parser.add_argument('--num-workers', default=0, type=int, help="Number of threads used during the preprocessing of the dataset")
    parser.add_argument('--threads', default=10, type=int, help="Number of threads used during the inference, used only when neural-network-type is set to adapt")
    parser.add_argument('--split-val', default=0.1, type=float, help="The split-val is used to divide the training set in training and validation with the following dimensions: train=train_images*(1-split_val)  valid=train_images*split_val")
    parser.add_argument('--act-bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight-bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias-bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake-quant', default=True, type=bool, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--neural-network', default="resnet8", type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--execution-type', default='transaxx', type=str, help="Leave it like this")
    parser.add_argument('--disable-aug', default=False, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--reload', default=True, type=bool, help="Set to True to reload a pretraind model, set to False to train a new one")
    parser.add_argument('--continue-training', default=False, type=bool, help="Set to True to continue the training for a number of epochs that is the difference between the already trained epochs and the total epochs")
    parser.add_argument('--conv-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--conv-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")
    parser.add_argument('--linear-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--linear-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")
    parser.add_argument('--log', default=0, type=int, help="Set to 0 to print the parameters necessary for gemmini")

    parser.add_argument('--transaxx-quant', default=8, type=int, help="")
    parser.add_argument('--data-execution-type', default='quant', type=str, help="")
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    
    labels_path = 'labels.txt'
    if args.log == 1:
        with open(labels_path, 'w') as f:
                f.write(f'const int appr_level<{len(conv_axx_levels)}> row_align(1) = {conv_axx_levels};\n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        exit("Dataset not supported")


    if args.data_execution_type == "quant":
        execution_type = "quant"
        namebit = "_a"+str(args.act_bit)+"_w"+str(args.weight_bit)+"_b"+str(args.bias_bit)
    else:
        if args.data_execution_type == "float":
            execution_type = "float"
            namebit = ""
        else:
            execution_type = "transaxx"
            namebit = f"_{args.transaxx_quant}x{args.transaxx_quant}"
        

    if args.execution_type == "quant" or args.execution_type == "transaxx":
        if args.fake_quant:
            namequant = "_fake"
        else:
            namequant = "_int"
    else:
        namequant = ""


    filename = model_dir + args.neural_network + namebit + namequant + f"_{args.data_execution_type}_" + args.dataset + "_" + args.activation_function + ".pth"
    print(f'Loading model parameters from {filename}')
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    print(f'batch size = {args.batch_size}')
    
    #val_data, calib_data = cifar10_data_loader(data_path="./data/", batch_size=128)
    _, val_data, _ = get_loaders_split(args.data_dir,
     batch_size=args.batch_size, dataset_type=args.dataset, num_workers=args.num_workers,
     split_val=args.split_val, disable_aug=args.disable_aug)

    print(f"Number of images in the test set: {len(val_data.dataset)}")

    mode = {"execution_type":args.execution_type, "act_bit":args.act_bit, "weight_bit":args.weight_bit, "bias_bit":args.bias_bit, "fake_quant":args.fake_quant, "classes":num_classes, "act_type":args.activation_function}

    base_mult = "bw_mult_9_9_"
    
    if args.neural_network == "resnet8":
        model = resnet8(mode).to(device)
    elif args.neural_network == "resnet20":
        model = resnet20(mode).to(device)
    elif args.neural_network == "resnet32":
        model = resnet32(mode).to(device)
    elif args.neural_network == "resnet56":
        model = resnet56(mode).to(device)
    else:
        exit("error unknown CNN model name")

    conv_axx_levels, linear_axx_levels = set_model_axx_levels(model, args.conv_axx_level_list, args.conv_axx_level, args.linear_axx_level_list, args.linear_axx_level)

    if args.data_execution_type == "transaxx":
        init_transaxx(model, conv_axx_levels, linear_axx_levels, args, args.transaxx_quant, device, fake_quant=False)
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if args.execution_type == "transaxx":
            init_transaxx(model, conv_axx_levels, linear_axx_levels, args, args.transaxx_quant, device, fake_quant=False)

    model.to(device)
    model.eval()

    test_loss, test_acc = evaluate_test_accuracy(val_data, model, device)
    print(f'Mult: {conv_axx_levels}, {linear_axx_levels} | test loss:{test_loss:.5f} | final test acc:{test_acc:.5f}')

if __name__ == "__main__":
    main()
