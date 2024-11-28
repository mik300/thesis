import argparse
import numpy as np
import torch
from pathlib import Path
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
from neural_networks.utils import get_loaders_split, evaluate_test_accuracy, calibrate_model, load_scaling_factors, save_scaling_factors, save_weights, save_activations
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
    parser.add_argument('--execution-type', default='adapt', type=str, help="Leave it like this")
    parser.add_argument('--disable-aug', default=False, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--reload', default=True, type=bool, help="Set to True to reload a pretraind model, set to False to train a new one")
    parser.add_argument('--continue-training', default=False, type=bool, help="Set to True to continue the training for a number of epochs that is the difference between the already trained epochs and the total epochs")
    parser.add_argument('--appr-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--appr-level-list', type=int, nargs=8, help="Exactly 8 integers specifying levels of approximation for each layer")
    parser.add_argument('--log', default=0, type=int, help="Set to 0 to print the parameters necessary for gemmini")
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.appr_level_list is None:
        approximation_levels = [args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level]
    else:
        approximation_levels = args.appr_level_list

    labels_path = 'labels.txt'
    if args.log == 1:
        with open(labels_path, 'w') as f:
                f.write(f'const int appr_level<{len(approximation_levels)}> row_align(1) = {approximation_levels};\n')

    device = "cuda"
    torch.set_num_threads(args.threads)

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        exit("Dataset not supported")

    namebit = "_a"+str(args.act_bit)+"_w"+str(args.weight_bit)+"_b"+str(args.bias_bit)

    if args.fake_quant:
        namequant = "_fake"
    else:
        namequant = "_int"


    filename = model_dir + args.neural_network + namebit + namequant + "_quant_" + args.dataset + "_" + args.activation_function + "_calibrated.pth"
    filename_sc = model_dir + args.neural_network + namebit + namequant + "_quant_" + args.dataset +"_" + args.activation_function + '_scaling_factors.pkl'

    print(filename_sc)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    print(f'batch size = {args.batch_size}')
    
    val_data, calib_data = cifar10_data_loader(data_path="./data/", batch_size=128)

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
    print(f"model type = {type(model)}")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    conv2d_layers = [(name, module) for name, module in model.named_modules() if (isinstance(module, torch.nn.Conv2d) or isinstance(module, AdaptConv2D) or isinstance(module, AdaPT_Conv2d)) and ("head" not in name and "reduction" not in name)]
    axx_list = [{'axx_mult' : 'mul8s_acc', 'axx_power' : 1.0, 'quant_bits' : 8, 'fake_quant' : False}]*len(conv2d_layers)
    axx_list[3:4] = [{'axx_mult' : 'bw_mult_9_9_0', 'axx_power' : 0.7082, 'quant_bits' : 8, 'fake_quant' : False}] * 1
    start = time.time()
    replace_conv_layers(model,  AdaptConv2D, axx_list, 0, 0, layer_count=[0], returned_power = [0], initialize = True)  
    print('Time to compile cuda extensions: ', time.time()-start)

    with torch.cuda.device(0):
        total_macs, total_params, layer_specs = get_model_complexity_info(model, (3, 32, 32),as_strings=False, print_per_layer_stat=True,
                                                            custom_modules_hooks={AdaptConv2D : conv_flops_counter_hook}, 
                                                            param_units='M', flops_units='MMac',
                                                            verbose=True)

    with torch.no_grad():
        stats = collect_stats(model, calib_data, num_batches=2, device=device)
        amax = compute_amax(model, method="percentile", percentile=99.99, device=device) 
    
    axx_list[0:len(conv2d_layers)] = [{'axx_mult' : 'bw_mult_9_9_50', 'axx_power' : 0.7082, 'quant_bits' : 9, 'fake_quant' : False}] * len(conv2d_layers)
    returned_power = [0]
    replace_conv_layers(model,  AdaptConv2D, axx_list, total_macs, total_params, layer_count=[0], returned_power = returned_power, initialize = False)  

    test_loss, test_acc = evaluate_test_accuracy(val_data, model, device)
    print(f'Mult: {approximation_levels}, test loss:{test_loss}, final test acc:{test_acc}')

if __name__ == "__main__":
    main()
