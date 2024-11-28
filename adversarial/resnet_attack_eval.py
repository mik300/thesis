import os
import argparse
import numpy as np
import torch
from pathlib import Path
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
from neural_networks.utils import get_loaders_split, evaluate_test_accuracy, calibrate_model, load_scaling_factors, save_scaling_factors, save_weights, save_activations
from neural_networks.adapt.approx_layers.axx_layers import AdaPT_Conv2d
from tqdm import tqdm
import torch.nn.functional as F
import torchattacks
from adversarial.utils import get_attack
import warnings


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
    parser.add_argument('--adv-data-dir', default="./adversarial/adv_data/", type=str, help="Directory in which the adversarial data is stored")
    parser.add_argument('--dataset', default="cifar10", type=str, help="Select cifar10 or cifar100")
    parser.add_argument('--num-workers', default=0, type=int, help="Number of threads used during the preprocessing of the dataset")
    parser.add_argument('--threads', default=10, type=int, help="Number of threads used during the inference, used only when neural-network-type is set to adapt")
    parser.add_argument('--split-val', default=0.1, type=float, help="The split-val is used to divide the training set in training and validation with the following dimensions: train=train_images*(1-split_val)  valid=train_images*split_val")
    parser.add_argument('--act-bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight-bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias-bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake-quant', default=True, type=bool, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--neural-network', default="resnet8", type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--execution-type', default="quant", type=str, help="Leave it like this")
    parser.add_argument('--disable-aug', default=False, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--reload', default=True, type=bool, help="Set to True to reload a pretraind model, set to False to train a new one")
    parser.add_argument('--continue-training', default=False, type=bool, help="Set to True to continue the training for a number of epochs that is the difference between the already trained epochs and the total epochs")
    parser.add_argument('--appr-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--appr-level-list', type=int, nargs=8, help="Exactly 8 integers specifying levels of approximation for each layer")
    parser.add_argument('--log', default=0, type=int, help="Set to 0 to print the parameters necessary for gemmini")
    parser.add_argument('--nb-attacks', default=1, type=int, help="Specify number of attacks")

    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--AT', default=False, type=bool, help="Set to true to use Adversarially Trained (AT) models")
    parser.add_argument('--AT-epsilon', default=8, type=int, help="This epsilon is unrelated to the attack; it's used to select the Adversarially Trained model")
    parser.add_argument('--AT-alpha', default=10, type=float, help="This alpha is unrelated to the attack; it's used to select the Adversarially Trained model")
    parser.add_argument('--AT-epochs', default=5, type=int, help="The number of epochs has no effect on this script; it used to select the Adversarially Trained model")

    parser.add_argument('--data-act-bit', type=int, help="")
    parser.add_argument('--data-weight-bit', type=int, help="")
    parser.add_argument('--data-bias-bit', type=int, help="")
    parser.add_argument('--data-fake-quant', type=bool, help="")
    parser.add_argument('--data-execution-type', type=str, help="")
    parser.add_argument('--data-activation-function', type=str, help="")
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    AT_model_dir = "./fast_adversarial/AT_models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.adv_data_dir).mkdir(parents=True, exist_ok=True)

    if args.data_act_bit is None:
        args.data_act_bit = args.act_bit

    if args.data_weight_bit is None:
        args.data_weight_bit = args.weight_bit

    if args.data_bias_bit is None:
        args.data_bias_bit = args.bias_bit

    if args.data_fake_quant is None:
        args.data_fake_quant = args.fake_quant

    if args.data_execution_type is None:
        args.execution_type = args.execution_type

    if args.data_activation_function is None:
        args.data_activation_function = args.activation_function
    

    if args.execution_type == 'adapt':
        device = "cpu"
        torch.set_num_threads(args.threads)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device used: {device}')

    if args.appr_level_list is None:
        approximation_levels = [args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level]
    else:
        approximation_levels = args.appr_level_list


    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        exit("Dataset not supported")

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

    #Find the .pth file with the desired parameters
    if args.data_execution_type == "quant" or args.data_execution_type == "adapt":
        data_execution_type = "quant"
        data_namebit = "_a"+str(args.act_bit)+"_w"+str(args.weight_bit)+"_b"+str(args.bias_bit)
    else:
        data_execution_type = "float"
        data_namebit = ""

    if args.data_execution_type == "quant" or args.data_execution_type == "adapt":
        if args.data_fake_quant:
            data_namequant = "_fake"
        else:
            data_namequant = "_int"
    else:
        data_namequant = ""

    if args.AT == True:
        filename = AT_model_dir + "AT_" + args.neural_network + data_namebit + data_namequant + "_" + data_execution_type + "_" + args.dataset + "_" + args.data_activation_function + "_opt" + args.opt_level + "_alpha" + str(args.AT_alpha) +"_epsilon" + str(args.AT_epsilon) + "_" + str(args.AT_epochs) + ".pth"
    else:
        filename = model_dir + args.neural_network + data_namebit + data_namequant + "_" + data_execution_type + "_" + args.dataset + "_" + args.data_activation_function + "_calibrated.pth"
    
    

    print(f'Model parameters are loaded from {filename}')
    if args.execution_type == "quant" or args.execution_type == "adapt":
        filename_sc = model_dir + args.neural_network + data_namebit + data_namequant + "_" + "quant" + "_" + args.dataset +"_" + args.data_activation_function + '_scaling_factors.pkl'
        print(f'Scaling factors loaded from {filename_sc} and assigned to the model')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_loader, valid_loader, test_loader = get_loaders_split(args.data_dir,
     batch_size=args.batch_size, dataset_type=args.dataset, num_workers=args.num_workers,
     split_val=args.split_val, disable_aug=args.disable_aug)
    
    mode= {"execution_type":args.execution_type, "act_bit":args.act_bit, "weight_bit":args.weight_bit, "bias_bit":args.bias_bit, "fake_quant":args.fake_quant, "classes":num_classes, "act_type":args.activation_function}
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

    print(f"Number of images in the test set: {len(valid_loader.dataset)}")
    base_mult = "bw_mult_9_9_"

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if args.execution_type == "quant" or args.execution_type == "adapt":
        load_scaling_factors(model, filename_sc, device)
    model.to(device)
    model.eval()
    
    update_model(model, base_mult, approximation_levels)

    if args.nb_attacks == 1:
        # Prompt attack type and parameters
        attack_type, params = get_attack()

        # Get the attack class dynamically and create the attack object
        AttackClass = getattr(torchattacks, attack_type)
        message = f'Executing {attack_type} attack with parameters: {params}'
        atk = AttackClass(model, **params)  # Pass the params as keyword arguments

        formatted_params = [f"{int(value)}" if isinstance(value, int) else f"{value:.3f}" for value in params.values()]
        # Join them with underscores to form the suffix
        attack_parameters = "_" + "_".join(formatted_params)
    else:
        attack_type_list = []
        atk_list = []
        for i in range(args.nb_attacks):
            attack_type, params = get_attack()
            attack_type_list.append(attack_type) #Used later to name the saved data 
            AttackClass = getattr(torchattacks, attack_type)
            atk_list.append(AttackClass(model, **params))
            print("")
        atk = torchattacks.MultiAttack(atk_list)
        attack_type = "_".join(attack_type_list)
        attack_parameters = ""
        message = f'Executing a MultiAttack with the following attack types: {attack_type_list}'

    atk.set_normalization_used(cifar10_mean, cifar10_std)
    adv_data_path = args.adv_data_dir + args.neural_network + namebit + namequant + "_" + execution_type + "_" + args.dataset + "_" + args.activation_function + "_" + attack_type + attack_parameters + ".pt"
    if not os.path.exists(adv_data_path):
        raise FileNotFoundError(f"Error: '{adv_data_path}' not found. Run generated_adv_data.py to generate the required data.")
    adv_loader = atk.load(load_path=adv_data_path)

    #adv_loader test_loader
    test_loss, test_acc = evaluate_test_accuracy(adv_loader, model, device)
    print(message)
    print(f'Mult: {approximation_levels}, test loss:{test_loss}, final test acc:{test_acc}')

if __name__ == "__main__":
    main()
