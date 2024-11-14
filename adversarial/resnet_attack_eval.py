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
    parser.add_argument('--batch_size', default=100, type=int, help="Number of images processed during each iteration")
    parser.add_argument('--data_dir', default="./data/", type=str, help="Directory in which the MNIST and FASHIONMNIST dataset are stored or should be downloaded")
    parser.add_argument('--AT', default=False, type=bool, help="Set to true to use Adversarially Trained (AT) models")
    parser.add_argument('--dataset', default="cifar10", type=str, help="Select cifar10 or cifar100")
    parser.add_argument('--epochs', default=128, type=int, help="Number of training epochs")
    parser.add_argument('--num_workers', default=0, type=int, help="Number of threads used during the preprocessing of the dataset")
    parser.add_argument('--threads', default=10, type=int, help="Number of threads used during the inference, used only when neural-network-type is set to adapt")
    parser.add_argument('--split_val', default=0.1, type=float, help="The split-val is used to divide the training set in training and validation with the following dimensions: train=train_images*(1-split_val)  valid=train_images*split_val")
    parser.add_argument('--act_bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight_bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias_bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake_quant', default=True, type=bool, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--neural_network', default="resnet8", type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--execution-type', default="quant", type=str, help="Leave it like this")
    parser.add_argument('--disable_aug', default=False, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--reload', default=True, type=bool, help="Set to True to reload a pretraind model, set to False to train a new one")
    parser.add_argument('--continue_training', default=False, type=bool, help="Set to True to continue the training for a number of epochs that is the difference between the already trained epochs and the total epochs")
    parser.add_argument('--appr-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--appr-level-list', type=int, nargs=8, help="Exactly 8 integers specifying levels of approximation for each layer")
    parser.add_argument('--log', default=0, type=int, help="Set to 0 to print the parameters necessary for gemmini")
    parser.add_argument('--attack', default="PGD", type=int, help="Choose a type of attack from: PGD, FGSM, BIM, CW... The complete list can be found in the Supported Attacks section in the README file")
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    AT_model_dir = "./fast_adversarial/AT_models/"
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

    device = "cpu"
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

    if args.AT == True:
        filename = AT_model_dir + "AT_" + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + dataset + "_" + args.activation_function + "_opt" + args.opt_level + "_alpha" + str(args.alpha) +"_epsilon" + str(args.epsilon) + "_" + str(args.epochs) + ".pth"
    else:
        filename = model_dir + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + args.dataset + "_" + args.activation_function + "_calibrated.pth"
    
    filename_sc = model_dir + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + args.dataset +"_" + args.activation_function + '_scaling_factors.pkl'

    print(f'Model is sourced from: {filename}')
    print(filename_sc)

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
    for idx, (images, labels) in enumerate(train_loader):
        attack_type = args.  
        # Dynamically get the attack class from torchattacks
        AttackClass = getattr(torchattacks, attack_type)
        atk = AttackClass(model, eps=8/255, alpha=2/255, steps=4)
        #atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
        adv_images = atk(images, labels)

    # Save
    atk.save(train_loader, save_path=f"./adversarial_attacks/data.pt", verbose=True)
    # Load
    adv_loader = atk.load(load_path="./adversarial_attacks/data.pt")


    base_mult = "bw_mult_9_9_"
    
    print(f"model type = {type(model)}")

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    #model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    load_scaling_factors(model, filename_sc, device)

    

    update_model(model, base_mult, approximation_levels)

    #adv_loader test_loader
    test_loss, test_acc = evaluate_test_accuracy(test_loader, model, args, device)
    print(f'Mult: {approximation_levels}, test loss:{test_loss}, final test acc:{test_acc}')

if __name__ == "__main__":
    main()
