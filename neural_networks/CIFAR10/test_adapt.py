import argparse
import numpy as np
import torch
from pathlib import Path
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
from neural_networks.utils import get_loaders_split, evaluate_test_accuracy, calibrate_model, load_scaling_factors, save_scaling_factors, save_weights, save_activations
from neural_networks.adapt.approx_layers.axx_layers import AdaPT_Conv2d

"""
This python file can be used to test the performance of the ResNet neural networks with different approximate multiplier configurations.
It also saves in a portable python format the weights and input activations of each layer for each multiplier configuration.
The code is organized as follows: 
1- loads the model and dataset
2- sets the requested approximation level to each layer (see functions "update_layer" and "update_model"
3- performs a dummy run with a single random image 
4- evaluates the test accuracy for each approximation level
"""


def update_layer(in_module, name, mult_type):
    new_module = False
    if isinstance(in_module, AdaPT_Conv2d):
        new_module = True
        in_module.axx_mult = mult_type
        in_module.update_kernel()
    return new_module

def update_model(model, mult_type="bw_mult_9_9_0"):
    for name, module in model.named_children():
        new_module = update_layer(module, name, mult_type)
        if not new_module:
            update_model(module, mult_type)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation-function', default="ReLU", type=str, help="Activation function used for each act layer.")
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
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

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


    filename = model_dir + args.neural_network + namebit + namequant + "_quant_" + args.dataset + "_" + args.activation_function + "_calibrated.pth"
    filename_sc = model_dir + args.neural_network + namebit + namequant + "_quant_" + args.dataset +"_" + args.activation_function + '_scaling_factors.pkl'


    print(filename)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_loader, valid_loader, test_loader = get_loaders_split(args.data_dir, batch_size=args.batch_size, dataset_type=args.dataset, num_workers=args.num_workers, split_val=args.split_val, disable_aug=args.disable_aug)

    mode= {"execution_type":args.execution_type, "act_bit":args.act_bit, "weight_bit":args.weight_bit, "bias_bit":args.bias_bit, "fake_quant":args.fake_quant, "classes":num_classes, "act_type":args.activation_function}

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

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    load_scaling_factors(model, filename_sc, device)
    filename_weights = model_dir + args.neural_network + namebit + namequant + "_quant_" + args.dataset + "_" + args.activation_function + '_weights.pkl'
    save_weights(model, filename_weights)
    input_tensor = torch.randn(1,3,32,32).to(next(model.parameters()).device)

    # for i in range(10):
    #     """ Use the same random input stimulus to obtain the intermediate input activations to each layer"""
    #     update_model(model, base_mult + str(i))
    #     filename_inputs = model_dir + args.neural_network + namebit + namequant + "_quant_" + args.dataset + "_" + args.activation_function + "_" + base_mult + str(i) + '_inputs.pkl'
    #     save_activations(model, input_tensor, filename_inputs)

    # for i in range(10):
    #     mult_type = base_mult + str(i)
    #     update_model(model, mult_type)
    #     test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
    #     print(f'Mult: {mult_type}, test loss:{test_loss}, final test acc:{test_acc}')
    
    appr = 15
    """ Use the same random input stimulus to obtain the intermediate input activations to each layer"""
    update_model(model, base_mult + str(appr))
    filename_inputs = model_dir + args.neural_network + namebit + namequant + "_quant_" + args.dataset + "_" + args.activation_function + "_" + base_mult + str(appr) + '_inputs.pkl'
    save_activations(model, input_tensor, filename_inputs)
    mult_type = base_mult + str(appr)
    update_model(model, mult_type)
    test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
    print(f'Mult: {mult_type}, test loss:{test_loss}, final test acc:{test_acc}')
    # mult_type = base_mult + str("i")
    # update_model(model, mult_type)
    # test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
    # print(f'Mult: {mult_type}, test loss:{test_loss}, final test acc:{test_acc}')


if __name__ == "__main__":
    main()
