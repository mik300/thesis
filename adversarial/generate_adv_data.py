import argparse
import numpy as np
import torch
from pathlib import Path
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
from neural_networks.utils import *
from neural_networks.adapt.approx_layers.axx_layers import AdaPT_Conv2d
from adversarial.utils import get_attack
from tqdm import tqdm
import torch.nn.functional as F
import torchattacks
import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    parser.add_argument('--appr-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--appr-level-list', type=int, nargs=8, help="Exactly 8 integers specifying levels of approximation for each layer")
    parser.add_argument('--nb-attacks', default=1, type=int, help="Specify number of attacks")

    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--AT', default=0, type=int, help="Set to true to use Adversarially Trained (AT) models")
    parser.add_argument('--AT-epsilon', default=8, type=int, help="This epsilon is unrelated to the attack; it's used to select the Adversarially Trained model")
    parser.add_argument('--AT-alpha', default=10, type=float, help="This alpha is unrelated to the attack; it's used to select the Adversarially Trained model")
    parser.add_argument('--AT-epochs', default=5, type=int, help="The number of epochs has no effect on this script; it used to select the Adversarially Trained model")

    parser.add_argument('--param-act-bit', type=int, help="")
    parser.add_argument('--param-weight-bit', type=int, help="")
    parser.add_argument('--param-bias-bit', type=int, help="")
    parser.add_argument('--param-fake-quant', type=bool, help="")
    parser.add_argument('--param-execution-type', type=str, help="")
    parser.add_argument('--param-activation-function', type=str, help="")
    parser.add_argument('--param-neural-network', type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")

    parser.add_argument('--conv-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--conv-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")
    parser.add_argument('--linear-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--linear-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")
    parser.add_argument('--transaxx-quant', default=8, type=int, help="")
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    AT_model_dir = "./fast_adversarial/AT_models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.adv_data_dir).mkdir(parents=True, exist_ok=True)

    if args.execution_type == 'adapt':
        device = "cpu"
        torch.set_num_threads(args.threads)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device used: {device}')


    if args.param_act_bit is None:
        args.param_act_bit = args.act_bit
    if args.param_weight_bit is None:
        args.param_weight_bit = args.weight_bit
    if args.param_bias_bit is None:
        args.param_bias_bit = args.bias_bit
    if args.param_fake_quant is None:
        args.param_fake_quant = args.fake_quant
    if args.param_execution_type is None:
        args.param_execution_type = args.execution_type
    if args.param_activation_function is None:
        args.param_activation_function = args.activation_function
    if args.param_neural_network is None:
        args.param_neural_network = args.neural_network

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        exit("Dataset not supported")

    if args.execution_type == "quant" or args.execution_type == "adapt":
        execution_type = "quant"
        namebit = "_a"+str(args.act_bit)+"_w"+str(args.weight_bit)+"_b"+str(args.bias_bit)
    elif args.execution_type == "float":
        execution_type = "float"
        namebit = ""
    else:
        execution_type = "transaxx"
        namebit = f"_{args.transaxx_quant}x{args.transaxx_quant}"

    if args.execution_type == "quant" or args.execution_type == "adapt" or args.execution_type == "transaxx":
        if args.fake_quant:
            namequant = "_fake"
        else:
            namequant = "_int"
    else:
        namequant = ""

    if args.param_execution_type == "quant" or args.param_execution_type == "adapt":
        param_execution_type = "quant"
        param_namebit = "_a"+str(args.param_act_bit)+"_w"+str(args.param_weight_bit)+"_b"+str(args.param_bias_bit)
    elif args.param_execution_type == "float":
        param_execution_type = "float"
        param_namebit = ""
    else:
        param_execution_type = "transaxx"
        param_namebit = f"_{args.transaxx_quant}x{args.transaxx_quant}"

    if args.param_execution_type == "quant" or args.param_execution_type == "adapt" or args.param_execution_type == "transaxx":
        if args.param_fake_quant:
            param_namequant = "_fake"
        else:
            param_namequant = "_int"
    else:
        param_namequant = ""


    if args.AT == 1:
        filename = AT_model_dir + "AT_" + args.param_neural_network + param_namebit + param_namequant + "_" + param_execution_type + "_" + args.dataset + "_" + args.param_activation_function + "_opt" + args.opt_level + "_alpha" + str(args.AT_alpha) +"_epsilon" + str(args.AT_epsilon) + "_" + str(args.AT_epochs) + ".pth"
    else:
        if args.execution_type == "transaxx":
            filename = model_dir + args.param_neural_network + param_namebit + param_namequant + "_" + param_execution_type + "_" + args.dataset + "_" + args.param_activation_function + ".pth"
        else:
            filename = model_dir + args.param_neural_network + param_namebit + param_namequant + "_" + param_execution_type + "_" + args.dataset + "_" + args.param_activation_function + "_calibrated.pth"
    
    if args.param_execution_type == "adapt":
        filename_sc = model_dir + args.param_neural_network + param_namebit + param_namequant + "_" + param_execution_type + "_" + args.dataset + "_" + args.param_activation_function + '_scaling_factors.pkl'
        print(f'Scaling factors are loaded from: {filename_sc}')

    print(f'Model parameters are loaded from: {filename}')
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_loader, valid_loader, test_loader = get_loaders_split(args.data_dir,
     batch_size=args.batch_size, dataset_type=args.dataset, num_workers=args.num_workers,
     split_val=args.split_val, disable_aug=args.disable_aug)

    print(f"train_loader size = {len(train_loader.dataset)}")
    print(f"valid_loader size = {len(valid_loader.dataset)}")
    print(f"test_loader size = {len(test_loader.dataset)}")

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

    conv_axx_levels, linear_axx_levels = set_model_axx_levels(model, args.conv_axx_level_list, args.conv_axx_level, args.linear_axx_level_list, args.linear_axx_level)
    if args.execution_type == "transaxx":
        conv_axx_levels_string = "_" + "_".join(map(str, conv_axx_levels))
    else:
        conv_axx_levels_string = ""


    if args.param_execution_type == "transaxx":
        init_transaxx_train(model, conv_axx_levels, args, args.transaxx_quant, device, args.fake_quant)
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if args.execution_type == "transaxx":
            init_transaxx_train(model, conv_axx_levels, args, args.transaxx_quant, device, args.fake_quant)


    if args.execution_type == "adapt":
        load_scaling_factors(model, filename_sc, device)
    

    
    if args.nb_attacks == 1:
        # Prompt attack type and parameters
        attack_type, params = get_attack()

        # Get the attack class dynamically and create the attack object
        AttackClass = getattr(torchattacks, attack_type)
        atk = AttackClass(model, **params)  # Pass the params as keyword arguments
        atk.set_normalization_used(cifar10_mean, cifar10_std)
        print(f'Executing {atk}')

        #formatted_params = [f"{int(value)}" if isinstance(value, int) else f"{value:.3f}" for value in params.values()]
        formatted_params = [
        f"{value}" if isinstance(value, int) else 
        f"{value:.3f}" if isinstance(value, float) else 
        f"{value}" 
        for value in params.values()
        ]
        # Join parameters with underscores to form the suffix
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
        atk.set_normalization_used(cifar10_mean, cifar10_std)
        print(f'Executing {atk}')

    
    print(f"Generating and saving adversarial images for {args.dataset}...")

    if args.AT == 1:
        AT_suffix = "AT_"
    else:
        AT_suffix = ""
    adv_data_path = args.adv_data_dir + AT_suffix + args.neural_network + namebit + namequant + "_" + args.execution_type + conv_axx_levels_string + "_" + args.dataset + "_" + args.activation_function + "_" + attack_type + attack_parameters + ".pt"
    atk.save(test_loader, save_path=adv_data_path, verbose=True, return_verbose=True)
    print(f"Adversarial data has been saves in {adv_data_path}")

    adv_loader = atk.load(load_path=adv_data_path, normalize=cifar10_mean_std)
    print(f"Adversarial data size is {len(adv_loader.dataset)} images")

    #print(model)
    print(f'Model parameters are loaded from {filename}')
    print(f"Adversarial data is loaded from {adv_data_path}")
    print(f'Adversarial data size: {len(adv_loader.dataset)}')
    test_loss, test_acc = evaluate_test_accuracy(adv_loader, model, device)
    print(f"Mult: {conv_axx_levels}, {linear_axx_levels} | test loss:{test_loss:.4f} | final test acc:{test_acc:.4f} (adversarial)")

if __name__ == "__main__":
    main()
