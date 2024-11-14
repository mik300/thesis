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
import argparse
from fractions import Fraction


attack_params = {
    'PGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'FGSM': {'eps': {'type': 'float', 'default': 8/255}},
    'BIM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'RFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'EOTPGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'eot_iter': {'type': 'int', 'default': 2}},
    'FFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 10/255}},
    'TPGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'MIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'decay': {'type': 'float', 'default': 1.0}, 'steps': {'type': 'int', 'default': 10}},
    'UPGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'random_start': {'type': 'bool', 'default': False}, 'loss': {'type': 'str', 'default': "ce"}, 'decay': {'type': 'float', 'default': 1.0}, 'eot_iter': {'type': 'int', 'default': 1}},
    'APGD': {'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 8/255}, 'steps': {'type': 'int', 'default': 10}, 'n_restarts': {'type': 'int', 'default': 1}, 'seed': {'type': 'int', 'default': 0}, 'loss': {'type': 'str', 'default': "ce"}, 'eot_iter': {'type': 'int', 'default': 1}, 'rho': {'type': 'float', 'default': 0.75}, 'verbose': {'type': 'bool', 'default': False}},
    'APGDT': {'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 8/255}, 'steps': {'type': 'int', 'default': 10}, 'n_restarts': {'type': 'int', 'default': 1}, 'seed': {'type': 'int', 'default': 0}, 'eot_iter': {'type': 'int', 'default': 1}, 'rho': {'type': 'float', 'default': 0.75}, 'verbose': {'type': 'bool', 'default': False}, 'n_classes': {'type': 'int', 'default': 10}},
    'DIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'decay': {'type': 'float', 'default': 0.0}, 'steps': {'type': 'int', 'default': 10}, 'resize_rate': {'type': 'float', 'default': 0.9}, 'diversity_prob': {'type': 'float', 'default': 0.5}, 'random_start': {'type': 'bool', 'default': False}},
    'TIFGSM':  {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'decay': {'type': 'float', 'default': 1.0}, 'kernel_name': {'type': 'str', 'default': "gaussian"}, 'len_kernel': {'type': 'int', 'default': 15}, 'nsig': {'type': 'int', 'default': 3}, 'resize_rate': {'type': 'float', 'default': 0.9}, 'diversity_prob': {'type': 'float', 'default': 0.5}, 'random_start': {'type': 'bool', 'default': False}},
    'Jitter': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'random_start': {'type': 'bool', 'default': True}},
    # 'NIFGSM':
    # 'PGDRS':
    # 'SINIFGSM':
    # 'VMIFGSM':
    # 'VNIFGSM':
    # 'SPSA':
    # 'JSMA':
    # 'EADL1':
    # 'EADEN':
    # 'PIFGSM':
    # 'PIFGSMPP':
    'CW': {'c': {'type': 'float'}, 'kappa': {'type': 'float'}, 'steps': {'type': 'int'}},
    # 'PGDL2':
    # 'DeepFool':
    # 'PGDRSL2':
    # 'SparseFool':
    # 'OnePixel':
    # 'Pixle':
    # 'FAB':
    'AutoAttack':{'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 0.3}, 'version': {'type': 'str', 'default': "standard"}, 'n_classes': {'type': 'int', 'default': 10}, 'seed': {'type': 'int', 'default': 0}, 'verbose': {'type': 'bool', 'default': False}},
    # 'Square':
    # 'MultiAttack':
}

def get_attack_parameters(attack_type):
    if attack_type in attack_params:
        params = {}
        for param, info in attack_params[attack_type].items():
            default_value = param_info.get('default')
            param_type = info['type']
            value_str = input(f"Enter value for {param} ({param_type}) : ")
            
            # Parse input based on specified type
            if value_str == "":
                value = default_value
            else:
                if param_type == 'int':
                    value = int(value_str)
                elif param_type == 'float':
                    try:
                        # If that fails, try parsing as a float
                        value = float(value_str)
                    except ValueError:
                        # If both int and float parsing fail, treat it as a fraction
                        value = float(Fraction(value_str))
                else:
                    raise ValueError(f"Unsupported parameter type '{param_type}' for {param}")

            params[param] = value
        return params
    else:
        raise ValueError(f"Attack type '{attack_type}' not recognized.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_function', default="ReLU", type=str, help="Activation function used for each act layer.")
    parser.add_argument('--batch-size', default=100, type=int, help="Number of images processed during each iteration")
    parser.add_argument('--data-dir', default="./data/", type=str, help="Directory in which the MNIST and FASHIONMNIST dataset are stored or should be downloaded")
    parser.add_argument('--adv-data-dir', default="./adversarial/adv_data/", type=str, help="Directory in which the adversarial data is stored")
    parser.add_argument('--AT', default=False, type=bool, help="Set to true to use Adversarially Trained (AT) models")
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
    parser.add_argument('--execution-type', default="quant", type=str, help="Leave it like this")
    parser.add_argument('--disable_aug', default=False, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--reload', default=True, type=bool, help="Set to True to reload a pretraind model, set to False to train a new one")
    parser.add_argument('--continue_training', default=False, type=bool, help="Set to True to continue the training for a number of epochs that is the difference between the already trained epochs and the total epochs")
    parser.add_argument('--appr-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--appr-level-list', type=int, nargs=8, help="Exactly 8 integers specifying levels of approximation for each layer")
    parser.add_argument('--log', default=0, type=int, help="Set to 0 to print the parameters necessary for gemmini")
    parser.add_argument('--attack', default="PGD", type=str, help="Choose a type of attack from: PGD, FGSM, BIM, CW... The complete list can be found in the Supported Attacks section in the README file")

    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    AT_model_dir = "./fast_adversarial/AT_models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.adv_data_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        exit("Dataset not supported")

    if args.execution_type == "quant":
        namebit = "_a"+str(args.act_bit)+"_w"+str(args.weight_bit)+"_b"+str(args.bias_bit)
    else:
        namebit = ""

    if args.execution_type == "quant":
        if args.fake_quant:
            namequant = "_fake"
        else:
            namequant = "_int"
    else:
        namequant = ""

    if args.AT == True:
        filename = AT_model_dir + "AT_" + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + dataset + "_" + args.activation_function + "_opt" + args.opt_level + "_alpha" + str(args.alpha) +"_epsilon" + str(args.epsilon) + "_" + str(args.epochs) + ".pth"
    else:
        filename = model_dir + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + args.dataset + "_" + args.activation_function + "_calibrated.pth"
    
    filename_sc = model_dir + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + args.dataset + "_" + args.activation_function + '_scaling_factors.pkl'

    print(f'Model parameters are loaded from: {filename}')
    print(f'Scaling factors are loaded from: {filename_sc}')

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
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    load_scaling_factors(model, filename_sc, device)
    print(f"Number of images in the test set: {len(test_loader.dataset)}")

    # Get the selected attack type
    attack_type = args.attack

    print(f"Running {attack_type} attack")
    # Retrieve the parameters from user input
    params = get_attack_parameters(attack_type)

    # Get the attack class dynamically and create the attack object
    AttackClass = getattr(torchattacks, attack_type)
    print(f'Executing {attack_type} attack with parameters: {params}')
    atk = AttackClass(model, **params)  # Pass the params as keyword arguments
    
    print(f"Generating adversarial images for {args.dataset} ...")
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        adv_images = atk(images, labels)

    formatted_params = [f"{int(value)}" if isinstance(value, int) else f"{value:.3f}" for value in params.values()]


    # Join them with underscores to form the suffix
    attack_parameters = "_".join(formatted_params)
    adv_data_path = args.adv_data_dir + args.neural_network + namebit + "_" + namequant + "_" + args.execution_type + "_" + args.dataset + "_" + args.activation_function + "_" + attack_type + "_" + attack_parameters + ".pt"
    atk.save(test_loader, save_path=adv_data_path, verbose=True)

if __name__ == "__main__":
    main()
