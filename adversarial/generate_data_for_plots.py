import subprocess
import argparse
from neural_networks.utils import set_model_axx_levels
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
import torch
import warnings
import subprocess
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation-function', default="ReLU", type=str, help="Activation function used for each act layer.")
    parser.add_argument('--dataset', default="cifar10", type=str, help="Select cifar10 or cifar100")
    parser.add_argument('--act-bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight-bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias-bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake-quant', default=True, type=bool, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--neural-network', default="resnet8", type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--execution-type', default="quant", type=str, help="Leave it like this")

    parser.add_argument('--param-act-bit', type=int, help="")
    parser.add_argument('--param-weight-bit', type=int, help="")
    parser.add_argument('--param-bias-bit', type=int, help="")
    parser.add_argument('--param-fake-quant', type=bool, help="")
    parser.add_argument('--param-execution-type', type=str, help="")
    parser.add_argument('--param-activation-function', type=str, help="")
    parser.add_argument('--param-neural-network', type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")

    parser.add_argument('--adv-data-dir', default="./adversarial/adv_data/", type=str, help="Directory in which the adversarial data is stored")
    parser.add_argument('--adv-act-bit', type=int, help="")
    parser.add_argument('--adv-weight-bit', type=int, help="")
    parser.add_argument('--adv-bias-bit', type=int, help="")
    parser.add_argument('--adv-fake-quant', type=bool, help="")
    parser.add_argument('--adv-execution-type', type=str, help="")
    parser.add_argument('--adv-activation-function', type=str, help="")
    parser.add_argument('--adv-neural-network', type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--adv-conv-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--adv-conv-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")
    parser.add_argument('--adv-linear-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--adv-linear-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")

    parser.add_argument('--conv-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--conv-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")
    parser.add_argument('--linear-axx-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--linear-axx-level-list', type=int, nargs='+', help="List of integers specifying levels of approximation for each convolutional layer")
    parser.add_argument('--transaxx-quant', default=8, type=int, help="")
    return parser.parse_args()

def main():
    args = get_args()

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
    else:
        exit("Dataset not supported")

    mode = {"execution_type":args.execution_type, "act_bit":args.act_bit, "weight_bit":args.weight_bit, "bias_bit":args.bias_bit, "fake_quant":args.fake_quant, "classes":num_classes, "act_type":args.activation_function}
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

    if args.adv_neural_network is None:
        args.adv_neural_network = args.neural_network


    conv_axx_levels, linear_axx_levels = set_model_axx_levels(model, args.conv_axx_level_list, args.conv_axx_level, args.linear_axx_level_list, args.linear_axx_level)
    adv_conv_axx_levels, adv_linear_axx_levels = set_model_axx_levels(model, args.adv_conv_axx_level_list, args.adv_conv_axx_level, args.adv_linear_axx_level_list, args.adv_linear_axx_level)

    #subprocess.run(["bash", "adversarial/generate_plots_data.sh", f"{conv_axx_levels}", f"{args.neural_network}", f"{args.adv_neural_network}", f"{adv_conv_axx_levels}"])

    conv_axx_levels_str = " ".join(map(str, conv_axx_levels))
    adv_conv_axx_levels_str = " ".join(map(str, adv_conv_axx_levels))
    

    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "float", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "float", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "float", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])

    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "quant", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "quant", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "quant", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])

    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "transaxx", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "transaxx", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "float", "--execution-type", "transaxx", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])



    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "float", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "float", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "float", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])

    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "quant", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "quant", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "quant", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])

    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "transaxx", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "transaxx", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "quant", "--execution-type", "transaxx", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])



    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "float", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "float", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "float", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])

    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "quant", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "quant", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "quant", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])

    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "transaxx", "--adv-execution-type", "float"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "transaxx", "--adv-execution-type", "quant"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])
    args1 = ["--prompt", "0", "--save-data", "1", "--conv-axx-level-list", *map(str, conv_axx_levels), "--neural-network", f"{args.neural_network}", "--adv-neural-network", f"{args.adv_neural_network}", "--adv-conv-axx-level-list", *map(str, adv_conv_axx_levels), "--param-execution-type", "transaxx", "--execution-type", "transaxx", "--adv-execution-type", "transaxx"]
    subprocess.run(["python", "adversarial/resnet_attack_eval.py", *args1])

    
if __name__ == "__main__":
    main()