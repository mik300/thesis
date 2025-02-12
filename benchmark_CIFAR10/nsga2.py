from pymoo.optimize import minimize
from neural_networks.CIFAR10.resnet import resnet8, resnet14, resnet20, resnet32, resnet50, resnet56
from neural_networks.utils import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from benchmark_CIFAR10.Utils.utils import transaxx_model_8x8
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.pm import PM
from pathlib import Path
import benchmark_CIFAR10.Utils.GA_utils as GA_utils
import numpy as np
import time
import torch
import pickle
import argparse
import torchattacks
from adversarial.utils import get_attack


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural-network', default="resnet32", type=str, help="Choose one from fashionmnist, mnist, resnet8, resnet14, resnet20, resnet32, resnet50, resnet56")
    parser.add_argument('--batch-size', default=100, type=int, help="Number of images processed during each iteration")
    parser.add_argument('--data-dir', default="./data/dataset/pytorch_only/", type=str, help="Directory in which the MNIST and FASHIONMNIST dataset are stored or should be downloaded")
    parser.add_argument('--dataset', default="cifar10", type=str, help="Select cifar10 or cifar100")
    parser.add_argument('--epochs', default=1, type=int, help="Number of retraining epochs executed for each individual during the NSGA search")
    parser.add_argument('--lr-max', default=1e-2, type=float, help="Maximum learning rate for 'cyclic' scheduler, standard learning rate for 'flat' scheduler")
    parser.add_argument('--lr-min', default=1e-4, type=float, help="Minimum learning rate for 'cyclic' scheduler")
    parser.add_argument('--lr-type', default="cyclic", type=str, help="Select learning rate scheduler, choose between 'cyclic' or 'flat'")
    parser.add_argument('--weight-decay', default=5e-4, type=float, help="Weight decay applied during the optimization step")
    parser.add_argument('--fname', default="baseline_model.pth", type=str, help="Name of the model, must include .pth")
    parser.add_argument('--num-workers', default=4, type=int, help="Number of threads used during the preprocessing of the dataset")
    parser.add_argument('--threads', default=16, type=int, help="Number of threads used during the inference, used only when neural-network-type is set to adapt")
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducible random initialization")
    parser.add_argument('--lr-momentum', default=0.9, type=float, help="Learning rate momentum")
    parser.add_argument('--split-val', default=0.1, type=float, help="The split-val is used to divide the training set in training and validation with the following dimensions: train=train_images*(1-split_val)  valid=train_images*split_val")
    parser.add_argument('--partial-val', default=10, type=float, help="The partial-val defines the portion of the training set used for partial retraining during the evaluation of each individual. The number of train images is defined as train=train_images*(1-split_val)/partial_val")
    parser.add_argument('--retrain-type', default='full', type=str, help="Defines whether each approximate neural network evaluated during the search is not retrained, retrained entirely or retrained by updating just the bias of convolutional layers. Choose between 'none', 'bias', and 'full'")
    parser.add_argument('--start-from-last', default=0, type=int, help="Set to true to reload a previous pareto front configuration")
    parser.add_argument('--axx-linear', default=False, type=bool, help="Set to True to enable approximate computing of linear layers. Ignored by default for CIFAR10 networks")
    parser.add_argument('--population', default=70, type=int, help="The number of new individuals evaluated at each iteration. Each individal is an approximate NN")
    parser.add_argument('--generations', default=80, type=int, help="The number of generations explored during the genetic search")
    parser.add_argument('--crossover-probability', default=0.8, type=float, help="Probability of performing a single-point crossover operation on an individual")
    parser.add_argument('--mutation-probability', default=0.8, type=float, help="Probability of gene mutation occurring on an individual")
    parser.add_argument('--axx-levels', default=255, type=int, help="Number of approximation levels supported by the multiplier, or number of LUT corresponding to different multipliers")
    parser.add_argument('--disable-aug', default=True, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--execution-type', default='transaxx', type=str, help="")
    parser.add_argument('--act-bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight-bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias-bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake-quant', default=True, type=bool, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--activation-function', default="ReLU", type=str, help="Activation function used for each act layer.")

    

    parser.add_argument('--log', default=0, type=int, help="Set to 0 to print the parameters necessary for gemmini")
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

    parser.add_argument('--adv-AT', default=0, type=int, help="Set to true to use adversarial data generated by Adversarially Trained (AT) models")
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
    params = get_args()
    model_dir = "./neural_networks/models/"
    AT_model_dir = "./fast_adversarial/AT_models/"
    train_args = {'epochs': params.epochs, 'lr_min': params.lr_min, 'lr_max': params.lr_max, 'batch': params.batch_size,
                  'weight_decay': params.weight_decay, 'num_workers': params.num_workers, 'lr_momentum': params.lr_momentum}
    #model_name = "./neural_networks/models/" + params.neural_network + "_quant_baseline_model.pth"

    if params.param_act_bit is None:
        params.param_act_bit = params.act_bit
    if params.param_weight_bit is None:
        params.param_weight_bit = params.weight_bit
    if params.param_bias_bit is None:
        params.param_bias_bit = params.bias_bit
    if params.param_fake_quant is None:
        params.param_fake_quant = params.fake_quant
    if params.param_execution_type is None:
        params.param_execution_type = params.execution_type
    if params.param_activation_function is None:
        params.param_activation_function = params.activation_function
    if params.param_neural_network is None:
        params.param_neural_network = params.neural_network

    if params.adv_act_bit is None:
        params.adv_act_bit = params.param_act_bit
    if params.adv_weight_bit is None:
        params.adv_weight_bit = params.param_weight_bit
    if params.adv_bias_bit is None:
        params.adv_bias_bit = params.param_bias_bit
    if params.adv_fake_quant is None:
        params.adv_fake_quant = params.param_fake_quant
    if params.adv_execution_type is None:
        params.adv_execution_type = params.param_execution_type
    if params.adv_activation_function is None:
        params.adv_activation_function = params.param_activation_function
    if params.adv_neural_network is None:
        params.adv_neural_network = params.param_neural_network

    if params.execution_type == 'adapt':
        device = "cpu"
        torch.set_num_threads(params.threads)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device used: {device}')

    if params.execution_type == "adapt":
        execution_type = "quant"
        namebit = "_a"+str(params.act_bit)+"_w"+str(params.weight_bit)+"_b"+str(params.bias_bit)
    else:
        execution_type = "transaxx"
        namebit = "_8x8"
        
    if params.fake_quant:
        namequant = "_fake"
    else:
        namequant = "_int"

    if params.dataset == "cifar10":
        num_classes = 10
    elif params.dataset == "cifar100":
        num_classes = 100
    else:
        exit("Dataset not supported")

    if params.adv_execution_type == "quant" or params.adv_execution_type == "adapt":
        adv_execution_type = "quant"
        adv_namebit = "_a"+str(params.adv_act_bit)+"_w"+str(params.adv_weight_bit)+"_b"+str(params.adv_bias_bit)
    elif params.adv_execution_type == "float":
        adv_execution_type = "float"
        adv_namebit = ""
    else:
        adv_execution_type = "transaxx"
        adv_namebit = f"_{params.transaxx_quant}x{params.transaxx_quant}"

    if params.adv_execution_type == "quant" or params.adv_execution_type == "adapt" or params.adv_execution_type == "transaxx":
        if params.adv_fake_quant:
            adv_namequant = "_fake"
        else:
            adv_namequant = "_int"
    else:
        adv_namequant = ""

    #Find the .pth file with the desired parameters
    if params.param_execution_type == "quant" or params.param_execution_type == "adapt":
        param_execution_type = "quant"
        param_namebit = "_a"+str(params.act_bit)+"_w"+str(params.weight_bit)+"_b"+str(params.bias_bit)
    elif params.param_execution_type == "float":
        param_execution_type = "float"
        param_namebit = ""
    else:
        param_execution_type = "transaxx"
        param_namebit = f"_{params.transaxx_quant}x{params.transaxx_quant}"

    if params.param_execution_type == "quant" or params.param_execution_type == "adapt" or params.param_execution_type == "transaxx":
        if params.param_fake_quant:
            param_namequant = "_fake"
        else:
            param_namequant = "_int"
    else:
        param_namequant = ""

    if params.AT == 1:
        model_name = AT_model_dir + "AT_" + params.neural_network + namebit + namequant + "_" + execution_type + "_" + params.dataset + "_" + params.activation_function + "_opt" + params.opt_level + "_alpha" + str(params.AT_alpha) +"_epsilon" + str(params.AT_epsilon) + "_" + str(params.AT_epochs) + ".pth"
    else:
        if params.execution_type == "transaxx":
            model_name = model_dir + params.neural_network + namebit + namequant + "_" + execution_type + "_" + params.dataset + "_" + params.activation_function + ".pth"
        else:
            model_name = model_dir + params.neural_network + namebit + namequant + "_" + execution_type + "_" + params.dataset + "_" + params.activation_function + "_calibrated.pth"

    #model_name = "./neural_networks/models/" + params.neural_network + namebit + namequant + "_" + execution_type + "_" + "cifar10" +"_" + params.activation_function + ".pth"
    filename_sc = f"./neural_networks/models/{params.neural_network}_a8_w8_b32_fake_quant_cifar10_ReLU_scaling_factors.pkl"
    torch.set_num_threads(params.threads)

    pkl_repo = './benchmark_CIFAR10/results_pkl/'
    Path(pkl_repo).mkdir(parents=True, exist_ok=True)
    in_file = pkl_repo + params.neural_network +'pareto_conf_'+ str(params.generations) + '_' + str(params.population) + '_Pc' + str(int(params.crossover_probability*10)) +'_Pm' + str(int(params.mutation_probability*10)) + '_seed1' + '.pkl'
    f_file = pkl_repo + params.neural_network + 'pareto_res_' + str(params.generations) +'_' + str(params.population) +'_Pc' + str(int(params.crossover_probability*10)) +'_Pm' + str(int(params.mutation_probability*10)) +'_seed1'+ '.pkl'

    imsize = (1,1,28,28) if (params.neural_network == 'mnist' or params.neural_network == 'fashionmnist' ) else (1,3,32,32)
    if params.start_from_last:
        print("loading last pareto front")
        file = open(pkl_repo + params.neural_network +'_backup_'+ str(params.generations) + '_' + str(params.population) + '_Pc' + str(int(params.crossover_probability*10)) +'_Pm' + str(int(params.mutation_probability*10)) + '_seed1' + '.pkl', 'rb')
        last_design = pickle.load(file)
        file.close()
        # Ng = Ng - last_design['current_gen']

    class ProblemWrapper(Problem):
        def _evaluate(self, designs, out, *args, **kwargs):
            res1 = []
            res2 = []
            print("eval solution ", len(designs))
            if self.data['current_gen'] == 0 and params.start_from_last:
                print(f"last_design = {last_design}")
                for i in range(len(last_design['designs'])):
                    designs[i] = last_design['designs'][i]

            print(f"current gen is {self.data['current_gen']+1} out of {params.generations}")
            file = open(pkl_repo + params.neural_network +'_backup_'+ str(params.generations) + '_' + str(params.population) + '_Pc' + str(int(params.crossover_probability*10)) +'_Pm' + str(int(params.mutation_probability*10)) + '_seed1' + '.pkl', 'wb')
            pickle.dump({'designs':designs, 'current_gen':self.data['current_gen'], 'Ng':params.generations, 'Np':params.population, 'Pc':params.crossover_probability, 'Pm':params.mutation_probability, 'net':params.neural_network}, file)
            file.close()
            for i in range(len(designs)):
                design_time = time.time()
                design = designs[i]
                design_id = "".join([str(a) for a in design])
                if design_id in self.data['previous_designs_results']:
                    # design already evaluated
                    f1 = self.data['previous_designs_results']['design_id']['f1']
                    f2 = self.data['previous_designs_results']['design_id']['f2']
                    res1.append(f1)
                    res2.append(f2)
                    print("skipped")
                else:
                    self.data['previous_designs_results']['design_id'] = {}
                    print(f"design = {design}")
                    checkpoint = torch.load(model_name, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    transaxx_model_8x8(model, design, axx_linear=params.axx_linear, retrain_type='full')  # model becomes transaxx model
                    # design is an individual, fitness function is the function to optimize
                    _, f1_inverse = evaluate_test_accuracy(adv_loader, model, device)
                    _, f2_inverse = evaluate_test_accuracy(test_dataloader, model, device)
                    f1 = 1/f1_inverse
                    f2 = 1/f2_inverse
                    self.data['previous_designs_results']['design_id']['f1'] = f1
                    self.data['previous_designs_results']['design_id']['f2'] = f2
                res1.append(f1)
                res2.append(f2)

                # print(f"evaluated design {i+1} out of {len(designs)}, time is {(time.time() - design_time)}")
                print(f"design: {i + 1}/{len(designs)} || gen: {self.data['current_gen']+1}/{params.generations} || adv test acc is: {(100/f1):0.2f} || test acc is: {(100/f2):0.2f}|| time is: {(time.time() - design_time):0.2f}")


            out['F'] = np.column_stack([ np.array(res1), np.array(res2)])
            self.data['current_gen'] += 1
    # start time
    start_time = time.time()

    if params.neural_network == 'mnist' or params.neural_network == "fashionmnist":
        train_dataloader, valid_dataloader, test_dataloader = get_loaders_split(params.data_dir,
                                                                                batch_size=train_args['batch'],
                                                                                dataset_type=params.neural_network,
                                                                                num_workers=train_args['num_workers'],
                                                                                split_val=params.split_val,
                                                                                disable_aug=params.disable_aug)
    else:
        train_dataloader, valid_dataloader, test_dataloader = get_loaders_split(params.data_dir,
                                                                                batch_size=train_args['batch'],
                                                                                dataset_type='cifar10',
                                                                                num_workers=train_args['num_workers'],
                                                                                split_val=params.split_val,
                                                                                disable_aug=params.disable_aug)

    
    

    mode = {"execution_type":execution_type, "act_bit":params.act_bit, "weight_bit":params.weight_bit, "bias_bit":params.bias_bit, "fake_quant":params.fake_quant, "classes":num_classes, "act_type":params.activation_function}

    if params.neural_network == "resnet8":
        model = resnet8(mode).to(device)
    elif params.neural_network == "resnet14":
        model = resnet14(mode).to(device)
    elif params.neural_network == "resnet20":
        model = resnet20(mode).to(device)
    elif params.neural_network == "resnet32":
        model = resnet32(mode).to(device)
    elif params.neural_network == "resnet50":
        model = resnet50(mode).to(device)
    elif params.neural_network == "resnet56":
        model = resnet56(mode).to(device)
    elif params.neural_network == "mnist" or params.neural_network == 'fashionmnist':
        model = adapt_mnist_net()
    else:
        exit("error unknown CNN model name")

    if execution_type == "transaxx":
        conv_axx_levels, linear_axx_levels = set_model_axx_levels(model, params.conv_axx_level_list, params.conv_axx_level, params.linear_axx_level_list, params.linear_axx_level)
        adv_conv_axx_levels, adv_linear_axx_levels = set_model_axx_levels(model, params.adv_conv_axx_level_list, params.adv_conv_axx_level, params.adv_linear_axx_level_list, params.adv_linear_axx_level)
        init_transaxx(model, conv_axx_levels, linear_axx_levels, params.batch_size, 8, device, fake_quant=True)

    if params.execution_type == "transaxx":
        conv_axx_levels_string = "_" + "_".join(map(str, conv_axx_levels))
        linear_axx_levels_string = "_" + "_".join(map(str, linear_axx_levels))
    else:
        conv_axx_levels_string = ""

    if params.adv_execution_type == "transaxx":
        adv_conv_axx_levels_string = "_" + "_".join(map(str, adv_conv_axx_levels))
        linear_axx_levels_string = "_" + "_".join(map(str, adv_linear_axx_levels))
    else:
        adv_conv_axx_levels_string = ""
    #load_scaling_factors(model, filename_sc, device = device)
    print(f"Model parameters are loaded from {model_name}")
    checkpoint = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    print(model)

    # Prompt attack type and parameters
    attack_type, atk_params = get_attack(prompt=False)

    # Get the attack class dynamically and create the attack object
    AttackClass = getattr(torchattacks, attack_type)
    message = f'Executing {attack_type} attack with parameters: {atk_params}'
    atk = AttackClass(model, **atk_params)  # Pass the attack params as keyword arguments
    atk.set_normalization_used(cifar10_mean, cifar10_std)

    #formatted_params = [f"{int(value)}" if isinstance(value, int) else f"{value:.3f}" for value in params.values()]
    formatted_params = [
    f"{value}" if isinstance(value, int) else 
    f"{value:.3f}" if isinstance(value, float) else 
    f"{value}" 
    for value in atk_params.values()
    ]
    # Join them with underscores to form the suffix
    attack_parameters = "_" + "_".join(formatted_params)

    if params.adv_AT == 1:
        AT_suffix = "AT_"
    else:
        AT_suffix = ""
    adv_data_path = params.adv_data_dir + AT_suffix + params.adv_neural_network + adv_namebit + adv_namequant + "_" + adv_execution_type + adv_conv_axx_levels_string + "_" + params.dataset + "_" + params.activation_function + "_" + attack_type + attack_parameters + ".pt"
    if not os.path.exists(adv_data_path):
        raise FileNotFoundError(f"Error: '{adv_data_path}' not found. Run generated_adv_data.py to generate the required data.")
    print(f"Adversarial data is loaded from {adv_data_path}")
    adv_loader = atk.load(load_path=adv_data_path, normalize=cifar10_mean_std)

    ## define the problem
    n_appr_levels, n_levels = GA_utils.encode_chromosome2(model, axx_linear=params.axx_linear, n_appr_levels=params.axx_levels)
    print(n_appr_levels, n_levels)
    xl = 0  #lista di n_levels elementi, contiene il lower limit di ogni variabile,
    #sono n_levels moltiplicatori, quindi n_levels variabili con valori da 0 a 255

    test_loss, test_acc = evaluate_test_accuracy(test_dataloader, model, device=device)
    adv_test_loss, adv_test_acc = evaluate_test_accuracy(adv_loader, model, device=device)
    print(f'Baseline test accuracy: {test_acc}') 
    print(f'Baseline adv accuracy: {adv_test_acc}') 

    problem = ProblemWrapper(n_var=n_levels, n_obj=2, xl=xl, xu=params.axx_levels, vtype=int, model=model, max_acc=test_acc, current_gen=0, previous_designs_results={})

    algorithm = NSGA2(pop_size=params.population, sampling=IntegerRandomSampling(),
                      crossover=SinglePointCrossover(prob=params.crossover_probability, repair=RoundingRepair()),
                      mutation=PM(prob=params.mutation_probability, eta=3.0, vtype=float, repair=RoundingRepair()),)

    stop_criteria = ('n_gen', params.generations)

    results = minimize(problem=problem, algorithm=algorithm, seed=1, termination=stop_criteria)

    print("--- %s seconds ---" % (time.time() - start_time))

    file = open(in_file, 'wb')
    pickle.dump(results.X, file)
    file.close()

    file = open(f_file, 'wb')
    pickle.dump(results.F, file)
    file.close()


if __name__ == "__main__":
    main()

