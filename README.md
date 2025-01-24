
# About
This repository aims to evaluate the robustness of quantized CNNs with approximate computing against adversarial attacks.
The repository includes:  

- Code for training ResNet models on the CIFAR10 and CIFAR100 datasets.  
- Support for fast adversarial training, available in the `fast_adversarial` directory.  
- Implementation of approximate convolutions using the **TransAxx** framework.  
- A directory dedicated to simulating adversarial attacks on CNN models. 

## Installation
Execute the following commands to install the necessary dependencies.

Open a terminal and intall ninja-build
```bash
sudo apt update
sudo apt install ninja-build
```
Install all the necessary dependecies using conda (requires anaconda or miniconda https://www.anaconda.com/download)
```bash

conda env create -f environment.yml
```

In order to use all 255 multipliers for the TransAxx model, it is necessary to install 7-zip and extract the `axx_mults_8x8.7z` file located un `transaxx/ext_modules/include/nn/cuda/axx_mults`. You can download 7-zip at https://www.7-zip.org/download.html. Execute the following command to extract the `.7z` file:
```bash
cd transaxx/ext_modules/include/nn/cuda/axx_mults/
7z x axx_mults_8x8.7z
```


Finally, modify the PYTHONPATH as follows:
```bash
export PYTHONPATH="${PYTHONPATH}:your_path_to_this_folder/transaxx/:your_path_to_this_folder/neural_networks:your_path_to_this_folder/benchmark_CIFAR10:your_path_to_this_folder/approximate_multiplier$"$
```

### python_quantization
In case `python_quantization` is not installed, execute these commands:
```bash
cd transaxx/pytorch-quantization
pip install --no-cache-dir --index-url https://pypi.nvidia.com --index-url https://pypi.org/simple pytorch-quantization==2.1.3
python setup.py install
```

Or you can try following the instructions in the `README` in the `transaxx/pytorch-quantization` directory, but this is what worked for me

### MARLIN Environment

In order to use the script contained in `benchmark_CIFAR10/`, run the following commands:
```bash
cd MARLIN
make build
```
This command will setup the required scripts and repository. Further information on the configuration can be found in the folder `./riscv_characterization`.

### Cuda Setup

Set the following environment variables in .bashrc for proper CUDA functioning:
```bash
export CUDA_HOME="/usr/local/cuda-11.8"
export CUDA_PATH="/usr/local/cuda-11.8"
```
# Usage

Focus on the `neural_networks`, `fast_adversarial`, and `adversarial` directories. The other directories primarily contain dependencies, and you won’t need to run any scripts from them.  

## Model Training and Evaluation
To train CIFAR-10 NNs you can use the script "train_cifar10.py", you can check the options by executing the following command: 
> python neural_networks/CIFAR10/train_cifar10.py --help


As an example, run the following command:
> python neural_networks/CIFAR10/train_cifar10.py --neural-network resnet8 --execution-type quant

The available execution types are the following:
- **`float`**: Models are defined using regular layers from `torch.nn`.
- **`quant`**: These are quantized models, the quantization is achieved using custom layer definitions in `neural_networks/custom_layers.py`.
- **`transaxx`**: These are quantized models that also support approximate multipliers. Quantization is done using `pytorch-quantization`, and custom layers are defined in `transaxx/layers`. For more details about the `transaxx` layer definitions, refer to the `transaxx/layers/` directory.

To evaluate a model’s accuracy, run the following command:
> python neural_networks/CIFAR10/test_transaxx.py --neural-network resnet32 --execution-type quant --param-execution-type float

In this example, we evaluate the accuracy of a `resnet32` model with `quant` execution type, using pre-trained parameters from a `resnet32` model with `float` execution type.

In general, there are three execution types available, and you can load pre-trained parameters from any of them. This allows for nine possible combinations, such as loading parameters from a `float` model into a `transaxx` model, or from `quant` into `transaxx`, and so on.
 

## Adversarial Training
The `fast_adversarial` directory contains training scripts using an FGSM adversary, with the goal of increasing the model's accuracy when under attack. For a more detailed overview of this repository see the README in the `fast_adversarial` directory.

For example running the following command:
> python fast_adversarial/CIFAR10/train_resnet.py --neural-network resnet32 --execution-type transaxx --epochs 112 --reload 1

Would instantiate a resnet32 model with pre-trained parameters (because reload is 1), and continue training for a number of epochs equal to the difference between the number of epochs passed as argument and the number of epochs used for the pre-trained parameters.
The adversarially trained parameters are saved in `fast_adversarial/AT_models`.

## Attack Simulation

To simulate attacks on CNNs, the first step is generating perturbed data (adversarial images). This can be done using the following command:

> python adversarial/generate_adv_data.py --neural-network resnet8 --execution-type quant 

In this command:
- A resnet8 model is instantiated.
- You are prompted to select the type of attack and its parameters.
- Adversarial images are generated based on this model and saved in the `adversarial/adv_data` directory.

To evaluate a model's accuracy under attack, use the resnet_attack_eval.py script. For instance:

> python adversarial/resnet_attack_eval.py --neural-network resnet32 --execution-type quant --adv-neural-network resnet8 --adv-execution-type quant

Here:
- A resnet32 model is instantiated and evaluated.
- The adversarial data generated by the resnet8 model in the previous step is used to test the resnet32 model's accuracy.


## Reproducing the Results
In this section, I will outline all the steps needed to recreate the plots, including retraining, adversarial retraining, adversarial data generation, evaluation, and plotting.

Retrain the TransAxx model for 2 epochs
```bash
python neural_networks/CIFAR10/train_cifar10.py --neural-network resnet32 --execution-type transaxx --reload 1 --continue-training 1 --epochs 112 --lr-type step --lr 0.0005 --lr-gamma 0.9 --step-size 2
```

Train the float model
```bash
python neural_networks/CIFAR10/train_cifar10.py --neural-network resnet32 --execution-type float
```

Adversarially train the float model for 2 epochs
```bash
python fast_adversarial/CIFAR10/train_resnet.py --neural-network resnet32 --execution-type float --lr-schedule step --epsilon 1 --epochs 105 --reload 1
```

Adversarially train the quant model for 2 epochs
```bash
python fast_adversarial/CIFAR10/train_resnet.py --neural-network resnet32 --execution-type quant --lr-schedule step --epsilon 1 --epochs 112 --reload 1
```

Adversarially train the transaxx model for 2 epochs
```bash
python fast_adversarial/CIFAR10/train_resnet.py --neural-network resnet32 --execution-type transaxx --lr-schedule step --epsilon 1 --epochs 112 --reload 1
```

Generate the adversarial data using the 3 execution types, the default attack type is used (PGD):
```bash
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type transaxx
```

Generate the adversarial data using the 3 execution types, but using the parameters generated form the adversarial training, same attack type:
```bash
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type float --AT 1 --AT-epochs 105 --AT-epsilon 1
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type quant --AT 1 --AT-epochs 112 --AT-epsilon 1
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type transaxx --AT 1 --AT-epochs 112 --AT-epsilon 1
```

Generate the data (adv. accuracy and regular accuracy) for the plots using the adv. data generated without adv. training
```bash
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105
```
Same thing but now data generated with adv. training
```bash
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-AT 1
```

Plot the results
```bash
python adversarial/plot_adv_acc.py --neural-network resnet32 --adv-neural-network resnet32 --adv-AT 0
```
or
```bash
python adversarial/plot_adv_acc.py --neural-network resnet32 --adv-neural-network resnet32 --adv-AT 1
```


To test the performance of resnet32 on adv. data generated by resnet8 (any 2 combination of resnet networks can be tested):
```bash
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet8
```

Plot:
```bash
python adversarial/plot_adv_acc.py --neural-network resnet32 --adv-neural-network resnet8 --adv-AT 0
```

The same commands work for other resnet models, the only parameter that needs to be changed is the `--epochs` parameter, set it so that it only retrains the model for 2 epochs.

## Acknowledgments
This project makes use of the following third-party libraries:

-[MARLIN](https://github.com/vlsi-lab/MARLIN)
Used for defining the CNN models and providing the script to train them. 

-[fast_adversarial](https://github.com/locuslab/fast_adversarial/tree/master)
Used for adversarial training of the CNN models. 

-[TransAxx](https://github.com/dimdano/transaxx)
Used for supporting approximate convolutional layers.

-[Adversarial-Attacks-PyTorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
Used to generate adversarial data for simulating adversarial attacks.
