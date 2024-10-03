
# About
This repository contains the code used to train some ResNet models for CIFAR10 and CIFAR100.
The training is done using fake quantization, followed by 99th percentile calibration for the activations, allowing fully quantized simulation of all convolutional and linear layers.
Thi repository also contains a basic support to approximate convolutions, implemented using AdaPT classes.
## Install instructions
Execute the following commands to install the necessary dependencies.

Open a terminal and intall ninja-build
```bash
sudo apt update
sudo apt install ninja-build
```
Install marlin environment (requires anaconda or miniconda https://www.anaconda.com/download)
```bash
cd MARLIN

conda env create -f marlin.yml
```
Inside each folder there are README files with additional information useful to install additional components, execute the code and replicate the data. 

In order to use the multipliers LUTs and pre-trained neural networks, it is necessary to install 7-zip and extract the files according to the README contained in each folder. You can download 7-zip at https://www.7-zip.org/download.html
When launching any python file, remember to keep the folder hierarchy as defined in this repository.
You can add the folders to the PYTHONPATH with the following command:
```bash
export PYTHONPATH="${PYTHONPATH}:your_path_to_this folder/:your_path_to_this folder/neural_networks:your_path_to_this folder/benchmark_CIFAR10:your_path_to_this folder/approximate_multiplier$"$
export PYTHONPATH="${PYTHONPATH}:/home/michael/thesis_fw/:/home/michael/thesis_fw/neural_networks:/home/michael/thesis_fw/benchmark_CIFAR10:/home/michael/thesis_fw/approximate_multiplier"
```
