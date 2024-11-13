import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
from neural_networks.utils import get_loaders_split, evaluate_test_accuracy, calibrate_model, load_scaling_factors, save_scaling_factors


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation-function', default="ReLU", type=str, help="Activation function used for each act layer.")
    parser.add_argument('--batch-size', default=100, type=int, help="Number of images processed during each iteration")
    parser.add_argument('--data-dir', default="./data/", type=str, help="Directory in which the MNIST and FASHIONMNIST dataset are stored or should be downloaded")
    parser.add_argument('--dataset', default="cifar10", type=str, help="Select cifar10 or cifar100")

    parser.add_argument('--epochs', default=128, type=int, help="Number of training epochs")
    parser.add_argument('--lr-max', default=1e-1, type=float, help="Maximum learning rate for 'cyclic' scheduler, standard learning rate for 'flat' scheduler")
    parser.add_argument('--lr-min', default=1e-4, type=float, help="Minimum learning rate for 'cyclic' scheduler")
    parser.add_argument('--lr-type', default="multistep", type=str, help="Select learning rate scheduler, choose between 'cyclic' or 'multistep'")
    parser.add_argument('--weight-decay', default=5e-4, type=float, help="Weight decay applied during the optimization step")
    parser.add_argument('--num-workers', default=4, type=int, help="Number of threads used during the preprocessing of the dataset")
    parser.add_argument('--threads', default=12, type=int, help="Number of threads used during the inference, used only when neural-network-type is set to adapt")
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducible random initialization")
    parser.add_argument('--lr-momentum', default=0.9, type=float, help="Learning rate momentum")
    parser.add_argument('--split-val', default=0.1, type=float, help="The split-val is used to divide the training set in training and validation with the following dimensions: train=train_images*(1-split_val)  valid=train_images*split_val")
    parser.add_argument('--act-bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight-bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias-bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake_quant', default=1, type=int, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--neural-network', default="resnet8", type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--execution-type', default='quant', type=str, help="Select type of neural network and precision. Options are: float, quant, adapt. \n float: the neural network is executed with floating point precision.\n quant: the neural network weight, bias and activations are quantized to 8 bit\n adapt: the neural network is quantized to 8 bit and processed with exact/approximate multipliers")
    parser.add_argument('--disable-aug', default=False, type=bool, help="Set to True to disable data augmentation to obtain deterministic results")
    parser.add_argument('--reload', default=False, type=bool, help="Set to True to reload a pretraind model, set to False to train a new one")
    parser.add_argument('--continue-training', default=False, type=bool, help="Set to True to continue the training for a number of epochs that is the difference between the already trained epochs and the total epochs")
    return parser.parse_args()


def main():
    args = get_args()
    model_dir = "./neural_networks/models/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.execution_type == 'adapt':
        device = "cpu"
        torch.set_num_threads(args.threads)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    if args.fake_quant:
        namequant = "_fake"
    else:
        namequant = "_int"

    filename = model_dir + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + args.dataset +"_" + args.activation_function + ".pth"
    filename_sc = model_dir + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + args.dataset +"_" + args.activation_function + '_scaling_factors.pkl'

    print(filename)
    print(filename_sc)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, valid_loader, test_loader = get_loaders_split(args.data_dir, batch_size=args.batch_size, dataset_type=args.dataset, num_workers=args.num_workers, split_val=args.split_val, disable_aug=args.disable_aug)

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
    print(f"args.reload = {args.reload}")
    if args.reload:
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
        print(f'final test loss:{test_loss}, final test acc:{test_acc}')
        if args.continue_training:
            args.epochs = args.epochs - checkpoint['epoch']
            print(f'continuing training for {args.epochs} epochs')
        else:
            calibrate_model(model, train_loader, device)
            test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
            print(f'Post-calibration test loss:{test_loss}, test acc:{test_acc}')
            model.train()
            opt = torch.optim.SGD(model.parameters(), lr=args.lr_min, momentum=args.lr_momentum,
                                  weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
            print(f'Post-calibration with fine-tuning test loss:{test_loss}, test acc:{test_acc}')
            calibrate_model(model, train_loader, device)
            test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
            print(f'Post-calibration with fine-tuning and final calibration test loss:{test_loss}, test acc:{test_acc}')
            save_scaling_factors(model, filename_sc)
            filename = model_dir + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + args.dataset + "_" + args.activation_function + "_calibrated.pth"

            torch.save({
                'epoch': checkpoint['epoch'] + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss / train_n,
                'train_acc': train_acc / train_n,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'device': device,
                'train_parameters': {'batch': args.batch_size, 'epochs': args.epochs, 'lr': args.lr_min,
                                     'wd': args.weight_decay}
            }, filename)
            model = None

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
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            model.eval()
            test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
            print(f'Reloaded model test loss:{test_loss}, test acc:{test_acc}')
            load_scaling_factors(model, filename_sc)
            test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
            print(f'Reloaded scaling factor test loss:{test_loss}, test acc:{test_acc}')

            exit()

    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.lr_momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_type == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    best_test_acc = 0
    # Training
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        model.eval()
        test_loss, test_acc = evaluate_test_accuracy(valid_loader, model, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss / train_n,
                'train_acc': train_acc / train_n,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'device': device,
                'train_parameters': {'batch': args.batch_size, 'epochs': args.epochs, 'lr': lr,
                                     'wd': args.weight_decay}
            }, filename)
        model.train()

        print(f'epoch:{epoch}, time:{epoch_time - start_epoch_time:.2f}, lr:{lr:.6f}, train loss:{train_loss / train_n:.4f}, train acc:{train_acc / train_n:.4f}, valid loss:{test_loss:.4f}, valid acc:{test_acc:.4f}')


    # Evaluation
    if args.neural_network == "resnet8":
        model = resnet8(mode).to(device)
    elif args.neural_network == "resnet20":
        model = resnet20(mode).to(device)
    elif args.neural_network == "resnet32":
        model = resnet32(mode).to(device)
    elif args.neural_network == "resnet56":
        model = resnet56(mode).to(device)
    model.load_state_dict(torch.load(filename, map_location='cpu')['model_state_dict'])
    model.float()
    model.to(device)
    model.eval()
    test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
    print(f'Pre-calibration | test loss:{test_loss}, test acc:{test_acc}')
    calibrate_model(model, train_loader, device)
    test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
    print(f'Post-calibration | test loss:{test_loss}, test acc:{test_acc}')
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_min, momentum=args.lr_momentum,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
    test_loss, test_acc = evaluate_test_accuracy(test_loader, model, device)
    print(f'Post-calibration with fine-tuning | test loss:{test_loss}, test acc:{test_acc}')

if __name__ == "__main__":
    main()
