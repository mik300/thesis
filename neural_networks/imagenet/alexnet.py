import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.adapt.approx_layers import axx_layers
from neural_networks.custom_layers import Conv2d, Linear, Act
import pickle
import numpy

biasflag = False
fc_biasflag = True
log = True
conv_inputs_path = "alexnet_conv_inputs.txt"
conv_outputs_mat = "alexnet_conv_output_mat.txt"
class AlexNet(nn.Module):
    bn_momentum = 0.1
    def __init__(self, num_classes=10, mode=None):
        print(f"num_classes in Alexnet = {num_classes}")
        super(AlexNet, self).__init__()
        print(f'mode = {mode}')
        # First convolutional layer
        if mode["execution_type"] == 'float':
            self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=biasflag)
        elif mode["execution_type"] == 'quant':
            self.conv1 = Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'])
        elif mode["execution_type"] == 'adapt':
            self.conv1 = axx_layers.AdaPT_Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4), padding=(0, 0), bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act1 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        # Max Pooling layer after conv1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.bn1 = nn.BatchNorm2d(96, momentum=self.bn_momentum)

        # Second convolutional layer
        if mode["execution_type"] == 'float':
            self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride = 1, padding=2, groups=1, bias=biasflag)
        elif mode["execution_type"] == 'quant':
            self.conv2 = Conv2d(96, 256, kernel_size=5,  stride = 1, padding=2, groups=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'])
        elif mode["execution_type"] == 'adapt':
            self.conv2 = axx_layers.AdaPT_Conv2d(96, 256, kernel_size=(5, 5), stride = 1, padding=(2, 2), groups=1, bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act2 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        # Max Pooling layer after conv2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.bn2 = nn.BatchNorm2d(256, momentum=self.bn_momentum)

        # Third convolutional layer
        if mode["execution_type"] == 'float':
            self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride = 1, padding=1, bias=biasflag)
        elif mode["execution_type"] == 'quant':
            self.conv3 = Conv2d(256, 384, kernel_size=3, stride = 1, padding=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'])
        elif mode["execution_type"] == 'adapt':
            self.conv3 = axx_layers.AdaPT_Conv2d(256, 384, kernel_size=3, stride = 1, padding=1, bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act3 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        # Fourth convolutional layer
        if mode["execution_type"] == 'float':
            self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride = 1, padding=1, groups=1, bias=biasflag)
        elif mode["execution_type"] == 'quant':
            self.conv4 = Conv2d(384, 384, kernel_size=3, stride = 1, padding=1, groups=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'])
        elif mode["execution_type"] == 'adapt':
            self.conv4 = axx_layers.AdaPT_Conv2d(384, 384, kernel_size=3, stride = 1, padding=1, groups=1, bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act4 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        # Fifth convolutional layer
        if mode["execution_type"] == 'float':
            self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=1, bias=biasflag)
        elif mode["execution_type"] == 'quant':
            self.conv5 = Conv2d(384, 256, kernel_size=3, padding=1, groups=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'])
        elif mode["execution_type"] == 'adapt':
            self.conv5 = axx_layers.AdaPT_Conv2d(384, 256, kernel_size=3, padding=1, groups=1, bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act5 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        # Max Pooling layer after conv5
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        # Fully connected layers
        if mode["execution_type"] == 'float':
            self.fc6 = nn.Linear(256 * 6 * 6, 4096, bias=fc_biasflag)
        elif mode["execution_type"] == 'quant':
            self.fc6 = Linear(256 * 6 * 6, 4096, bias=fc_biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'])
        elif mode["execution_type"] == 'adapt':
            self.fc6 = axx_layers.AdaPT_Linear(256 * 6 * 6, 4096, bias=fc_biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act6 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        if mode["execution_type"] == 'float':
            self.fc7 = nn.Linear(4096, 4096, bias=fc_biasflag)
        elif mode["execution_type"] == 'quant':
            self.fc7 = Linear(4096, 4096, bias=fc_biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'])
        elif mode["execution_type"] == 'adapt':
            self.fc7 = axx_layers.AdaPT_Linear(4096, 4096, bias=fc_biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act7 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        if mode["execution_type"] == 'float':
            self.fc8 = nn.Linear(4096, num_classes, bias=fc_biasflag)
        elif mode["execution_type"] == 'quant':
            self.fc8 = Linear(4096, num_classes, bias=fc_biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'])
        elif mode["execution_type"] == 'adapt':
            self.fc8 = axx_layers.AdaPT_Linear(4096, num_classes, bias=fc_biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        

        # Softmax activation at the end
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        init_log()
        log_to_file(x, f'static elem_t alexnet_images<{x.shape[0]}><{x.shape[2]}><{x.shape[3]}><{x.shape[1]}> row_align(1)', conv_inputs_path,'conv1')
        out = self.conv1(x)
        out = self.act1(out)
        out = self.pool1(out)
        #log_output(out, 'conv1')
        #out = self.bn1(out)
        #print(f'out shape = {out.shape}')
        #log_to_file(out, f'static elem_t conv_1_out_pooled<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_inputs_path,'conv1')
        out = self.conv2(out)
        out = self.act2(out)
        out = self.pool2(out)
        #log_output(out, 'conv2')
        #out = self.bn2(out)

        out = self.conv3(out)
        out = self.act3(out)
        #log_output(out, 'conv3')

        out = self.conv4(out)
        out = self.act4(out)
        #log_output(out, 'conv4')

        out = self.conv5(out)
        out = self.act5(out)
        out = self.pool3(out)
        #log_output(out, 'conv5')
        #print(f'before flatten = {out.shape}')
        out = self.flatten(out)
        #log_output(out, 'conv5')
        #print(f'flatten = {out.shape}')
        #log_to_file(out, f'static elem_t average<{out.shape[0]}><{out.shape[1]}> row_align(1)', conv_inputs_path,'conv5')
 
        out = self.fc6(out)
        out = self.act6(out)
        #log_output(out, 'act6')

        out = self.fc7(out)
        out = self.act7(out)
        log_output(out, 'act7')
        #print(f'out shape = {out.shape}')
        #log_to_file(out, f'static elem_t fc8_in<{out.shape[0]}><{out.shape[1]}> row_align(1)', conv_inputs_path,'fc8')
        out = self.fc8(out)
        #print(f'out shape = {out.shape}')
        #log_output(out, 'fc8')
        #out = self.softmax(out)

        return out

def alexnet(mode=None):
    return AlexNet(mode=mode)


def log_to_file(output, layer_name, filepath, conv_name):
    if log == True:
        if torch.is_tensor(output):
            scaling_factor = scale_to_int8(conv_name)
            tensor_scaled = torch.clamp(torch.round(scaling_factor * output), min=-128, max=127).to(torch.int8)
            if tensor_scaled.dim() == 4:
                tensor_scaled = tensor_scaled.permute(0, 2, 3, 1) #permute the dimensions to match the definition of inputs in gemmini 
        with open(filepath, 'a') as f:
            # Convert the tensor to a NumPy array for clean output
            tensor_scaled = tensor_scaled.detach().cpu().numpy()  # Move to CPU and convert to NumPy
            # Format the output as a string
            tensor_scaled = tensor_scaled.tolist()  # Convert to list for better formatting
            list_string = str(tensor_scaled).replace('\n', '').replace('  ', ' ').replace(' ', '')  # Replace newlines if any
            list_string = list_string + ";"
            f.write(f'{layer_name} = {list_string}\n')



def log_output(output, conv_name):
    if log == True:
        tensor_scaled = output
        if torch.is_tensor(output):
            scaling_factor = scale_to_int8(conv_name)
            #print(f'scaling_factor.type = {scaling_factor.type}')
            tensor_scaled = torch.clamp(torch.round(scaling_factor * output), min=-128, max=127).to(torch.int8)
            #print(f'tensor_scaled.dim = {tensor_scaled.dim()}')
            #print(f'tensor_scaled.shape = {tensor_scaled.shape}')
            if tensor_scaled.dim() == 4:
                tensor_scaled = tensor_scaled.permute(0, 2, 3, 1) #permute the dimensions to match the definition of inputs in gemmini 
            np_array = tensor_scaled.cpu().numpy()
        with open(conv_outputs_mat, 'a') as f:
            if "fc" not in conv_name and "act" not in conv_name:
                f.write("output_mat:\n")
                for batch in range(np_array.shape[0]):
                    for wrow in range(np_array.shape[1]):
                        for wcol in range(np_array.shape[2]):
                            f.write("[")
                            for och in range(np_array.shape[3]):
                                if och == np_array.shape[3] - 1:
                                    f.write(f"{np_array[batch][wrow][wcol][och]}")
                                else:
                                    f.write(f"{np_array[batch][wrow][wcol][och]},")
                            f.write("]\n")
                f.write("\n\n")
            else:
                f.write("output_mat:\n")
                for orow in range(np_array.shape[0]):
                    f.write("[")
                    for ocol in range(np_array.shape[1]):
                        if ocol == np_array.shape[1] - 1:
                            f.write(f"{np_array[orow][ocol]}")
                        else:
                            f.write(f"{np_array[orow][ocol]},")
                    f.write("]\n")


def scale_to_int8(conv_name):
    filename_sc = './neural_networks/models/alexnet_a8_w8_b32_fake_quant_cifar10_ReLU_scaling_factors.pkl'
    with open(filename_sc, 'rb') as f:
        scaling_factors = pickle.load(f)
    
    # Move all scaling factors to CPU
    for key, value in scaling_factors.items():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            scaling_factors[key] = value.cpu()
    
    return scaling_factors[conv_name]

def init_log():
    if log == True:
        with open(conv_inputs_path, 'w') as f:
            f.write("")
        with open(conv_outputs_mat, 'w') as f:
            f.write("")