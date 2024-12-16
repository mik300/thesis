import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.adapt.approx_layers import axx_layers
from neural_networks.custom_layers import Conv2d, Linear, Act
import pickle
import numpy
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

biasflag = False
log = True
torch.set_printoptions(threshold=float('inf'), precision=16, sci_mode=False)
conv_inputs_path = "conv_inputs.txt"
conv_outputs_path = "conv_outputs.txt"
conv_outputs_mat = "conv_output_mat.txt"

class ResidualModule(nn.Module):
    ratio = 1
    bn_momentum = 0.1
    layer_index = -11

    def __init__(self, input_channels, output_channels, stride=1, mode=None):
        
        super(ResidualModule, self).__init__()

        
        self.bn1 = nn.BatchNorm2d(input_channels, momentum=self.bn_momentum)

        if mode["execution_type"] == 'float' or mode["execution_type"] == 'transaxx':
            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=biasflag)

        elif mode["execution_type"] == 'quant':
            self.conv1 = Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        elif mode["execution_type"] == 'adapt':
            self.conv1 = axx_layers.AdaPT_Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False, axx_mult="bw_mult_9_9_0", scaling_factor=None, calibrate=False)

        else:
            exit("unknown layer command")

        if mode["execution_type"] == 'float' or mode["execution_type"] == 'transaxx':
            self.act1 = nn.ReLU()
        else:
            self.act1 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        self.bn2 = nn.BatchNorm2d(output_channels, momentum=self.bn_momentum)

        if mode["execution_type"] == 'float' or mode["execution_type"] == 'transaxx':
            self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)

        elif mode["execution_type"] == 'quant':
            self.conv2 = Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=biasflag, act_bit=mode['act_bit'],
                                weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        elif mode["execution_type"] == 'adapt':
            self.conv2 = axx_layers.AdaPT_Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False, axx_mult="bw_mult_9_9_0", scaling_factor=None, calibrate=False)

        else:
            exit("unknown layer command")
        if mode["execution_type"] == 'float' or mode["execution_type"] == 'transaxx':
            self.act2 = nn.ReLU()
        else:
            self.act2 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)
        if stride != 1 or input_channels != self.ratio * output_channels:
            self.shortcut = True

    def shortcut_identity(self, prex, x):
        """
        Identity mapping implementation of the shortcut version without parameters.
        @param prex: activations tensor before the activation layer
        @param x: input tensor
        @return: sum of input tensors
        """
        if x.shape != prex.shape:
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return prex + torch.cat((d, p), dim=1)
        else:
            return prex + x

    def forward(self, x):
        conv_name = 'layer' + str(ResidualModule.layer_index) + '.0.'
        #print(f'conv_name = {conv_name}')
        out = self.act1(self.bn1(x))
        #log_to_file(out, f'static elem_t {conv_name}conv1_in<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_inputs_path, conv_name + 'conv1')
        log_quantized_act(out, f'static elem_t  {conv_name}conv_1_in<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_inputs_path, conv_name + 'conv1')
        out = self.conv1(out)
        #print(f"out = {out[0][0][0][0]}")
        #log_quantized_output(out, f'static elem_t  {conv_name}conv_1_out<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_outputs_path, conv_name + 'conv1')
        #log_to_file(out, f'static elem_t {conv_name}conv1_out<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_outputs_path, conv_name + 'conv1')
        #log_output(out, conv_outputs_mat, conv_name + 'conv1')
        #print(f'conv1 out shape = {out.shape}')
        out = self.act2(self.bn2(out))
        #log_to_file(out, f'static elem_t {conv_name}conv2_in<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_inputs_path, conv_name + 'conv2')
        log_quantized_act(out, f'static elem_t {conv_name}conv2_in<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_inputs_path, conv_name + 'conv2')
        out = self.conv2(out)
        #log_quantized_output(out, f'static elem_t {conv_name}conv2_out<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_outputs_path, conv_name + 'conv2')
        #log_to_file(out, f'static elem_t {conv_name}conv2_out<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_outputs_path, conv_name + 'conv2')
        #log_output(out, conv_outputs_mat, conv_name + 'conv2')
        #print(f'conv2 out shape = {out.shape}')
        out = self.shortcut_identity(out, x) if hasattr(self, 'shortcut') else out + x
        ResidualModule.layer_index += 1
        return out


class ResNet(nn.Module):
    def __init__(self, res_block, num_res_blocks, num_classes=10, mode=None):
        super(ResNet, self).__init__()
        print(f'mode = {mode}')
        self.input_channels = 16
        self.bn_momentum = 0.1
        if mode["execution_type"] == 'float' or mode["execution_type"] == 'transaxx':
            self.conv1 = nn.Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        elif mode["execution_type"] == 'quant':
            self.conv1 = Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)
        elif mode["execution_type"] == 'adapt':
            self.conv1 = axx_layers.AdaPT_Conv2d(in_channels=3, out_channels=self.input_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, dilation=1, bias=False, axx_mult="bw_mult_9_9_0", scaling_factor=None, calibrate=False)

        else:
            exit("unknown layer command")

        if mode["execution_type"] == 'float' or mode["execution_type"] == 'transaxx':
            self.act = nn.ReLU()
        else:
            self.act = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        self.layer1 = self._make_res(res_block, 16, num_res_blocks[0], stride=1, mode=mode)
        self.layer2 = self._make_res(res_block, 32, num_res_blocks[1], stride=2, mode=mode)
        self.layer3 = self._make_res(res_block, 64, num_res_blocks[2], stride=2, mode=mode)

        self.bn = nn.BatchNorm2d(64 * res_block.ratio, momentum=self.bn_momentum)

        self.flatten = nn.Flatten()
        if mode["execution_type"] == 'float' or mode["execution_type"] == 'transaxx':
            self.linear = nn.Linear(64 * res_block.ratio, mode['classes'])
        else:
            self.linear = Linear(64 * res_block.ratio, mode['classes'], act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'])

    def _make_res(self, res_block, output_channels, num_res_blocks, stride, mode=None):
        strides = [stride] + [1] * (num_res_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(res_block(self.input_channels, output_channels, stride, mode))
            self.input_channels = output_channels * res_block.ratio
        return nn.Sequential(*layers)

    def forward(self, x):
        init_log()
        log_quantized_act(x, f'static elem_t conv_1_in<{x.shape[0]}><{x.shape[2]}><{x.shape[3]}><{x.shape[1]}> row_align(1)', conv_inputs_path, 'conv1')
        out = self.conv1(x)
        log_quantized_output(out, f'static elem_t conv_1_out<{out.shape[0]}><{out.shape[2]}><{out.shape[3]}><{out.shape[1]}> row_align(1)', conv_outputs_path, 'conv1')
        #log_output(out, conv_outputs_mat, 'conv1')
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = self.flatten(out)
        #log_to_file(out, f'static elem_t linear_in<{out.shape[0]}><{out.shape[1]}> row_align(1)', conv_inputs_path, 'linear')
        log_quantized_act(out, f'static elem_t linear_in<{out.shape[0]}><{out.shape[1]}> row_align(1)', conv_inputs_path, 'linear')
        out = self.linear(out)
        #log_to_file(out, f'static elem_t linear_out<{out.shape[0]}><{out.shape[1]}> row_align(1)', conv_outputs_path, 'linear')
        #log_output(out, conv_outputs_mat, 'linear')
        #log_quantized_output(out, f'static elem_t linear_out<{out.shape[0]}><{out.shape[1]}> row_align(1)', conv_outputs_path, 'linear')
        return out


def log_to_file(output, layer_name, filepath, conv_name):
    if log == True:
        if torch.is_tensor(output):
            scaling_factor = scale_to_int8(conv_name)
            tensor_scaled = torch.clamp(torch.round(scaling_factor * output.detach()), min=-128, max=127).to(torch.int8)
            #tensor_scaled = output.detach()
            if tensor_scaled.dim() == 4:
                tensor_scaled = tensor_scaled.permute(0, 2, 3, 1) #permute the dimensions to match the definition of inputs in gemmini 
        with open(filepath, 'a') as f:
            # Convert the tensor to a NumPy array for clean output
            tensor_scaled = tensor_scaled.detach().cpu().numpy()  # Move to CPU and convert to NumPy
            # Format the output as a string
            tensor_scaled = tensor_scaled.tolist()  # Convert to list for better formatting
            list_string = str(tensor_scaled).replace('\n', '').replace('  ', ' ')  # Replace newlines if any
            list_string = list_string + ";"
            f.write(f'{layer_name} = {list_string}\n')

def log_quantized_output(output, layer_name, filepath, conv_name):
    if log == True:
        if torch.is_tensor(output):
            # quant_bits = 8
            # quant_limit = pow(2,quant_bits-1)-1
            # fake_quant = False
            # quant_desc = QuantDescriptor(num_bits=quant_bits, fake_quant=fake_quant, unsigned=False, calib_method='histogram')
            # quantizer = TensorQuantizer(quant_desc)
            tensor_scaled = output.detach()
            quant_output = tensor_scaled
            #quant_output = quantizer(tensor_scaled)
            if quant_output.dim() == 4:
                quant_output = quant_output.permute(0, 2, 3, 1) #permute the dimensions to match the definition of inputs in gemmini 
        with open(filepath, 'a') as f:
            quant_output = quant_output.detach().cpu().numpy()  # Move to CPU and convert to NumPy
            quant_output = quant_output.tolist()  # Convert to list for better formatting
            list_string = str(quant_output).replace('\n', '').replace('  ', ' ')  # Replace newlines if any
            list_string = list_string + ";"
            f.write(f'{layer_name} = {list_string}\n')

def log_quantized_act(input_tensor, layer_name, filepath, conv_name):
    if log == True:
        if torch.is_tensor(input_tensor):
            quant_bits = 8
            quant_limit = pow(2,quant_bits-1)-1
            fake_quant = False
            quant_desc = QuantDescriptor(num_bits=quant_bits, fake_quant=fake_quant, unsigned=False, calib_method='histogram')
            quantizer = TensorQuantizer(quant_desc)
            tensor_in = input_tensor.detach()
            quant_input = quantizer(tensor_in)
            if quant_input.dim() == 4:
                quant_input = quant_input.permute(0, 2, 3, 1) #permute the dimensions to match the definition of inputs in gemmini 
        with open(filepath, 'a') as f:
            # Convert the tensor to a NumPy array for clean input_tensor
            quant_input = quant_input.cpu().numpy()  # Move to CPU and convert to NumPy
            # Format the input_tensor as a string
            quant_input = quant_input.tolist()  # Convert to list for better formatting
            list_string = str(quant_input).replace('\n', '').replace('  ', ' ')  # Replace newlines if any
            list_string = list_string + ";"
            f.write(f'{layer_name} = {list_string}\n')

def log_output(output, filepath, conv_name):
    if log == True:
        tensor_scaled = output.detach()
        if torch.is_tensor(output):
            scaling_factor = scale_to_int8(conv_name)
            #print(f'scaling_factor.type = {scaling_factor.type}')
            tensor_scaled = torch.clamp(torch.round(scaling_factor * output.detach()), min=-128, max=127).to(torch.int8)
            #tensor_scaled = output.detach()
            #print(f'tensor_scaled.dim = {tensor_scaled.dim()}')
            #print(f'tensor_scaled.shape = {tensor_scaled.shape}')
            if tensor_scaled.dim() == 4:
                tensor_scaled = tensor_scaled.permute(0, 2, 3, 1) #permute the dimensions to match the definition of inputs in gemmini 
            np_array = tensor_scaled.cpu().detach().numpy()
        with open(conv_outputs_mat, 'a') as f:
            if conv_name != 'linear':
                f.write("output_mat:\n")
                for och in range(np_array.shape[0]):
                    for wrow in range(np_array.shape[1]):
                        for wcol in range(np_array.shape[2]):
                            f.write("[")
                            for ich in range(np_array.shape[3]):
                                if ich == np_array.shape[3] - 1:
                                    f.write(f"{np_array[och][wrow][wcol][ich]}")
                                else:
                                    f.write(f"{np_array[och][wrow][wcol][ich]},")
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
    filename_sc = './neural_networks/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU_scaling_factors.pkl'
    with open(filename_sc, 'rb') as f:
        scaling_factors = pickle.load(f)
    
    # Move all scaling factors to CPU
    for key, value in scaling_factors.items():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            scaling_factors[key] = value.cpu()
    
    return scaling_factors[conv_name]

def init_log():
    with open(conv_inputs_path, 'w') as f:
        f.write("")
    with open(conv_outputs_path, 'w') as f:
        f.write("")
    with open(conv_outputs_mat, 'w') as f:
        f.write("")

def resnet8(mode=None):
    return ResNet(ResidualModule, [1, 1, 1], mode=mode)


def resnet14(mode=None):
    return ResNet(ResidualModule, [2, 2, 2], mode=mode)


def resnet20(mode=None):
    return ResNet(ResidualModule, [3, 3, 3], mode=mode)


def resnet32(mode=None):
    return ResNet(ResidualModule, [5, 5, 5], mode=mode)


def resnet50(mode=None):
    return ResNet(ResidualModule, [8, 8, 8], mode=mode)


def resnet56(mode=None):
    return ResNet(ResidualModule, [9, 9, 9], mode=mode)


def resnet110(mode=None):
    return ResNet(ResidualModule, [18, 18, 18], mode=mode)