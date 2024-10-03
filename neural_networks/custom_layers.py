import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from neural_networks.quantization import act_quantizer, param_quantizer, QuantCalibrator, fake_scale_out_funct, scale_out_funct

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False, act_bit=8, weight_bit=8, bias_bit=8, fake_quant=True, scaling_factor=None, calibrate=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.act_bit = act_bit  # activation precision
        self.weight_bit = weight_bit  # weight precision
        self.bias_bit = bias_bit  # bias precision
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1
        self.max_v_a = 2 ** (self.act_bit - 1) - 1
        self.scaling_factor = scaling_factor
        if self.act_bit <= 8:
            self.datatype = torch.int8
        elif self.act_bit <= 16:
            self.datatype = torch.int16
        elif self.act_bit <= 32:
            self.datatype = torch.int32
        self.aq = act_quantizer(-self.max_v_a, self.max_v_a, fake_quant=fake_quant, scaling_factor=self.scaling_factor, dtype=self.datatype) # use scaling factor computed using
        self.wq = param_quantizer(-self.max_v_w, self.max_v_w, fake_quant=fake_quant, dtype=self.datatype)
        self.bq = param_quantizer(-self.max_v_b, self.max_v_b, fake_quant=fake_quant,dtype= self.datatype)
        self.calibrate=calibrate
        self.calibrator = QuantCalibrator(th=0.001, act_bit=self.act_bit)
        if fake_quant:
            self.scale_out = fake_scale_out_funct.apply
        else:
            self.scale_out = scale_out_funct.apply

    def forward(self, x):
        if not self.training and self.calibrate:
            self.calibrator.calibrate_funct(x.detach())


        x =  F.conv2d(self.aq(x), self.wq(self.weight), None, self.stride, self.padding, self.dilation, self.groups)
        x = self.scale_out(x, self.aq.scaling_factor, self.wq.scaling_factor)
        if self.bias is not None:
            x = x + self.scale_out(self.bq(self.bias).view(1, -1, 1, 1), self.bq.scaling_factor, 1)
            # x = x + self.scale_out(self.bq(self.bias), self.bq.scaling_factor, 1)
        return x

    def update_act_scale(self, scaling_factor=None):
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor
        else:
            self.scaling_factor = self.max_v_a / self.calibrator.calibrated_value
        self.aq.scaling_factor = self.scaling_factor

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, act_bit=8, weight_bit=8, bias_bit=8, fake_quant=True, scaling_factor=None, calibrate=False):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.act_bit = act_bit  # activation precision
        self.weight_bit = weight_bit  # weight precision
        self.bias_bit = bias_bit  # bias precision
        self.max_v_w = 2 ** (self.weight_bit - 1) - 1
        self.max_v_b = 2 ** (self.bias_bit - 1) - 1
        self.max_v_a = 2 ** (self.act_bit - 1) - 1
        self.scaling_factor = scaling_factor
        if self.act_bit <= 8:
            self.datatype = torch.int8
        elif self.act_bit <= 16:
            self.datatype = torch.int16
        elif self.act_bit <= 32:
            self.datatype = torch.int32
        self.aq = act_quantizer(-self.max_v_a, self.max_v_a, fake_quant=fake_quant, scaling_factor=self.scaling_factor, dtype=self.datatype)
        self.wq = param_quantizer(-self.max_v_w, self.max_v_w, fake_quant=fake_quant, dtype=self.datatype)
        self.bq = param_quantizer(-self.max_v_b, self.max_v_b, fake_quant=fake_quant, dtype=self.datatype)
        self.calibrate=calibrate
        self.calibrator = QuantCalibrator(th=0.001, act_bit=self.act_bit)
        if fake_quant:
            self.scale_out = fake_scale_out_funct.apply
        else:
            self.scale_out = scale_out_funct.apply

    def forward(self, x):
        if not self.training and self.calibrate:
            self.calibrator.calibrate_funct(x.detach())

        x =  F.linear(self.aq(x), self.wq(self.weight), None)
        x = self.scale_out(x, self.aq.scaling_factor, self.wq.scaling_factor)
        if self.bias is not None:
            x = x + self.scale_out(self.bq(self.bias).view(1, -1), self.bq.scaling_factor, 1)
            # x = x + self.scale_out(self.bq(self.bias), self.bq.scaling_factor, 1)
        return x

    def update_act_scale(self, scaling_factor=None):
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor
        else:
            self.scaling_factor = self.max_v_a / self.calibrator.calibrated_value
        self.aq.scaling_factor = self.scaling_factor


class Act(nn.ReLU):
    def __init__(self, act_bit=8, fake_quant=True, scaling_factor=None, calibrate=False):
        """
        Activation function wrapper, check activations.py for the documentation
        @param act_type: Name of the activation function, can be any of the functions in the act_lib dictionary
        @param act_params a list of parameters for the activation function.
        """
        super(Act, self).__init__()

        self.act_bit = act_bit  # activation precision
        self.max_v_a = 2 ** (self.act_bit - 1) - 1
        self.scaling_factor = scaling_factor
        if self.act_bit <= 8:
            self.datatype = torch.int8
        elif self.act_bit <= 16:
            self.datatype = torch.int16
        elif self.act_bit <= 32:
            self.datatype = torch.int32
        self.aq = act_quantizer(-self.max_v_a, self.max_v_a, fake_quant=fake_quant, scaling_factor=self.scaling_factor, dtype=self.datatype) # use scaling factor computed using
        self.calibrate=calibrate
        self.calibrator = QuantCalibrator(th=0.001, act_bit=self.act_bit)
        if fake_quant:
            self.scale_out = fake_scale_out_funct.apply
        else:
            self.scale_out = scale_out_funct.apply


    def forward(self, x):
        if not self.training and self.calibrate:
            self.calibrator.calibrate_funct(x.detach())
        return F.relu(self.aq(x))

    def update_act_scale(self, scaling_factor=None):
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor
        else:
            self.scaling_factor = self.max_v_a / self.calibrator.calibrated_value
        self.aq.scaling_factor = self.scaling_factor

