import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.adapt.approx_layers import axx_layers
from neural_networks.custom_layers import Conv2d, Linear, Act

biasflag = False
class ResidualModule(nn.Module):
    ratio = 1
    bn_momentum = 0.1

    def __init__(self, input_channels, output_channels, stride=1, mode=None):
        
        super(ResidualModule, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_channels, momentum=self.bn_momentum)

        if mode["execution_type"] == 'float':
            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=biasflag)

        elif mode["execution_type"] == 'quant':
            self.conv1 = Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        elif mode["execution_type"] == 'adapt':
            self.conv1 = axx_layers.AdaPT_Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False, axx_mult="bw_mult_9_9_0", scaling_factor=None, calibrate=False)

        else:
            exit("unknown layer command")

        self.act1 = Act(act_bit=8, fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        self.bn2 = nn.BatchNorm2d(output_channels, momentum=self.bn_momentum)

        if mode["execution_type"] == 'float':
            self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)

        elif mode["execution_type"] == 'quant':
            self.conv2 = Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=biasflag, act_bit=mode['act_bit'],
                                weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        elif mode["execution_type"] == 'adapt':
            self.conv2 = axx_layers.AdaPT_Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False, axx_mult="bw_mult_9_9_0", scaling_factor=None, calibrate=False)

        else:
            exit("unknown layer command")

        self.act2 = Act(act_bit=8, fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

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
        out = self.act1(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        out = self.shortcut_identity(out, x) if hasattr(self, 'shortcut') else out + x
        return out


class ResNet(nn.Module):
    def __init__(self, res_block, num_res_blocks, num_classes=10, mode=None):
        super(ResNet, self).__init__()
        self.input_channels = 16
        self.bn_momentum = 0.1
        if mode["execution_type"] == 'float':
            self.conv1 = nn.Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        elif mode["execution_type"] == 'quant':
            self.conv1 = Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1, bias=biasflag, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'], fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)
        elif mode["execution_type"] == 'adapt':
            self.conv1 = axx_layers.AdaPT_Conv2d(in_channels=3, out_channels=self.input_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, dilation=1, bias=False, axx_mult="bw_mult_9_9_0", scaling_factor=None, calibrate=False)

        else:
            exit("unknown layer command")
        print(mode.keys())
        self.act = Act(act_bit=8, fake_quant=mode['fake_quant'], scaling_factor=None, calibrate=False)

        self.layer1 = self._make_res(res_block, 16, num_res_blocks[0], stride=1, mode=mode)
        self.layer2 = self._make_res(res_block, 32, num_res_blocks[1], stride=2, mode=mode)
        self.layer3 = self._make_res(res_block, 64, num_res_blocks[2], stride=2, mode=mode)

        self.bn = nn.BatchNorm2d(64 * res_block.ratio, momentum=self.bn_momentum)

        self.flatten = nn.Flatten()
        if mode["execution_type"] == 'float':
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
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = self.flatten(out)
        out = self.linear(out)
        return out


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