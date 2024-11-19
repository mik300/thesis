import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.adapt.approx_layers import axx_layers
from neural_networks.custom_layers import Conv2d, Linear, Act
import pickle
import numpy

biasflag = True
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, mode=None):
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

        # Fully connected layers
        if mode["execution_type"] == 'float':
            self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        elif mode["execution_type"] == 'quant':
            self.fc1 = Linear(256 * 6 * 6, 4096, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'])
        elif mode["execution_type"] == 'adapt':
            self.fc1 = axx_layers.AdaPT_Linear(256 * 6 * 6, 4096, bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act6 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        if mode["execution_type"] == 'float':
            self.fc2 = nn.Linear(4096, 4096)
        elif mode["execution_type"] == 'quant':
            self.fc2 = Linear(4096, 4096, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'])
        elif mode["execution_type"] == 'adapt':
            self.fc2 = axx_layers.AdaPT_Linear(4096, 4096, bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        self.act7 = Act(act_bit=mode['act_bit'], fake_quant=mode['fake_quant'])

        if mode["execution_type"] == 'float':
            self.fc3 = nn.Linear(4096, num_classes)
        elif mode["execution_type"] == 'quant':
            self.fc3 = Linear(4096, num_classes, act_bit=mode['act_bit'], weight_bit=mode['weight_bit'], bias_bit=mode['bias_bit'])
        elif mode["execution_type"] == 'adapt':
            self.fc3 = axx_layers.AdaPT_Linear(4096, num_classes, bias=biasflag, axx_mult="bw_mult_9_9_0")
        else:
            exit("unknown layer command")
        

        # Softmax activation at the end
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.act2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.conv5(out)
        out = self.act5(out)
        out = self.pool3(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.act6(out)
        out = self.fc2(out)
        out = self.act7(out)
        out = self.fc3(out)
        #out = self.softmax(out)

        return out

def alexnet(mode=None):
    return AlexNet(mode=mode)