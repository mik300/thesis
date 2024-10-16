import torch
from torch.autograd import Function

class param_quantizer(torch.nn.Module):
    def __init__(self, min_val, max_val, fake_quant=True, dtype=torch.int32):
        super(param_quantizer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        if fake_quant:
            self.quantize = quantize_tensor.apply
        else:
            self.quantize = int_quantize_tensor.apply
        self.scaling_factor = 1.0
        self.dtype = dtype
    def forward(self, x):
        self.scaling_factor = compute_scaling_factor(x, self.max_val)
        #print(f'param scaling_factor = {self.scaling_factor}')
        return self.quantize(x,self.min_val,self.max_val, self.scaling_factor, self.dtype)


class act_quantizer(torch.nn.Module):
    def __init__(self, min_val, max_val, scaling_factor=None, fake_quant=True, dtype=torch.int32):
        super(act_quantizer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        if fake_quant:
            self.quantize = quantize_tensor.apply
        else:
            self.quantize = int_quantize_tensor.apply
        self.scaling_factor = scaling_factor
        self.dtype = dtype
        self.calibrated = False
        self.runtime_sc = 1.0
        
    def forward(self, x):
        if self.scaling_factor is None and not self.calibrated:
            scaling_factor = compute_scaling_factor(x, self.max_val)
        else:
            scaling_factor = self.scaling_factor
        #print(f'act scaling_factor = {self.scaling_factor}')
        return self.quantize(x,self.min_val,self.max_val, scaling_factor, self.dtype)


class quantize_tensor(Function):
    @staticmethod
    def forward(ctx, real_tensor, min_v, max_v, scaling_factor,dtype):
        ctx.save_for_backward(real_tensor)
        quantized_tensor = torch.clamp(torch.round(scaling_factor * real_tensor), min=min_v, max=max_v) / scaling_factor
        return quantized_tensor

    @staticmethod
    def backward(ctx, grad_out_tensor):  # straight-through estimator
        return grad_out_tensor, None, None, None, None


class int_quantize_tensor(Function):
    @staticmethod
    def forward(ctx, real_tensor, min_v, max_v, scaling_factor, dtype):
        ctx.save_for_backward(real_tensor)
        return torch.clamp(torch.round(scaling_factor * real_tensor), min=min_v, max=max_v).to(dtype)

    @staticmethod
    def backward(ctx, grad_out_tensor):  # straight-through estimator
        return grad_out_tensor, None, None, None, None


class scale_out_funct(Function):
    @staticmethod
    def forward(ctx, x, s1, s2):
        ctx.save_for_backward(x)
        return x.to(dtype=torch.float32) / (s1 * s2)

    @staticmethod
    def backward(ctx, grad_out_tensor):  # straight-through estimator
        return grad_out_tensor, None, None


class fake_scale_out_funct(Function):
    @staticmethod
    def forward(ctx, x, s1, s2):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_out_tensor):  # straight-through estimator
        return grad_out_tensor, None, None


def compute_scaling_factor(real_tensor, max_v):
    t_max = torch.max(torch.abs(torch.min(real_tensor)), torch.abs(torch.max(real_tensor))).item()
    # print(f't_max = {t_max}')
    if t_max == 0.0:   # avoid division by zero, should never happen in practice
        t_max = 1.0
    return max_v / t_max  # best usage of quantized range


def np2(value, bitwidth):
    power = 0
    while (1 / (2 ** (power + 1))) > value and (power + 1) < bitwidth:
        power += 1
    lower_value = 1 / (2 ** power)
    upper_value = 1 / (2 ** (power + 1))
    # Choose the closest value
    if abs(value - lower_value) <= abs(value - upper_value):
        closest_value = lower_value
    else:
        closest_value = upper_value
    return closest_value


class QuantCalibrator:
    def __init__(self, th=0.01, act_bit=32):
        """
        Calibration class for quantization, it calibrates the quantization parameters for activations iterating over the dataset. Three iterations are required.
        The calibrator is controlled by using the status variable, which should be set manually. The status could be any of the following: ["itminmax", "binedges", "ithists", "computeth", "itxaboveth", "compthvalue"]. Before the status is changed, it is necessary to execute the previous steps.
        @param th:
        @param act_bit:
        """
        super(QuantCalibrator, self).__init__()
        self.min_value = 1e9
        self.max_value = -1e9
        self.hist = None
        self.bin_edges = None
        self.num_bins = 2048
        self.calibrated_value = 1
        self.percentile_threshold = None
        self.values_above_threshold = []
        self.status = None  # status of the calibration process  ["minmax", "binedges", "hists", "computeth", "cumxaboveth", "compthvalue"]
        self.th_value = th
        self.act_bit = act_bit

    def cumulative_min_max(self, x):  # iterate over batches
        self.min_value = min(self.min_value, x.min().item())
        self.max_value = max(self.max_value, x.max().item())

    def compute_bin_edges(self):
        self.bin_edges = torch.linspace(self.min_value, self.max_value, self.num_bins+1)

    def cumulative_hist(self, x):  # iterate over batches
        if self.hist is None:
            self.hist = torch.histc(x.view(-1), bins= self.num_bins, min=self.min_value, max=self.max_value)
        else:
            self.hist += torch.histc(x.view(-1), bins= self.num_bins, min=self.min_value, max=self.max_value)

    def compute_threshold(self):
        cumulative_hist = torch.cumsum(self.hist, dim=0)
        percentile_threshold_index = torch.searchsorted(cumulative_hist, (1 - self.th_value) * cumulative_hist[-1].item())
        self.percentile_threshold = self.bin_edges[percentile_threshold_index]
        # print(f"percentile threshold is {self.percentile_threshold.item()}")

    def cumulative_x_above_th(self, x):  # iterate over batches
        values = x.view(-1)
        self.values_above_threshold.append(values[values > self.percentile_threshold])
        # print(f"values above threshold {len(self.values_above_threshold[-1])} out of {len(values)}")

    def compute_th_value(self):
        sorted_values = torch.sort(torch.cat(self.values_above_threshold)).values
        self.calibrated_value = sorted_values[-int(self.th_value * len(sorted_values))]
        self.hist = None
        self.bin_edges = None
        self.percentile_threshold = None
        self.values_above_threshold = []

    def calibrate_funct(self, x=None):
        # ["itminmax", "binedges", "ithists", "computeth", "itxaboveth", "compthvalue"]
        if self.status == "itminmax":
            self.cumulative_min_max(x)
        elif self.status == "binedges":
            self.compute_bin_edges()
        elif self.status == "ithists":
            self.cumulative_hist(x)
        elif self.status == "computeth":
            self.compute_threshold()
        elif self.status == "itxaboveth":
            self.cumulative_x_above_th(x)
        elif self.status == "compthvalue":
            self.compute_th_value()
        else:
            exit("unknown status")
