from torch.nn import *
from torch.nn import ReflectionPad1d, ConstantPad1d
import torch
from math import log2, ceil, sqrt, floor


class DenseBlock(Module):

    def __init__(self, dilation, in_filters: int, out_filters: int):
        super().__init__()
        self.dilatation = dilation
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.causal_conv1 = Sequential(
            ConstantPad1d((dilation, 0), 0),
            Conv1d(in_filters, out_filters, kernel_size=2,
                                   dilation=dilation)
        )
        self.causal_conv2 = Sequential(
            ConstantPad1d((dilation, 0), 0),
            Conv1d(in_filters, out_filters, kernel_size=2, dilation=dilation)
        )        
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        xf, xg = self.causal_conv1(input), self.causal_conv2(input)
        activations = self.tanh(xf) * self.sigmoid(xg)
        return torch.cat([input, activations], dim=1)


def TCBlock(seq_len: int, in_filters: int, filters: int):
    log_seq_len = int(ceil(log2(seq_len)))
    model = Sequential()
    for i in range(log_seq_len):
        block = DenseBlock(2 ** i, (in_filters + i * filters), filters)
        model.add_module(f'dense_{i + 1}', block)
    return model


class AttentionBlock(Module):

    def __init__(self, input_size, key_size, value_size):
        super().__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.log_key_size = log2(key_size)
        self.sqrt_key_size = sqrt(key_size)
        self.key_layer = Linear(input_size, key_size)
        self.query_layer = Linear(input_size, key_size)
        self.value_layer = Linear(input_size, value_size)
        self.softmax = Softmax(dim=1)

    def forward(self, input):
        input = input.permute(0, 2, 1)  # bs x t x channels
        keys = self.key_layer(input)  # bs x t x ks
        query = self.query_layer(input)  # bs x t x ks
        logits = query.bmm(keys.permute(0, 2, 1))  # bs x t x t
        mask = torch.ones(logits.shape[-1], logits.shape[-1], dtype=torch.bool) 
        for i in range(logits.shape[-1]):
            mask[i, i:] = False
        logits[:, mask] = - float('inf')
        probs = self.softmax(logits / self.sqrt_key_size)  # bs x t x t
        values = self.value_layer(input)  # bs x t x vs
        read = probs.matmul(values)  # bs x t vs
        output = torch.cat([input, read], dim=-1)  # bs x t x (channels + vs)
        output = output.permute(0, 2, 1)  # bs x (channels + vs) x t
        return output


class ResidualBlockImageNet(Module):

    def __init__(self, in_filters: int, out_filters: int, num_convs: int = 3):
        super(ResidualBlockImageNet, self).__init__()
        self.main_road = Sequential()
        for i in range(num_convs):
            in_conv = in_filters if i == 0 else out_filters
            block = Sequential(
                Conv2d(in_conv, out_filters, kernel_size=3, padding=1),
                BatchNorm2d(out_filters, affine=False),
                LeakyReLU(0.1))
            self.main_road.add_module(f'residual_block_{i}', block)
        self.main_road = Sequential(*self.main_road)
        self.conv1x1 = Conv2d(in_filters, out_filters, kernel_size=1)
        self.maxpool = MaxPool2d(2)
        self.dropout = Dropout2d(0.9)

    def forward(self, input):
        output = input
        output = self.main_road(output)
        output2 = self.conv1x1(input)
        output += output2
        output = self.maxpool(output)
        output = self.dropout(output)
        return output


def ConvBlockOmniglot(in_filters: int, out_filters: int):
    return Sequential(
        Conv2d(in_filters, out_filters, kernel_size=3, padding=1),
        BatchNorm2d(out_filters),
        ReLU(),
        MaxPool2d(2)
    )


class Flatten(Module):

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return torch.flatten(input, self.start_dim, self.end_dim)
