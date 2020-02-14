from torch.nn import *
import torch
from math import log2, ceil, sqrt


class DenseBlock(Module):

    def __init__(self, dilation, in_filters: int, out_filters: int):
        super().__init__()
        self.dilatation = dilation
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.causal_conv1 = Conv1d(in_filters, out_filters, kernel_size=2, dilation=dilation)
        self.causal_conv2 = Conv1d(in_filters, out_filters, kernel_size=2, dilation=dilation)

    def forward(self, input):
        xf, xg = self.causal_conv1(input), self.causal_conv2(input)
        activations = functional.tanh(xf) * functional.sigmoid(xg)
        return torch.cat([input, activations], dim=1)


class TCBlock(Module):

    def __init__(self, seq_len: int, in_filters: int, filters: int):
        super().__init__()
        self.filters = filters
        self.seq_len = seq_len
        self.log_seq_len = int(ceil(log2(seq_len)))
        self.blocks = [DenseBlock(2 ** i, (in_filters if i == 0 else filters), filters) for i in
                       range(self.log_seq_len)]

    def forward(self, input):
        output = input
        for block in self.blocks:
            output = block(output)
        return output


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

    def forward(self, input):
        keys = self.key_layer(input)
        query = self.query_layer(input)
        logits = query.matmul(keys.transpose(1, 2))
        mask = torch.ones(logits.shape[-1], logits.shape[-1], dtype=torch.bool)
        for i in range(logits.shape[-1]):
            mask[i, i:] = False
        logits[:, mask] = - torch.inf
        probs = functional.softmax(logits / self.sqrt_key_size, dim=1)
        values = self.value_layer(input)
        read = probs.matmul(values)
        return torch.cat([input, read], dim=1)


class ResidualBlockImageNet(Module):

    def __init__(self, in_filters: int, out_filters: int, num_convs: int = 3):
        super(ResidualBlockImageNet, self).__init__()
        self.layers = Sequential()
        for i in range(num_convs):
            block = Sequential(
                Conv2d(in_filters if i == 0 else out_filters, out_filters, kernel_size=3, padding='same'),
                BatchNorm2d(out_filters),
                LeakyReLU(0.1))
            self.layers.add_module(f'residual_block_{i}', block)
        self.layers = Sequential(*self.layers)
        self.conv1x1 = Conv2d(in_filters, out_filters, kernel_size=1)
        self.dropout = Dropout2d(0.9)
        self.maxpool = MaxPool2d(2)

    def forward(self, input):
        output = input
        for l in self.layers:
            output = l(output)
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
    def forward(self, input):
        return input.view(input.size(0), -1)
