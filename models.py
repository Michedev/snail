from modules import *

def build_embedding_network_omniglot():
    conv_layers = Sequential()
    for i in range(4):
        block = ConvBlockOmniglot(1 if i == 0 else 64, 64)
        conv_layers.add_module(f'conv_block_{i}', block)
    network = Sequential(conv_layers, Flatten(), Linear(64, 64))
    return network

def build_embedding_network_miniimagenet():
    residual_layers = Sequential(
        ResidualBlockImageNet(3, 64),
        ResidualBlockImageNet(64, 96),
        ResidualBlockImageNet(96, 128),
        ResidualBlockImageNet(128, 256)
    )
    network = Sequential(residual_layers, Conv2d(256, 2048, kernel_size=1), AvgPool2d(6), ReLU(), Dropout2d(0.9), Conv1d(2048, 384, kernel_size=1), Flatten())
    return network

def build_snail(in_filters, n, t):
    log2_t = int(ceil(log2(t)))
    model = Sequential()
    filters = in_filters
    model.add_module('attn1', AttentionBlock(filters + n, 64, 32))  # n x t x 96
    filters += 32
    model.add_module('tc1', TCBlock(t, filters, 128))  # n x t x 864
    filters += 128 * log2_t
    model.add_module('attn2', AttentionBlock(filters, 256, 128))  # n x t x 992
    filters += 128
    model.add_module('tc2', TCBlock(t, filters, 128)) # n x t x 1760
    filters += 128 * log2_t
    model.add_module('attn3', AttentionBlock(filters, 512, 256)) # n x t x 2016
    filters += 256
    model.add_module('conv1x1', Conv1d(filters, n, kernel_size=1))
    return model

def build_snail_omniglot(n, t):
    return build_snail(64, n, t)

def build_snail_miniimagenet(n, t):
    return build_snail(384, n, t)
