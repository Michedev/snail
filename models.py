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
    network = Sequential(residual_layers, Conv2d(256, 2048, kernel_size=1), AvgPool2d(5), ReLU(),
                         Flatten(1), Dropout(0.9), Linear(2048, 384))
    return network


def build_snail(in_filters, n, t):
    log2_t = int(ceil(log2(t)))
    model = Sequential()
    softmax = Softmax(dim=1)
    filters = in_filters + n
    model.add_module('attn1', AttentionBlock(filters, 64, 32))  # n x t x 96
    filters += 32
    model.add_module('tc1', TCBlock(t, filters, 128))  # n x t x 864
    filters += 128 * log2_t
    model.add_module('attn2', AttentionBlock(filters, 256, 128))  # n x t x 992
    filters += 128
    model.add_module('tc2', TCBlock(t, filters, 128))  # n x t x 1760
    filters += 128 * log2_t
    model.add_module('attn3', AttentionBlock(filters, 512, 256))  # n x t x 2016
    filters += 256
    model.add_module('conv1x1', Conv1d(filters, n, kernel_size=1))
    model.add_module('softmax', softmax)
    return model


def build_snail_omniglot(n, t):
    return build_snail(64, n, t)


def build_snail_miniimagenet(n, t):
    return build_snail(384, n, t)


class Snail(Module):

    def __init__(self, n: int, k: int, dataset: str):
        super(Snail, self).__init__()
        assert dataset in ['omniglot', 'miniimagenet']
        t = n * k + 1
        self.embedding_network = None
        self.snail = None
        if dataset == 'omniglot':
            self.embedding_network = build_embedding_network_omniglot()
            self.snail = build_snail_omniglot(n, t)
        else:
            self.snail = build_snail_miniimagenet(n, t)
            self.embedding_network = build_embedding_network_miniimagenet()
        self.dataset = dataset
        self.t = t

    def forward(self, X, y):
        batch_size = X.shape[0]
        y = y.permute(0, 2, 1)
        X = X.reshape(X.size(0) * X.size(1), X.size(4), X.size(2), X.size(3))
        X_embedding = self.embedding_network(X)
        X_embedding = X_embedding.reshape(batch_size, X_embedding.size(1), self.t)
        X_embedding = torch.cat([X_embedding, y], dim=1)
        yhat = self.snail(X_embedding)
        return yhat



__all__ = ['build_snail_miniimagenet', 'build_embedding_network_miniimagenet',
           'build_snail_omniglot', 'build_embedding_network_omniglot', 'build_snail', 'predict']
