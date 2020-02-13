from typing import List

import numpy as np
import torchvision
from path import Path
from tensorboardX import SummaryWriter

from dataset import sample_batch
from models import *
from fire import Fire
from random import seed as set_seed



class Snail:

    def __init__(self, n: int, k: int, dataset: str, track_loss=True, track_layers=True):
        self.t = n * k + 1
        self.n = n
        self.k = k
        self.dataset = dataset
        self.is_omniglot = dataset == 'omniglot'
        self.is_miniimagenet = dataset == 'miniimagenet'
        self.ohe_matrix = torch.eye(n)
        if self.is_omniglot:
            self.model = build_snail_omniglot(n, self.t)
            self.embedding_network = build_embedding_network_omniglot()
        elif self.is_miniimagenet:
            self.model = build_snail_miniimagenet(n, self.t)
            self.embedding_network = build_embedding_network_miniimagenet()
        self.opt = torch.optim.Adam(self.model.parameters())
        self.loss = CrossEntropyLoss()
        self.logger = SummaryWriter('log_' + dataset) if self.track_layers or self.track_loss else None
        self.track_layers = track_layers
        self.track_loss = track_loss



    def train(self, episodes: int, batch_size: int, train_classes: List):
        for episode in range(episodes):
            X, y, y_last = sample_batch(batch_size, train_classes, self.t, self.n, self.k, self.ohe_matrix)
            self.opt.zero_grad()
            X_embedding = self.embedding_network(X)
            X_embedding = torch.cat([X_embedding, y], dim=-1)
            yhat = self.model(X_embedding)
            yhat_last = yhat[:, -1, :]
            loss_value = self.loss(yhat_last, y_last)
            loss_value.backward()
            self.opt.step()
            if self.track_loss:
                self.logger.add_scalar(f'loss_{self.dataset}_last', loss_value, episode)

def main(dataset='omniglot', n=5, k=5, trainsize=1200, episodes=5_000, batch_size=32, seed=13):
    assert dataset in ['omniglot', 'miniimagenet']
    np.random.seed(seed)
    set_seed(seed)
    DATAFOLDER = Path(__file__).parent / 'data'
    data = torchvision.datasets.Omniglot(DATAFOLDER, download=True)
    OMNIGLOTFOLDER = DATAFOLDER / 'omniglot-py'
    classes = list(OMNIGLOTFOLDER.listdir())
    index_classes = np.arange(len(classes))
    np.random.shuffle(index_classes)
    index_train = index_classes[:trainsize]
    index_test = index_classes[trainsize:]
    train_classes = classes[index_train]
    test_classes = classes[index_test]
    # todo add generation of new classes by rotation
    model = Snail(n, k, dataset)
    model.train(episodes, batch_size, train_classes)


if __name__ == '__main__':
    Fire(main)