from typing import List

import numpy as np
from path import Path
from tensorboardX import SummaryWriter

from dataset import sample_batch, get_train_test_classes, pull_data_omniglot
from models import *
from fire import Fire
from random import seed as set_seed

ROOT = Path(__file__).parent
DATAFOLDER = ROOT / 'data'
OMNIGLOTFOLDER = DATAFOLDER / 'omniglot-py'


class Snail:

    def __init__(self, n: int, k: int, dataset: str, track_loss=True, track_layers=True, freq_track_layers=100,
                 device='cuda'):
        self.t = n * k + 1
        self.n = n
        self.k = k
        self.dataset = dataset
        self.is_omniglot = dataset == 'omniglot'
        self.is_miniimagenet = dataset == 'miniimagenet'
        self.ohe_matrix = torch.eye(n)
        if self.is_omniglot:
            self.model = build_snail_omniglot(n, self.t).to(device)
            self.embedding_network = build_embedding_network_omniglot().to(device)
        elif self.is_miniimagenet:
            self.model = build_snail_miniimagenet(n, self.t).to(device)
            self.embedding_network = build_embedding_network_miniimagenet().to(device)
        self.opt = torch.optim.Adam(self.model.parameters())
        self.loss = CrossEntropyLoss()
        self.track_layers = track_layers
        self.track_loss = track_loss
        self.logger = SummaryWriter('log_' + dataset) if self.track_layers or self.track_loss else None
        self.freq_track_layers = freq_track_layers
        self.device = device

    def train(self, episodes: int, batch_size: int, train_classes: List):
        for episode in range(episodes):
            X, y, y_last = sample_batch(batch_size, train_classes, self.t, self.n, self.k, self.ohe_matrix)
            y = y.permute(0, 2, 1)
            self.opt.zero_grad()
            X = X.reshape(X.size(0) * X.size(1), X.size(4), X.size(2), X.size(3))
            X = X.to(self.device)
            y = y.to(self.device)
            y_last = y_last.to(self.device)
            X_embedding = self.embedding_network(X)
            X_embedding = X_embedding.reshape(batch_size, X_embedding.size(1), self.t)
            X_embedding = torch.cat([X_embedding, y], dim=1)
            yhat = self.model(X_embedding)
            yhat_last = yhat[:, :, -1]
            loss_value = self.loss(yhat_last, y_last)
            loss_value.backward()
            self.opt.step()
            loss_value = float(loss_value)
            if self.track_loss:
                self.logger.add_scalar(f'loss_{self.dataset}_last', loss_value, global_step=episode)
            if self.track_layers and episode % self.freq_track_layers == 0:
                for i, l in enumerate(self.model.parameters(recurse=True)):
                    self.logger.add_histogram(f'layer_{i}', l, global_step=episode)
            print('loss episode:', loss_value)

    def save_weights(self, folder=''):
        torch.save(self.embedding_network.state_dict(), f'{folder}embedding_network_{self.dataset}.pth')
        torch.save(self.model.state_dict(), f'{folder}snail_{self.dataset}.pth')


def main(dataset='omniglot', n=5, k=5, trainsize=1200, episodes=5_000, batch_size=32, seed=13,
         force_download=False, device='cuda', use_tensorboard=True, save_destination='./'):
    """
    Download the dataset if not present and train SNAIL (Simple Neural Attentive Meta-Learner).
    When training is successfully finished, the embedding network weights and snail weights are saved, as well
    the path of classes used for training/test in train_classes.txt/test_classes.txt
    :param dataset: Dataset used for training,  can be only {'omniglot', 'miniimagenet'} (defuult 'omniglot')
    :param n: the N in N-way in meta-learning i.e. number of class sampled in each row of the dataset (default 5)
    :param k: the K in K-shot in meta-learning i.e. number of observations for each class (default 5)
    :param trainsize: number of class used in training (default 1200)
    :param episodes: time of model updates (default 5000)
    :param batch_size: size of a training batch (default 32)
    :param seed: seed for reproducibility (default 13)
    :param force_download: :bool redownload data even if folder is present (default True)
    :param device: : device used in pytorch for training, can be "cuda*" or "cpu" (default 'cuda')
    :param use_tensorboard: :bool save metrics in tensorboard (default True)
    :param save_destination: :string location of model weights (default './')
    """
    assert dataset in ['omniglot', 'miniimagenet']
    assert 'cuda' in device or device == 'cpu'
    if not torch.cuda.is_available():
        device = 'cpu'
    np.random.seed(seed)
    set_seed(seed)
    pull_data_omniglot(force_download)
    classes = list(OMNIGLOTFOLDER.glob('*/*/'))
    print(len(classes))
    train_classes_file = ROOT / 'train_classes.txt'
    test_classes_file = ROOT / 'test_classes'
    train_classes, test_classes = get_train_test_classes(classes, test_classes_file, train_classes_file, trainsize)


    model = Snail(n, k, dataset, device=device, track_loss=use_tensorboard, track_layers=use_tensorboard)
    model.train(episodes, batch_size, train_classes)
    model.save_weights(save_destination)
    with open('train_classes.txt', 'w') as f:
        f.write(', '.join(train_classes))
    with open('test_classes.txt', 'w') as f:
        f.write(', '.join(test_classes))


if __name__ == '__main__':
    Fire(main)
