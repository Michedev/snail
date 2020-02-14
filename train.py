from typing import List

import numpy as np
from path import Path
from tensorboardX import SummaryWriter

from dataset import sample_batch
from models import *
from fire import Fire
from random import seed as set_seed
import os

DATAFOLDER = Path(__file__).parent / 'data'
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
            X = X.reshape(X.size(0) * X.size(1), X.size(4), X.size(2), X.size(3), )
            X_embedding = self.embedding_network(X)
            X_embedding = X_embedding.reshape(batch_size, X_embedding.size(1), self.t)
            X_embedding = torch.cat([X_embedding, y], dim=1)
            yhat = self.model(X_embedding)
            yhat_last = yhat[:, :, -1]
            loss_value = self.loss(yhat_last, y_last)
            loss_value.backward()
            self.opt.step()
            if self.track_loss:
                self.logger.add_scalar(f'loss_{self.dataset}_last', loss_value, episode)
            if self.track_layers and episode % self.freq_track_layers == 0:
                for i, l in enumerate(self.model.parameters(recurse=True)):
                    self.logger.add_histogram(f'layer_{i}', l, global_step=episode)

    def save_weights(self, folder=''):
        torch.save(self.embedding_network.state_dict(), f'{folder}embedding_network_{self.dataset}.pth')
        torch.save(self.model.state_dict(), f'{folder}snail_{self.dataset}.pth')


def pull_data(force):
    if force or not OMNIGLOTFOLDER.exists():
        archives = ['images_background', 'images_evaluation']
        for archive_name in archives:
            os.system(f'wget https://github.com/brendenlake/omniglot/raw/master/python/{archive_name}.zip')
        os.system('rm -rf data/omniglot-py/*')
        os.system('mkdir -p data/omniglot-py')
        for archive in archives:
          os.system(f'unzip -q {archive}.zip -d data/omniglot-py/')
          os.system(f'rm {archive}.zip')
        os.system('mv data/omniglot-py/*/* data/omniglot-py')
        os.system('rm -rf data/omniglot-py/images_*/')

def main(dataset='omniglot', n=5, k=5, trainsize=1200, episodes=5_000, batch_size=32, seed=13,
         force_download=False, device='cuda', use_tensorboard=True, save_destination='./'):
    assert dataset in ['omniglot', 'miniimagenet']
    assert 'cuda' in device or device == 'cpu'
    if not torch.cuda.is_available():
        device = 'cpu'
    np.random.seed(seed)
    set_seed(seed)
    pull_data(force_download)
    classes = list(OMNIGLOTFOLDER.glob('*/*/'))
    print(len(classes))
    index_classes = np.arange(len(classes))
    np.random.shuffle(index_classes)
    index_train = index_classes[:trainsize]
    index_test = index_classes[trainsize:]
    train_classes = [classes[i_train] for i_train in index_train]
    test_classes = [classes[i_test] for i_test in index_test]

    train_classes = list(train_classes)
    test_classes = list(test_classes)
    with open('train_classes.txt', 'w') as f:
        f.write(', '.join(train_classes))
    with open('test_classes.txt', 'w') as f:
        f.write(', '.join(test_classes))
    model = Snail(n, k, dataset, device=device, track_loss=use_tensorboard, track_layers=use_tensorboard)
    model.train(episodes, batch_size, train_classes)
    model.save_weights(save_destination)


if __name__ == '__main__':
    Fire(main)
