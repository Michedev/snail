from typing import List
from itertools import chain
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
from torch.utils.data import DataLoader, RandomSampler

from dataset import OmniglotMetaLearning, get_train_test_classes, pull_data_omniglot, pull_data_miniimagenet
from models import build_embedding_network_miniimagenet, build_embedding_network_omniglot, build_snail_miniimagenet, \
    build_snail_omniglot
from fire import Fire
from random import seed as set_seed
import torch
from torch.nn import *


from paths import ROOT, OMNIGLOTFOLDER, WEIGHTSFOLDER, MINIIMAGENETFOLDER


class Snail:

    def __init__(self, n: int, k: int, dataset: str, track_loss=True, track_layers=True, freq_track_layers=100,
                 device='cuda', track_loss_freq=10, random_rotation=True):
        assert dataset in ['omniglot','miniimagenet']
        self.t = n * k + 1
        self.n = n
        self.k = k
        self.device = torch.device(device)
        self.dataset = dataset
        self.is_omniglot = dataset == 'omniglot'
        self.is_miniimagenet = dataset == 'miniimagenet'
        self.ohe_matrix = torch.eye(n)
        if self.is_omniglot:
            self.embedding_network = build_embedding_network_omniglot()
            self.model = build_snail_omniglot(n, self.t)
        elif self.is_miniimagenet:
            self.model = build_snail_miniimagenet(n, self.t)
            self.embedding_network = build_embedding_network_miniimagenet()
        self.embedding_network.to(self.device)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(chain(self.embedding_network.parameters(), self.model.parameters()), lr=0.0003)
        self.loss = CrossEntropyLoss()
        self.track_layers = track_layers
        self.track_loss = track_loss
        self.logger = SummaryWriter('log_' + dataset) if self.track_layers or self.track_loss else None
        self.freq_track_layers = freq_track_layers
        self.random_rotation = random_rotation
        self.track_loss_freq = track_loss_freq

    def train(self, epochs: int, batch_size: int, train_classes, test_classes=None):
        self.embedding_network.train()
        self.model.train()
        train_data = OmniglotMetaLearning(train_classes, self.n, self.k, self.random_rotation)
        data_loader = DataLoader(train_data, batch_size=batch_size,
                                 shuffle=True, num_workers=cpu_count(),
                                 drop_last=True)
        if test_classes:
            test_data = OmniglotMetaLearning(test_classes, self.n, self.k, self.random_rotation)
            test_loader = DataLoader(test_data, sampler=RandomSampler(test_data, replacement=True),
                                     batch_size=batch_size)
            test_data.shuffle()
        global_step = 0
        for epoch in range(epochs):
            print('epoch', epoch)
            if test_classes:
                test_iter = iter(test_loader)
            for X, y, y_last in data_loader:
                logging_step = self.track_loss and global_step % self.track_loss_freq == 0
                if logging_step:
                    loss_value, accuracy_train = self.calc_loss(X, y, y_last, logging_step)
                else:
                    loss_value = self.calc_loss(X, y, y_last, logging_step)
                loss_value.backward()
                self.opt.step()
                loss_value = float(loss_value)
                if logging_step:
                    losses_dict = dict(train=loss_value)
                    acc_dict = dict(train=accuracy_train)
                    self.logger.add_scalar(f'Train/loss_{self.dataset}_last', loss_value, global_step=global_step)
                    self.logger.add_scalar(f'Train/acc_{self.dataset}_last', accuracy_train, global_step=global_step)

                    if test_classes and logging_step:
                        with torch.set_grad_enabled(False):
                            X_test, y_test, y_last_test = next(test_iter)
                            test_loss, accuracy_test = self.calc_loss(X_test, y_test, y_last_test, also_accuracy=True)
                            losses_dict['test'] = test_loss
                            acc_dict['test'] = accuracy_test
                        self.logger.add_scalar(f'Test/loss_{self.dataset}_last', test_loss, global_step=global_step)
                        self.logger.add_scalar(f'Test/acc_{self.dataset}_last', accuracy_test,
                                               global_step=global_step)
                    self.logger.add_scalars(f'loss_{self.dataset}_last', losses_dict, global_step=global_step)
                    self.logger.add_scalars(f'accuracy_{self.dataset}_last', acc_dict, global_step=global_step)
                if self.track_layers and global_step % self.freq_track_layers == 0:
                    for i, l in enumerate(self.model.modules()):
                        if hasattr(l, 'parameters'):
                            for param_name, param in l.named_parameters():
                                self.logger.add_histogram(f'{type(l)}_{i}/{param_name}', param, global_step=global_step)
                if global_step % 100 == 0:
                    print(f'loss step {global_step}:', loss_value)
                    if logging_step:
                        print(f'accuracy step {global_step}:', accuracy_train.item())
                global_step += 1
            train_data.shuffle()
            if epoch % 100 == 0:
                self.save_weights()

    def calc_loss(self, X, y, y_last, also_accuracy=True):
        X = X.to(self.device)
        y = y.to(self.device)
        y_last = y_last.to(self.device)
        for tensor in [X, y, y_last]:
            tensor.squeeze_(dim=0)
        yhat = self.predict(X, y)
        p_yhat_last = yhat[:, :, -1]
        loss_value = self.loss(p_yhat_last, y_last)
        if not also_accuracy:
            return loss_value
        yhat_last = p_yhat_last.argmax(dim=1)
        accuracy = (yhat_last == y_last).float().mean()
        return loss_value, accuracy

    def predict(self, X, y):
        batch_size = X.shape[0]
        y = y.permute(0, 2, 1)
        self.opt.zero_grad()
        X = X.reshape(X.size(0) * X.size(1), X.size(4), X.size(2), X.size(3))
        X_embedding = self.embedding_network(X)
        X_embedding = X_embedding.reshape(batch_size, X_embedding.size(1), self.t)
        X_embedding = torch.cat([X_embedding, y], dim=1)
        yhat = self.model(X_embedding)
        return yhat

    @property
    def embedding_network_fname(self):
        return f'embedding_network_{self.dataset}_{self.n}_{self.k}.pth'

    @property
    def embedding_network_path(self):
        return WEIGHTSFOLDER / self.embedding_network_fname

    @property
    def snail_fname(self):
        return f'snail_{self.dataset}_{self.n}_{self.k}.pth'

    @property
    def snail_path(self):
        return WEIGHTSFOLDER / self.snail_fname

    def load_if_exists(self):
        if self.embedding_network_path.exists():
            self.embedding_network.load_state_dict(torch.load(self.embedding_network_path))
        if self.snail_path.exists():
            self.model.load_state_dict(torch.load(self.snail_path))

    def save_weights(self):
        self.embedding_network.eval()
        self.model.eval()
        torch.save(self.embedding_network.state_dict(),
                   WEIGHTSFOLDER / self.embedding_network_fname)
        torch.save(self.model.state_dict(),
                   WEIGHTSFOLDER / self.snail_fname)


def main(dataset='omniglot', n=5, k=5, trainsize=1200, epochs=200, batch_size=32, random_rotation=True,
         seed=13, force_download=False, device='cuda', use_tensorboard=True,
         eval_test=True, track_loss_freq=10, load_weights=True):
    """
    Download the dataset if not present and train SNAIL (Simple Neural Attentive Meta-Learner).
    When training is successfully finished, the embedding network weights and snail weights are saved, as well
    the path of classes used for training/test in train_classes.txt/test_classes.txt
    :param dataset: Dataset used for training,  can be only {'omniglot', 'miniimagenet'} (defuult 'omniglot')
    :param n: the N in N-way in meta-learning i.e. number of class sampled in each row of the dataset (default 5)
    :param k: the K in K-shot in meta-learning i.e. number of observations for each class (default 5)
    :param trainsize: number of class used in training (default 1200) (the remaining classes are for test)
    :param epochs: times that model see the dataset (default 200)
    :param batch_size: size of a training batch (default 32)
    :param random_rotation: :bool rotate the class images by multiples of 90 degrees (default True)
    :param seed: seed for reproducibility (default 13)
    :param force_download: :bool redownload data even if folder is present (default True)
    :param device: : device used in pytorch for training, can be "cuda*" or "cpu" (default 'cuda')
    :param use_tensorboard: :bool save metrics in tensorboard (default True)
    :param eval_test: :bool after test_loss_freq batch calculate loss and accuracy on test set (default True)
    :param track_loss_freq: :int step frequency of loss/accuracy save into tensorboard (default 10)
    :param load_weights: :bool if available load under model_weights snail and embedding network weights (default True)
    """
    assert dataset in ['omniglot', 'miniimagenet']
    assert device.startswith('cuda') or device == 'cpu'
    if not torch.cuda.is_available():
        print('Warning: cuda is not available, fall back to cpu')
        device = 'cpu'
    np.random.seed(seed)
    set_seed(seed)
    if dataset == 'omniglot':
        pull_data_omniglot(force_download)
        classes = list(OMNIGLOTFOLDER.glob('*/*/'))
    else:
        pull_data_miniimagenet(force_download)
        classes = list(MINIIMAGENETFOLDER.glob('*/*/'))
    print(len(classes))
    train_classes_file = ROOT / f'train_classes_{dataset}.txt'
    test_classes_file = ROOT / f'test_classes_{dataset}.txt'
    train_classes, test_classes = get_train_test_classes(classes, test_classes_file, train_classes_file, trainsize)

    model = Snail(n, k, dataset, device=device, track_loss=use_tensorboard,
                  track_layers=use_tensorboard,
                  track_loss_freq=track_loss_freq, random_rotation=random_rotation)
    if load_weights:
        model.load_if_exists()
    model.train(epochs, batch_size, train_classes, None if not eval_test else test_classes)
    model.save_weights()
    with open('train_classes.txt', 'w') as f:
        f.write(', '.join(train_classes))
    with open('test_classes.txt', 'w') as f:
        f.write(', '.join(test_classes))


if __name__ == '__main__':
    Fire(main)
