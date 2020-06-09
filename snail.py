from itertools import chain
from multiprocessing import cpu_count

import torch
from ignite.engine import Engine, Events
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import OmniglotMetaLearning, MiniImageNetMetaLearning
from models import build_embedding_network_omniglot, build_snail_omniglot, build_snail_miniimagenet, \
    build_embedding_network_miniimagenet
from paths import WEIGHTSFOLDER


class Snail:

    def __init__(self, n: int, k: int, dataset: str, track_loss=True, track_layers=True, freq_track_layers=100,
                 device='cuda', track_loss_freq=3, track_params_freq=1000, random_rotation=True):
        assert dataset in ['omniglot', 'miniimagenet']
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
        self.track_params_freq = track_params_freq

    def train(self, epochs: int, batch_size: int, train_classes, test_classes=None):
        self.embedding_network.train()
        self.model.train()
        train_data = OmniglotMetaLearning(train_classes, self.n, self.k, self.random_rotation) if self.is_omniglot else \
            MiniImageNetMetaLearning(train_classes, self.n, self.k, self.random_rotation)
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=cpu_count(),
                                  drop_last=True)
        if test_classes:
            test_data = OmniglotMetaLearning(test_classes, self.n, self.k, self.random_rotation) \
                        if self.is_omniglot else \
                        MiniImageNetMetaLearning(test_classes, self.n, self.k, self.random_rotation)
            test_data.shuffle()
            test_loader = DataLoader(test_data, shuffle=True, num_workers=cpu_count(),
                                     batch_size=2, drop_last=True)
        train_engine = Engine(lambda engine, batch: self.opt_step(*batch, return_accuracy=False))

        @train_engine.on(Events.EPOCH_COMPLETED(every=self.track_loss_freq))
        def eval_test(engine):
            if self.track_loss:
                self.tb_log(train_loader, self.logger, engine.state.epoch, is_train=True)
                if test_classes:
                    self.tb_log(test_loader, self.logger, engine.state.epoch, is_train=False)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def save_weights(engine):
            torch.save(self.model.state_dict(), self.snail_path)
            torch.save(self.embedding_network.state_dict(), self.embedding_network_path)

        @train_engine.on(Events.ITERATION_COMPLETED(every=self.track_params_freq))
        def tb_log_histogram_params(engine):
            if self.track_layers:
                for name, params in self.model.named_parameters():
                    self.logger.add_histogram(name.replace('.', '/'), params, engine.state.iteration)
                    if params.grad is not None:
                        self.logger.add_histogram(name.replace('.', '/') + '/grad', params.grad, engine.state.iteration)

        train_engine.run(train_loader, max_epochs=epochs)

    def tb_log(self, dataloader, logger, epoch, is_train):
        eval_engine = Engine(lambda engine, batch: self.calc_loss(*batch, also_accuracy=True, grad=False))
        label = 'train' if is_train else 'test'

        @eval_engine.on(Events.EPOCH_STARTED)
        def init_stats(engine):
            engine.state.sum_loss = 0.0
            engine.state.sum_acc = 0.0
            engine.state.steps = 0

        @eval_engine.on(Events.ITERATION_COMPLETED)
        def update_stats(engine):
            loss, acc = engine.state.output
            engine.state.sum_loss += loss
            engine.state.sum_acc += acc
            engine.state.steps += 1

        @eval_engine.on(Events.EPOCH_COMPLETED)
        def log_stats(engine):
            mean_loss = engine.state.sum_loss / engine.state.steps
            mean_acc = engine.state.sum_acc / engine.state.steps
            logger.add_scalar(f'epoch_loss/{label}', mean_loss, epoch)
            logger.add_scalar(f'epoch_acc/{label}', mean_acc, epoch)
            print(label, 'epoch loss', mean_loss.item())
            print(label, 'epoch accuracy', mean_acc.item())

        eval_engine.run(dataloader, 1)

    def calc_loss(self, X, y, y_last, also_accuracy=True, grad=True):
        X = X.to(self.device)
        y = y.to(self.device)
        y_last = y_last.to(self.device)
        for tensor in [X, y, y_last]:
            tensor.squeeze_(dim=0)
        with torch.set_grad_enabled(grad):
            yhat = self.predict(X, y)
            p_yhat_last = yhat[:, :, -1]
            loss_value = self.loss(p_yhat_last, y_last)
        if not also_accuracy:
            return loss_value
        yhat_last = p_yhat_last.argmax(dim=1)
        accuracy = (yhat_last == y_last).float().mean()
        return loss_value, accuracy

    def opt_step(self, X, y, y_last, return_accuracy=False):
        self.opt.zero_grad()
        loss_value = self.calc_loss(X, y, y_last, return_accuracy)
        if return_accuracy:
            loss_value, accuracy = loss_value
        loss_value.backward()
        self.opt.step()
        if return_accuracy:
            return accuracy

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