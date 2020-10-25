from itertools import chain
from multiprocessing import cpu_count

import torch
from ignite.engine import Engine, Events
from torch.nn import CrossEntropyLoss, Sequential, Linear, ReLU, BatchNorm1d, Dropout, LeakyReLU
from ignite.metrics import RunningAverage
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from dataset import OmniglotMetaLearning, MiniImageNetMetaLearning
from models import Snail
from paths import WEIGHTSFOLDER, PRETRAINED_EMBEDDING_PATH


class SnailTrain:

    def __init__(self, n: int, k: int, dataset: str, track_loss=True, track_layers=True, freq_track_layers=100,
                 device='cuda', track_loss_freq=3, track_params_freq=1000, random_rotation=True, lr=10e-4, trainpbar=True):
        assert dataset in ['omniglot', 'miniimagenet']
        self.t = n * k + 1
        self.n = n
        self.k = k
        self.device = torch.device(device)
        self.dataset = dataset
        self.is_omniglot = dataset == 'omniglot'
        self.is_miniimagenet = dataset == 'miniimagenet'
        self.ohe_matrix = torch.eye(n)
        self.model = Snail(n, k, dataset)
        if self.is_miniimagenet:
            self.model.embedding_network.load_state_dict(torch.load(PRETRAINED_EMBEDDING_PATH, map_location=torch.device('cpu')))
            print('Load pretrained embedding MiniImagenet')
        self.model = self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max', factor=0.1, patience=5)
        self.loss = CrossEntropyLoss(reduction='mean')
        self.track_layers = track_layers
        self.track_loss = track_loss
        self.logger = SummaryWriter('tb/log_' + dataset + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) \
                        if self.track_layers or self.track_loss else None
        self.freq_track_layers = freq_track_layers
        self.random_rotation = random_rotation
        self.track_loss_freq = track_loss_freq
        self.trainpbar = trainpbar
        self.track_params_freq = track_params_freq

    def train(self, epochs: int, train_loader, test_loader=None, eval_length=None):
        self.model.train()
        train_engine = Engine(lambda engine, batch: self.opt_step(*batch, return_accuracy=False))

        @train_engine.on(Events.EPOCH_COMPLETED(every=self.track_loss_freq))
        def eval_test(engine):
            if self.track_loss:
                self.tb_log(train_loader, self.logger, engine.state.epoch, is_train=True, eval_length=eval_length)
                if test_loader:
                    self.tb_log(test_loader, self.logger, engine.state.epoch, is_train=False, eval_length=eval_length)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def save_state(engine):
            torch.save(self.model.state_dict(), self.snail_path)
            torch.save(self.opt.state_dict(), self.snail_opt_path)

        @train_engine.on(Events.ITERATION_COMPLETED(every=self.track_params_freq))
        def tb_log_histogram_params(engine):
            if self.track_layers:
                for name, params in self.model.named_parameters():
                    self.logger.add_histogram(name.replace('.', '/'), params, engine.state.iteration)
                    if params.grad is not None:
                        self.logger.add_histogram(name.replace('.', '/') + '/grad', params.grad, engine.state.iteration)
        if self.trainpbar:
            RunningAverage(output_transform=lambda x: x).attach(train_engine, 'loss')
            p = ProgressBar()
            p.attach(train_engine, ['loss'])
        train_engine.run(train_loader, max_epochs=epochs)

    def tb_log(self, dataloader, logger: SummaryWriter, epoch, is_train, eval_length=None):
        eval_engine = Engine(lambda engine, batch: self.calc_loss(*batch, also_accuracy=True, grad=False))
        label = 'train' if is_train else 'test'

        @eval_engine.on(Events.EPOCH_STARTED)
        def init_stats(engine):
            engine.state.losses = []
            engine.state.accs = []

        @eval_engine.on(Events.ITERATION_COMPLETED)
        def update_stats(engine):
            loss, acc = engine.state.output
            engine.state.losses.append(loss)
            engine.state.accs.append(acc)

        @eval_engine.on(Events.EPOCH_COMPLETED)
        def log_stats(engine):
            losses = torch.FloatTensor(engine.state.losses)
            accs = torch.FloatTensor(engine.state.accs)
            mean_loss = losses.mean().item()
            mean_acc = accs.mean().item()
            std_loss = losses.std().item()
            std_acc = accs.std().item()
            if not is_train:
                self.lr_plateau.step(mean_acc)
            logger.add_scalar(f'epoch_loss/mean_{label}', mean_loss, epoch)
            logger.add_scalar(f'epoch_acc/mean_{label}', mean_acc, epoch)
            logger.add_scalar(f'epoch_loss/std_{label}', std_loss, epoch)
            logger.add_scalar(f'epoch_acc/std_{label}', std_acc, epoch)
            print(label, 'epoch loss', mean_loss.item())
            print(label, 'epoch accuracy', mean_acc.item())

        print('-' * 100)
        print('Epoch', epoch)
        self.model.eval()
        eval_engine.run(dataloader, 1, eval_length)
        self.model.train()

    def calc_loss(self, X, y, y_last, also_accuracy=True, grad=True):
        X = X.to(self.device)
        y = y.to(self.device)
        y_last = y_last.to(self.device)
        for tensor in [X, y, y_last]:
            tensor.squeeze_(dim=0)
        with torch.set_grad_enabled(grad):
            yhat = self.model(X, y) # bs x n x t
            p_yhat_last = yhat[:, :, -1]
            loss_value = self.loss(p_yhat_last, y_last)
        if not also_accuracy:
            return loss_value
        yhat_last = p_yhat_last.argmax(dim=1)
        accuracy = (yhat_last == y_last).float().mean()
        return loss_value, accuracy

    def opt_step(self, X, y, y_last, return_accuracy=False):
        self.opt.zero_grad()
        loss_value = self.calc_loss(X, y, y_last, return_accuracy, grad=True)
        if return_accuracy:
            loss_value, accuracy = loss_value
        loss_value.backward()
        self.opt.step()
        if return_accuracy:
            return accuracy
        return loss_value


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
    def snail_opt_fname(self):
        return f'snail_opt_{self.dataset}_{self.n}_{self.k}.pth'

    @property
    def snail_opt_path(self):
        return WEIGHTSFOLDER / self.snail_opt_fname


    @property
    def snail_path(self):
        return WEIGHTSFOLDER / self.snail_fname

    def load_if_exists(self):
        if self.snail_path.exists():
            self.model.load_state_dict(torch.load(self.snail_path, map_location=self.device))
        if self.snail_opt_path.exists():
            self.opt.load_state_dict(torch.load(self.snail_opt_path, map_location=self.device))

    def save_weights(self):
        self.model.eval()
        torch.save(self.model.state_dict(),
                   WEIGHTSFOLDER / self.snail_fname)