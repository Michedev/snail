from datetime import datetime
from path import Path
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from models import Snail
from paths import WEIGHTSFOLDER


class ModelSaver:

    def __init__(self, model, savepath: Path, mode='min'):
        assert mode in ['min', 'max']
        assert savepath.endswith('.pth')
        self.model = model
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.savepath = savepath

    def step(self, curr_value):
        if curr_value > self.best_value:
            self.best_value = curr_value
            torch.save(self.model.state_dict(), self.savepath)


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
        self.model = self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = CrossEntropyLoss(reduction='mean')
        self.ohe_matrix = torch.eye(n, device=device)
        self.track_layers = track_layers
        self.track_loss = track_loss
        best_model_path = WEIGHTSFOLDER / (self.model.fname.replace('snail', 'snail_best_test'))
        self.saver = ModelSaver(self.model, best_model_path, mode='max')
        self.logger = SummaryWriter('tb/log_' + dataset + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) \
                        if self.track_layers or self.track_loss else None
        self.freq_track_layers = freq_track_layers
        self.random_rotation = random_rotation
        self.track_loss_freq = track_loss_freq
        self.trainpbar = trainpbar
        self.track_params_freq = track_params_freq

    def train(self, epochs: int, train_loader, test_loader=None, trainsize=None, valsize=None):
        self.model.train()
        train_engine = Engine(lambda e, b: self.train_step(b))

        @train_engine.on(Events.EPOCH_COMPLETED(every=self.track_loss_freq))
        def eval_test(engine):
            if self.track_loss:
                self.tb_log(train_loader, self.logger, engine.state.epoch, is_train=True, eval_length=valsize)
                if test_loader is not None:
                    self.tb_log(test_loader, self.logger, engine.state.epoch, is_train=False, eval_length=valsize)

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
        train_engine.run(train_loader, max_epochs=epochs, epoch_length=trainsize)

    def tb_log(self, dataloader, logger, epoch, is_train, eval_length=None):
        eval_engine = Engine(lambda engine, batch: self.test_step(batch, also_accuracy=True, grad=False))
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
            if not is_train:
                self.saver.step(mean_acc)
            logger.add_scalar(f'epoch_loss/{label}', mean_loss, epoch)
            logger.add_scalar(f'epoch_acc/{label}', mean_acc, epoch)
            print(label, 'epoch loss', mean_loss.item())
            print(label, 'epoch accuracy', mean_acc.item())

        print('-' * 100)
        print('Epoch', epoch)
        self.model.eval()
        eval_engine.run(dataloader, 1, eval_length)
        self.model.train()

    def test_step(self, batch, also_accuracy=True, grad=True):
        X_train, y_train = batch['train']
        X_test, y_test = batch['test']
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        y_train_ohe = self.ohe_matrix[y_train]
        with torch.set_grad_enabled(grad):
            p_yhat_last = self.model(X_train, y_train_ohe, X_test) # bs x n
            loss_value = self.loss(p_yhat_last, y_test.squeeze(1))
        if not also_accuracy:
            return loss_value
        yhat_last = p_yhat_last.argmax(dim=1)
        accuracy = (yhat_last == y_test).float().mean()
        return loss_value, accuracy

    def train_step(self, batch, return_accuracy=False):
        self.opt.zero_grad()
        loss_value = self.test_step(batch, return_accuracy, grad=True)
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
