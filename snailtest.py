from operator import itemgetter

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from torch.nn import CrossEntropyLoss


class Snailtest:

    def __init__(self, model, device, n: int, loss_f=None):
        self.model = model.to(device)
        self.device = device
        self.n = n
        self.loss_f = loss_f if loss_f is None else CrossEntropyLoss(reduction='mean')
        self.ohe_matrix = torch.eye(n).to(device)

    def test_step(self, batch, also_accuracy=True, grad=True, only_test_last=False):
        X_train, y_train = batch['train']
        X_test, y_test = batch['test']
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device) if not only_test_last else X_test[:, :1].to(self.device)
        y_test = y_test.to(self.device) if not only_test_last else y_test[:, :1].to(self.device)
        y_train_ohe = self.ohe_matrix[y_train]
        with torch.set_grad_enabled(grad):
            p_yhat_last = self.model(X_train, y_train_ohe, X_test) # bs x n
            loss_value = self.loss_f(p_yhat_last, y_test)
        if not also_accuracy:
            return loss_value
        yhat_last = p_yhat_last.argmax(dim=1)
        accuracy = (yhat_last == y_test).float().mean()
        return loss_value, accuracy

    def test(self, test_dataloader, testsize=None, dataset_name='Test', pbar=False):
        engine = Engine(lambda e, b: self.test_step(b, also_accuracy=True, grad=False, only_test_last=True))

        @engine.on(Events.EPOCH_STARTED)
        def init_lists(e):
            e.state.losses = []
            e.state.accuracies = []

        @engine.on(Events.ITERATION_COMPLETED)
        def accumulate_vars(e):
            loss, acc = e.state.output
            e.state.losses.append(loss)
            e.state.accuracies.append(acc)

        @engine.on(Events.EPOCH_COMPLETED)
        def get_results(e):
            e.state.losses = torch.FloatTensor(e.state.losses)
            e.state.accuracies = torch.FloatTensor(e.state.accuracies)
            mean_loss, std_loss = e.state.losses.mean().item(), e.state.losses.std().item()
            mean_acc, std_acc = e.state.accuracies.mean().item(), e.state.accuracies.std().item()
            print(f'{dataset_name} loss = ', mean_loss, '+-', std_loss)
            print(f'{dataset_name} acc', mean_acc, '+-', std_acc)
            return [mean_loss, std_loss, mean_acc, std_acc]

        if pbar:
            RunningAverage(output_transform=itemgetter(0)).attach(engine, 'loss')
            RunningAverage(output_transform=itemgetter(1)).attach(engine, 'accuracy')

            ProgressBar().attach(engine, ['loss', 'accuracy'])

        self.model.eval()
        engine.run(test_dataloader, 1, testsize)
        return engine.state.output




