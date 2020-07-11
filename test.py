from path import Path
from models import Snail
from dataset import OmniglotMetaLearning, MiniImageNetMetaLearning
from torch.utils.data import DataLoader
from fire import Fire
import torch
from sklearn.metrics import accuracy_score
import gc
from ignite.engine import Engine, Events
from tqdm import tqdm
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from paths import MINIIMAGENETFOLDER


def main(dataset='omniglot',
         n=5, k=5, batch_size=32,
         device='cpu', n_sample=500,
         calc_accuracy=True):
    assert dataset in ['omniglot', 'miniimagenet']
    use_cuda = 'cuda' in device
    if dataset == 'omniglot':
        with open('test_classes.txt') as f:
            test_classes = f.read().split(', ')
        data = OmniglotMetaLearning(test_classes, n, k,
                 random_rotation=False, length=n_sample)
    else:
        test_classes = (MINIIMAGENETFOLDER / 'test').dirs()
        data = MiniImageNetMetaLearning(test_classes, n, k, False, length=n_sample)
    data_loader = DataLoader(data, batch_size=batch_size, pin_memory=use_cuda, drop_last=True)
    snail = Snail(n, k, dataset)
    snail.load_state_dict(torch.load(snail.path, map_location=torch.device(device)))
    snail = snail.to(device)
    snail = snail.eval()
    snail.requires_grad_(False)
    loss = torch.nn.CrossEntropyLoss()
    loss_values = torch.zeros(n_sample // batch_size, dtype=torch.float32)
    if calc_accuracy:
        acc_values = torch.zeros(n_sample // batch_size, dtype=torch.float32)

    def predict_step(X, y, y_last):
        X = X.to(device)
        y = y.to(device)
        y_last = y_last.to(device)
        with torch.no_grad():
            yhat = snail(X, y)
        yhat_last = yhat[:, :, -1]
        return yhat_last.cpu(), y_last.cpu()

    predictor = Engine(lambda e, b: predict_step(*b))

    @predictor.on(Events.ITERATION_COMPLETED)
    def eval_loss(engine):
        y_pred, y_true = engine.state.output
        loss_value = loss(y_pred, y_true)
        loss_values[engine.state.iteration-1] = loss_value

    if calc_accuracy:
            @predictor.on(Events.ITERATION_COMPLETED)
            def eval_accuracy(engine):
                y_pred, y_true = engine.state.output
                y_pred = y_pred.argmax(dim=1)
                accuracy = y_pred == y_true
                accuracy = accuracy.float().mean()
                acc_values[engine.state.iteration-1] = accuracy

    p = ProgressBar()
    p.attach(predictor)

    predictor.run(data_loader, max_epochs=1)
    
    print('\n', '='*80, '\n', sep='', end='')
    print('loss:', loss_values.mean().item(),
             '+-', loss_values.std(unbiased=True).item())
    if calc_accuracy:
        print('\n', '='*80, '\n', sep='', end='')
        print('avg acc:', acc_values.mean().item(),
                '+-', acc_values.std(unbiased=True).item())
        print('\n', '='*80, '\n', sep='', end='')
    print('accuracy', acc_values)

if __name__ == '__main__':
    Fire(main)
