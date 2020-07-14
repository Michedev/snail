from path import Path
from models import *
from dataset import sample_batch, pull_data_omniglot
from fire import Fire
import torch
import gc
from tqdm import tqdm
from ignite.engine import Engine


def main(dataset='omniglot',
         n=5, k=5, batch_size=32,
         model_path='model_weights/snail_omniglot_5_5.pth',
         device='cpu', n_sample=500):
    assert dataset in ['omniglot', 'miniimagenet']
    pull_data_omniglot(force=False)
    t = n * k + 1
    snail = Snail(n, k, dataset)
    snail.load_state_dict(torch.load(snail.path))
    snail.requires_grad_(False)
    with open(test_classes_path) as f:
        test_classes = f.read()
    test_classes = list(map(Path, test_classes.split(', ')))
    loss = torch.nn.CrossEntropyLoss()
    loss_values = torch.zeros(n_sample)
    acc_values = torch.zeros(n_sample)
    ohe_matrix = torch.eye(n)
    loss_f = torch.nn.CrossEntropyLoss(reduction='mean')

    def predict_batch(X, y, y_last):
        p_yhat = snail(X,y)
        p_yhat_last = p_yhat[:, :, -1]
        loss_value = loss_f(p_yhat_last, y_last)
        return loss_value
    
    predictor = Engine(lambda e, b: predict_batch(*b))

    print('\n', '='*80, '\n', sep='', end='')
    print('avg loss:', loss_values.mean().item())
    print('std loss:', loss_values.std(unbiased=True).item())
    print('\n', '='*80, '\n', sep='', end='')
    print('avg acc:', acc_values.mean().item())
    print('std acc:', acc_values.std(unbiased=True).item())
    print('\n', '='*80, '\n', sep='', end='')


if __name__ == '__main__':
    Fire(main)