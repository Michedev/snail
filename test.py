from path import Path
from models import Snail
from dataset import sample_batch, pull_data_omniglot
from fire import Fire
import torch
from sklearn.metrics import accuracy_score
import gc
from ignite.engine import Engine, Events
from tqdm import tqdm


def main(dataset='omniglot',
         n=5, k=5, batch_size=32,
         device='cpu', n_sample=500,
         calc_accuracy=True):
    assert dataset in ['omniglot', 'miniimagenet']
    snail = Snail(n, k, dataset)
    snail.load_state_dict(torch.load(snail.path))
    embedding_network = embedding_network.eval()
    snail = snail.eval()
    snail.requires_grad_(False)
    loss = torch.nn.CrossEntropyLoss()
    loss_values = torch.zeros(n_sample)
    if calc_accuracy:
        acc_values = torch.zeros(n_sample)

    def predict_step(X, y, y_last):
        yhat = snail(X, y)
        yhat_last = yhat[:, :, -1]
        return yhat_last, y_last

    predictor = Engine(lambda e, b: predict_step(*b))

    @predictor.on(Events.ITERATION_COMPLETED)
    def eval_loss(engine):
        y_pred, y_true = engine.state.output
        loss_value = loss(y_pred, y_true)
        loss_values[engine.state.iteration] = loss_value

    if calc_accuracy:
            @predictor.on(Events.ITERATION_COMPLETED)
            def eval_accuracy(engine):
                y_pred, y_true = engine.state.output
                y_pred = y_pred.argmax(dim=1)
                accuracy = accuracy_score(y_true, y_pred)
                acc_values[engine.state.iteration] = accuracy



    
    print('\n', '='*80, '\n', sep='', end='')
    print('loss:', loss_values.mean().item(),
             '+-', loss_values.std(unbiased=True).item())
    print('\n', '='*80, '\n', sep='', end='')
    if calc_accuracy:
        print('avg acc:', acc_values.mean().item(),
                '+-', acc_values.std(unbiased=True).item())
        print('\n', '='*80, '\n', sep='', end='')


if __name__ == '__main__':
    Fire(main)