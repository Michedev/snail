from path import Path
from models import *
from dataset import sample_batch
from fire import Fire
import torch


def main(dataset='omniglot',
         n=5, k=5, batch_size=32,
         model_path='model_weights/snail_omniglot.pth',
         embedding_path='model_weights/embedding_network_omniglot.pth',
         test_classes_path='test_classes.txt',
         device='cpu', n_sample=500):
    assert dataset in ['omniglot', 'miniimagenet']
    t = n * k + 1
    if dataset == 'omniglot':
        embedding_network = build_embedding_network_omniglot().to(device)
        snail = build_snail_omniglot(n, k).to(device)
    else:
        embedding_network = build_embedding_network_miniimagenet().to(device)
        snail = build_snail_miniimagenet(n, k).to(device)

    with open(test_classes_path) as f:
        test_classes = f.read()
    test_classes = list(map(Path, test_classes.split(', ')))
    loss = torch.nn.CrossEntropyLoss()
    loss_values = torch.zeros(n_sample)
    acc_values = torch.zeros(n_sample)
    for i in range(n_sample):
        X, y, y_last = sample_batch(batch_size, test_classes, t, n, k)
        yhat = predict(embedding_network, snail, X, y)
        p_yhat_last: torch.Tensor = yhat[:, :, -1]
        loss_value = loss(p_yhat_last, y_last)
        yhat_last = p_yhat_last.argmax(1)
        acc = (yhat_last == y_last).mean()
        loss_values[i] = loss_value
        acc_values[i] = acc

if __name__ == '__main__':
    Fire(main)