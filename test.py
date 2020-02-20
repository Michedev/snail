from path import Path
from models import *
from dataset import sample_batch, pull_data_omniglot
from fire import Fire
import torch
import gc


def main(dataset='omniglot',
         n=5, k=5, batch_size=32,
         model_path='model_weights/snail_omniglot_5_5.pth',
         embedding_path='model_weights/embedding_network_omniglot_5_5.pth',
         test_classes_path='test_classes.txt',
         device='cpu', n_sample=500):
    assert dataset in ['omniglot', 'miniimagenet']
    pull_data_omniglot(force=False)
    t = n * k + 1
    if dataset == 'omniglot':
        embedding_network = build_embedding_network_omniglot().to(device)
        snail = build_snail_omniglot(n, t).to(device)
    else:
        embedding_network = build_embedding_network_miniimagenet().to(device)
        snail = build_snail_miniimagenet(n, t).to(device)
    # embedding_network.load_state_dict(torch.load(embedding_path))
    #todo fix import of snail weights
    # snail.load_state_dict(torch.load(model_path))
    embedding_network = embedding_network.eval()
    snail = snail.eval()
    embedding_network.requires_grad_(False)
    snail.requires_grad_(False)

    with open(test_classes_path) as f:
        test_classes = f.read()
    test_classes = list(map(Path, test_classes.split(', ')))
    loss = torch.nn.CrossEntropyLoss()
    loss_values = torch.zeros(n_sample)
    acc_values = torch.zeros(n_sample)
    ohe_matrix = torch.eye(n)
    for i in range(n_sample):
        X, y, y_last = sample_batch(batch_size, test_classes, t, n, k, ohe_matrix)
        yhat = predict(embedding_network, snail, X, y)
        p_yhat_last: torch.Tensor = yhat[:, :, -1]
        loss_value = loss(p_yhat_last, y_last)
        yhat_last = p_yhat_last.argmax(1)
        acc = (yhat_last == y_last).float().mean()
        loss_values[i] = loss_value
        acc_values[i] = acc
        gc.collect()
    print('\n', '='*100, '\n', sep='', end='')
    print('loss values:', loss_values)
    print('avg loss:', loss_values.mean())
    print('\n', '='*100, '\n', sep='', end='')
    print('acc values:', acc_values)
    print('avg acc:', acc_values.mean())
    print('\n', '='*100, '\n', sep='', end='')


if __name__ == '__main__':
    main()