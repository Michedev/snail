import numpy as np

from dataset import get_train_test_classes, pull_data_omniglot, pull_data_miniimagenet
from fire import Fire
from random import seed as set_seed
import torch

from paths import ROOT, OMNIGLOTFOLDER, MINIIMAGENETFOLDER
from snailtrain import SnailTrain


def main(dataset='omniglot', n=5, k=5, trainsize=None, testsize=None, epochs=200, batch_size=32, random_rotation=True,
         seed=13, force_download=False, device='cuda', use_tensorboard=True,
         eval_test=True, track_loss_freq=1, track_weights=True, track_weights_freq=100, load_weights=True):
    """
    Download the dataset if not present and train SNAIL (Simple Neural Attentive Meta-Learner).
    When training is successfully finished, the embedding network weights and snail weights are saved, as well
    the path of classes used for training/test in train_classes.txt/test_classes.txt
    :param dataset: Dataset used for training,  can be only {'omniglot', 'miniimagenet'} (defuult 'omniglot')
    :param n: the N in N-way in meta-learning i.e. number of class sampled in each row of the dataset (default 5)
    :param k: the K in K-shot in meta-learning i.e. number of observations for each class (default 5)
    :param trainsize: [omniglot-only] number of class used in training (default 1200) while the remaining classes are for test.
    :param epochs: times that model see the dataset (default 200)
    :param batch_size: size of a training batch (default 32)
    :param random_rotation: :bool rotate the class images by multiples of 90 degrees (default True)
    :param seed: seed for reproducibility (default 13)
    :param force_download: :bool redownload data even if folder is present (default True)
    :param device: : device used in pytorch for training, can be "cuda*" or "cpu" (default 'cuda')
    :param use_tensorboard: :bool save metrics in tensorboard (default True)
    :param eval_test: :bool after test_loss_freq batch calculate loss and accuracy on test set (default True)
    :param track_loss_freq: :int epoch frequency of loss/accuracy saving inside tensorboard (default 1)
    :param track_weights: :bool when True log parameters histogram inside tensorboard (default True)
    :param track_weights_freq: :int steps frequency of saving parameters and gradients histograms inside tensorboard (default 100)
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
        train_classes_file = ROOT / f'train_classes_{dataset}.txt'
        test_classes_file = ROOT / f'test_classes_{dataset}.txt'
        train_classes, test_classes = get_train_test_classes(classes, test_classes_file, train_classes_file, trainsize)
    else:
        pull_data_miniimagenet(force_download)
        train_classes = (MINIIMAGENETFOLDER / 'train').dirs()
        test_classes = (MINIIMAGENETFOLDER / 'test').dirs()
    print('train classes', len(train_classes))
    print('test classes', len(test_classes))
    model = SnailTrain(n, k, dataset, device=device, track_loss=use_tensorboard,
                       track_layers=track_weights and use_tensorboard, track_loss_freq=track_loss_freq,
                       track_params_freq=track_weights_freq, random_rotation=random_rotation)
    if load_weights:
        model.load_if_exists()
    model.train(epochs, batch_size, train_classes, None if not eval_test else test_classes, trainsize, testsize)
    model.save_weights()
    with open('train_classes.txt', 'w') as f:
        f.write(', '.join(train_classes))
    with open('test_classes.txt', 'w') as f:
        f.write(', '.join(test_classes))


if __name__ == '__main__':
    Fire(main)
