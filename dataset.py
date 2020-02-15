import zipfile

import numpy as np
import torch
from random import sample, randint

from os.path import Path

import wget
from skimage import io, transform

from train import OMNIGLOTFOLDER


def sample_batch(batch_size, train_classes, t, n, k, ohe_matrix=None):
    X = torch.zeros(batch_size, t, 28, 28, 1)
    y = torch.zeros(batch_size, t, n)
    y_last_class = torch.zeros(batch_size, dtype=torch.int64)
    if ohe_matrix is None:
      ohe_matrix = torch.eye(n)
    batch_classes = [sample(train_classes, n) for _ in range(batch_size)]
    for i_batch in range(batch_size):
        image_names_batch = []
        for i_class in range(n):
            name_images = sample(batch_classes[i_batch][i_class].files(), k)
            image_names_batch += name_images
            y[i_batch, i_class * k: (i_class + 1) * k] = ohe_matrix[[i_class] * k]
            for i_img, name_image in enumerate(name_images):
                img = load_and_transform(name_image)
                X[i_batch, i_class * k + i_img, :, :, :] = torch.from_numpy(img).unsqueeze(-1)
                del img
        i_last_class = randint(0, n-1)
        last_class = batch_classes[i_batch][i_last_class]
        last_class_images = last_class.files()
        last_img = None
        while not last_img or last_img in image_names_batch:
            last_img = sample(last_class_images, 1)[0]
        last_img = load_and_transform(last_img)
        X[i_batch, -1] = torch.from_numpy(last_img).unsqueeze(dim=-1)
        y_last_class[i_batch] = i_last_class
    return X, y, y_last_class


def load_and_transform(name_image):
    img = io.imread(name_image, as_gray=True)
    img = transform.resize(img, (28, 28))
    return img


def get_train_test_classes(classes, test_classes_file, train_classes_file, trainsize):
    if not train_classes_file.exists() or test_classes_file.exists():
        index_classes = np.arange(len(classes))
        np.random.shuffle(index_classes)
        index_train = index_classes[:trainsize]
        index_test = index_classes[trainsize:]
        train_classes = [classes[i_train] for i_train in index_train]
        test_classes = [classes[i_test] for i_test in index_test]
    else:

        with open(train_classes_file) as f:
            train_classes = f.read()
        train_classes = train_classes.split(', ')
        train_classes = list(map(Path, train_classes))

        with open(test_classes_file) as f:
            test_classes = f.read()
        test_classes = test_classes.split(', ')
        test_classes = list(map(Path, test_classes))
    return train_classes, test_classes


def pull_data_omniglot(force):
    if force or not OMNIGLOTFOLDER.exists():
        archives = ['images_background', 'images_evaluation']
        for archive_name in archives:
            wget.download(f'https://github.com/brendenlake/omniglot/raw/master/python/{archive_name}.zip')
        if OMNIGLOTFOLDER.exists():
            for el in OMNIGLOTFOLDER.files(): el.remove()
        OMNIGLOTFOLDER.makedirs_p()
        for archive in archives:
            with zipfile.ZipFile(f'{archive}.zip') as z:
                z.extractall(OMNIGLOTFOLDER)
            Path(f'{archive}.zip').remove()
        for folder in OMNIGLOTFOLDER.glob('/*/*'):
            folder.move(OMNIGLOTFOLDER)
        for folder in OMNIGLOTFOLDER.dirs('images_*'):
            folder.removedirs()