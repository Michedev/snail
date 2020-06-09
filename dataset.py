import zipfile
from typing import List

import numpy as np
import torch
from random import sample, randint, shuffle

from path import Path
import wget
from skimage import io, transform
from torchvision.datasets.utils import download_file_from_google_drive

import os
from paths import OMNIGLOTFOLDER, MINIIMAGENETFOLDER


def sample_batch(batch_size, train_classes, t, n, k, random_rotation=True, ohe_matrix=None):
    X = torch.zeros(batch_size, t, 28, 28, 1)
    y = torch.zeros(batch_size, t, n)
    y_last_class = torch.zeros(batch_size, dtype=torch.int64)
    if ohe_matrix is None:
        ohe_matrix = torch.eye(n)
    batch_classes = [sample(train_classes, n) for _ in range(batch_size)]
    for i_batch in range(batch_size):
        image_names_batch = []
        rotations = {}
        for i_class in range(n):
            name_images = sample(batch_classes[i_batch][i_class].files(), k)
            image_names_batch += name_images
            y[i_batch, i_class * k: (i_class + 1) * k] = ohe_matrix[[i_class] * k]
            rotation = 0 if not random_rotation else 90 * randint(0, 3)
            rotations[i_class] = rotation
            for i_img, name_image in enumerate(name_images):
                img = load_and_transform(name_image, rotation)
                X[i_batch, i_class * k + i_img, :, :, :] = torch.from_numpy(img).unsqueeze(-1)
                del img
        i_last_class = randint(0, n - 1)
        last_class = batch_classes[i_batch][i_last_class]
        last_class_images = last_class.files()
        last_img = None
        rotation_last = rotations[i_last_class]
        while not last_img or last_img in image_names_batch:
            last_img = sample(last_class_images, 1)[0]
        last_img = load_and_transform(last_img, rotation_last, X.shape[1:-1])
        X[i_batch, -1] = torch.from_numpy(last_img).unsqueeze(dim=-1)
        y_last_class[i_batch] = i_last_class
    return X, y, y_last_class


def fit_task(classes: List[Path], k: int, random_rotation=True, ohe_matrix=None):
    n = len(classes)
    t = n * k + 1
    X = torch.zeros(t, 28, 28, 1)
    y = torch.zeros(t, n)
    image_names_batch, rotations = fit_train_task(X, y, classes, k, n, ohe_matrix, random_rotation)
    i_last_class = fit_last_image(X, classes, image_names_batch, n, rotations)
    return X, y, i_last_class


def fit_last_image(X, classes, image_names_batch, n, rotations):
    i_last_class = randint(0, n - 1)
    last_class = classes[i_last_class]
    last_class_images = last_class.files()
    last_img = None
    rotation_last = rotations[i_last_class]
    while not last_img or last_img in image_names_batch:
        last_img = sample(last_class_images, 1)[0]
    last_img = load_and_transform(last_img, rotation_last, X.shape[1:-1])
    X[-1] = torch.from_numpy(last_img).unsqueeze(dim=-1)
    return i_last_class


def fit_train_task(X, y, classes, k, n, ohe_matrix, random_rotation):
    image_names_batch = []
    rotations = {}
    if ohe_matrix is None:
        ohe_matrix = torch.eye(n)
    for i_class, class_name in enumerate(classes):
        name_images = sample(class_name.files(), k)
        image_names_batch += name_images
        y[i_class * k: (i_class + 1) * k, :] = ohe_matrix[[i_class] * k]
        rotation = 0 if not random_rotation else 90 * randint(0, 3)
        rotations[i_class] = rotation
        for i_img, name_image in enumerate(name_images):
            img = load_and_transform(name_image, rotation, X.shape[1:-1])
            X[i_class * k + i_img, :, :, :] = torch.from_numpy(img).unsqueeze(-1)
            del img
    return image_names_batch, rotations


class MetaLearningDataset(torch.utils.data.Dataset):

    def __init__(self, class_pool, n, k, random_rotation, image_size):
        self.class_pool = class_pool
        self.n = n
        self.k = k
        self.t = n * k + 1
        self.ohe = torch.eye(n)
        self.image_size = list(image_size)
        self.random_rotation = random_rotation

    def __len__(self):
        return len(self.class_pool) // self.n - 1

    def __getitem__(self, i):
        classes = self.class_pool[i * self.n: (i + 1) * self.n]
        n = len(classes)
        t = n * self.k + 1
        X = torch.zeros([t] + self.image_size)
        y = torch.zeros(t, n)
        image_names_batch, rotations = fit_train_task(X, y, classes, self.k, n, self.ohe, random_rotation=True)
        i_last_class = fit_last_image(X, classes, image_names_batch, n, rotations)
        return X, y, i_last_class

    def shuffle(self):
        shuffle(self.class_pool)


class OmniglotMetaLearning(MetaLearningDataset):

    def __init__(self, class_pool, n, k, random_rotation):
        super(OmniglotMetaLearning, self).__init__(class_pool, n, k, random_rotation, image_size=[28, 28, 1])


class MiniImageNetMetaLearning(MetaLearningDataset):

    def __init__(self, class_pool, n, k, random_rotation):
        super(MiniImageNetMetaLearning, self).__init__(class_pool, n, k, random_rotation, image_size=[84, 84, 3])


def load_and_transform(name_image, rotation, image_size):
    img = io.imread(name_image, as_gray=True)
    img = transform.resize(img, image_size)
    img = transform.rotate(img, rotation)
    return img


def get_train_test_classes(classes, test_classes_file, train_classes_file, trainsize):
    if not train_classes_file.exists() or not test_classes_file.exists():
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
        for idiom_folder in OMNIGLOTFOLDER.dirs():
            for char_folder in idiom_folder.dirs():
                char_folder.move(OMNIGLOTFOLDER)
        for folder in OMNIGLOTFOLDER.dirs('images_*'):
            folder.removedirs()


def pull_data_miniimagenet(force):
    test_id = '1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v'
    train_id = '107FTosYIeBn5QbynR46YG91nHcJ70whs'
    if not MINIIMAGENETFOLDER.exists():
        MINIIMAGENETFOLDER.makedirs()
    for zipfname, url in [('train.tar', train_id), ('test.tar', test_id)]:
        zippath = MINIIMAGENETFOLDER / zipfname
        dstfolder = MINIIMAGENETFOLDER / zipfname.split('.')[0]
        if not dstfolder.exists() or force:
            download_file_from_google_drive(url, MINIIMAGENETFOLDER, zipfname)
            if dstfolder.exists() and force:
                dstfolder.removedirs()
            os.system(f'tar -xvf {MINIIMAGENETFOLDER}')
            zippath.remove()
