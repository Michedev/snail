from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torchmeta.datasets.helpers import miniimagenet, omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision import transforms
from torchmeta.transforms.target_transforms import TargetTransform
from paths import DATAFOLDER

class OheTransform(TargetTransform):

    def __init__(self, n):
        self.n = n
        self.ohe = torch.eye(n)

    def __call__(self, target):
        return self.ohe[target]




def miniimagenet_transforms():
    return transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_meta_miniimagenet(n: int, supp_size: int, query_size: int, split: str):
    assert split in ['train', 'val', 'test']
    return miniimagenet(DATAFOLDER, supp_size, n, test_shots=query_size, meta_split=split,
                        download=True, target_transform=OheTransform(n))


def get_train_miniimagenet(n: int, supp_size: int, query_size: int):
    return get_meta_miniimagenet(n, supp_size, query_size, 'train')


def get_val_miniimagenet(n: int, supp_size: int, query_size: int):
    return get_meta_miniimagenet(n, supp_size, query_size, 'val')


def get_test_miniimagenet(n: int, supp_size: int, query_size: int):
    return get_meta_miniimagenet(n, supp_size, query_size, 'test')


def get_meta_omniglot(n: int, k_train: int, k_test: int, split: str):
    assert split in ['train', 'val', 'test']
    return omniglot(DATAFOLDER, k_train, n, meta_split=split)


def get_train_omniglot(n: int, k_train: int, k_test: int):
    return get_meta_omniglot(n, k_train, k_test, 'train')


def get_val_omniglot(n: int, k_train: int, k_test: int):
    return get_meta_omniglot(n, k_train, k_test, 'val')


def get_test_omniglot(n: int, k_train: int, k_test: int):
    return get_meta_omniglot(n, k_train, k_test, 'test')


class MetaLearningDataLoader(ABC):

    def __init__(self, batch_size: int, n: int, n_s: int, n_q: int,
                 train_len: int, val_len: int, test_len: int, device: str, num_workers: int = None):
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len
        self.n_s = n_s
        self.n_q = n_q
        self.n = n
        self.batch_size = batch_size
        self.pin_mem = 'cuda' in device
        if num_workers is None:
            self.cpus = 0
        else:
            self.cpus = num_workers
        self.prepare_data()

    def prepare_data(self) -> None:
        self.train_data = self._train_dataset()
        self.val_data = self._val_dataset()
        self.test_data = self._test_dataset()

    def train_dataloader(self) -> DataLoader:
        return BatchMetaDataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.cpus,
                                   pin_memory=self.pin_mem)

    def val_dataloader(self):
        return BatchMetaDataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                                   num_workers=self.cpus, pin_memory=self.pin_mem)

    def test_dataloader(self) -> DataLoader:
        return BatchMetaDataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                   num_workers=self.cpus, pin_memory=self.pin_mem)

    @abstractmethod
    def _train_dataset(self):
        pass

    @abstractmethod
    def _val_dataset(self):
        pass

    @abstractmethod
    def _test_dataset(self):
        pass


class OmniglotDataLoader(MetaLearningDataLoader):

    def _train_dataset(self):
        return get_train_omniglot(self.n, self.n_s, self.n_q)

    def _val_dataset(self):
        return get_val_omniglot(self.n, self.n_s, self.n_q)

    def _test_dataset(self):
        return get_test_omniglot(self.n, self.n_s, self.n_q)


class MiniImagenetDataLoader(MetaLearningDataLoader):

    def _train_dataset(self):
        return get_train_miniimagenet(self.n, self.n_s, self.n_q)

    def _val_dataset(self):
        return get_val_miniimagenet(self.n, self.n_s, self.n_q)

    def _test_dataset(self):
        return get_test_miniimagenet(self.n, self.n_s, self.n_q)
