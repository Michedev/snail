import torch
from path import Path
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, BatchNorm2d, \
    LeakyReLU, Softmax, ModuleDict, CrossEntropyLoss, Module, ConvTranspose2d, Conv2d, MSELoss
from models import build_embedding_network_miniimagenet
from torch.utils.data import DataLoader
from collections import OrderedDict
from ignite.metrics import Accuracy, Loss
from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from itertools import chain
from paths import MINIIMAGENETFOLDER, WEIGHTSFOLDER, TRAIN_MINIIMAGENET, \
    PRETRAIN, PRETRAINED_EMBEDDING_CLASSIFIER_PATH, PRETRAINED_EMBEDDING_PATH, PRETRAINED_EMBEDDING_AE_PATH
from PIL import Image
from ignite.metrics import RunningAverage
from torchvision import transforms
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from fire import Fire


class UnsupervisedMiniImagenet(torch.utils.data.Dataset):

    def __init__(self, noise=True):
        super().__init__()
        self.noise = noise
        self.files = list(TRAIN_MINIIMAGENET.walkfiles('*.jpg'))
        self.preprocess_image = transforms.Compose([
            transforms.Resize([84, 84]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path_img = self.files[index]
        img = Image.open(path_img)
        img = self.preprocess_image(img)
        noise = torch.empty_like(img)
        noise.normal_(0.0, 0.02)
        img_noise = img + noise
        return img_noise, img


class SupervisedMiniImagenet(torch.utils.data.Dataset):

    def __init__(self, classes: list):
        super().__init__()
        self.n = len(classes)
        self.dict_classes = {cl: i for i, cl in enumerate(classes)}
        self.files = [[cl_file, cl] for cl in classes for cl_file in cl.files()]
        self.preprocess_image = transforms.Compose([
            transforms.Resize([84, 84]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path_img, y = self.files[index]
        y = self.dict_classes[y]
        img = Image.open(path_img)
        img = self.preprocess_image(img)
        return img, y


class MiniImageNetAE(Module):

    def __init__(self):
        super().__init__()
        self.embedding_nn = build_embedding_network_miniimagenet()
        self.decoder1 = Sequential(Linear(384, 32), ReLU())
        self.decoder2 = Sequential(Conv2d(32, 16, 3, padding=1), BatchNorm2d(16), ReLU(),
                                   Conv2d(16, 16, 3, padding=1), BatchNorm2d(16), ReLU(),
                                   Conv2d(16, 3, 3, padding=1))

    def forward(self, x):
        xhat = self.decoder1(self.embedding_nn(x)).unsqueeze(-1).unsqueeze(-1)
        repeat_array = ([1] * (len(xhat.shape) - 1)) + [84, 84]
        xhat = xhat.repeat(repeat_array).squeeze(0)
        return self.decoder2(xhat)


def train_model(model, classes, device, epochs, batch_size):
    opt = torch.optim.Adam(model.parameters())
    loss = MSELoss()
    trainer = create_supervised_trainer(model, opt, loss, device=torch.device(device))
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.1, patience=5)
    dataset = UnsupervisedMiniImagenet()
    train_len = int(len(dataset) * 0.8)
    test_len = len(dataset) - train_len
    train_data, test_data = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    p = ProgressBar()
    p.attach(trainer, ['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_test(engine):
        model.eval()
        print('Epoch', engine.state.epoch)
        metrics = dict(mse=Loss(MSELoss()))
        evaluator = create_supervised_evaluator(model, metrics, device=torch.device(device))
        evaluator.run(train_loader, max_epochs=1)
        print('Train mse', evaluator.state.metrics['mse'])
        evaluator = create_supervised_evaluator(model, metrics, device=torch.device(device))
        evaluator.run(test_loader, max_epochs=1)
        print('Test mse', evaluator.state.metrics['mse'])
        model.train()

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_embedding(engine):
        torch.save(model.embedding_nn.state_dict(), PRETRAINED_EMBEDDING_PATH)
        torch.save(model.state_dict(), PRETRAINED_EMBEDDING_AE_PATH)

    trainer.run(train_loader, max_epochs=epochs)


def main(epochs, batch_size, device='cuda'):
    train_classes = TRAIN_MINIIMAGENET.dirs()
    model = MiniImageNetAE()
    model = model.to(device)
    if PRETRAINED_EMBEDDING_AE_PATH.exists():
        model.load_state_dict(torch.load(PRETRAINED_EMBEDDING_AE_PATH, map_location=torch.device(device)))
        print('loaded embedding from', PRETRAINED_EMBEDDING_AE_PATH)
    train_model(model, train_classes, device, epochs, batch_size)


if __name__ == "__main__":
    Fire(main)
