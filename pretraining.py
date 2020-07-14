import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, LeakyReLU, Softmax, ModuleDict, CrossEntropyLoss
from models import build_embedding_network_miniimagenet
from torch.utils.data import DataLoader
from collections import OrderedDict
from ignite.metrics import Accuracy, Loss
from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from itertools import chain
from paths import MINIIMAGENETFOLDER, WEIGHTSFOLDER
from PIL import Image
from ignite.metrics import RunningAverage
from torchvision import transforms
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from fire import Fire


TRAIN_MINIIMAGENET = MINIIMAGENETFOLDER / 'train'
EMBEDDING_PATH = WEIGHTSFOLDER / 'embedding_miniimagenet.pth'
EMBEDDING_CLASSIFIER_PATH = WEIGHTSFOLDER / 'embedding_classifier_miniimagenet.path'

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

        

def build_model_pretraining(num_classes):
    embedding_nn = build_embedding_network_miniimagenet()
    classifier = Sequential(Linear(384, 100), BatchNorm1d(100), ReLU(), Linear(100, num_classes), Softmax(1))
    model = Sequential()
    model.add_module('embedding_nn', embedding_nn)
    model.add_module('classifier', classifier)
    return model

def train_model(model, classes, device, epochs, batch_size):
    opt = torch.optim.Adam(model.parameters())
    loss = CrossEntropyLoss()
    trainer = create_supervised_trainer(model, opt, loss, device=torch.device(device))

    dataset = SupervisedMiniImagenet(classes)
    train_len = int(len(dataset)*0.8)
    test_len = len(dataset) - train_len
    train_data, test_data = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    p = ProgressBar()
    p.attach(trainer, ['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_test(engine):
        print('Epoch', engine.state.epoch)
        metrics = dict(accuracy=Accuracy(), crossentropy=Loss(CrossEntropyLoss()))
        evaluator = create_supervised_evaluator(model, metrics, device=torch.device(device))
        evaluator.run(train_loader, max_epochs=1)
        acc = evaluator.state.metrics['accuracy']
        print('Train accuracy', evaluator.state.metrics['accuracy'])
        print('Train crossentropy', evaluator.state.metrics['crossentropy'])
        evaluator = create_supervised_evaluator(model, metrics, device=torch.device(device))
        evaluator.run(test_loader, max_epochs=1)
        acc = evaluator.state.metrics['accuracy']
        print('Test accuracy', evaluator.state.metrics['accuracy'])
        print('Test crossentropy', evaluator.state.metrics['crossentropy'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_embedding(engine):
        torch.save(model[0].state_dict(), EMBEDDING_PATH)
        torch.save(model.state_dict(), EMBEDDING_CLASSIFIER_PATH)

    trainer.run(train_loader, max_epochs=epochs)



def main(epochs, batch_size, device='cuda'):
    train_classes = TRAIN_MINIIMAGENET.dirs()
    model = build_model_pretraining(len(train_classes))
    model = model.to(device)
    if EMBEDDING_CLASSIFIER_PATH.exists():
        model.load_state_dict(torch.load(EMBEDDING_CLASSIFIER_PATH, map_location=torch.device(device)))
        print('loaded embedding from', EMBEDDING_CLASSIFIER_PATH)
    train_model(model, train_classes, device, epochs, batch_size)

if __name__ == "__main__":
    Fire(main)
