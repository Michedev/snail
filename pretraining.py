import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, LeakyReLU, Softmax, ModuleDict, CrossEntropyLoss
from models import build_embedding_network_miniimagenet
from torch.utils.data import DataLoader
from collections import OrderedDict
from ignite.metrics import Accuracy
from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from itertools import chain
from paths import MINIIMAGENETFOLDER, WEIGHTSFOLDER
from PIL import Image
from ignite.metrics import RunningAverage
from torchvision import transforms
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from fire import Fire


TRAIN_MINIIMAGENET = MINIIMAGENETFOLDER / 'train'


class SupervisedMiniImagenet(torch.utils.data.Dataset):
    
    def __init__(self, classes: list):
        super().__init__()
        self.n = len(classes)
        self.dict_classes = {cl: i for i, cl in enumerate(classes)}
        self.files = [[cl_file, cl] for cl_file in cl.files() for cl in classes]
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
    d = OrderedDict()
    d['embedding_nn'] = build_embedding_network_miniimagenet()
    d['classifier'] = Sequential(Linear(384, 100), BatchNorm1d(100), ReLU(), Linear(100, num_classes), Softmax(1))
    return ModuleDict(d)

def train_model(model, classes, device, epochs, batch_size):
    opt = torch.optim.Adam(model.paramters())
    loss = CrossEntropyLoss()
    trainer = create_supervised_trainer(model, opt, loss, device=torch.device(device))

    dataset = SupervisedMiniImagenet(classes)
    train_len = int(len(dataset)*0.8)
    test_len = len(dataset) - train_len
    train_data, test_data = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    Accuracy().attach(trainer, 'accuracy')
    p = ProgressBar()
    p.attach(trainer, ['loss', 'accuracy'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_test(engine):
        metrics = dict(accuracy=Accuracy(), crossentropy=CrossEntropyLoss(reduction='mean'))
        evaluator = create_supervised_evaluator(model, metrics)
        evaluator.run(test_loader, max_epochs=1)
        acc = evaluator.state.metrics['accuracy']
        print('Test accuracy', evaluator.state.metrics['accuracy'])
        print('Test crossentropy', evaluator.state.metrics['crossentropy'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_embedding(engine):
        torch.save(model['embedding_nn'], WEIGHTSFOLDER / 'embedding_miniimagenet.pth')

    trainer.run(train_data, max_epochs=epochs)



def main(epochs, batch_size, device='cuda'):
    model = build_model_pretraining()
    model = model.to(device)
    train_classes = TRAIN_MINIIMAGENET.dirs()
    train_model(model, train_classes, device, epochs, batch_size)

if __name__ == "__main__":
    Fire(main)