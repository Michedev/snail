import torch
from fire import Fire

from dataset import OmniglotDataLoader, MiniImagenetDataLoader
from models import Snail
from snailtest import Snailtest


def main(dataset='omniglot',
         n=5, k=1, batch_size=32,
         device='cpu', n_sample=500):
    assert dataset in ['omniglot', 'miniimagenet']
    if dataset == 'omniglot':
        dataloader = OmniglotDataLoader(batch_size, n, k, 1, device)
    else:
        dataloader = MiniImagenetDataLoader(batch_size, n, k, 1, device)
    snail = Snail(n, k, dataset)
    snail.load_state_dict(torch.load(snail.path, map_location=torch.device(device)))
    print('Loaded', snail.path)
    snail = snail.to(device)
    snail = snail.eval()
    snail.requires_grad_(False)
    loss = torch.nn.CrossEntropyLoss()
    tester = Snailtest(snail, device, n, loss)
    tester.test(dataloader.test_dataloader(), n_sample, pbar=True)

if __name__ == '__main__':
    Fire(main)
