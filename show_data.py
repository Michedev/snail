from path import Path
import dataset
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from ignite.engine import Engine

classes = Path(__file__).parent / 'data' / 'miniimagenet' / 'train'
classes = classes.dirs()
print('num classes', len(classes))
nitems = 10

miniimagenet = dataset.MiniImageNetMetaLearning(classes, n=5, k=1, random_rotation=False, length=10000)
train_loader = DataLoader(miniimagenet, batch_size=nitems, shuffle=True)

fig, axs = plt.subplots(6, nitems, figsize=(30, 10))

iter_loader = iter(train_loader)

X_batch, y_batch, y_last_batch = next(iter_loader)

print(X_batch.shape)
print(y_batch.shape)
print(y_last_batch.shape)
for j in range(nitems):
    X,y,y_last = X_batch[j],y_batch[j],y_last_batch[j]
    for i in range(len(X)):
        axs[i, j].imshow(X[i].permute(1,2,0), aspect='auto')
        i_class = (y[i] * torch.arange(y.shape[1])).sum().int().item()
        if y[i].sum() == 0:
            axs[i, j].set_title('class true ' + str(y_last.item()))
        else:
            axs[i, j].set_title('class ' + str(i_class))
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
plt.tight_layout()
plt.show()
