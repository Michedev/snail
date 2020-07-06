from path import Path
import dataset
import torch
import matplotlib.pyplot as plt

classes = Path(__file__).parent / 'data' / 'miniimagenet' / 'train'
classes = classes.dirs()

data = dataset.MiniImageNetMetaLearning(classes, n=5, k=1, random_rotation=False, size=[84, 84, 3])

X,y,y_last = data[9]

fig, axs = plt.subplots(6, figsize=(20, 20))
axs = axs.reshape(-1)

for i in range(len(X)):
    axs[i].imshow(X[i])
    i_class = (y[i] * torch.arange(y.shape[1])).sum().int().item()
    axs[i].set_title('class ' + str(i_class))
    axs[i].set_xticks([])
    axs[i].set_yticks([])

plt.show()