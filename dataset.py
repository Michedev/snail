import torch
from random import sample, randint
from skimage import io, transform


def sample_batch(batch_size, train_classes, t, n, k, ohe_matrix=None):
    X = torch.zeros(batch_size, t, 28, 28, 1)
    y = torch.zeros(batch_size, t, n, dtype=torch.int16)
    y_last_class = torch.zeros(batch_size, dtype=torch.int16)
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