import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, X, y, transform=None):
        # assert all(X.size(0) == tensor.size(0) for tensor in X)
        # assert all(y.size(0) == tensor.size(0) for tensor in y)

        self.X = X
        self.y = y

        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]

        if self.transform:
            x = self.transform(x)

        y = self.y[index]

        return x, y

    def __len__(self):
        return len(self.X)

