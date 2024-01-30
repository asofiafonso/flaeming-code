import numpy as np
from torch.utils.data import Dataset

from flaeming.data.datasets import (
    FlaemingDataset,
    FlaemingDatasetFITS,
    FlaemingDatasetNPY,
    FlaemingDatasetRGB,
)


class TorchDataset(Dataset, FlaemingDataset):
    def __init__(self, sample_number: int, filter_list: list[str], transform=None):
        super().__init__(sample_number=sample_number, filter_list=filter_list)
        self.transform = transform

        # all data is loaded in a channels-last format,
        # move to channels-first format below for pytorch
        self.images = np.moveaxis(np.asarray(self.images).astype(np.float32), -1, 1)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.labels)


class TorchDatasetRGB(TorchDataset, FlaemingDatasetRGB):
    pass


class TorchDatasetNPY(TorchDataset, FlaemingDatasetNPY):
    pass


class TorchDatasetFITS(TorchDataset, FlaemingDatasetFITS):
    pass
