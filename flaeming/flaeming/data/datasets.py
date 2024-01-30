import os
from abc import ABC

import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split

from flaeming.data import cutouts_folder, samples_folder
from flaeming.data import static_variables as sv
from flaeming.data.images import create_numpy_array
from flaeming.data.tables import COSMOSSamplingTable, SC4KTable


class FlaemingDataset(ABC):
    def __init__(self, sample_number: int, filter_list: list[str]):
        self.filter_list = filter_list
        self.filter_string = "".join(filter_list)
        self.sample_number = sample_number
        self.load_tables()
        self.load_images()

        # Remove NaNs from array
        self.images[np.isnan(self.images)] = 0

        self.create_dataset_split()

    def load_tables(self):
        sc4k = SC4KTable().table
        cosmos = COSMOSSamplingTable(sample_number=self.sample_number).table
        self.ids = pd.concat(
            [sc4k[SC4KTable.ID], cosmos[COSMOSSamplingTable.ID]]
        ).to_numpy()
        self.labels = pd.concat([sc4k[sv.LAE_CLASS], cosmos[sv.LAE_CLASS]]).to_numpy()
        self.labels = np.eye(2, dtype="uint8")[self.labels].astype(np.float32)

    def create_dataset_split(
        self, test_fraction: float = 0.2, dev_fraction: float = 0.2
    ):
        (
            self.images_train_dev,
            self.images_test,
            self.labels_train_dev,
            self.labels_test,
        ) = train_test_split(self.images, self.labels, test_size=test_fraction)

        rescaled_dev_fraction = dev_fraction / (1 - test_fraction)

        (
            self.images_train,
            self.images_dev,
            self.labels_train,
            self.labels_dev,
        ) = train_test_split(
            self.images_train_dev,
            self.labels_train_dev,
            test_size=rescaled_dev_fraction,
        )
        return


class FlaemingDatasetRGB(FlaemingDataset):
    IMAGE_DATA_FOLDER = cutouts_folder
    EXTENSION = "jpg"

    def load_images(self):
        self.images = []
        for galaxy in self.ids:
            image_name = f"rgb_{self.filter_string}.{self.EXTENSION}"
            image_path = os.path.join(self.IMAGE_DATA_FOLDER, galaxy, image_name)
            self.images.append(imread(image_path))

        self.images = np.asarray(self.images) / 255.0


class FlaemingDatasetNPY(FlaemingDataset):
    IMAGE_DATA_FOLDER = samples_folder

    def load_images(self):
        image_array_path_sc4k = os.path.join(
            self.IMAGE_DATA_FOLDER, f"SC4K_{self.filter_string}.npy"
        )

        image_array_path_cosmos = os.path.join(
            self.IMAGE_DATA_FOLDER,
            f"{sv.NONLAE_TABLE_BASENAME}_{self.sample_number}_{self.filter_string}.npy",
        )

        self.images = np.concatenate(
            [np.load(image_array_path_sc4k), np.load(image_array_path_cosmos)]
        )

        self.images = self.images / np.nanmax(self.images, axis=(1, 2, 3)).reshape(
            [len(self.images), 1, 1, 1]
        )


class FlaemingDatasetFITS(FlaemingDataset):
    IMAGE_DATA_FOLDER = cutouts_folder
    EXTENSION = "fits"
    SURVEYS = ["subaru", "uvista"]

    def load_images(self, **kwargs):
        self.images = create_numpy_array(
            self.IMAGE_DATA_FOLDER, self.ids, self.filter_list, self.SURVEYS, **kwargs
        )
