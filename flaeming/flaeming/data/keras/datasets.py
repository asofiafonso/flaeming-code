import tensorflow as tf

from flaeming.data.datasets import (
    FlaemingDataset,
    FlaemingDatasetFITS,
    FlaemingDatasetNPY,
    FlaemingDatasetRGB,
)


class KerasDataset(FlaemingDataset):
    def __init__(self, sample_number: int, filter_list: list[str]):
        super().__init__(sample_number=sample_number, filter_list=filter_list)

    @staticmethod
    def __get_tf_dataset(X, y, batch_size):
        return (
            tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(buffer_size=len(X))
            .batch(batch_size)
        )

    def get_all_datasets(self, batch_size: int = 64):
        datasets = {}
        datasets["train"] = self.__get_tf_dataset(
            self.images_train, self.labels_train, batch_size
        )
        datasets["dev"] = self.__get_tf_dataset(
            self.images_dev, self.labels_dev, batch_size
        )
        datasets["test"] = self.__get_tf_dataset(
            self.images_test, self.labels_test, batch_size
        )
        return datasets


class KerasDatasetRGB(KerasDataset, FlaemingDatasetRGB):
    pass


class KerasDatasetNPY(KerasDataset, FlaemingDatasetNPY):
    pass


class KerasDatasetFITS(KerasDataset, FlaemingDatasetFITS):
    pass
