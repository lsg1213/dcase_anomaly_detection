from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import util


class DcaseDataset(torch.utils.data.Dataset):
    def __init__(self, files, config, transform=None):
        self.transform = transform
        self.config = config
        for file_id, file_name in tqdm(enumerate(files)):
            # shape = (#frames, #dims)
            features = util.extract_feature(file_name, config=self.config["feature"])
            features = features[:: self.config["feature"]["n_hop_frames"], :]

            if file_id == 0:
                # shape = (#total frames over all audio files, #dim. of feature vector)
                dataset = np.zeros(
                    (
                        features.shape[0] * len(files),
                        self.config["feature"]["n_mels"] * self.config["feature"]["n_frames"],
                    ),
                    np.float32,
                )

            dataset[
                features.shape[0] * file_id : features.shape[0] * (file_id + 1), :
            ] = features

        self.feat_data = dataset

        train_size = int(len(dataset) * (1.0 - self.config["training"]["validation_split"]))
        print(
            "train_size: %d, val_size: %d"
            % (
                train_size,
                int(len(dataset) * self.config["training"]["validation_split"]),
            )
        )

    def __len__(self):
        return self.feat_data.shape[0]  # return num of samples

    def __getitem__(self, index):
        sample = self.feat_data[index, :]  # return vector

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloader(dataset, config):
    """
    Make dataloader from dataset for training.
    """
    train_size = int(len(dataset) * (1.0 - config["training"]["validation_split"]))
    data_loader_train = torch.utils.data.DataLoader(
        Subset(dataset, list(range(0, train_size))),
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        Subset(dataset, list(range(train_size, len(dataset)))),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    return data_loader_train, data_loader_val

    