import sys

import torch
import numpy as np
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import torchaudio

import util


def extract_feature(file_name, config):
    """
    Extract feature vectors.

    file_name : str
        target audio file

    config : str
        configuration for feature extraction

    return : numpy.array( numpy.array( float ) )
        vector array
        dataset.shape = (dataset_size, feature_vector_length)
    """

    n_mels = config["n_mels"]
    n_frames = config["n_frames"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    power = config["power"]

    # calculate the number of dimensions
    dims = n_mels * n_frames
    
    device = util.get_device()

    # generate melspectrogram using librosa
    audio, sample_rate = torchaudio.load(file_name)
    melspectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power).to(device)
    audio = audio.to(device)
    mel_spectrogram = melspectrogram(audio)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = (
        20.0 / power * torch.log10(torch.maximum(mel_spectrogram, torch.ones_like(mel_spectrogram) * sys.float_info.epsilon))
    )

    # calculate total vector size
    n_vectors = log_mel_spectrogram.shape[-1] - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return torch.empty((0, 0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = torch.zeros((n_vectors, dims), dtype=log_mel_spectrogram.dtype, device=log_mel_spectrogram.device)
    for frame in range(n_frames):
        vectors[:, n_mels * frame : n_mels * (frame + 1)] = log_mel_spectrogram[..., frame : frame + n_vectors].transpose(-1,-2).squeeze(0)

    return vectors.cpu()


class DcaseDataset(torch.utils.data.Dataset):
    def __init__(self, files, config, transform=None):
        self.transform = transform
        self.config = config
        for file_id, file_name in tqdm(enumerate(files)):
            # shape = (#frames, #dims)
            features = extract_feature(file_name, config=self.config["feature"])
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

    