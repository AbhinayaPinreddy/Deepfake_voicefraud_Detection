import os
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocess import preprocess
from features import extract_features

def load_asvspoof(protocol_file, audio_dir, limit=None):
    file_paths = []
    labels = []

    with open(protocol_file, 'r') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            parts = line.strip().split()
            file_name = parts[1]
            label = parts[-1]

            file_path = os.path.join(audio_dir, file_name + ".flac")

            file_paths.append(file_path)
            labels.append(0 if label == "bonafide" else 1)

    return file_paths, labels


def pad_features(features, max_len=300):
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)))
    else:
        features = features[:, :max_len]
    return features


class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio = preprocess(self.file_paths[idx])

        features = extract_features(audio)
        features = pad_features(features)

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return features.unsqueeze(0), label