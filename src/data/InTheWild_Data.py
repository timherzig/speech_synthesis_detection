import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf

from torch.utils.data.dataloader import Dataset


class InTheWildDataset(Dataset):
    def __init__(
        self, protocol_file_path, data_path, data_type="time_frame", weighted=False
    ):
        self.protocol = pd.read_csv(protocol_file_path)
        self.data_path = data_path
        self.data_type = data_type

        if weighted:
            set = "test"
            if "val" in protocol_file_path:
                set = "validate"
            elif "train" in protocol_file_path:
                set = "train"

            subset = os.path.join(
                "/".join(protocol_file_path.split("/")[:-1]),
                "subset",
                f"{set}_subset.csv",
            )
            subset = np.genfromtxt(subset, delimiter=",", dtype=str)

            self.protocol["tmp"] = self.protocol["file"].apply(
                lambda x: x.split(".")[0]
            )
            self.protocol = self.protocol[self.protocol["tmp"].isin(subset)]

            print(f"Length of InTheWild subset: {len(self.protocol.index)}")

    def __len__(self):
        return len(self.protocol.index)

    def __getitem__(self, index):
        data_file_path = os.path.join(self.data_path, self.protocol.iloc[index, 0])

        if self.data_type == "time_frame":
            sample, _ = sf.read(data_file_path)
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.protocol.iloc[index, 2]
            label = label_encode(label)
            sub_class = torch.tensor(-1, dtype=torch.int64)
            return sample, label, sub_class

        if self.data_type == "CQT":
            sample = torch.load(data_file_path)
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.protocol.iloc[index, 2]
            label = label_encode(label)
            sub_class = torch.tensor(-1, dtype=torch.int64)
            return sample, label, sub_class

    def get_weights(self):
        label_info = self.protocol.iloc[:, 2]
        num_zero_class = (label_info == "bona-fide").sum()
        num_one_class = (label_info == "spoof").sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


def label_encode(label):
    if label == "bona-fide":
        label = torch.tensor(0, dtype=torch.int64)
    elif label == "human":
        label = torch.tensor(0, dtype=torch.int64)
    else:
        label = torch.tensor(1, dtype=torch.int64)
    return label
