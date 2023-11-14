import torch
import numpy as np
import torchaudio.transforms as T


def aug_tf(sample, config):
    sample = T.TimeMasking(time_mask_param=4000)(sample)
    sample = T.FrequencyMasking(freq_mask_param=30)(sample)
    return sample


def augment_time_frame_data(samples, config):
    samples = [
        aug_tf(sample, config) if np.random.rand() < config.data.aug_prob else sample
        for sample in samples
    ]
    return samples


def aug_CQT(sample, config):
    return sample


def augment_CQT_data(samples, config):
    samples = [
        aug_CQT(sample, config) if np.random.rand() < config.data.aug_prob else sample
        for sample in samples
    ]
    return samples


def collate_function(batch, config):
    samples, label, sub_class = zip(*batch)

    if config.data.data_type == "time_frame":
        samples = augment_time_frame_data(samples, config)
    elif config.data.data_type == "CQT":
        samples = augment_CQT_data(samples, config)

    return torch.stack(samples), torch.stack(label), torch.stack(sub_class)
