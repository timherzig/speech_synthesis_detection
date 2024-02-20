import scipy
import torch
import librosa
import numpy as np
import torchaudio.transforms as T


def mix_up(samples, label, sub_class, config, alpha=0.2):
    # Not implemented yet
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    return samples, label, sub_class


def mms(wave, window_size):
    window = scipy.signal.get_window("hann", window_size, fftbins=True)
    wave_windowed = np.convolve(wave, window, mode="same")
    spectrum = np.fft.fft(wave_windowed)
    return librosa.amplitude_to_db(np.abs(spectrum)) / 80


def get_mms(wave, hahn_window_sizes=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]):
    return torch.tensor(
        np.array([mms(wave, window_size) for window_size in hahn_window_sizes])
    ).float()


def spectogram(wave, window_length):
    return librosa.feature.melspectrogram(
        y=wave,
        sr=16000,
        win_length=window_length,
    )


def get_spectogram(wave, window_lengths=[8, 16, 32, 64, 128, 256, 512, 1024, 2048]):
    return torch.tensor(
        np.array([spectogram(wave, window_length) for window_length in window_lengths])
    ).float()


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
    elif config.data.data_type == "mel":
        # TODO: power to db!
        pass

    if config.data.mms:
        samples = [get_mms(samples[i].squeeze().numpy()) for i in range(len(samples))]

    if config.data.spectogram:
        samples = [
            get_spectogram(samples[i].squeeze().numpy()) for i in range(len(samples))
        ]

    if config.data.mix_up:
        samples, label, sub_class = mix_up(samples, label, sub_class, config)

    return torch.stack(samples), torch.stack(label), torch.stack(sub_class)
