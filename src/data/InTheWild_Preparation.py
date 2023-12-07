import os
import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from argparse import ArgumentParser
from tqdm import tqdm


def gen_time_frame(
    protocol_path, read_audio_path, write_audio_path, duration, status: str, sr=16000
):
    sub_path = os.path.join(write_audio_path, status + "_" + str(duration))
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path)  # .values
    file_index = protocol["file"]
    num_files = len(protocol.index)
    total_sample_count = 0

    for i in tqdm(range(num_files)):
        x, fs = sf.read(os.path.join(read_audio_path, file_index[i]))
        if sr != fs:
            x = librosa.resample(x, fs, sr)
            fs = sr
        if len(x) < duration * fs:
            x = np.tile(x, int((duration * fs) // len(x)) + 1)
        x = x[0 : (int(duration * fs))]
        total_sample_count += 1
        sf.write(os.path.join(sub_path, file_index[i]), x, fs)
    print(
        "{} pieces {}-second {} samples generated.".format(
            total_sample_count, duration, status
        )
    )


def gen_cqt(
    protocol_path, read_audio_path, write_audio_path, duration, status: str, sr=16000
):
    sub_path = os.path.join(write_audio_path, status + "_" + str(duration) + "_cqt")
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path)
    file_index = protocol["file"]
    num_files = len(protocol.index)
    total_sample_count = 0
    fs = 16000

    for i in tqdm(range(num_files)):
        x, fs = sf.read(os.path.join(read_audio_path, file_index[i]))
        if sr != fs:
            x = librosa.resample(x, fs, sr)
            fs = sr
        len_sample = int(duration * fs)

        if len(x) < len_sample:
            x = np.tile(x, int(len_sample // len(x)) + 1)

        x = x[0 : int(len_sample - 256)]

        x = signal.lfilter([1, -0.97], [1], x)
        x_cqt = librosa.cqt(
            x,
            sr=fs,
            hop_length=256,
            n_bins=432,
            bins_per_octave=48,
            window="hann",
            fmin=15,
        )
        pow_cqt = np.square(np.abs(x_cqt))
        log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)
        total_sample_count += 1
        torch.save(
            log_pow_cqt, os.path.join(sub_path, file_index[i].split(".")[0] + ".pt")
        )
    print(
        "{} {} CQT features of {}*{} generated.".format(
            total_sample_count, status, 432, int((duration * fs) // 256)
        )
    )


if __name__ == "__main__":
    # directory info of InTheWild dataset
    parser = ArgumentParser()
    parser.add_argument(
        "--root_path", type=str, required=True, help="root path of InTheWild dataset"
    )
    args = parser.parse_args()

    root_path = args.root_path
    eval_protocol_path = os.path.join(root_path, "meta.csv")
    eval_data_path = os.path.join(root_path, "release_in_the_wild")

    # create folders for new types of data
    new_data_path = os.path.join(root_path, "data")
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    # generate equal-duration time-frames examples
    print("Generating time-frame data...")
    time_dur = 6  # in seconds
    gen_time_frame(eval_protocol_path, eval_data_path, new_data_path, time_dur, "eval")

    # generate cqt feature per sample
    print("Generating CQT data...")
    cqt_dur = 6.4  # in seconds, default ICASSP 2021 setting
    gen_cqt(eval_protocol_path, eval_data_path, new_data_path, cqt_dur, "eval")

    print("Data preparation finished.")
