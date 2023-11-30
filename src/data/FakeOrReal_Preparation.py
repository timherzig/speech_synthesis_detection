import os
import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from argparse import ArgumentParser


def gen_time_frame(
    protocol_path, read_audio_path, write_audio_path, duration, status: str
):
    sub_path = os.path.join(write_audio_path, status + "_" + str(duration))
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path)  # .values
    file_index = protocol["file"]
    label_index = protocol["label"]
    num_files = len(protocol.index)
    total_sample_count = 0

    for i in range(num_files):
        try:
            label_dir = "fake" if label_index[i] == "spoof" else "real"
            x, fs = sf.read(os.path.join(read_audio_path, label_dir, file_index[i]))
            if len(x) < duration * fs:
                x = np.tile(x, int((duration * fs) // len(x)) + 1)
            x = x[0 : (int(duration * fs))]
            total_sample_count += 1
            sf.write(os.path.join(sub_path, file_index[i]), x, fs)
        except:
            protocol = protocol.drop([i])

    os.remove(protocol_path)
    protocol.to_csv(protocol_path, index=False)
    print(
        "{} pieces {}-second {} samples generated.".format(
            total_sample_count, duration, status
        )
    )


def gen_cqt(protocol_path, read_audio_path, write_audio_path, duration, status: str):
    sub_path = os.path.join(write_audio_path, status + "_" + str(duration) + "_cqt")
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path)
    file_index = protocol["file"]
    label_index = protocol["label"]
    num_files = len(protocol.index)
    total_sample_count = 0
    fs = 16000

    for i in range(num_files):
        label_dir = "fake" if label_index[i] == "spoof" else "real"
        x, fs = sf.read(os.path.join(read_audio_path, label_dir, file_index[i]))
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


def build_data_set(root_path, split):
    split_path = os.path.join(root_path, split)

    fake_list = os.listdir(os.path.join(split_path, "fake"))
    real_list = os.listdir(os.path.join(split_path, "real"))

    fake_df = pd.DataFrame(fake_list, columns=["file"])
    fake_df["label"] = "spoof"

    real_df = pd.DataFrame(real_list, columns=["file"])
    real_df["label"] = "bonafide"

    df = pd.concat([fake_df, real_df], ignore_index=True)

    if split == "testing":
        split = "test"
    elif split == "training":
        split = "train"
    elif split == "validation":
        split = "val"

    df.to_csv(os.path.join(root_path, f"{split}.csv"), index=False)


if __name__ == "__main__":
    # directory info of FakeOrReal dataset
    parser = ArgumentParser()
    parser.add_argument(
        "--root_path", type=str, required=True, help="root path of FakeOrReal dataset"
    )
    args = parser.parse_args()

    # build_data_set(args.root_path, "testing")
    # build_data_set(args.root_path, "training")
    # build_data_set(args.root_path, "validation")

    root_path = args.root_path
    train_protocol_path = os.path.join(root_path, "train.csv")
    train_data_path = os.path.join(root_path, "training")
    val_protocol_path = os.path.join(root_path, "val.csv")
    val_data_path = os.path.join(root_path, "validation")
    eval_protocol_path = os.path.join(root_path, "test.csv")
    eval_data_path = os.path.join(root_path, "testing")

    # create folders for new types of data
    new_data_path = os.path.join(root_path, "data")
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    # generate equal-duration time-frames examples
    print("Generating time-frame data...")
    time_dur = 6  # in seconds
    gen_time_frame(
        train_protocol_path, train_data_path, new_data_path, time_dur, "train"
    )
    gen_time_frame(val_protocol_path, val_data_path, new_data_path, time_dur, "val")
    gen_time_frame(eval_protocol_path, eval_data_path, new_data_path, time_dur, "eval")

    # generate cqt feature per sample
    print("Generating CQT data...")
    cqt_dur = 6.4  # in seconds, default ICASSP 2021 setting
    gen_cqt(train_protocol_path, train_data_path, new_data_path, cqt_dur, "train")
    gen_cqt(val_protocol_path, val_data_path, new_data_path, cqt_dur, "val")
    gen_cqt(eval_protocol_path, eval_data_path, new_data_path, cqt_dur, "eval")

    print("Data preparation finished.")
