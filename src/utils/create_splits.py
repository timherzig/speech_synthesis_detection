import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def split_csv(file):
    root_dir = "/".join(file.split("/")[:-1])
    df = pd.read_csv(file)
    train, val, test = np.split(
        df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))]
    )

    train.to_csv(os.path.join(root_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(root_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(root_dir, "test.csv"), index=False)


def split_txt(file):
    root_dir = "/".join(file.split("/")[:-1])
    with open(file, "r") as f:
        lines = f.readlines()
    train, val, test = np.split(lines, [int(0.6 * len(lines)), int(0.8 * len(lines))])

    with open(os.path.join(root_dir, "train.txt"), "w") as f:
        f.writelines(train)
    with open(os.path.join(root_dir, "val.txt"), "w") as f:
        f.writelines(val)
    with open(os.path.join(root_dir, "test.txt"), "w") as f:
        f.writelines(test)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    if args.file.endswith(".csv"):
        split_csv(args.file)
    elif args.file.endswith(".txt"):
        split_txt(args.file)
