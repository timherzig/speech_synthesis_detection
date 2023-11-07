import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Speech Synthesis Detection")

    # General
    parser.add_argument(
        "--task",
        default="train",
        type=str,
        help="Task to run",
    )

    # Model
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Path to experiment config",
    )

    return parser.parse_args()
