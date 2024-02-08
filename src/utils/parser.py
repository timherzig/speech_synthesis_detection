import argparse


def none_or_str(value):
    print(value)
    if value == "None":
        return None
    return value


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

    # Checkpoint path for testing
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=none_or_str,
        nargs="?",
        help="Path to checkpoint to test",
    )

    # Which tests to perform
    parser.add_argument(
        "--test",
        default="all",
        type=str,
        help="Which tests to perform (19/21/all)",
    )

    # Vocal (if all information should be printed)
    parser.add_argument(
        "--vocal",
        default=False,
        type=bool,
        help="Print all information",
    )

    return parser.parse_args()
