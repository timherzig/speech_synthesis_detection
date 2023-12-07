import torch
import pytest
from omegaconf import OmegaConf

from src.data.data import get_dataloaders

DATASETS = [
    {"name": "LA_19", "root": "ds/audio/LA_19/"},
    {"name": "LA_21", "root": "ds/audio/LA_21/"},
    {"name": "InTheWild", "root": "ds/audio/InTheWild/"},
    {"name": "FakeOrReal", "root": "ds/audio/FakeOrReal/"},
]

DEFAULT_CONFIG = "tests/test_config/default.yaml"


def test_dataloader_batch_dimension():
    """
    Test if the dataloader is returning the correct batch dimension
    """

    config = OmegaConf.load(DEFAULT_CONFIG)

    for entry in DATASETS:
        dataloader = get_dataloaders(config, entry["name"], entry["root"])

        for batch in dataloader:
            # Correct batch dimensions
            if config.data.data_type == "time_frame":
                assert batch[0].shape == torch.Size(
                    [config.data.batch_size, 1, 6 * 16000]
                )
            elif config.data.data_type == "CQT":
                assert batch[0].shape == torch.Size(
                    [config.data.batch_size, 1, 432, 400]
                )

            break
