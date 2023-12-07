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

    # get CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(DEFAULT_CONFIG)

    for entry in DATASETS:
        config.data.data_type = "time_frame"
        _, _, dataloader, _ = get_dataloaders(config, device)

        for batch in dataloader:
            assert batch[0].shape == torch.Size([config.batch_size, 1, 6 * 16000])
            break

        config.data.data_type = "CQT"
        _, _, dataloader, _ = get_dataloaders(config, device)

        for batch in dataloader:
            assert batch[0].shape == torch.Size([config.batch_size, 1, 432, 400])
            break
