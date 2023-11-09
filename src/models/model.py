from src.models.architectures.build_model import build_model


def get_model(config):
    return build_model(config)
