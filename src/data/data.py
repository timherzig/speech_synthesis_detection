from torch.utils.data.dataloader import DataLoader
from src.data.LA_Data import PrepASV15Dataset, PrepASV19Dataset, PrepASV21Dataset
from src.data.collate import collate_function


def get_dataloaders(config, device):
    root_path = config.data.root_dir

    if config.data.data_type == "time_frame":
        if config.data.version == 15:
            train_protocol_file_path = root_path + "CM_protocol/cm_train.trn.txt"
            dev_protocol_file_path = root_path + "CM_protocol/cm_develop.ndx.txt"
            eval_protocol_file_path = root_path + "CM_protocol/cm_evaluation.ndx.txt"
            train_data_path = root_path + "data/train_6/"
            dev_data_path = root_path + "data/dev_6/"
            eval_data_path = root_path + "data/eval_6/"
        elif config.data.version == 19:
            train_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
            )
            dev_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
            )
            eval_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
            )
            train_data_path = root_path + "data/train_6/"
            dev_data_path = root_path + "data/dev_6/"
            eval_data_path = root_path + "data/eval_6/"
        else:
            eval_protocol_file_path = root_path + "keys/LA/CM/trial_metadata.txt"
            eval_data_path = root_path + "data/eval_6/"

    elif config.data.data_type == "CQT":
        if config.data.version == 15:
            train_protocol_file_path = root_path + "CM_protocol/cm_train.trn.txt"
            dev_protocol_file_path = root_path + "CM_protocol/cm_develop.ndx.txt"
            eval_protocol_file_path = root_path + "CM_protocol/cm_evaluation.ndx.txt"
            train_data_path = root_path + "data/train_6.4_cqt/"
            dev_data_path = root_path + "data/dev_6.4_cqt/"
            eval_data_path = root_path + "data/eval_6.4_cqt/"
        elif config.data.version == 19:
            train_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
            )
            dev_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
            )
            eval_protocol_file_path = (
                root_path
                + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
            )
            train_data_path = root_path + "data/train_6.4_cqt/"
            dev_data_path = root_path + "data/dev_6.4_cqt/"
            eval_data_path = root_path + "data/eval_6.4_cqt/"
        else:
            eval_protocol_file_path = root_path + "keys/LA/CM/trial_metadata.txt"
            eval_data_path = root_path + "data/eval_6/"
    else:
        print("Program only supports 'time_frame' and 'CQT' data types.")
        sys.exit()

    # TODO: Prepare data and set training parameters
    if config.data.version == 15:
        train_set = PrepASV15Dataset(
            train_protocol_file_path, train_data_path, data_type=config.data.data_type
        )
        dev_set = PrepASV15Dataset(
            dev_protocol_file_path, dev_data_path, data_type=config.data.data_type
        )
        eval_set = PrepASV15Dataset(
            eval_protocol_file_path, eval_data_path, data_type=config.data.data_type
        )
    elif config.data.version == 19:
        train_set = PrepASV19Dataset(
            train_protocol_file_path, train_data_path, data_type=config.data.data_type
        )
        dev_set = PrepASV19Dataset(
            dev_protocol_file_path, dev_data_path, data_type=config.data.data_type
        )
        eval_set = PrepASV19Dataset(
            eval_protocol_file_path, eval_data_path, data_type=config.data.data_type
        )
    else:
        eval_set = PrepASV21Dataset(
            eval_protocol_file_path, eval_data_path, data_type=config.data.data_type
        )

    eval_loader = DataLoader(
        eval_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    if config.data.version == 21:
        return None, None, eval_loader, None

    weights = train_set.get_weights().to(device)  # weight used for WCE
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: collate_function(x, config),
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, dev_loader, eval_loader, weights
