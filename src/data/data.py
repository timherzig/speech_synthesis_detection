import sys
import torch
from torch.utils.data.dataloader import DataLoader
from src.data.collate import collate_function
from src.data.InTheWild_Data import InTheWildDataset
from src.data.FakeOrReal_Data import FakeOrRealDataset
from src.data.LA_Data import PrepASV15Dataset, PrepASV19Dataset, PrepASV21Dataset


def get_dataloaders(config, device, pooled=False):
    if type(config.data.version) == str:
        config.data.version = [config.data.version]
        config.data.root_dir = [config.data.root_dir]

    train_ds = []
    dev_ds = []
    eval_ds = []
    weights = []

    for i in range(len(config.data.version)):
        version = config.data.version[i]
        root_path = config.data.root_dir[i]

        if config.data.data_type == "time_frame":
            if version == "LA_15":
                train_protocol_file_path = root_path + "CM_protocol/cm_train.trn.txt"
                dev_protocol_file_path = root_path + "CM_protocol/cm_develop.ndx.txt"
                eval_protocol_file_path = (
                    root_path + "CM_protocol/cm_evaluation.ndx.txt"
                )
                train_data_path = root_path + "data/train_6/"
                dev_data_path = root_path + "data/dev_6/"
                eval_data_path = root_path + "data/eval_6/"
            elif version == "LA_19":
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
            elif version == "LA_21":
                # eval_protocol_file_path = root_path + "keys/LA/CM/trial_metadata.txt"
                train_protocol_file_path = root_path + "keys/LA/CM/train.txt"
                train_data_path = root_path + "data/eval_6/"
                dev_protocol_file_path = root_path + "keys/LA/CM/val.txt"
                dev_data_path = root_path + "data/eval_6/"
                eval_protocol_file_path = root_path + "keys/LA/CM/test.txt"
                eval_data_path = root_path + "data/eval_6/"
            elif version == "InTheWild":
                # eval_protocol_file_path = root_path + "meta.csv"
                eval_protocol_file_path = root_path + "test.csv"
                eval_data_path = root_path + "data/eval_6/"
                train_protocol_file_path = root_path + "train.csv"
                train_data_path = root_path + "data/eval_6/"
                dev_protocol_file_path = root_path + "val.csv"
                dev_data_path = root_path + "data/eval_6/"
            else:  # "FakeOrReal"
                train_data_path = root_path + "data/train_6/"
                train_protocol_file_path = root_path + "train.csv"
                dev_data_path = root_path + "data/val_6/"
                dev_protocol_file_path = root_path + "val.csv"
                eval_data_path = root_path + "data/test_6/"
                eval_protocol_file_path = root_path + "test.csv"

        elif config.data.data_type == "CQT":
            if version == "LA_15":
                train_protocol_file_path = root_path + "CM_protocol/cm_train.trn.txt"
                dev_protocol_file_path = root_path + "CM_protocol/cm_develop.ndx.txt"
                eval_protocol_file_path = (
                    root_path + "CM_protocol/cm_evaluation.ndx.txt"
                )
                train_data_path = root_path + "data/train_6.4_cqt/"
                dev_data_path = root_path + "data/dev_6.4_cqt/"
                eval_data_path = root_path + "data/eval_6.4_cqt/"
            elif version == "LA_19":
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
            elif version == "LA_21":
                train_protocol_file_path = root_path + "keys/LA/CM/train.txt"
                train_data_path = root_path + "data/eval_6.4_cqt/"
                dev_protocol_file_path = root_path + "keys/LA/CM/val.txt"
                dev_data_path = root_path + "data/eval_6.4_cqt/"
                test_protocol_file_path = root_path + "keys/LA/CM/test.txt"
                test_data_path = root_path + "data/eval_6.4_cqt/"
            elif version == "InTheWild":
                eval_protocol_file_path = root_path + "val.csv"
                eval_data_path = root_path + "data/eval_6.4_cqt/"
                train_protocol_file_path = root_path + "train.csv"
                train_data_path = root_path + "data/eval_6.4_cqt/"
                dev_protocol_file_path = root_path + "val.csv"
                dev_data_path = root_path + "data/eval_6.4_cqt/"
            else:  # "FakeOrReal" CQT not yet implemented for this dataset
                raise NotImplementedError
        elif config.data.data_type == "mel":
            if version == "LA_15":
                raise NotImplementedError
            elif version == "LA_19":
                raise NotImplementedError
            elif version == "LA_21":
                raise NotImplementedError
            elif version == "InTheWild":
                raise NotImplementedError
            else:  # "FakeOrReal"
                train_protocol_file_path = root_path + "train.csv"
                dev_protocol_file_path = root_path + "val.csv"
                eval_protocol_file_path = root_path + "test.csv"
                train_data_path = root_path + "data/train_6_mel/"
                dev_data_path = root_path + "data/val_6_mel/"
                eval_data_path = root_path + "data/test_6_mel/"

        else:
            print("Program only supports 'time_frame', 'CQT' and 'mel' data types.")
            sys.exit()

        # TODO: Prepare data and set training parameters
        if version == "LA_15":
            train_ds.append(
                PrepASV15Dataset(
                    train_protocol_file_path,
                    train_data_path,
                    data_type=config.data.data_type,
                )
            )
            dev_ds.append(
                PrepASV15Dataset(
                    dev_protocol_file_path,
                    dev_data_path,
                    data_type=config.data.data_type,
                )
            )
            eval_ds.append(
                PrepASV15Dataset(
                    eval_protocol_file_path,
                    eval_data_path,
                    data_type=config.data.data_type,
                )
            )
        elif version == "LA_19":
            train_ds.append(
                PrepASV19Dataset(
                    train_protocol_file_path,
                    train_data_path,
                    data_type=config.data.data_type,
                )
            )
            dev_ds.append(
                PrepASV19Dataset(
                    dev_protocol_file_path,
                    dev_data_path,
                    data_type=config.data.data_type,
                )
            )
            eval_ds.append(
                PrepASV19Dataset(
                    eval_protocol_file_path,
                    eval_data_path,
                    data_type=config.data.data_type,
                )
            )
        elif version == "LA_21":
            train_ds.append(
                PrepASV21Dataset(
                    train_protocol_file_path,
                    train_data_path,
                    data_type=config.data.data_type,
                )
            )
            dev_ds.append(
                PrepASV21Dataset(
                    dev_protocol_file_path,
                    dev_data_path,
                    data_type=config.data.data_type,
                )
            )
            eval_ds.append(
                PrepASV21Dataset(
                    eval_protocol_file_path,
                    eval_data_path,
                    data_type=config.data.data_type,
                )
            )
        elif version == "InTheWild":
            train_ds.append(
                InTheWildDataset(
                    train_protocol_file_path,
                    train_data_path,
                    data_type=config.data.data_type,
                )
            )
            dev_ds.append(
                InTheWildDataset(
                    dev_protocol_file_path,
                    dev_data_path,
                    data_type=config.data.data_type,
                )
            )
            eval_ds.append(
                InTheWildDataset(
                    eval_protocol_file_path,
                    eval_data_path,
                    data_type=config.data.data_type,
                )
            )
        else:  # "FakeOrReal"
            train_ds.append(
                FakeOrRealDataset(
                    train_protocol_file_path,
                    train_data_path,
                    data_type=config.data.data_type,
                )
            )
            dev_ds.append(
                FakeOrRealDataset(
                    dev_protocol_file_path,
                    dev_data_path,
                    data_type=config.data.data_type,
                )
            )
            eval_ds.append(
                FakeOrRealDataset(
                    eval_protocol_file_path,
                    eval_data_path,
                    data_type=config.data.data_type,
                )
            )

    if pooled:
        print(f"pre pool len: {len(eval_ds)}")
        eval_ds.extend(dev_ds)
        print(f"post pool len: {len(eval_ds)}")

    eval_set = torch.utils.data.ConcatDataset(eval_ds)
    print(f"Number of eval samples in {config.data.version}: {len(eval_set)}")

    eval_loader = DataLoader(
        eval_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda x: collate_function(x, config),
    )

    if (len(train_ds) == 0 and len(dev_ds) == 0) or pooled:
        return None, None, eval_loader, None

    for i in range(len(train_ds)):
        weights.append(train_ds[i].get_weights())

    weights = torch.mean(torch.stack(weights), dim=0).to(device)

    train_set = torch.utils.data.ConcatDataset(train_ds)
    print(f"Number of train samples in {config.data.version}: {len(train_set)}")
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: collate_function(x, config),
    )

    dev_set = torch.utils.data.ConcatDataset(dev_ds)
    print(f"Number of dev samples in {config.data.version}: {len(dev_set)}")
    dev_loader = DataLoader(
        dev_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda x: collate_function(x, config),
    )

    return train_loader, dev_loader, eval_loader, weights
