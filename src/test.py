import torch

from src.utils.metrics import asv_cal_accuracies, cal_roc_eer
from src.utils.temperature_scaling import ModelWithTemperature
from src.data.data import get_dataloaders
from src.models.model import get_model

LA_19_ROOT = "/ds/audio/LA_19/"
LA_21_ROOT = "/ds/audio/LA_21/"


def test_all_datasets(Net, test_out_file, config, checkpoint, device):
    # First test LA_19
    config.data.root_dir = LA_19_ROOT
    config.data.version = 19
    print(f"---------LA_19---------")

    # Get dataloaders
    _, dev_loader, eval_loader, _ = get_dataloaders(config, device)

    accuracy, probabilities = asv_cal_accuracies(
        net=Net,
        device=device,
        data_loader=eval_loader,
    )

    pre_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    print(
        "EER without temperature scaling: {:.2f}% for {}.".format(
            pre_ts_eer * 100, checkpoint
        )
    )

    # Temperature scaling
    Net_ts = ModelWithTemperature(Net)
    Net_ts.set_temperature(dev_loader)

    accuracy, probabilities = asv_cal_accuracies(
        net=Net_ts,
        device=device,
        data_loader=eval_loader,
    )

    post_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    print(
        "EER with temperature scaling: {:.2f}% for {}.".format(
            post_ts_eer * 100, checkpoint
        )
    )

    with open(test_out_file, "w") as f:
        f.write(
            f"Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.\n"
        )
        f.write(
            "EER without temperature scaling: {:.2f}% for {}.\n".format(
                pre_ts_eer * 100, checkpoint
            )
        )
        f.write(
            "EER with temperature scaling: {:.2f}% for {}.\n".format(
                post_ts_eer * 100, checkpoint
            )
        )

    # Then test LA_21
    config.data.root_dir = LA_21_ROOT
    config.data.version = 21
    print(f"---------LA_21---------")

    # Get dataloaders
    _, _, eval_loader, _ = get_dataloaders(config, device)

    accuracy, probabilities = asv_cal_accuracies(
        net=Net,
        device=device,
        data_loader=eval_loader,
    )

    pre_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    print(
        "EER without temperature scaling: {:.2f}% for {}.".format(
            pre_ts_eer * 100, checkpoint
        )
    )

    # Temperature scaling
    # Net = ModelWithTemperature(Net)
    # Net.set_temperature(dev_loader)

    accuracy, probabilities = asv_cal_accuracies(
        net=Net_ts,
        device=device,
        data_loader=eval_loader,
    )

    post_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    print(
        "EER with temperature scaling: {:.2f}% for {}.".format(
            post_ts_eer * 100, checkpoint
        )
    )

    with open(test_out_file, "w") as f:
        f.write(
            f"Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.\n"
        )
        f.write(
            "EER without temperature scaling: {:.2f}% for {}.\n".format(
                pre_ts_eer * 100, checkpoint
            )
        )
        f.write(
            "EER with temperature scaling: {:.2f}% for {}.\n".format(
                post_ts_eer * 100, checkpoint
            )
        )


def test_LA(Net, test_out_file, config, checkpoint, device):
    # Get dataloaders
    _, dev_loader, eval_loader, _ = get_dataloaders(config, device)

    accuracy, probabilities = asv_cal_accuracies(
        net=Net,
        device=device,
        data_loader=eval_loader,
    )

    pre_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    print(
        "EER without temperature scaling: {:.2f}% for {}.".format(
            pre_ts_eer * 100, checkpoint
        )
    )

    # Temperature scaling
    Net = ModelWithTemperature(Net)
    Net.set_temperature(dev_loader)

    accuracy, probabilities = asv_cal_accuracies(
        net=Net,
        device=device,
        data_loader=eval_loader,
    )

    post_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    print(
        "EER with temperature scaling: {:.2f}% for {}.".format(
            post_ts_eer * 100, checkpoint
        )
    )

    with open(test_out_file, "w") as f:
        f.write(
            f"Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.\n"
        )
        f.write(
            "EER without temperature scaling: {:.2f}% for {}.\n".format(
                pre_ts_eer * 100, checkpoint
            )
        )
        f.write(
            "EER with temperature scaling: {:.2f}% for {}.\n".format(
                post_ts_eer * 100, checkpoint
            )
        )

    return


def test(config, checkpoint, dataset="all"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_out_file = "/".join(checkpoint.split("/")[:-2]) + "/test.txt"

    Net = get_model(config).to(device)

    num_total_learnable_params = sum(
        i.numel() for i in Net.parameters() if i.requires_grad
    )
    print("Number of params: {}.".format(num_total_learnable_params))
    check_point = torch.load(checkpoint)
    Net.load_state_dict(check_point["model_state_dict"])

    print("Checkpoint loaded...")

    if dataset == "all":
        test_all_datasets(Net, test_out_file, config, checkpoint, device)
    else:
        test_LA(Net, test_out_file, config, checkpoint, device)
