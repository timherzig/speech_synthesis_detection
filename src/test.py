import os
import torch

from src.utils.metrics import asv_cal_accuracies, cal_roc_eer
from src.utils.temperature_scaling import ModelWithTemperature
from src.data.data import get_dataloaders
from src.models.model import get_model

LA_19_ROOT = "/ds/audio/LA_19/"
LA_21_ROOT = "/ds/audio/LA_21/"
FakeOrReal_ROOT = "/ds/audio/FakeOrReal/"
InTheWild_ROOT = "/ds/audio/InTheWild/"


def test_all_datasets(Net, test_out_file, config, checkpoint, device):
    # First test LA_19
    config.data.root_dir = LA_19_ROOT
    config.data.version = "LA_19"
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
            f"----------------Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.----------------\n"
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
        f.write(
            "------------------------------------------------------------------------------------------------------------------------\n"
        )
        f.close()

    # Test LA_21
    config.data.root_dir = LA_21_ROOT
    config.data.version = "LA_21"
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

    with open(test_out_file, "a") as f:
        f.write(
            f"----------------Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.----------------\n"
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
        f.write(
            "------------------------------------------------------------------------------------------------------------------------\n"
        )
        f.close()

    # Test FakeOrReal
    config.data.root_dir = FakeOrReal_ROOT
    config.data.version = "FakeOrReal"
    print(f"---------FakeOrReal---------")

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

    with open(test_out_file, "a") as f:
        f.write(
            f"----------------Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.----------------\n"
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
        f.write(
            "------------------------------------------------------------------------------------------------------------------------\n"
        )
        f.close()

    # Test InTheWild
    config.data.root_dir = InTheWild_ROOT
    config.data.version = "InTheWild"
    print(f"---------InTheWild---------")

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

    with open(test_out_file, "a") as f:
        f.write(
            f"----------------Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.----------------\n"
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
        f.write(
            "------------------------------------------------------------------------------------------------------------------------\n"
        )
        f.close()


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

    with open(test_out_file, "a") as f:
        f.write(
            f"----------------Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.----------------\n"
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
        f.write(
            "------------------------------------------------------------------------------------------------------------------------\n"
        )
        f.close()

    return


def test_FakeOrReal(Net, test_out_file, config, checkpoint, device):
    # Test FakeOrReal
    config.data.root_dir = FakeOrReal_ROOT
    config.data.version = "FakeOrReal"
    print(f"---------FakeOrReal---------")

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

    # We do not have a dev set to tune the temperature scaling

    # accuracy, probabilities = asv_cal_accuracies(
    #     net=Net_ts,
    #     device=device,
    #     data_loader=eval_loader,
    # )

    # post_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    # print(
    #     "EER with temperature scaling: {:.2f}% for {}.".format(
    #         post_ts_eer * 100, checkpoint
    #     )
    # )

    with open(test_out_file, "a") as f:
        f.write(
            f"----------------Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.----------------\n"
        )
        f.write(
            "EER without temperature scaling: {:.2f}% for {}.\n".format(
                pre_ts_eer * 100, checkpoint
            )
        )
        # f.write(
        #     "EER with temperature scaling: {:.2f}% for {}.\n".format(
        #         post_ts_eer * 100, checkpoint
        #     )
        # )
        f.write(
            "------------------------------------------------------------------------------------------------------------------------\n"
        )
        f.close()


def test_InTheWild(Net, test_out_file, config, checkpoint, device):
    # Test InTheWild
    config.data.root_dir = InTheWild_ROOT
    config.data.version = "InTheWild"
    print(f"---------InTheWild---------")

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

    # We do not have a dev set to tune the temperature scaling

    # accuracy, probabilities = asv_cal_accuracies(
    #     net=Net_ts,
    #     device=device,
    #     data_loader=eval_loader,
    # )

    # post_ts_eer = cal_roc_eer(probabilities, show_plot=False)

    # print(
    #     "EER with temperature scaling: {:.2f}% for {}.".format(
    #         post_ts_eer * 100, checkpoint
    #     )
    # )

    with open(test_out_file, "a") as f:
        f.write(
            f"----------------Evaluation results using {config.data.root_dir.split('/')[-1]}_{config.data.version} dataset.----------------\n"
        )
        f.write(
            "EER without temperature scaling: {:.2f}% for {}.\n".format(
                pre_ts_eer * 100, checkpoint
            )
        )
        # f.write(
        #     "EER with temperature scaling: {:.2f}% for {}.\n".format(
        #         post_ts_eer * 100, checkpoint
        #     )
        # )
        f.write(
            "------------------------------------------------------------------------------------------------------------------------\n"
        )
        f.close()


def test(config, checkpoint, dataset="all"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("/".join(checkpoint.split("/")[:-2]) + "/tests/", exist_ok=True)
    test_file = ".".join(checkpoint.split("/")[-1].split(".")[:-1]) + ".txt"
    test_out_file = "/".join(checkpoint.split("/")[:-2]) + "/tests/" + test_file

    Net = get_model(config).to(device)

    num_total_learnable_params = sum(
        i.numel() for i in Net.parameters() if i.requires_grad
    )
    print("Number of params: {}.".format(num_total_learnable_params))
    check_point = torch.load(checkpoint)
    Net.load_state_dict(check_point["model_state_dict"])

    print("Checkpoint loaded...")

    if dataset.lower() == "all":
        test_all_datasets(Net, test_out_file, config, checkpoint, device)
    elif dataset.lower() == "la":
        test_LA(Net, test_out_file, config, checkpoint, device)
    elif dataset.lower() == "fakeorreal":
        test_FakeOrReal(Net, test_out_file, config, checkpoint, device)
    elif dataset.lower() == "inthewild":
        test_InTheWild(Net, test_out_file, config, checkpoint, device)
    else:
        raise NotImplementedError(f"{dataset} dataset testing not implemented")
