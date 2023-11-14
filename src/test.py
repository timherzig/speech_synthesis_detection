import torch

from src.utils.metrics import asv_cal_accuracies, cal_roc_eer
from src.utils.temperature_scaling import ModelWithTemperature
from src.data.data import get_dataloaders
from src.models.model import get_model


def test(config, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_out_file = "/".join(checkpoint.split("/")[:-2]) + "/test.txt"

    # Get dataloaders
    _, dev_loader, eval_loader, _ = get_dataloaders(config, device)

    Net = get_model(config).to(device)

    num_total_learnable_params = sum(
        i.numel() for i in Net.parameters() if i.requires_grad
    )
    print("Number of params: {}.".format(num_total_learnable_params))
    check_point = torch.load(checkpoint)
    Net.load_state_dict(check_point["model_state_dict"])

    print("Checkpoint loaded...")

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
