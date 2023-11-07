import os
import torch
import torch.optim as optim
import torch.nn.functional as F

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.metrics import asv_cal_accuracies, cal_roc_eer
from src.data.data import get_dataloaders
from src.models.model import get_model


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./trained_models/"):
        os.makedirs("./trained_models/")

    # Get dataloaders
    train_loader, dev_loader, eval_loader = get_dataloaders(config, device)

    Net = get_model(config).to(device)

    num_total_learnable_params = sum(
        i.numel() for i in Net.parameters() if i.requires_grad
    )
    print("Number of learnable params: {}.".format(num_total_learnable_params))

    optimizer = optim.Adam(Net.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_type = "WCE"  # {'WCE', 'mixup'}

    print(
        "Training data: {} {}, Date type: {}. Training started...".format(
            config.data.root_dir, config.data.version, config.data.data_type
        )
    )

    num_epoch = config.num_epochs
    loss_per_epoch = torch.zeros(
        num_epoch,
    )
    best_d_eer = [0.09, 0]

    time_name = time.ctime()
    time_name = time_name.replace(" ", "_")
    time_name = time_name.replace(":", "_")

    path = "./trained_models/LA_{}/{}/{}/".format(
        config.data.version, config.model.architecture, config.model.size
    )

    log_path = "{}/log".format(path)

    save_path = "{}/model".format(path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    f = open("{}/{}.csv".format(log_path, time_name), "w+")

    # for epoch in range(check_point['epoch']+1, num_epoch):
    print("Training started...")
    for epoch in range(num_epoch):
        Net.train()
        t = time.time()
        total_loss = 0
        counter = 0
        for batch in train_loader:
            counter += 1
            # forward
            samples, labels, _ = batch
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if loss_type == "mixup":
                # mixup
                alpha = 0.1
                lam = np.random.beta(alpha, alpha)
                lam = torch.tensor(lam, requires_grad=False)
                index = torch.randperm(len(labels))
                samples = lam * samples + (1 - lam) * samples[index, :]
                preds = Net(samples)
                labels_b = labels[index]
                loss = lam * F.cross_entropy(preds, labels) + (
                    1 - lam
                ) * F.cross_entropy(preds, labels_b)
            else:
                preds = Net(samples)
                loss = F.cross_entropy(preds, labels, weight=weights)
                # loss = F.cross_entropy(preds, labels)

            # backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_per_epoch[epoch] = total_loss / counter

        dev_accuracy, d_probs = asv_cal_accuracies(
            net=Net,
            device=device,
            data_loader=dev_loader,
        )

        d_eer = cal_roc_eer(d_probs, show_plot=False)
        if d_eer <= best_d_eer[0]:
            best_d_eer[0] = d_eer
            best_d_eer[1] = int(epoch)

            eval_accuracy, e_probs = asv_cal_accuracies(
                net=Net,
                device=device,
                data_loader=eval_loader,
            )
            e_eer = cal_roc_eer(e_probs, show_plot=False)
        else:
            e_eer = 0.99
            eval_accuracy = 0.00

        net_str = (
            config.data.data_type
            + "_"
            + str(epoch)
            + "_"
            + "ASVspoof20"
            + str(config.data.version)
            + "_LA_Loss_"
            + str(round(total_loss / counter, 4))
            + "_dEER_"
            + str(round(d_eer * 100, 2))
            + "%_eEER_"
            + str(round(e_eer * 100, 2))
            + "%.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": Net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss_per_epoch,
            },
            ("{}/{}".format(save_path, net_str)),
        )

        elapsed = time.time() - t

        print_str = (
            "Epoch: {}, Elapsed: {:.2f} mins, lr: {:.3f}e-3, Loss: {:.4f}, d_acc: {:.2f}%, e_acc: {:.2f}%, "
            "dEER: {:.2f}%, eEER: {:.2f}%, best_dEER: {:.2f}% from epoch {}.".format(
                epoch,
                elapsed / 60,
                optimizer.param_groups[0]["lr"] * 1000,
                total_loss / counter,
                dev_accuracy * 100,
                eval_accuracy * 100,
                d_eer * 100,
                e_eer * 100,
                best_d_eer[0] * 100,
                int(best_d_eer[1]),
            )
        )
        print(print_str)
        df = pd.DataFrame([print_str])
        df.to_csv(
            log_path + time_name + ".csv", sep=" ", mode="a", header=False, index=False
        )

        scheduler.step()

    f.close()
    plt.plot(torch.log10(loss_per_epoch))
    plt.show()

    print("End of training.")
