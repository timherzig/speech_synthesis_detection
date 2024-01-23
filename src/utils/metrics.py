import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt


def asv_cal_accuracies(net, device, data_loader):
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        softmax_acc = 0
        num_files = 0
        probs = torch.empty(0, 3).to(device)
        sub_classes = torch.empty(0).to(device)

        for batch in data_loader:
            # load batch and infer
            sample, label, sub_class = batch

            num_files += len(label)
            sample = sample.to(device)
            label = label.to(device)
            sub_class = sub_class.to(device)
            zeroes_label = torch.zeros_like(label).to(device)
            infer, _ = net((sample, zeroes_label))

            # obtain output probabilities
            t1 = F.softmax(infer, dim=1)
            t2 = label.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)
            sub_classes = torch.cat((sub_classes, sub_class), dim=0)

            # calculate example level accuracy
            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(label).sum().item()
            softmax_acc += batch_acc

        softmax_acc = softmax_acc / num_files

    return softmax_acc, probs.to(device), sub_classes.to(device)


def cal_roc_eer(probs, show_plot=True):
    """
    probs: tensor, number of samples * 3, containing softmax probabilities
    row wise: [genuine prob, fake prob, label]
    TP: True Fake
    FP: False Fake
    """
    all_labels = probs[:, 2]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
    one_index = torch.nonzero(all_labels).squeeze(-1)
    zero_probs = probs[zero_index, 0]
    one_probs = probs[one_index, 0]

    threshold_index = torch.linspace(-0.1, 1.01, 10000)
    tpr = torch.zeros(
        len(threshold_index),
    )
    fpr = torch.zeros(
        len(threshold_index),
    )
    cnt = 0
    for i in threshold_index:
        # TODO: Ask Arnab about this, what to do if there are no samples for respective class (zero division)
        tpr[cnt] = (
            one_probs.le(i).sum().item() / len(one_probs) if len(one_probs) > 0 else 1
        )
        fpr[cnt] = (
            zero_probs.le(i).sum().item() / len(zero_probs)
            if len(zero_probs) > 0
            else 1
        )
        cnt += 1

    sum_rate = tpr + fpr
    distance_to_one = torch.abs(sum_rate - 1)
    eer_index = distance_to_one.argmin(dim=0).item()
    out_eer = 0.5 * (fpr[eer_index] + 1 - tpr[eer_index]).numpy()

    if show_plot:
        print("EER: {:.4f}%.".format(out_eer * 100))
        plt.figure(1)
        plt.plot(
            torch.linspace(-0.2, 1.2, 1000),
            torch.histc(zero_probs, bins=1000, min=-0.2, max=1.2) / len(zero_probs),
        )
        plt.plot(
            torch.linspace(-0.2, 1.2, 1000),
            torch.histc(one_probs, bins=1000, min=-0.2, max=1.2) / len(one_probs),
        )
        plt.xlabel("Probability of 'Genuine'")
        plt.ylabel("Per Class Ratio")
        plt.legend(["Real", "Fake"])
        plt.grid()

        plt.figure(3)
        plt.scatter(fpr, tpr)
        plt.xlabel("False Positive (Fake) Rate")
        plt.ylabel("True Positive (Fake) Rate")
        plt.grid()
        plt.show()

    return out_eer


def cal_roc_eer_sub_class(probs, sub_classes, show_plot=True):
    assert probs.shape[0] == sub_classes.shape[0], "Number of samples not equal."

    for sub_class in torch.unique(sub_classes):
        if sub_class == 0.0 or sub_class == -1.0:
            print("Skipping sub class 0")
            continue
        sub_class_index = torch.nonzero(sub_classes == sub_class).squeeze(-1)
        zero_class_index = torch.nonzero(sub_classes == 0.0).squeeze(-1)
        neg_class_index = torch.nonzero(sub_classes == -1.0).squeeze(-1)
        index = torch.cat((sub_class_index, zero_class_index, neg_class_index), dim=0)
        sub_class_probs = probs[index, :]
        eer = cal_roc_eer(sub_class_probs, show_plot=show_plot)
        print(f"Sub Class {sub_class} EER: {eer*100:.4f}")
