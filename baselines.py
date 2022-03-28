import argparse
import os

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from sklearn.linear_model import LogisticRegression

from train_utils import (
    get_device,
    train,
    test,
    save_checkpoint,
    load_checkpoint,
    save_trained_model,
    plot_learning_curve,
)
from data import get_data, get_scatter_transform, get_scattered_loader
from models import ScatterLinear, get_num_params
from dp_utils import (
    ORDERS,
    get_privacy_spent,
    get_renyi_divergence,
    scatter_normalization,
)
from log import Logger


def main(
    dataset,
    augment=False,
    batch_size=2048,
    mini_batch_size=256,
    sample_batches=False,
    lr=1,
    optim="SGD",
    momentum=0.9,
    nesterov=False,
    noise_multiplier=1,
    max_grad_norm=0.1,
    epochs=100,
    input_norm=None,
    num_groups=None,
    bn_noise_multiplier=None,
    max_epsilon=None,
    logdir=None,
    checkpoint_save_path=None,
    save_checkpoint_per_epoch=None,
    resume_training_from=False,
):

    logdir = logdir + "/baseline_" + dataset + "_" + optim + "_epoch" + str(epochs) + "_nm" + str(noise_multiplier) + "_cn" + str(max_grad_norm)
    logger = Logger(logdir)
    device = get_device()

    train_data, test_data = get_data(dataset, augment=augment)
    scattering, K, (h, w) = get_scatter_transform(dataset)
    scattering.to(device)

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size

    # Batch accumulation and data augmentation with Poisson sampling isn't implemented
    if sample_batches:
        assert n_acc_steps == 1
        assert not augment

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=mini_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    rdp_norm = 0
    if input_norm == "BN":
        # compute noisy data statistics or load from disk if pre-computed
        save_dir = f"bn_stats/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = scatter_normalization(
            train_loader,
            scattering,
            K,
            device,
            len(train_data),
            len(train_data),
            noise_multiplier=bn_noise_multiplier,
            orders=ORDERS,
            save_dir=save_dir,
        )
        model = ScatterLinear(K, (h, w), input_norm="BN", bn_stats=bn_stats)
    else:
        model = ScatterLinear(K, (h, w), input_norm=input_norm, num_groups=num_groups)

    model.to(device)

    if augment:
        model = nn.Sequential(scattering, model)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=mini_batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )
    else:
        # if there is no data augmentation, pre-compute the scattering transform
        train_loader = get_scattered_loader(
            train_loader,
            scattering,
            device,
            drop_last=True,
            sample_batches=sample_batches,
        )
        test_loader = get_scattered_loader(test_loader, scattering, device)

    # baseline Logistic Regression without privacy
    if optim == "LR":
        assert not augment
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for data, target in train_loader:
            with torch.no_grad():
                data = data.to(device)
                X_train.append(data.cpu().numpy().reshape(len(data), -1))
                y_train.extend(target.cpu().numpy())

        for data, target in test_loader:
            with torch.no_grad():
                data = data.to(device)
                X_test.append(data.cpu().numpy().reshape(len(data), -1))
                y_test.extend(target.cpu().numpy())

        import numpy as np

        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        for idx, C in enumerate([0.01, 0.1, 1.0, 10, 100]):
            clf = LogisticRegression(C=C, fit_intercept=True)
            clf.fit(X_train, y_train)

            train_acc = 100 * clf.score(X_train, y_train)
            test_acc = 100 * clf.score(X_test, y_test)
            print(
                f"C={C}, "
                f"Acc train = {train_acc: .2f}, "
                f"Acc test = {test_acc: .2f}"
            )

            logger.log_epoch(idx, 0, train_acc, 0, test_acc, None)
        return

    print(f"model has {get_num_params(model)} parameters")

    if optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine(
        model,
        sample_rate=bs / len(train_data),
        alphas=ORDERS,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)

    start_epoch = 0
    learning_history = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    COMMON_DIR_SUFFIX = (
        "baseline_"
        +dataset
        + "_"
        + optimizer.__class__.__name__
        + "_epoch"
        + str(epochs)
        + "_nm"
        + str(noise_multiplier)
        + "_cn"
        + str(max_grad_norm)
    )

    if resume_training_from == "best_model":
        best_model_dir = (
            checkpoint_save_path
            + "/best_model_"
            + COMMON_DIR_SUFFIX
            + "/"
            + "best_model.pt"
        )
        model, optimizer, start_epoch = load_checkpoint(
            best_model_dir, model, optimizer
        )

    elif resume_training_from == "last_epoch":
        checkpoints_dir = (
            checkpoint_save_path
            + "/checkpoints_"
            + COMMON_DIR_SUFFIX
            + "/"
            + "checkpoint_last.pt"
        )
        model, optimizer, start_epoch = load_checkpoint(
            checkpoints_dir, model, optimizer
        )
    for epoch in range(start_epoch, epochs):
        #Note: Not sure if we need the best_model thing here but keeping this flag var here for now to be able to add in future if needed
        is_best_model = False
        print(f"\nEpoch: {epoch} of {epochs}")
        train_loss, train_acc = train(
            model, train_loader, optimizer, n_acc_steps=n_acc_steps
        )
        test_loss, test_acc = test(model, test_loader)
        # Storing all losses and accuracies for plotting
        learning_history["train_losses"].append(train_loss)
        learning_history["train_accuracy"].append(train_acc)
        learning_history["val_losses"].append(test_loss)
        learning_history["val_accuracy"].append(test_acc)

        if noise_multiplier > 0:
            rdp_sgd = (
                get_renyi_divergence(
                    privacy_engine.sample_rate, privacy_engine.noise_multiplier
                )
                * privacy_engine.steps
            )
            epsilon, _ = get_privacy_spent(rdp_norm + rdp_sgd)
            epsilon2, _ = get_privacy_spent(rdp_sgd)
            print(f"ε = {epsilon:.3f} (sgd only: ε = {epsilon2:.3f})")

            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None

        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, epsilon)
        logger.log_scalar("epsilon/train", epsilon, epoch)

        if (epoch + 1) % int(save_checkpoint_per_epoch) == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(
                checkpoint, is_best_model, checkpoint_save_path, COMMON_DIR_SUFFIX
            )

    save_trained_model(
        checkpoint_save_path, model, COMMON_DIR_SUFFIX
    )
    plot_save_path = (
        checkpoint_save_path + "/plots/" + "baseline_plot_" + COMMON_DIR_SUFFIX + ".png"
    )
    title = "Baseline: "+ dataset + " Sigma: " + str(noise_multiplier) + " CN: " + str(max_grad_norm)

    plot_learning_curve(
        plot_save_path,
        title,
        "EPOCHS",
        len(learning_history["train_accuracy"]),
        "Accuracy",
        learning_history["train_accuracy"],
        "Training Accuracy",
        learning_history["val_accuracy"],
        "Validation Accuracy",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "fmnist", "mnist"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--optim", type=str, default="SGD", choices=["SGD", "Adam", "LR"]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--noise_multiplier", type=float, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input_norm", default=None, choices=["GroupNorm", "BN"])
    parser.add_argument("--num_groups", type=int, default=81)
    parser.add_argument("--bn_noise_multiplier", type=float, default=6)
    parser.add_argument("--max_epsilon", type=float, default=None)
    parser.add_argument("--sample_batches", action="store_true")
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--checkpoint_save_path", default=None)
    parser.add_argument("--save_checkpoint_per_epoch", default=5)
    parser.add_argument(
        "--resume_training_from", choices=["best_model", "last_epoch"], default=None
    )
    args = parser.parse_args()
    main(**vars(args))
