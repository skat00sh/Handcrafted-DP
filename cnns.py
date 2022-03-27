import argparse
import os

import torch
import torch.nn as nn
from opacus import PrivacyEngine

import train_utils
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
from models import CNNS, get_num_params
from dp_utils import (
    ORDERS,
    get_privacy_spent,
    get_renyi_divergence,
    scatter_normalization,
)
from log import Logger

import time


def main(
    dataset,
    augment=False,
    use_scattering=False,
    size=None,
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
    early_stop=True,
    checkpoint_save_path=None,
    save_checkpoint_per_epoch=None,
    resume_training_from=False,
):
    logdir = logdir + "/" + dataset + "_" + optim + "_epoch" + str(epochs) + "_nm" + str(noise_multiplier) + "_cn" + str(max_grad_norm)
    logger = Logger(logdir)
    device = get_device()

    train_data, test_data = get_data(dataset, augment=augment)

    if use_scattering:
        scattering, K, _ = get_scatter_transform(dataset)
        scattering.to(device)
    else:
        scattering = None
        K = 3 if len(train_data.data.shape) == 4 else 1

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
        model = CNNS[dataset](K, input_norm="BN", bn_stats=bn_stats, size=size)
    else:
        model = CNNS[dataset](
            K, input_norm=input_norm, num_groups=num_groups, size=size
        )

    model.to(device)

    if use_scattering and augment:
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
        # pre-compute the scattering transform if necessery
        train_loader = get_scattered_loader(
            train_loader,
            scattering,
            device,
            drop_last=True,
            sample_batches=sample_batches,
        )
        test_loader = get_scattered_loader(test_loader, scattering, device)

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

    best_acc = 0
    flat_count = 0
    start_epoch = 0
    learning_history = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    COMMON_DIR_SUFFIX = (
        dataset
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
        is_best_model = False
        epoch_start_time = time.time()
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

        # stop if we're not making progress
        if test_acc > best_acc:
            best_acc = test_acc
            is_best_model = True
            flat_count = 0
        else:
            flat_count += 1
            if flat_count >= 20 and early_stop:
                print("plateau...")
                break

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

        print("Time per epoch", time.time() - epoch_start_time)

    save_trained_model(
        checkpoint_save_path, model, dataset, optimizer.__class__.__name__, epochs
    )
    plot_save_path = (
        checkpoint_save_path + "/plots/" + "plot_" + COMMON_DIR_SUFFIX + ".png"
    )
    plot_learning_curve(
        plot_save_path,
        "Training and Validation Accuracy",
        "EPOCHS",
        epochs,
        "Accuracy",
        learning_history["train_accuracy"],
        "Training Accuracy",
        learning_history["val_accuracy"],
        "Validation Accuracy",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "fmnist", "mnist"])
    parser.add_argument("--size", default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--use_scattering", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optim", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--noise_multiplier", type=float, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input_norm", default=None, choices=["GroupNorm", "BN"])
    parser.add_argument("--num_groups", type=int, default=81)
    parser.add_argument("--bn_noise_multiplier", type=float, default=6)
    parser.add_argument("--max_epsilon", type=float, default=None)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--sample_batches", action="store_true")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--checkpoint_save_path", default=None)
    parser.add_argument("--save_checkpoint_per_epoch", default=5)
    parser.add_argument(
        "--resume_training_from", choices=["best_model", "last_epoch"], default=None
    )
    args = parser.parse_args()
    total_time_start = time.time()
    main(**vars(args))
    print(f"Total time for 100 Epochs", time.time() - total_time_start)
