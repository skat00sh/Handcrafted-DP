import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train(model, train_loader, optimizer, n_acc_steps=1):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    bs = (
        train_loader.batch_size
        if train_loader.batch_size is not None
        else train_loader.batch_sampler.batch_size
    )
    print(f"training on {num_batches} batches of size {bs}")

    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()

        if ((batch_idx + 1) % n_acc_steps == 0) or (
            (batch_idx + 1) == len(train_loader)
        ):
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                # accumulate per-example gradients but don't take a step yet
                optimizer.virtual_step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction="sum").item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100.0 * correct / num_examples

    print(
        f"Train set: Average loss: {train_loss:.4f}, "
        f"Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)"
    )

    return train_loss, train_acc


def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100.0 * correct / num_examples

    print(
        f"Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)"
    )

    return test_loss, test_acc


def save_checkpoint(
    state: dict, is_best: bool, checkpoint_path: str, COMMON_DIR_SUFFIX: str
):

    best_model_dir = checkpoint_path + "/best_model_" + COMMON_DIR_SUFFIX + "/"
    checkpoints_dir = checkpoint_path + "/checkpoints_" + COMMON_DIR_SUFFIX + "/"
    if not os.path.exists(best_model_dir):
        os.mkdir(best_model_dir)
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    if is_best:
        checkpoint_name = best_model_dir + "/best_model" + ".pt"
        torch.save(state, checkpoint_name)

    checkpoint_name = (
        checkpoints_dir + "/checkpoint_EPOCH_" + str(state["epoch"]) + ".pt"
    )
    torch.save(state, checkpoint_name)
    # Saving a copy with a different suffix _last so as to allow for resuming from last EPOCH option
    checkpoint_name = checkpoints_dir + "/checkpoint_last.pt"
    torch.save(state, checkpoint_name)


def load_checkpoint(checkpoint_fpath: str, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    if os.path.exists(checkpoint_fpath):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["epoch"]
    else:
        raise FileNotFoundError("No model checkpoint exists at given path")


def save_trained_model(model_save_path, model, COMMON_DIR_SUFFIX):
    trained_model_save_dir = model_save_path + "/trained_models/"
    if not os.path.exists(trained_model_save_dir):
        os.mkdir(trained_model_save_dir)
    trained_model_save_path = (
        trained_model_save_dir + COMMON_DIR_SUFFIX + ".pt"
    )
    print(f"Trained Model Save Path: {trained_model_save_path}")
    torch.save(model, trained_model_save_path)


def plot_learning_curve(
    save_path: str,
    title: str,
    x_label: str,
    epochs: int,
    y_label: str,
    values_1: list,
    values1_title: str,
    value_2: list,
    values2_title: str,
):
    plt.plot([i for i in range(0, epochs)], values_1, "g", label=values1_title)
    plt.plot([i for i in range(0, epochs)], value_2, "b", label=values2_title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)
