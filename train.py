import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100

from metric_accumulation import accuracy
from utils import build_datasets
from syncbn import SyncBatchNorm

torch.set_num_threads(1)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self, norm_type: str = "custom"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        if norm_type == "custom":
            self.bn1 = nn.SyncBatchNorm(128, affine=False)  # to be replaced with SyncBatchNorm
        elif norm_type == "lib":
            self.bn1 = SyncBatchNorm(128)
        else:
            raise ValueError(f"Unexpected norm type = {norm_type}")

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size, args):
    torch.manual_seed(0)

    dataset, val_dataset = build_datasets(add_val=True)

    loader = DataLoader(dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=64)

    model = Net(args.norm_type)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)
    acc_train, acc_val = 0, 0
    for _ in range(args.n_epoch):
        epoch_loss = torch.zeros((1,), device=device)

        model.train()
        train_accs = []
        for idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()
            if (idx - 1) % args.grad_accum == 0 or idx == len(loader) - 1:
                average_gradients(model)
                optimizer.step()

            acc = (output.argmax(dim=1) == target).float().mean().item()
            train_accs.append(acc)

            if rank == 0:
                print(f"Rank {dist.get_rank()}, loss: {epoch_loss.item() / num_batches}, acc: {acc}")
            epoch_loss = 0

        acc_train = sum(train_accs) / len(loader)

        if args.run_val:
            acc_val = accuracy(rank, size, model, val_dataset, device)
            if rank == 0:
                print(f"Rank {dist.get_rank()}, train_acc: , val_acc: {acc_val}")

    if args.run_val:
        return acc_train, torch.cuda.max_memory_allocated()

    return acc_train, acc_val, torch.cuda.max_memory_allocated()
