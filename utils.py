from torchvision.datasets import CIFAR100
from torchvision import transforms


def load_data(train: bool = True):
    return CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=train
    )


def build_datasets(add_val: bool = False):
    if not add_val:
        return load_data()
    return load_data(), load_data(train=False)


def add_spaces(s: str, length: int = 12):
    return s + max(length - len(s), 0) * ' '
