# datasets/cifar.py
from typing import Tuple, List
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.datasets import CIFAR10, CIFAR100

def load_cifar(
    dataset_name: str,
    processor,                   # transform/processor đã chuẩn bị sẵn (Compose hoặc callable)
    batch_size: int = 100,
    num_workers: int = 4,
    root: str = "./data",
):
    """
    Trả về (loader, classes, num_classes) cho Cifar10/Cifar100.
    Không dùng lambda/closure để tránh lỗi pickling khi num_workers>0.
    """
    assert dataset_name in ["Cifar10", "Cifar100"], "dataset_name phải là 'Cifar10' hoặc 'Cifar100'"

    if dataset_name == "Cifar10":
        ds = CIFAR10(root=root, transform=processor, train=False, download=True)
        # Giữ nguyên danh sách classes như file gốc của bạn (dạng số nhiều).
        classes = [
            "airplanes",
            "cars",
            "birds",
            "cats",
            "deers",
            "dogs",
            "frogs",
            "horses",
            "ships",
            "trucks",
        ]
        num_classes = 10
    else:  # "Cifar100"
        ds = CIFAR100(root=root, transform=processor, train=False, download=True)
        # Với CIFAR100, dùng chính labels chuẩn đi kèm dataset (không tự liệt kê 100 lớp).
        classes = list(ds.classes)
        num_classes = 100

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=SequentialSampler(ds),
    )
    return loader, classes, num_classes
