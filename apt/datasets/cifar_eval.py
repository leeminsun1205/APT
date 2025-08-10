# datasets/cifar.py
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

def load_cifar(dataset_name: str, processor, batch_size: int, num_workers: int = 4):
    
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        dataset_class = CIFAR10
    elif dataset_name == 'cifar100':
        dataset_class = CIFAR100
    else:
        raise ValueError("Tên bộ dữ liệu không hợp lệ. Vui lòng chọn 'cifar10' hoặc 'cifar100'.")

    # Tải tập dữ liệu test
    testset = dataset_class(
        root='./data', 
        train=False, 
        download=True, 
        transform=processor
    )
    
    # Lấy thông tin về lớp từ chính bộ dữ liệu
    classes = testset.classes
    num_classes = len(classes)

    # Tạo DataLoader
    loader = DataLoader(
        testset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=SequentialSampler(testset)
    )

    return loader, classes, num_classes