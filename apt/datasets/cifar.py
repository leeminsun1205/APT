# apt/datasets/cifar.py
import os
import pickle
import random
from typing import List, Tuple

from PIL import Image
from torchvision import datasets as tvds
from torchvision.transforms.functional import to_pil_image

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from .oxford_pets import OxfordPets


def _export(ds, out_root: str, split_name: str):
    img_root = os.path.join(out_root, "images")
    mkdir_if_missing(img_root)
    classes = ds.classes
    for c in classes:
        mkdir_if_missing(os.path.join(img_root, c))
    for idx in range(len(ds)):
        img, y = ds[idx]
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        cname = classes[y]
        outp = os.path.join(img_root, cname, f"{split_name}_{idx:06d}.png")
        if not os.path.exists(outp):
            img.save(outp)


def _collect(img_root: str, classes: List[str]) -> List[Tuple[str, int, str]]:
    items = []
    for y, cname in enumerate(classes):
        cdir = os.path.join(img_root, cname)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.startswith("."):
                continue
            items.append((os.path.join(cdir, fn), y, cname))
    return items


def _split_train_val(all_items, seed: int, val_ratio: float):
    rnd = random.Random(seed)
    byc = {}
    for p, y, c in all_items:
        if os.path.basename(p).startswith("train_"):
            byc.setdefault(y, []).append((p, y, c))
    tr, va = [], []
    for y, arr in byc.items():
        rnd.shuffle(arr)
        n = len(arr)
        nval = max(1, int(round(n * val_ratio)))
        va.extend(arr[:nval])
        tr.extend(arr[nval:])
    train = [Datum(impath=p, label=y, classname=c) for p, y, c in tr]
    val = [Datum(impath=p, label=y, classname=c) for p, y, c in va]
    return train, val


class _CIFARBase(DatasetBase):
    dataset_dir = None
    tv_name = None
    split_filename = None
    data_temp = "data_temp"

    def __init__(self, cfg):
        assert self.dataset_dir and self.tv_name and self.split_filename

        # torchvision cache (đọc/tải), luôn ghi được
        cache_dir = getattr(cfg.DATASET, "ROOT", None) or os.path.join(self.data_temp, "torchvision")
        mkdir_if_missing(cache_dir)

        # nơi export ảnh/split (luôn ghi được)
        out_root = os.path.join(self.data_temp, self.dataset_dir)
        self.image_dir = os.path.join(out_root, "images")
        self.split_path = os.path.join(out_root, self.split_filename)
        self.split_fewshot_dir = os.path.join(self.data_temp, "split_fewshot", self.dataset_dir)
        mkdir_if_missing(out_root)
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            if self.tv_name == "CIFAR10":
                ds_tr = tvds.CIFAR10(cache_dir, train=True, download=True)
                ds_te = tvds.CIFAR10(cache_dir, train=False, download=True)
            else:
                ds_tr = tvds.CIFAR100(cache_dir, train=True, download=True)
                ds_te = tvds.CIFAR100(cache_dir, train=False, download=True)

            classes = ds_tr.classes
            _export(ds_tr, out_root, "train")
            _export(ds_te, out_root, "test")

            all_items = _collect(self.image_dir, classes)
            seed = cfg.SEED
            val_ratio = float(getattr(cfg.DATASET, "VAL_RATIO", 0.1))
            train_x, val = _split_train_val(all_items, seed, val_ratio)
            test_items = [(p, y, c) for (p, y, c) in all_items if os.path.basename(p).startswith("test_")]
            test = [Datum(impath=p, label=y, classname=c) for p, y, c in test_items]

            OxfordPets.save_split(train_x, val, test, self.split_path, self.image_dir)
            train, val, test = train_x, val, test

        # few-shot
        num_shots = int(getattr(cfg.DATASET, "NUM_SHOTS", 0))
        if num_shots >= 1:
            pre = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{cfg.SEED}.pkl")
            if os.path.exists(pre):
                with open(pre, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                tr_fs = self.generate_fewshot_dataset(train, num_shots=num_shots)
                va_fs = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                with open(pre, "wb") as f:
                    pickle.dump({"train": tr_fs, "val": va_fs}, f, protocol=pickle.HIGHEST_PROTOCOL)
                train, val = tr_fs, va_fs

        # subsample
        subsample = getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all")
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)


@DATASET_REGISTRY.register()
class APT_CIFAR10(_CIFARBase):
    dataset_dir = "cifar10-data"
    tv_name = "CIFAR10"
    split_filename = "split_zhou_CIFAR10.json"


@DATASET_REGISTRY.register()
class APT_CIFAR100(_CIFARBase):
    dataset_dir = "cifar100-data"
    tv_name = "CIFAR100"
    split_filename = "split_zhou_CIFAR100.json"
