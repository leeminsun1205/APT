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


def _writable(path: str) -> bool:
    try:
        mkdir_if_missing(path)
        testfile = os.path.join(path, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        return True
    except Exception:
        return False


def _export_cifar_to_folders(ds, out_root: str, split_name: str):
    img_root = os.path.join(out_root, "images")
    mkdir_if_missing(img_root)

    classes = ds.classes
    for c in classes:
        mkdir_if_missing(os.path.join(img_root, c))

    for idx in range(len(ds)):
        img, label = ds[idx]
        cls_name = classes[label]
        out_path = os.path.join(img_root, cls_name, f"{split_name}_{idx:06d}.png")
        if os.path.exists(out_path):
            continue
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        img.save(out_path)


def _collect_items_from_folders(img_root: str, classes: List[str]) -> List[Tuple[str, int, str]]:
    items = []
    for y, cname in enumerate(classes):
        cdir = os.path.join(img_root, cname)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            if fname.startswith("."):
                continue
            impath = os.path.join(cdir, fname)
            items.append((impath, y, cname))
    return items


def _split_train_val(items: List[Tuple[str, int, str]], seed: int, val_ratio: float):
    rnd = random.Random(seed)
    by_class = {}
    for impath, y, cname in items:
        base = os.path.basename(impath)
        if not base.startswith("train_"):
            continue
        by_class.setdefault(y, []).append((impath, y, cname))

    train_list, val_list = [], []
    for y, samples in by_class.items():
        rnd.shuffle(samples)
        n = len(samples)
        n_val = max(1, int(round(n * val_ratio)))
        val_list.extend(samples[:n_val])
        train_list.extend(samples[n_val:])

    train_x = [Datum(impath=p, label=y, classname=c) for (p, y, c) in train_list]
    val = [Datum(impath=p, label=y, classname=c) for (p, y, c) in val_list]
    return train_x, val


class _CIFARBase(DatasetBase):
    dataset_dir = None
    torchvision_name = None
    split_filename = None

    data_temp = "data_temp"

    def __init__(self, cfg):
        assert self.dataset_dir and self.torchvision_name and self.split_filename

        root_cfg = getattr(cfg.DATASET, "ROOT", None)
        if root_cfg and _writable(root_cfg):
            cache_dir = root_cfg
        else:
            cache_dir = os.path.join(self.data_temp, "torchvision")
        mkdir_if_missing(cache_dir)

        out_root = os.path.join(self.data_temp, self.dataset_dir)
        self.image_dir = os.path.join(out_root, "images")
        self.split_path = os.path.join(out_root, self.split_filename)
        self.split_fewshot_dir = os.path.join(self.data_temp, "split_fewshot", self.dataset_dir)
        mkdir_if_missing(self.split_fewshot_dir)
        mkdir_if_missing(out_root)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)

        else:
            if self.torchvision_name == "CIFAR10":
                ds_train = tvds.CIFAR10(cache_dir, train=True, download=True)
                ds_test = tvds.CIFAR10(cache_dir, train=False, download=True)
            else:
                ds_train = tvds.CIFAR100(cache_dir, train=True, download=True)
                ds_test = tvds.CIFAR100(cache_dir, train=False, download=True)

            classes = ds_train.classes
            _export_cifar_to_folders(ds_train, out_root, split_name="train")
            _export_cifar_to_folders(ds_test, out_root, split_name="test")

            all_items = _collect_items_from_folders(self.image_dir, classes)

            seed = cfg.SEED
            val_ratio = float(getattr(cfg.DATASET, "VAL_RATIO", 0.1))
            train_x, val = _split_train_val(all_items, seed=seed, val_ratio=val_ratio)

            test_items = [(p, y, c) for (p, y, c) in all_items if os.path.basename(p).startswith("test_")]
            test = [Datum(impath=p, label=y, classname=c) for (p, y, c) in test_items]

            OxfordPets.save_split(train_x, val, test, self.split_path, self.image_dir)
            train, val, test = train_x, val, test

        num_shots = int(getattr(cfg.DATASET, "NUM_SHOTS", 0))
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"[CIFAR] Loading preprocessed few-shot: {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train_fs = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val_fs = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                print(f"[CIFAR] Saving preprocessed few-shot: {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump({"train": train_fs, "val": val_fs}, f, protocol=pickle.HIGHEST_PROTOCOL)
                train, val = train_fs, val_fs

        subsample = getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all")
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)


@DATASET_REGISTRY.register()
class CIFAR10(_CIFARBase):
    dataset_dir = "cifar10-data"
    torchvision_name = "CIFAR10"
    split_filename = "split_zhou_CIFAR10.json"


@DATASET_REGISTRY.register()
class CIFAR100(_CIFARBase):
    dataset_dir = "cifar100-data"
    torchvision_name = "CIFAR100"
    split_filename = "split_zhou_CIFAR100.json"
