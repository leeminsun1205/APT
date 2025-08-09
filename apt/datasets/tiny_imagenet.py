import os
import pickle
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


def _read_words(words_txt):
    """wnid -> readable name (fallback: wnid)"""
    mapping = {}
    if os.path.isfile(words_txt):
        with open(words_txt, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    wnid, name = parts[0], parts[1]
                    mapping[wnid] = name
    return mapping


def _read_train_items(train_dir, wnid2name):
    """Scan train/<wnid>/images/*.JPEG"""
    items = []
    wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
    for wnid in wnids:
        img_dir = os.path.join(train_dir, wnid, "images")
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            if fn.startswith(".") or not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            impath = os.path.join(img_dir, fn)
            y = wnid_to_label[wnid]
            cname = wnid2name.get(wnid, wnid)
            items.append(Datum(impath=impath, label=y, classname=cname))
    return items, wnid_to_label


def _read_val_items(val_dir, wnid_to_label, wnid2name):
    """Use val/val_annotations.txt to map images in val/images to wnid."""
    anno = os.path.join(val_dir, "val_annotations.txt")
    img_dir = os.path.join(val_dir, "images")
    items = []
    if not os.path.isfile(anno):
        raise FileNotFoundError(f"Missing {anno}")
    with open(anno, "r") as f:
        for line in f:
            # <filename> <wnid> <x1> <y1> <x2> <y2>
            parts = line.strip().split("\t")[0].split()
            if len(parts) < 2:
                continue
            fn, wnid = parts[0], parts[1]
            impath = os.path.join(img_dir, fn)
            if wnid not in wnid_to_label:
                # Some distributions include extra wnids in val; ignore unknowns
                continue
            y = wnid_to_label[wnid]
            cname = wnid2name.get(wnid, wnid)
            items.append(Datum(impath=impath, label=y, classname=cname))
    return items


def _split_train_val(train_items, seed, val_ratio):
    """Stratified split from train_items -> (train_x, val)."""
    rnd = random.Random(seed)
    by_label = defaultdict(list)
    for item in train_items:
        by_label[item.label].append(item)
    train_x, val = [], []
    for y, arr in by_label.items():
        rnd.shuffle(arr)
        n = len(arr)
        n_val = max(1, int(round(n * val_ratio)))
        val.extend(arr[:n_val])
        train_x.extend(arr[n_val:])
    return train_x, val


@DATASET_REGISTRY.register()
class TinyImagenet(DatasetBase):
    """
    Tiny ImageNet-200 as supervised dataset for Dassl:
    - train_x: from train/
    - val: split from train_x by VAL_RATIO
    - test: from val/ (since test has no labels)
    Few-shot/subsample handled like OxfordPets.
    """

    dataset_dir = "tiny-imagenet-200"   # folder name under ROOT
    data_temp = "data_temp"             # writable place for split/fewshot cache

    def __init__(self, cfg):
        # base root (read-only on Kaggle if /kaggle/input)
        root_cfg = "/kaggle/input/"
        self.dataset_dir = os.path.join(root_cfg, self.dataset_dir)

        # where we store split JSON & few-shot cache (writable)
        out_root = os.path.join(self.data_temp, "tiny-imagenet")
        mkdir_if_missing(out_root)
        self.split_path = os.path.join(out_root, "split_zhou_TinyImageNet.json")
        self.split_fewshot_dir = os.path.join(self.data_temp, "split_fewshot", "tiny-imagenet")
        mkdir_if_missing(self.split_fewshot_dir)

        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "val")
        words_txt = os.path.join(self.dataset_dir, "words.txt")

        if os.path.exists(self.split_path):
            # Note: save_split/read_split use path_prefix; here we pass dataset root
            train, val, test = OxfordPets.read_split(self.split_path, path_prefix=self.dataset_dir)
        else:
            wnid2name = _read_words(words_txt)
            # 1) read train set
            train_all, wnid_to_label = _read_train_items(train_dir, wnid2name)
            # 2) read val set (as our test), with labels via val_annotations
            test = _read_val_items(val_dir, wnid_to_label, wnid2name)
            # 3) split train into train_x/val
            seed = cfg.SEED
            val_ratio = float(getattr(cfg.DATASET, "VAL_RATIO", 0.1))
            train_x, val = _split_train_val(train_all, seed, val_ratio)
            # 4) save split (store relative to dataset root)
            OxfordPets.save_split(train_x, val, test, self.split_path, path_prefix=self.dataset_dir)
            train, val = train_x, val

        # few-shot
        num_shots = int(getattr(cfg.DATASET, "NUM_SHOTS", 0))
        if num_shots >= 1:
            seed = cfg.SEED
            cache = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(cache):
                print(f"[TinyIN] Loading few-shot cache: {cache}")
                with open(cache, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                tr_fs = self.generate_fewshot_dataset(train, num_shots=num_shots)
                va_fs = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                with open(cache, "wb") as f:
                    pickle.dump({"train": tr_fs, "val": va_fs}, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[TinyIN] Saved few-shot cache: {cache}")
                train, val = tr_fs, va_fs

        # subsample (all/base/new)
        subsample = getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all")
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
