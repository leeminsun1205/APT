# apt/datasets/cifar.py
import os
import pickle
import random
from typing import List, Tuple
import numpy as np

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


@DATASET_REGISTRY.register()
class APT_CIFAR10_1(_CIFARBase):
    dataset_dir = "cifar10.1"

    def __init__(self, cfg):
        # Không gọi super().__init__ cũ vì logic khác hoàn toàn (chỉ có test set)
        
        # 1. Setup paths
        self.dataset_dir = "cifar10.1"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir_path = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir_path, "images")
        self.db_dir = os.path.join(self.dataset_dir_path, "databases")
        
        # 2. Check and Extract if needed
        self._check_and_download()
            
        # 3. Create splits
        # CIFAR10.1 chỉ dùng để test/eval OOD cho CIFAR10 model
        # Nhưng DataManager yêu cầu train không rỗng, nên ta dùng vài sample đầu làm dummy
        test = self._read_data()
        
        # Use first 10 samples as dummy train/val to satisfy DataManager
        train = test[:5] if len(test) >= 5 else test[:1]
        val = test[5:10] if len(test) >= 10 else test[:1]

        super(DatasetBase, self).__init__() # Gọi grandparent init để setup transforms cơ bản nếu có
        
        self._train_x = train
        self._train_u = []  # No unlabeled data for CIFAR10.1
        self._val = val
        self._test = test
        self._num_classes = 10
        self._lab2cname = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
        self._classnames = [self._lab2cname[i] for i in range(10)]

    def _check_and_download(self):
        mkdir_if_missing(self.db_dir)
        
        # URLs for v6 dataset
        urls = {
            "cifar10.1_v6_data.npy": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
            "cifar10.1_v6_labels.npy": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy"
        }
        
        # Check download
        for filename, url in urls.items():
            fpath = os.path.join(self.db_dir, filename)
            if not os.path.exists(fpath):
                print(f"Downloading {filename} from {url}...")
                try:
                    self._download_file(url, fpath)
                    print(f"Successfully downloaded {filename}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download {filename}: {e}")

        # Check extraction
        if not os.path.exists(self.image_dir):
            print(f"Extracting CIFAR10.1 images to {self.image_dir}...")
            self._extract_images()

    def _download_file(self, url, fpath):
        import requests
        # Use requests for better handling
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(fpath, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
        else:
            raise RuntimeError(f"HTTP Error {r.status_code} for {url}")

    def _extract_images(self):
        data_path = os.path.join(self.db_dir, "cifar10.1_v6_data.npy")
        label_path = os.path.join(self.db_dir, "cifar10.1_v6_labels.npy")
        
        if not os.path.exists(data_path) or not os.path.exists(label_path):
             # Should be caught by _check_and_download, but as a safeguard
             raise FileNotFoundError(f"Missing data files in {self.db_dir}")

        data = np.load(data_path) # (2000, 32, 32, 3)
        labels = np.load(label_path) # (2000,)

        mkdir_if_missing(self.image_dir)
        
        # Define class names mapping (CIFAR10 order)
        # 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 
        # 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
        cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                           "dog", "frog", "horse", "ship", "truck"]

        for cname in cifar10_classes:
            mkdir_if_missing(os.path.join(self.image_dir, cname))

        for i in range(len(data)):
            img_arr = data[i]
            label = int(labels[i])
            cname = cifar10_classes[label]
            
            im = Image.fromarray(img_arr)
            # Filename: index_label.png
            save_path = os.path.join(self.image_dir, cname, f"{i:05d}.png")
            im.save(save_path)

    def _read_data(self):
        items = []
        cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                           "dog", "frog", "horse", "ship", "truck"]
        
        for label, cname in enumerate(cifar10_classes):
            class_dir = os.path.join(self.image_dir, cname)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                impath = os.path.join(class_dir, img_name)
                item = Datum(impath=impath, label=label, classname=cname)
                items.append(item)
        
        return items
