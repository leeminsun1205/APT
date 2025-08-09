from torchvision import datasets
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os

def export(ds, out_root, split):
    classes = ds.classes
    for c in classes:
        os.makedirs(os.path.join(out_root, "images", c), exist_ok=True)
    for i in range(len(ds)):
        img, y = ds[i]
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        cls = classes[y]
        outp = os.path.join(out_root, "images", cls, f"{split}_{i:06d}.png")
        if not os.path.exists(outp):
            img.save(outp)

root_torch = "/kaggle/working/data_temp/torchvision"
out10 = "/kaggle/working/data_temp/cifar10-data"
out100 = "/kaggle/working/data_temp/cifar100-data"

os.makedirs(root_torch, exist_ok=True)

print("📦 Exporting CIFAR10...")
export(datasets.CIFAR10(root_torch, train=True, download=True),  out10,  "train")
export(datasets.CIFAR10(root_torch, train=False, download=True), out10,  "test")

print("📦 Exporting CIFAR100...")
export(datasets.CIFAR100(root_torch, train=True, download=True),  out100, "train")
export(datasets.CIFAR100(root_torch, train=False, download=True), out100, "test")

print("✅ Done. CIFAR10 at", out10)
print("✅ Done. CIFAR100 at", out100)
