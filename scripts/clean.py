import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

for root, dirs, files in os.walk(args.path):
    if os.path.basename(root) == "seed0":
        # Kiểm tra xem đã có file zip chưa để đảm bảo an toàn
        if "output.zip" in files:
            for item in os.listdir(root):
                item_path = os.path.join(root, item)
                
                # Không xóa file output.zip
                if item == "output.zip":
                    continue
                
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            
            print(f"Đã dọn dẹp: {root}")