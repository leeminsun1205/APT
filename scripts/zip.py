import os
import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

for root, dirs, files in os.walk(args.path):
    if os.path.basename(root) == "seed0" and "cfg.yaml" in files:
        cfg_path = os.path.join(root, "cfg.yaml")
        output_dir = ""

        with open(cfg_path, 'r') as f:
            for line in f:
                if "OUTPUT_DIR:" in line:
                    output_dir = line.split("OUTPUT_DIR:")[1].strip().strip("'").strip('"')
                    break
        
        if not output_dir:
            continue

        zip_path = os.path.join(root, "output.zip")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for sub_root, sub_dirs, sub_files in os.walk(root):
                for file in sub_files:
                    if file == "output.zip":
                        continue
                    full_path = os.path.join(sub_root, file)
                    rel_path_in_seed0 = os.path.relpath(full_path, root)
                    arcname = os.path.join(output_dir, rel_path_in_seed0)
                    zipf.write(full_path, arcname)
        
        print(f"Success: {zip_path}")