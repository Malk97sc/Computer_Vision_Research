import os
import shutil

from mapping import SHEET_CLASSES

def clean_dataset(raw_path, out_path_dir):
    for cls, info in SHEET_CLASSES.items():
        r0, r1 = info["range"]

        out_path = os.path.join(out_path_dir, cls)
        os.makedirs(out_path, exist_ok=True)

        print(f"Cleaning {cls} (Label: {info["label"]})")

        for i in range(r0, r1 + 1):
            fname = f"{i}.jpg"
            src = os.path.join(raw_path, fname)
            dst = os.path.join(out_path, fname)
            #print(f"SRC: {src}, TO: {dst}")

            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"WARNING {src}")

    print("Dataset clean ready to use")


def run():
    RAW_DIR = "raw/"
    OUT_DIR = "data/"
    clean_dataset(RAW_DIR, OUT_DIR)

if __name__ == "__main__":
    run()