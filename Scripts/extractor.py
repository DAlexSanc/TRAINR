import os
import shutil

# 🔧 CHANGE THIS to your dataset main folder
ROOT_DIR = r"D:\SWT_RAWData\Basic Model"

def get_unique_path(dest_path):
    """Avoid overwriting files with same name"""
    base, ext = os.path.splitext(dest_path)
    counter = 1
    new_path = dest_path

    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1

    return new_path


def flatten_dataset(root_dir):
    moved_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root folder itself
        if dirpath == root_dir:
            continue

        for file in filenames:
            src_path = os.path.join(dirpath, file)
            dest_path = os.path.join(root_dir, file)

            dest_path = get_unique_path(dest_path)

            try:
                shutil.move(src_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {src_path}: {e}")

    print(f"\n✅ Done. {moved_count} files moved to main folder.")

if __name__ == "__main__":
    flatten_dataset(ROOT_DIR)
