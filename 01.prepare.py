import os
import shutil
from pathlib import Path
from PIL import Image
import uuid
import sys


def convert_to_jpeg(src, dest):
    try:
        img = Image.open(src)
        img.save(dest, 'JPEG')
        return True
    except Exception:
        return False


def process_images(src_dir, dest_dir):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    ignored_count = 0
    corrupt_count = 0
    file_count = 0

    total_files = sum(len(files) for _, _, files in os.walk(src_path))

    for root, dirs, files in os.walk(src_path):
        for file in files:
            src_file = Path(root) / file
            if src_file.stat().st_size < 1024 * 400:  # Ignore files smaller than 400 KB
                ignored_count += 1
                continue

            dest_file = dest_path / f"{uuid.uuid4()}.jpg"

            if convert_to_jpeg(src_file, dest_file):
                copied_count += 1
            else:
                corrupt_count += 1

            file_count += 1
            sys.stdout.write(
                f"\rProcessing file {file_count} of {total_files}")
            sys.stdout.flush()

    return copied_count, ignored_count, corrupt_count


if __name__ == "__main__":
    source_dir = "/mnt/data/photos/original"
    destination_dir = "/mnt/data/photos/prepared"

    copied, ignored, corrupt = process_images(source_dir, destination_dir)

    print(f"\nNumber of files copied: {copied}")
    print(f"Number of files ignored (size < 400 KB): {ignored}")
    print(f"Number of corrupt files: {corrupt}")
