import os
import hashlib
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

source_dir = '/mnt/data/photos/original/'
destination_dir = '/mnt/data/photos/prepared/'


def process_image(file_path):
    try:
        with Image.open(file_path) as img:
            # Check file size (width * height * 3 for RGB) and ignore files smaller than 400 KB
            if img.size[0] * img.size[1] * 3 >= 400 * 1024:
                img = img.convert('RGB')

                # Calculate the file hash
                file_hash = hashlib.md5(img.tobytes()).hexdigest()

                # Create the destination file path using the hash
                destination_file = os.path.join(
                    destination_dir, f"{file_hash}.jpg")

                # Check if the file already exists in the destination directory
                if not os.path.exists(destination_file):
                    img.save(destination_file)
                    return 'processed'

                return 'duplicate'

            return 'ignored'
    except Exception:
        return 'corrupt'


# Create the destination directory if it does not exist
os.makedirs(destination_dir, exist_ok=True)

# Calculate the total number of files to be processed
file_paths = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

total_files = len(file_paths)
file_counter = 0
processed_counter = 0
ignored_counter = 0
corrupt_counter = 0
duplicate_counter = 0

# Process the files using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    for file_path in file_paths:
        file_counter += 1
        result = executor.submit(process_image, file_path).result()

        if result == 'processed':
            processed_counter += 1
        elif result == 'ignored':
            ignored_counter += 1
        elif result == 'corrupt':
            corrupt_counter += 1
        elif result == 'duplicate':
            duplicate_counter += 1
        print(
            f"Processed {processed_counter} of {total_files} files", end='\r')

print(f"\nTotal files processed: {processed_counter}")
print(f"Total files ignored: {ignored_counter}")
print(f"Total files corrupt: {corrupt_counter}")
print(f"Total files duplicate: {duplicate_counter}")
