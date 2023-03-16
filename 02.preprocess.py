import os
import uuid
from PIL import Image
import torchvision.transforms as transforms
import torch
from concurrent.futures import ThreadPoolExecutor

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

source_dir = '/mnt/data/photos/prepared/'
destination_dir = '/mnt/data/photos/preprocessed/'


def preprocess_image(file_path):
    try:
        image = Image.open(file_path).convert('RGB')
        input_tensor = transform(image)  # Transform the image to a tensor
        destination_file = os.path.join(destination_dir, f"{uuid.uuid4()}.pt")
        torch.save(input_tensor, destination_file)  # Save the tensor to disk
        return 'processed'
    except Exception:
        return 'corrupt'


# Create the destination directory if it does not exist
os.makedirs(destination_dir, exist_ok=True)

# Calculate the total number of files to be processed
file_paths = [os.path.join(source_dir, f) for f in os.listdir(
    source_dir) if f.lower().endswith('.jpg')]
total_files = len(file_paths)
file_counter = 0
processed_counter = 0
corrupt_counter = 0

# Process the files using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    for file_path in file_paths:
        file_counter += 1
        result = executor.submit(preprocess_image, file_path).result()

        if result == 'processed':
            processed_counter += 1
        elif result == 'corrupt':
            corrupt_counter += 1
        print(
            f"Processed {processed_counter} of {total_files} files", end='\r')

print(f"\nTotal files processed: {processed_counter}")
print(f"Total files corrupt: {corrupt_counter}")
