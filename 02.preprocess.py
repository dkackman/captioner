import os
import uuid
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys

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
file_counter = 0
processed_counter = 0

# Create the destination directory if it does not exist
os.makedirs(destination_dir, exist_ok=True)

for root, _, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_counter += 1
            file_path = os.path.join(root, file)
            try:
                image = Image.open(file_path).convert('RGB')
                # Transform the image to a tensor
                input_tensor = transform(image)
                destination_file = os.path.join(
                    destination_dir, f"{uuid.uuid4()}.pt")
                # Save the tensor to disk
                torch.save(input_tensor, destination_file)
                processed_counter += 1
                sys.stdout.write(
                    f"\rProcessing file {processed_counter} of {file_counter}")
            except Exception as e:
                pass

print(f"\nTotal files processed: {processed_counter}")
print(f"Total files skipped: {file_counter - processed_counter}")
