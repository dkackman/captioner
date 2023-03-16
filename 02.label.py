import os
import torch
import csv
import torch

from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration


source_dir = '/mnt/data/photos/prepared/'
csv_file_path = '/mnt/data/photos/labels.csv'
model_name = "Salesforce/blip2-flan-t5-xl"

print("Loading model..")
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Blip2ForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16)
model.to(device)

processor = AutoProcessor.from_pretrained(
    model_name)


def get_image_labels(file_path):
    try:
        image = Image.open(file_path)

        inputs = processor(images=image, return_tensors="pt").to(
            device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    except Exception:
        return None


# Calculate the total number of files to be processed
file_paths = [os.path.join(source_dir, f) for f in os.listdir(
    source_dir) if f.lower().endswith('.jpg')]
total_files = len(file_paths)
file_counter = 0
processed_counter = 0

# Process the files sequentially and store the results in a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['file_path', 'caption'])

    for file_path in file_paths:
        file_counter += 1
        top_label_name = get_image_labels(file_path)

        if top_label_name is not None:
            processed_counter += 1
            csv_writer.writerow([file_path, top_label_name])
        print(
            f"Processed {processed_counter} of {total_files} files", end='\r')

print(f"\nTotal files processed: {processed_counter}")
