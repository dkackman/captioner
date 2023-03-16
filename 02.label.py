import os
import torch
import csv
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

source_dir = '/mnt/data/photos/prepared/'
csv_file_path = '/mnt/data/photos/labels.csv'

model_name = "Salesforce/blip2-flan-t5-xxl"

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Blip2ForConditionalGeneration.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True)  # .to(device)
processor = AutoProcessor.from_pretrained(model_name)


def get_image_labels(file_paths):
    try:
        images = [Image.open(file_path) for file_path in file_paths]
        inputs = processor(images=images, return_tensors="pt").to(
            device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True)
        return [text.strip() for text in generated_texts]
    except Exception:
        return None


# Calculate the total number of files to be processed
file_paths = [os.path.join(source_dir, f) for f in os.listdir(
    source_dir) if f.lower().endswith('.jpg')]
total_files = len(file_paths)
file_counter = 0
processed_counter = 0

# Process the files in pairs and store the results in a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['file_path', 'label_name'])

    batch_size = 32
    for i in range(0, total_files, batch_size):
        batch_file_paths = file_paths[i:i + batch_size]
        label_names = get_image_labels(batch_file_paths)

        if label_names is not None:
            for file_path, label_name in zip(batch_file_paths, label_names):
                processed_counter += 1
                csv_writer.writerow([file_path, label_name])
        print(
            f"Processed {processed_counter} of {total_files} files", end='\r')

print(f"\nTotal files processed: {processed_counter}")
