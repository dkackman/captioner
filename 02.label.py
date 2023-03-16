import os
import torch
import csv
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

source_dir = '/mnt/data/photos/prepared/'
csv_file_path = '/mnt/data/photos/labels.csv'

# need 24 GB of GPU memory for this one
xxl = "Salesforce/blip2-flan-t5-xxl"
xl = "Salesforce/blip2-flan-t5-xl"

# change this to swapr models
# in gneral bigger is better, slower, and requires better gpus
model_name = xl

# this controls the number of images sent to the gpu at once
batch_size = 32

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == xxl:
    # use 8 bit inference if the model is xxl
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", load_in_8bit=True)
else:
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16).to(device)

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
    source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
total_files = len(file_paths)
file_counter = 0
processed_counter = 0

# Process the files in pairs and store the results in a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['file_path', 'label_name'])

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
