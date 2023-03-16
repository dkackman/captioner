import os
import torch
import torchvision.models as models

# Load the pre-trained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# If you have a GPU, move the model to the GPU for faster processing
if torch.cuda.is_available():
    resnet50 = resnet50.cuda()


def predict_label(input_tensor):
    input_batch = input_tensor.unsqueeze(0)

    # Move the input data to the GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()

    # Predict the class probabilities
    with torch.no_grad():
        output = resnet50(input_batch)

    # Get the predicted class index
    _, predicted_class_idx = torch.max(output, 1)

    # Convert the class index to the corresponding label
    label = str(predicted_class_idx.item())
    return label


# Set the path to the directory with your preprocessed family photos
photos_directory = '/mnt/data/photos/prepared/'

# Iterate through your preprocessed family photos and predict their labels
for root, _, files in os.walk(photos_directory):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            input_tensor = torch.load(file_path)
            label = predict_label(input_tensor)
            print(f"Predicted label for {file}: {label}")
