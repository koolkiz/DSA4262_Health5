import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

def generate_embeddings(image_name):
    # Load Pretrained ResNet-50 Model
    model = models.resnet50(pretrained=True)
    model.eval()  # Set model to evaluation mode

    # Remove the last fully connected layer to get feature embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer

    # Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and Preprocess Image
    image_name = image_name.strip()
    image_path = f"../../data/images/all/{image_name}.jpg"  # Change to your image path
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Generate Image Embedding
    with torch.no_grad():
        embedding = model(image)

    # Flatten the feature vector
    embedding = embedding.view(embedding.size(0), -1)

    return embedding.flatten()
# Read the CSV file
import pandas as pd
import numpy as np
csv_path = '../../data/clinical_data.csv'
df = pd.read_csv(csv_path)

# Filter to only keep rows with "CM" and "MLO" in image_name
df = df[df['Image_name'].str.contains('CM') & df['Image_name'].str.contains('MLO')]
df['Pathology Classification/ Follow up'] = df['Pathology Classification/ Follow up'].replace({'Benign, Normal': 'Benign'})
df['Pathology Classification/ Follow up'] = df['Pathology Classification/ Follow up'].replace({'Malignant, Normal': 'Malignant'})
df_filtered = df[["Image_name", "Pathology Classification/ Follow up"]]
df_filtered["resnet50_embedding"] = df_filtered["Image_name"].apply(generate_embeddings)
df_filtered.to_csv("../../data/features/images/resnet50.csv", index=False)  # Saves without the index column
