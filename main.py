import os
from PIL import Image
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.quantization

# Flask App
app = Flask(__name__)

# Dataset Class
class architectureDS(Dataset):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.label_map = {}

        for label in os.listdir(rootdir):
            labeldir = os.path.join(rootdir, label)
            for file in os.listdir(labeldir):
                self.image_files.append(os.path.join(labeldir, file))
                self.labels.append(label)
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)

    def __getitem__(self, i):
        imgname = self.image_files[i]
        image = Image.open(imgname).convert("RGB")
        label = self.label_map[self.labels[i]]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_files)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
rootdir = 'path_to_your_architectural_dataset'
dataset = architectureDS(rootdir=rootdir, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model and Modify
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 25)  # Assume 25 classes
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Quantize Model
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
quantized_model.eval()

# Training Setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training Loop (Simplified)
def train_model(epochs=15):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader)}")

        # Early stopping (if you have a function for it)
        scheduler.step()

# Evaluation Metrics
def evaluate_model():
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Flask API to predict architectural style
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = quantized_model(image)
        _, predicted = torch.max(outputs, 1)

    class_names = ['Achaemenid', 'American Foursquare', 'Art Deco', 'Byzantine', '...']  # List all 25 classes
    return jsonify({"predicted_style": class_names[predicted.item()]})

if __name__ == '__main__':
    # Uncomment if you want to train the model
    train_model()
    
    evaluate_model()
    # Run Flask app
    app.run(debug=True)
