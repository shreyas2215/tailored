import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms as torchvision_transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify

# Initialize global variables
model = None
train_loader = None
test_loader = None
class_names = None
initialized = False  # Initialization flag

# Data Augmentation
transforms = torchvision_transforms.Compose([
    torchvision_transforms.Resize((224, 224)),
    torchvision_transforms.RandomHorizontalFlip(),
    torchvision_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    torchvision_transforms.ToTensor(),
    torchvision_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset class for the architecture dataset
class ArchitectureDataset(Dataset):
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

# Function to initialize dataset and model
def initialize(rootdir):
    global model, train_loader, test_loader, class_names, initialized

    if initialized:
        print("Already initialized.")
        return  # Skip reinitialization if already done

    # Prepare the dataset
    full_dataset = ArchitectureDataset(rootdir=rootdir, transform=transforms)

    # Split dataset into training and testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Prepare class names
    class_names = list(full_dataset.label_map.keys())

    # Model Selection and Modification
    model = models.resnet50(pretrained=True)
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda()  # Move model to GPU

    initialized = True  # Mark as initialized
    print("Initialization complete.")

def save_model(model, path='your_model.pth'):
    torch.save(model.state_dict(), path)
    print("Model saved successfully.")

# Training function
def train_model(model, train_loader, num_epochs=5):
    model.train()  # Set the model to training mode
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    save_model(model, 'your_model.pth')

# Evaluation function
def evaluate_model(model, test_loader):
    if model is None or test_loader is None:
        print("Model or test_loader not initialized. Please call initialize() first.")
        return

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

# Flask app for serving the model
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(file).convert("RGB")
    image = transforms(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image.cuda())  # Use the GPU for prediction
        _, predicted = torch.max(outputs, 1)

    return jsonify({"predicted_style": class_names[predicted.item()]})

# Main execution
if __name__ == "__main__":
    rootdir = 'path_to_your_architectural_dataset'  # Update with your dataset path
    initialize(rootdir)  # Ensure initialization happens first
    train_model(model, train_loader, num_epochs=5)  # Train the model
    evaluate_model(model, test_loader)  # Evaluate the model

    # Run the Flask API for prediction
    # app.run(debug=True)
