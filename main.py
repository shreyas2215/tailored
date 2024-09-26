import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms as torchvision_transforms
import deeplake
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

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

# Custom Dataset class for Deep Lake

class WikiArtDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds  # Deep Lake dataset
        self.transform = transform  # Transformations to apply

    def __len__(self):
        return len(self.ds['images'])  # Return the total number of images

    
    from torchvision.transforms import ToPILImage

    def __getitem__(self, idx):
        try:
            image = self.ds['images'][idx]  # Deep Lake tensor
            label = self.ds['labels'][idx]   # Deep Lake tensor

            if isinstance(image, deeplake.core.tensor.Tensor):
                image = image.numpy()  # Convert to NumPy array first

            if isinstance(image, np.ndarray):
                image = torch.tensor(image)  # Convert to PyTorch tensor

            # Check the shape of the image tensor
            if len(image.shape) == 3:  # Expecting HWC format
                height, width, channels = image.shape

                # Convert to CHW format
                image = image.permute(2, 0, 1)  # Change to CHW format

                if channels not in {1, 3}:
                    print(f"Unexpected number of channels: {channels} at index {idx}")

            else:
                raise ValueError(f"Unexpected number of dimensions for image at index {idx}: {image.shape}")

            # Convert tensor to PIL Image for transformations
            image = ToPILImage()(image)  # Convert tensor to PIL Image

            # Apply transformations
            if self.transform:
                try:
                    image = self.transform(image)  # Apply the transformation to the PIL image
                except Exception as e:
                    print(f"Error applying transformation: {e}")
                    return None  # Skip this sample on error

            # Convert label to PyTorch tensor
            label = torch.tensor(label.numpy())  # Ensure label is in tensor format

            return image, label  # Return processed image and label
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}")
            return None  # Skip this sample on error

            # Function to filter out None values in DataLoader
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None  # Handle case where the whole batch is invalid
    return torch.utils.data.dataloader.default_collate(batch)

# Function to initialize dataset and model
def initialize():
    global model, train_loader, test_loader, class_names, initialized

    if initialized:
        print("Already initialized.")
        return  # Skip reinitialization if already done
    

    # Load the Deep Lake dataset
    ds = deeplake.load('hub://activeloop/wiki-art')
    

    
    # Get the labels and class names from the dataset
    unique_labels = ds['labels'].info.class_names
    class_names = unique_labels

    # Prepare the dataset
    full_dataset = WikiArtDataset(ds, transform=transforms)

    # Split dataset into training and testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Optimized DataLoader settings with prefetching
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=custom_collate_fn,
        prefetch_factor=2,  # Adjust based on your hardware (2 is a common starting point)
        persistent_workers=True  # Keep worker processes alive between batches
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=custom_collate_fn,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Model Selection and Modification
    # Model Selection and Modification
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Use ResNet-50
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda()  # Move model to GPU


    initialized = True  # Mark as initialized
    print("Initialization complete.")
    



# Function to save the model
def save_model(model, path='your_model.pth'):
    torch.save(model.state_dict(), path)
    print("Model saved successfully.")

# Training function with gradient accumulation
def train_model(model, train_loader, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available
    model.to(device)  # Move the model to the GPU

    model.train()  # Set the model to training mode
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch + 1}/{num_epochs} started")
        
        for inputs, labels in train_loader:
            if inputs is None or labels is None:  # Skip invalid samples
                continue

          

    # Adjust labels if necessary
            if len(labels.shape) > 1:  # Check if labels are one-hot encoded
                labels = torch.argmax(labels, dim=1)

            
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the GPU

            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluation function
def evaluate_model(model, test_loader):

    model.eval()
    all_preds = set()  # Use a set for predictions
    all_labels = []
    print("Starting model evaluation...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs is None or labels is None:  # Check for None values
                print("Received None input or labels, skipping...")
                continue
            
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.update(preds.cpu().numpy())  # Update the set with predictions
            all_labels.extend(labels.cpu().numpy())

            # Free up memory after each batch
            del inputs, labels, outputs
            torch.cuda.empty_cache()

    if not all_labels or not all_preds:
        print("No predictions or labels collected.")
        return

    accuracy = accuracy_score(all_labels, list(all_preds))  # Convert set to list for accuracy calculation
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, list(all_preds), average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

# Flask app for serving the model
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    if not data or 'file_path' not in data:
        return jsonify({"error": "No file path provided"}), 400
    
    file_path = data['file_path']  # Extract the file path from the JSON data

    # Check if the file exists
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        image = Image.open(file_path).convert("RGB")  # Open the image
    except Exception as e:
        return jsonify({"error": f"Failed to open image: {e}"}), 500

    # Here you would need to transform the image and make predictions
    # Assuming you have a function 'transform_image' and a model 'model'
    image = transforms(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image.cuda())  # Use the GPU for prediction
        _, predicted = torch.max(outputs, 1)

    return jsonify({"predicted_style": class_names[predicted.item()]})
# Main execution
if __name__ == "__main__":
    initialize()  # Ensure initialization happens first
    train_model(model, train_loader, num_epochs=5) 
    evaluate_model(model, test_loader)  # Evaluate the model

    # Run the Flask API for prediction
    app.run(debug=True)

