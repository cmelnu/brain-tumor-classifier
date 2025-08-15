# # ================================
# bt_classifier.py
# Preprocessing and EDA pipeline
# ================================

import kagglehub
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

def count_images_per_class(dataset):
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)
    for idx, class_name in enumerate(dataset.classes):
        print(f"{class_name}: {class_counts.get(idx, 0)} images")

# ----------------------------------------
# Step 1: Download dataset from KaggleHub
# ----------------------------------------
base_dir = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
base_path = Path(base_dir)

# Define training and testing directories
train_dir = base_path / "Training"
test_dir = base_path / "Testing"

# -----------------------------------------------
# Step 2: Define image transformations (Grayscale)
# -----------------------------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),    # Convert to grayscale
    transforms.Resize((112, 112)),                      # Resize
    transforms.ToTensor(),                              # Convert to tensor [1, 224, 224]
    transforms.Normalize(mean=[0.5], std=[0.5])          # Normalize grayscale
])

# ----------------------------------------
# Step 3: Load datasets
# ----------------------------------------
train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)

print("\nTraining set image counts per class:")
count_images_per_class(train_dataset)

print("\nTest set image counts per class:")
count_images_per_class(test_dataset)
# ----------------------------------------
# Step 4: Create DataLoaders
# ----------------------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------------------
# Step 5: Print detected classes
# ----------------------------------------
print("Detected classes:", train_dataset.classes)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), # Output: [8, 112, 112]
    nn.Flatten(),  
    nn.Linear(16 * 56 * 56, len(train_dataset.classes))  # Capa final
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Calculate weights inversely proportional to class frequency
train_targets = [label for _, label in train_dataset.samples]
class_sample_count = Counter(train_targets)
weights = [1.0 / class_sample_count[i] for i in range(len(train_dataset.classes))]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------------------------------
# Step 6: Training loop with early stopping (patience=3) and progress bar
# -----------------------------------------------------
num_epochs = 10
patience = 3
best_loss = float('inf')
epochs_no_improve = 0

train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for each epoch
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # Early stopping logic
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs).")
            break

# ----------------------------------------
# Step 7: Evaluation on test set
# ----------------------------------------
model.eval()
test_correct = 0
test_total = 0
test_losses = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_losses.append(loss.item() * images.size(0))
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_loss = sum(test_losses) / test_total
test_accuracy = test_correct / test_total
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
