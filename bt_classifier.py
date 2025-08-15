# # ================================
# bt_classifier.py
# Preprocessing and EDA pipeline
# ================================

import kagglehub
from pathlib import Path
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 1: Download dataset from KaggleHub
# ----------------------------------------
base_dir = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
base_path = Path(base_dir)

# Define training and testing directories
train_dir = base_path / "Training"
test_dir = base_path / "Testing"

# ----------------------------------------
# Step 2: Define image transformations (Grayscale)
# ----------------------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),    # Convert to grayscale
    transforms.Resize((224, 224)),                      # Resize
    transforms.ToTensor(),                              # Convert to tensor [1, 224, 224]
    transforms.Normalize(mean=[0.5], std=[0.5])          # Normalize grayscale
])

# ----------------------------------------
# Step 3: Load datasets
# ----------------------------------------
train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)

# ----------------------------------------
# Step 4: Create DataLoaders
# ----------------------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------------------
# Step 5: Print detected classes
# ----------------------------------------
print("Detected classes:", train_dataset.classes)

# ----------------------------------------
# Step 6: Plot class distribution (EDA)
# ----------------------------------------
labels = [label for _, label in train_dataset]
counter = Counter(labels)

plt.bar(counter.keys(), counter.values(), tick_label=train_dataset.classes)
plt.title("Class distribution (training set)")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.show()

# ----------------------------------------
# Step 7: Display one image per class
# ----------------------------------------

# Reverse normalization for grayscale visualization
inv_normalize = transforms.Normalize(
    mean=[-1.0],
    std=[2.0]
)

class_images = {}
for img, label in train_dataset:
    class_name = train_dataset.classes[label]
    if class_name not in class_images:
        class_images[class_name] = img
    if len(class_images) == len(train_dataset.classes):
        break

# Show normalized and de-normalized images (grayscale)
n_classes = len(class_images)
plt.figure(figsize=(n_classes * 2.5, 5))

for idx, (class_name, img) in enumerate(class_images.items()):
    # Normalized image
    plt.subplot(2, n_classes, idx + 1)
    plt.imshow(img.squeeze(0).numpy(), cmap='gray')  # [1, 224, 224] → [224, 224]
    plt.title(f"{class_name}\n(normalized)")
    plt.axis('off')

    # De-normalized image
    img_inv = inv_normalize(img)
    img_inv = torch.clamp(img_inv, 0, 1)
    plt.subplot(2, n_classes, n_classes + idx + 1)
    plt.imshow(img_inv.squeeze(0).numpy(), cmap='gray')
    plt.title(f"{class_name}\n(original)")
    plt.axis('off')

plt.tight_layout()
plt.show()


import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),


    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(512  * 1 * 1  , len(train_dataset.classes))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


n_epochs = 20

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.2f}%")



model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")