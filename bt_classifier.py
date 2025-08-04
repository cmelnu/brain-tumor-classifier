# Preprocesing data

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure RGB
    transforms.Resize((224, 224)),                      # Resize to 224x224
    transforms.ToTensor(),                              # Convert to tensor
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)      # Normalize to [-1,1]
])

# Step 2: Define relative paths
base_dir = Path.cwd()
train_dir = base_dir / "archive" / "training"
test_dir = base_dir / "archive" / "testing"

# Step 3: Load datasets with transformations
train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)

# Step 4: Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 5: Check the detected classes
print("Detected classes:", train_dataset.classes)


# Exploratory Data Analysis

from collections import Counter
import matplotlib.pyplot as plt

labels = [label for _, label in train_dataset]
counter = Counter(labels)

plt.bar(counter.keys(), counter.values(), tick_label=train_dataset.classes)
plt.title("Class distribution (training set)")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.show()

# Reverse normalization: from [-1,1] to [0,1]
inv_normalize = transforms.Normalize(
    mean=[-1.0, -1.0, -1.0],
    std=[2.0, 2.0, 2.0]
)

# Collect one image per class
class_images = {}
for img, label in train_dataset:
    class_name = train_dataset.classes[label]
    if class_name not in class_images:
        class_images[class_name] = img
    if len(class_images) == len(train_dataset.classes):
        break

# Display: row 1 -> normalized, row 2 -> de-normalized
n_classes = len(class_images)
plt.figure(figsize=(n_classes * 2.5, 5))

for idx, (class_name, img) in enumerate(class_images.items()):
    # Row 1: Normalized image
    plt.subplot(2, n_classes, idx + 1)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title(f"{class_name}\n(normalized)")
    plt.axis('off')

    # Row 2: De-normalized image (to visualize properly)
    img_inv = inv_normalize(img)
    img_inv = torch.clamp(img_inv, 0, 1)  # ensure values are in [0,1]
    plt.subplot(2, n_classes, n_classes + idx + 1)
    plt.imshow(img_inv.permute(1, 2, 0).numpy())
    plt.title(f"{class_name}\n(original)")
    plt.axis('off')

plt.tight_layout()
plt.show()