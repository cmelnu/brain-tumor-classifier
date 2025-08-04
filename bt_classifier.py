# ================================
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
# Step 2: Define image transformations
# ----------------------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure RGB
    transforms.Resize((224, 224)),                      # Resize
    transforms.ToTensor(),                              # To tensor
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)      # Normalize to [-1, 1]
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

# Reverse normalization for visualization
inv_normalize = transforms.Normalize(
    mean=[-1.0, -1.0, -1.0],
    std=[2.0, 2.0, 2.0]
)

class_images = {}
for img, label in train_dataset:
    class_name = train_dataset.classes[label]
    if class_name not in class_images:
        class_images[class_name] = img
    if len(class_images) == len(train_dataset.classes):
        break

# Show normalized and de-normalized images
n_classes = len(class_images)
plt.figure(figsize=(n_classes * 2.5, 5))

for idx, (class_name, img) in enumerate(class_images.items()):
    # Normalized image
    plt.subplot(2, n_classes, idx + 1)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title(f"{class_name}\n(normalized)")
    plt.axis('off')

    # De-normalized image
    img_inv = inv_normalize(img)
    img_inv = torch.clamp(img_inv, 0, 1)
    plt.subplot(2, n_classes, n_classes + idx + 1)
    plt.imshow(img_inv.permute(1, 2, 0).numpy())
    plt.title(f"{class_name}\n(original)")
    plt.axis('off')

plt.tight_layout()
plt.show()
