import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [8, 112, 112]

    nn.Flatten(),  
    nn.Linear(8 * 112 * 112, len(train_dataset.classes))  # Capa final
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
