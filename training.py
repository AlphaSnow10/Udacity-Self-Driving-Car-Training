import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center_image_path = self.data.iloc[idx, 0]
        left_image_path = self.data.iloc[idx, 1]
        right_image_path = self.data.iloc[idx, 2]

        # Load images
        center_image = Image.open(center_image_path)
        left_image = Image.open(left_image_path)
        right_image = Image.open(right_image_path)

        # Apply transformations if specified
        if self.transform:
            center_image = self.transform(center_image)
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        # Get labels
        steering = float(self.data.iloc[idx, 3])
        throttle = float(self.data.iloc[idx, 4])
        reverse = float(self.data.iloc[idx, 5])
        speed = float(self.data.iloc[idx, 6])

        return center_image, left_image, right_image, torch.tensor([steering, throttle, reverse, speed])

# Define the complex neural network model
class MyComplexModel(nn.Module):
    def __init__(self):
        super(MyComplexModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # Output layer with 4 outputs (steering, throttle, reverse, speed)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor before fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Set random seed for reproducibility
torch.manual_seed(0)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load and split data
data = CustomDataset('D:/ASU/Formula Ai/Task 3/beta_simulator_windows/Data/driving_log.csv', transform=transform)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# Create an instance of the complex model
model = MyComplexModel()

# Define loss function
criterion = nn.MSELoss()

# Create a weight tensor for loss weights
# Adjust the weights based on the importance of each output
loss_weights = torch.tensor([1.0, 1.0, 0.0, 0.0])  # Adjust the weights

# Define optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# Define a learning rate schedule
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)  # Adjust the step_size and gamma as needed

# Lists to store training and validation losses for each output
train_losses = []
val_losses = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (center_image, left_image, right_image, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(center_image)

        # Compute weighted loss
        loss = torch.mean(loss_weights * (outputs - labels) ** 2)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {running_loss / len(train_loader)}")

    # Print individual losses
    steering_loss = torch.mean(loss_weights[0] * (outputs[:, 0] - labels[:, 0]) ** 2).item()
    throttle_loss = torch.mean(loss_weights[1] * (outputs[:, 1] - labels[:, 1]) ** 2).item()
    reverse_loss = torch.mean(loss_weights[2] * (outputs[:, 2] - labels[:, 2]) ** 2).item()
    speed_loss = torch.mean(loss_weights[3] * (outputs[:, 3] - labels[:, 3]) ** 2).item()

    print(f"Steering Loss: {steering_loss}, Throttle Loss: {throttle_loss}")

    # Store the total loss
    train_losses.append(running_loss / len(train_loader))

    # Adjust the learning rate
    scheduler.step()

# Validation loop
model.eval()
val_loss = 0.0
val_steering_loss = 0.0
val_throttle_loss = 0.0
val_reverse_loss = 0.0
val_speed_loss = 0.0

with torch.no_grad():
    for batch_idx, (center_image, left_image, right_image, labels) in enumerate(val_loader):
        outputs = model(center_image)

        # Compute weighted loss for each output in validation
        steering_loss = torch.mean(loss_weights[0] * (outputs[:, 0] - labels[:, 0]) ** 2)
        throttle_loss = torch.mean(loss_weights[1] * (outputs[:, 1] - labels[:, 1]) ** 2)
        reverse_loss = torch.mean(loss_weights[2] * (outputs[:, 2] - labels[:, 2]) ** 2)
        speed_loss = torch.mean(loss_weights[3] * (outputs[:, 3] - labels[:, 3]) ** 2)

        val_loss += steering_loss + throttle_loss + reverse_loss + speed_loss
        val_steering_loss += steering_loss.item()
        val_throttle_loss += throttle_loss.item()
        val_reverse_loss += reverse_loss.item()
        val_speed_loss += speed_loss.item()

# Compute the average validation loss for each output
avg_val_steering_loss = val_steering_loss / len(val_loader)
avg_val_throttle_loss = val_throttle_loss / len(val_loader)
avg_val_reverse_loss = val_reverse_loss / len(val_loader)
avg_val_speed_loss = val_speed_loss / len(val_loader)

# Print the validation losses for each output
print(f"Validation Steering Loss: {avg_val_steering_loss}")
print(f"Validation Throttle Loss: {avg_val_throttle_loss}")