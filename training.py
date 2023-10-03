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

        # Convolutional layers for center image
        self.conv1_center = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_center = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_center = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool_center = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layers for left image
        self.conv1_left = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_left = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_left = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool_left = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layers for right image
        self.conv1_right = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_right = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_right = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool_right = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # Output layer with 4 outputs (steering, throttle, reverse, speed)

    def forward(self, center_image, left_image, right_image):
        # Forward pass for center image
        x_center = self.pool_center(F.relu(self.conv1_center(center_image)))
        x_center = self.pool_center(F.relu(self.conv2_center(x_center)))
        x_center = self.pool_center(F.relu(self.conv3_center(x_center)))

        # Forward pass for left image
        x_left = self.pool_left(F.relu(self.conv1_left(left_image)))
        x_left = self.pool_left(F.relu(self.conv2_left(x_left)))
        x_left = self.pool_left(F.relu(self.conv3_left(x_left)))

        # Forward pass for right image
        x_right = self.pool_right(F.relu(self.conv1_right(right_image)))
        x_right = self.pool_right(F.relu(self.conv2_right(x_right)))
        x_right = self.pool_right(F.relu(self.conv3_right(x_right)))

        # Concatenate features from all three images
        x = torch.cat((x_center, x_left, x_right), dim=1)

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
        outputs = model(center_image, left_image, right_image)

        # Compute weighted loss for steering and throttle
        steering_throttle_loss = torch.mean(loss_weights[:2] * (outputs[:, :2] - labels[:, :2]) ** 2)

        # Backpropagation
        steering_throttle_loss.backward()
        optimizer.step()

        running_loss += steering_throttle_loss.item()

    # Print the loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {running_loss / len(train_loader)}")

    # Print individual losses
    steering_loss = torch.mean(loss_weights[0] * (outputs[:, 0] - labels[:, 0]) ** 2).item()
    throttle_loss = torch.mean(loss_weights[1] * (outputs[:, 1] - labels[:, 1]) ** 2).item()

    print(f"Steering Loss: {steering_loss}, Throttle Loss: {throttle_loss}")

    # Store the total loss
    train_losses.append(running_loss / len(train_loader))

    # Adjust the learning rate
    scheduler.step()

# Validation loop
model.eval()
val_loss = 0.0

with torch.no_grad():
    for batch_idx, (center_image, left_image, right_image, labels) in enumerate(val_loader):
        outputs = model(center_image, left_image, right_image)

        # Compute weighted loss for steering and throttle in validation
        steering_throttle_loss = torch.mean(loss_weights[:2] * (outputs[:, :2] - labels[:, :2]) ** 2)

        val_loss += steering_throttle_loss.item()

# Compute the average validation loss for steering and throttle
avg_val_loss = val_loss / len(val_loader)

# Print the validation loss for steering and throttle
print(f"Validation Steering and Throttle Loss: {avg_val_loss}")
