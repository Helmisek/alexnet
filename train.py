import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from model import AlexNet
from torch.cuda.amp import GradScaler, autocast


# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms for the training data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transforms for the validation data
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet dataset using Hugging Face datasets
dataset = load_dataset("imagenet-1k")

# Define a function to apply transforms to our dataset
def transform_dataset(examples):
    examples["pixel_values"] = [train_transform(image.convert("RGB")) for image in examples["image"]]
    return examples

# Apply the transforms to our dataset
train_dataset = dataset["train"].with_transform(transform_dataset)
val_dataset = dataset["validation"].with_transform(lambda examples: {"pixel_values": [val_transform(image.convert("RGB")) for image in examples["image"]]})

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# Initialize the model
model = AlexNet(num_classes=1000)
model = model.to(device)

# Define loss function and optimizer
lossFn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Using AMP GradScaler

scaler = GradScaler("cuda")

# Training loop
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    model_loss = 0.0

    # Prepare training data
    for i, batch in enumerate(train_loader):
        inputs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

    # Reset the gradients
    optimizer.zero_grad()

    # Using AMP autocasting
    with autocast("cuda"):
        outputs = model(inputs)
        loss = lossFn(outputs, labels) 

    # Use the scaler to perform the backward pass and update weights
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    model_loss += loss.item()
    if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {model_loss / 100:.3f}')
            model_loss = 0.0
    
    # Perform data validation (no need for AMP here)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the validation images: {100 * correct / total}%')

print('Finished Training')