import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, num_classes=9):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        for param in self.efficientnet.parameters():
            param.requires_grad = False

        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        self.fc1 = nn.Linear(in_features=1280, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        
        x = self.efficientnet(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.softmax(x)
        
        return x

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet18FeatureExtractor, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        for param in self.resnet18.parameters():
            param.requires_grad = False

        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.softmax(x)
        
        return x
    

class CNNModel(nn.Module):
    def __init__(self, num_classes=9, img_height=224, img_width=224):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.10)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.20)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same')
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout5 = nn.Dropout(0.30)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=512 * (img_height // 32) * (img_width // 32), out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = self.dropout5(x)
        
        x = self.flatten(x)
        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    train_dir = "E:/CourseWork/DDI-Code/last_try/Data/new_train"
    val_dir = "E:/CourseWork/DDI-Code/last_try/Data/new_val"
    test_dir = "E:/CourseWork/DDI-Code/last_try/Data/Test"


    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = EfficientNetFeatureExtractor(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.load_state_dict(torch.load("efb025.pth"))
    # model.eval()
    # model.to("cpu")
    # test_loss, test_acc = evaluate_model(model, test_loader, criterion, "cpu")
    # print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    # exit(0)

    num_epochs = 25
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Over Epochs")

    plt.tight_layout()
    plt.show()
