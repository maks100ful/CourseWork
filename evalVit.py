import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import timm
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, num_classes=9):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b1(pretrained=True)

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
        self.resnet18 = models.resnet50(pretrained=True)

        for param in self.resnet18.parameters():
            param.requires_grad = False

        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
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
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    top1_acc = epoch_acc
    recall = recall_score(all_labels, all_preds, average="weighted")
    top5_acc = top_k_accuracy(all_probs, all_labels, k=5)
    
    return epoch_loss, top1_acc, top5_acc, precision, recall, all_labels, all_preds

def top_k_accuracy(output, target, k=1):
    max_k_preds = np.argsort(output, axis=1)[:, -k:]
    correct = np.sum([1 if target[i] in max_k_preds[i] else 0 for i in range(len(target))])
    return correct / len(target)

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    train_dir = "/home/max/courseVa/new_train"
    val_dir = "/home/max/courseVa/new_val"
    # test_dir = "E:/CourseWork/DDI-Code/last_try/Data/Test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    # test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=6)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # model = CNNModel(num_classes=9).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

    new_head = nn.Sequential(
        nn.Linear(model.head.in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 9)
    )
    model.head = new_head
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    model.load_state_dict(torch.load("Vit.pth"))
    model.eval()
    model.to("cpu")
    test_loss, top1_acc, top5_acc, precision, recall, all_labels, all_preds = evaluate_model(model, val_loader, criterion, "cpu")
    
    cm = confusion_matrix(all_labels, all_preds)
    class_names = train_dataset.classes

    print(f"Test loss: {test_loss:.4f}")
    print(f"Top-1 accuracy: {top1_acc:.4f}")
    print(f"Top-5 accuracy: {top5_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

    exit(0)
