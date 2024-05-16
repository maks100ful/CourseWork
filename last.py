from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
import torch.optim as optim
from sklearn.metrics import precision_score
import numpy as np
import torch.nn as nn
import Augmentor
import torchvision.models as models
import matplotlib.pyplot as plt
from collections import Counter

transform = transforms.Compose([
    transforms.Resize(256),               
    transforms.CenterCrop(224),           
    transforms.ToTensor(),                 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
full_train_dataset  = datasets.ImageFolder('E:\\CourseWork\\DDI-Code\\last_try\\Data\\new_train', transform=transform)
test_dataset = datasets.ImageFolder('E:\\CourseWork\\DDI-Code\\last_try\\Data\\new_val', transform=transform)

def count_images_by_category(dataset):
    category_counts = Counter()
    for _, label in dataset:
        category_counts[label] += 1
    return category_counts

# Count images in the training and test datasets
train_category_counts = count_images_by_category(full_train_dataset)
test_category_counts = count_images_by_category(test_dataset)

# Convert to lists for plotting
train_categories = [full_train_dataset.classes[category] for category in train_category_counts.keys()]
train_counts = list(train_category_counts.values())

test_categories = [test_dataset.classes[category] for category in test_category_counts.keys()]
test_counts = list(test_category_counts.values())

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training dataset
axes[0].bar(train_categories, train_counts, color='skyblue')
axes[0].set_title('Number of Images per Category in Training Dataset')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Number of Images')
axes[0].tick_params(axis='x', rotation=45)

# Test dataset
axes[1].bar(test_categories, test_counts, color='lightgreen')
axes[1].set_title('Number of Images per Category in Test Dataset')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Number of Images')
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.show()

train_dataset, val_dataset = random_split(full_train_dataset , [0.8, 0.2])

# for i in full_train_dataset.classes:
#     p = Augmentor.Pipeline('E:\\CourseWork\\DDI-Code\\last_try\\Data\\Train\\' + i, output_directory=f'E:\\CourseWork\\DDI-Code\\last_try\\Data\\out\\{i}')
#     p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
#     p.zoom(probability=0.5, min_factor=1.1, max_factor=1.25)
#     p.sample(500) 

