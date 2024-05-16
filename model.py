import os
import shutil
import random
from pathlib import Path


random.seed(42)

data_dir = Path("E:/CourseWork/DDI-Code/last_try/Data/out")
train_dir = Path("E:/CourseWork/DDI-Code/last_try/Data/new_train")
val_dir = Path("E:/CourseWork/DDI-Code/last_try/Data/new_val")


train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)


class_names = [d.name for d in data_dir.iterdir() if d.is_dir()]


for class_name in class_names:

    class_dir = data_dir / class_name
    train_class_dir = train_dir / class_name
    val_class_dir = val_dir / class_name


    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(class_dir.glob('*'))
    
    random.shuffle(image_files)
    
    split_index = int(0.8 * len(image_files))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for file in train_files:
        shutil.move(str(file), str(train_class_dir / file.name))
        
    for file in val_files:
        shutil.move(str(file), str(val_class_dir / file.name))

print("Dataset split completed successfully.")
