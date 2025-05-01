import os
import random
from shutil import copy2

output_folder = "./yolo"
train_ratio = 0.85

images_path = os.path.join(output_folder, "images")
labels_path = os.path.join(output_folder, "labels")

train_images = os.path.join(images_path, "train")
val_images = os.path.join(images_path, "val")
train_labels = os.path.join(labels_path, "train")
val_labels = os.path.join(labels_path, "val")

os.makedirs(train_images, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

images = [f for f in os.listdir(images_path) if f.endswith(".png")]
random.shuffle(images)

train_count = int(len(images) * train_ratio)
train_files = images[:train_count]
val_files = images[train_count:]

for file in train_files:
    copy2(os.path.join(images_path, file), train_images)
    label_file = file.replace(".png", ".txt")
    copy2(os.path.join(labels_path, label_file), train_labels)

for file in val_files:
    copy2(os.path.join(images_path, file), val_images)
    label_file = file.replace(".png", ".txt")
    copy2(os.path.join(labels_path, label_file), val_labels)