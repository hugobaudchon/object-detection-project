from ultralytics import YOLO
import random
import yaml
import os
import shutil


#Split images into training, validation and test sets
images_folder = "/images"
labels_folder = "/labels"

# Destination base folder
dest_base = "/dataset_split"
train_images_folder = os.path.join(dest_base, "train/images")
train_labels_folder = os.path.join(dest_base, "train/labels")
val_images_folder = os.path.join(dest_base, "val/images")
val_labels_folder = os.path.join(dest_base, "val/labels")

# Create destination folders
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

images = sorted([f for f in os.listdir(images_folder) if f.endswith(".jpg")])
labels = sorted([f for f in os.listdir(labels_folder) if f.endswith(".txt")])
paired_files = [(img, img.replace(".jpg", ".txt")) for img in images if os.path.exists(os.path.join(labels_folder, img.replace(".jpg", ".txt")))]

random.shuffle(paired_files)

# Split data: 80% train/ 20% val/ 
total_files = len(paired_files)
train_split = int(0.8 * total_files)

train_files = paired_files[:train_split]
val_files = paired_files[train_split:]

def copy_files(file_list, dest_images, dest_labels):
    for img, lbl in file_list:
        shutil.copy(os.path.join(images_folder, img), dest_images)
        shutil.copy(os.path.join(labels_folder, lbl), dest_labels)

# Copy files to respective folders
copy_files(train_files, train_images_folder, train_labels_folder)
copy_files(val_files, val_images_folder, val_labels_folder)

# Define the data to write to the YAML file that is used in the YOLO model
data = {
    'train': '/dataset_split/train/images',
    'val': '/dataset_split/val/images'

    'nc': 6,
    'names': [
    "rc car",
    "yellow rubber duckie",
    "small orange traffic cone",
    "wooden toy house",
    "qr code",
    "road sign"
]
}

# Specify the file path
file_path = '/dataset_split/dataset.yaml'

# Write the data to a YAML file
with open(file_path, 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)


# Train yolo nano version 5:
#For other versions of YOLO (yolov8n.pt, yolov9t.pt yolo10n.py ,yolo11n.pt)
model = YOLO("yolov5nu.pt")

results = model.train(data="/dataset_split/dataset.yaml",
                         epochs = 100,
                           plots = True,patience = 10)
