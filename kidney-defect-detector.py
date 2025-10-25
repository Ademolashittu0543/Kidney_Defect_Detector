#!/usr/bin/env python
# coding: utf-8

# # Loading Dependencies and Dataset

# In[1]:


import kagglehub
import os, glob, random, shutil
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import kagglehub

# Download latest version
path = kagglehub.dataset_download("nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone")

print("Path to dataset files:", path)


# In[2]:


base_path = "/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone"
# List everything inside
print(os.listdir(base_path))


# ##  Root of dataset

# In[3]:


# Root of dataset
root = os.path.join(base_path, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

# Check CSV for metadata (optional, if needed for custom loading)
csv_path = os.path.join(base_path, "kidneyData.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("CSV columns:", df.columns)
    print("CSV head:\n", df.head())

# Automatically find folder with class subdirectories
def find_deepest_category_dir(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(d.lower() in ['normal', 'cyst', 'tumor', 'stone'] for d in dirnames):
            return dirpath
    return root  

source_dir = find_deepest_category_dir(root)
print("✅ Found source directory:", source_dir)


# ## Split images into train/validation

# In[4]:


# Output directories
base_dir = "/kaggle/working/organized_data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Detect categories
categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
print("✅ Detected categories:", categories)

# Create train/val subfolders
for dir_path in [train_dir, val_dir]:
    for category in categories:
        os.makedirs(os.path.join(dir_path, category), exist_ok=True)

# Split images into train/validation (80/20)
for category in categories:
    category_path = os.path.join(source_dir, category)
    image_files = glob.glob(os.path.join(category_path, "**", "*.jpg"), recursive=True)
    random.shuffle(image_files)

    split = int(0.8 * len(image_files))
    for f in image_files[:split]:
        shutil.copy(f, os.path.join(train_dir, category))
    for f in image_files[split:]:
        shutil.copy(f, os.path.join(val_dir, category))

print("\n✅ Dataset successfully reorganized and split!")
print(f"Train folder: {train_dir}")
print(f"Validation folder: {val_dir}")


# # Data cleaning and augmentation

# ## Remove extra/unwanted folder

# In[5]:


# Remove extra/unwanted folders
for folder in ["CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"]:
    for base in [train_dir, val_dir]:
        path = os.path.join(base, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Removed extra folder: {path}")


# ## Remove possible corrupted images

# In[6]:


# Remove corrupted images
def remove_corrupted_images(base_path):
    num_deleted = 0
    for root, _, files in os.walk(base_path):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError):
                os.remove(file_path)
                num_deleted += 1
    print(f"🧹 Removed {num_deleted} corrupted images from {base_path}")

remove_corrupted_images(train_dir)
remove_corrupted_images(val_dir)


# ## Data augmentation and rescaling

# In[7]:


# Data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load images
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,  
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

print("\nDetected classes (indices):")
print(train_data.class_indices)


# # Data visualization and Exploration

# In[8]:


# Get mapping of class names and indices
class_indices = train_data.class_indices
classes = list(class_indices.keys())

# Prepare dictionary to store one image per class
sample_images = {}
sample_labels = {}

# Loop through batches until we get one sample from each class
for images, labels in train_data:
    for img, label in zip(images, labels):
        label_idx = int(label) if labels.ndim == 1 else np.argmax(label)
        class_name = classes[label_idx]

        if class_name not in sample_images:
            sample_images[class_name] = img
            sample_labels[class_name] = class_name

        # Stop when we have one image for each class
        if len(sample_images) == len(classes):
            break
    if len(sample_images) == len(classes):
        break

# Plot one sample per class
plt.figure(figsize=(15, 12))
for i, (label, img) in enumerate(sample_images.items()):
    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(label, fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[9]:


images, labels = next(train_data)
print("images_train_data_shape:", images.shape)
print("laels_train_data_shape:", labels.shape)


# In[10]:


train_dir = "/kaggle/working/organized_data/train"
val_dir = "/kaggle/working/organized_data/validation"

def count_images_per_class(directory):
    class_counts = {}
    for cls in os.listdir(directory):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            class_counts[cls] = len(os.listdir(cls_path))
    return class_counts

train_counts = count_images_per_class(train_dir)
val_counts = count_images_per_class(val_dir)

# Combine into DataFrame
df_counts = pd.DataFrame({
    'Class': list(train_counts.keys()),
    'Train': list(train_counts.values()),
    'Validation': [val_counts.get(k, 0) for k in train_counts.keys()]
})

df_counts['Total'] = df_counts['Train'] + df_counts['Validation']
df_counts = df_counts.sort_values('Total', ascending=False)
df_counts


# In[11]:


plt.figure(figsize=(12,6))
sns.barplot(x='Class', y='Total', data=df_counts, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title("Image Count per Fracture Type (Train + Validation)")
plt.xlabel("Fracture Type")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()


# In[12]:


plt.figure(figsize=(8,8))
plt.pie(df_counts['Total'], labels=df_counts['Class'], autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette("Set3", len(df_counts)))
plt.title("Fracture Type Distribution (%)", fontsize=14)
plt.tight_layout()
plt.show()


# In[13]:


df_counts_plot = df_counts.set_index('Class')[['Train', 'Validation']]
df_counts_plot.plot(kind='bar', stacked=False, figsize=(12,6), colormap='viridis')
plt.title("Train vs Validation Image Count per Class", fontsize=14)
plt.xlabel("Fracture Type")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # 

# # Compute class weights for imbalance

# In[14]:


# Compute class weights for imbalance
labels = []
for _, label_batch in train_data:
    labels.extend(np.argmax(label_batch, axis=1))
    if len(labels) >= train_data.samples:
        break

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(zip(np.unique(labels), class_weights))
print("Class weights:", class_weight_dict)


# # Build model and Train

# ## Model preparation

# In[15]:


# Load ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tune last 50 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Build model
model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-5),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ## Model training

# In[16]:


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1e-7)

# Train model
history = model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)


# # Model validation and performance check

# ## Model Accuracy and Loss Curve

# In[17]:


# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[18]:


val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}")


# ## Confusion matrix and Classification report

# In[22]:


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Get true labels and class names
y_true = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Predict (probabilities)
y_pred = model.predict(val_data)

# Convert predictions to class indices
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

