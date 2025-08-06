# Install SuperGradients if not already installed
# !pip install super-gradients
#import torch
from super_gradients.training import models, Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
import os

# ---- 1. Dataset Parameters ----
DATA_DIR = './dataset'
CLASS_NAMES = ['Ear Protectors', 'Glasses', 'Gloves', 'Helmet', 'Mask', 'Person', 'Safety_shoes', 'Shoes', 'Vest', 'Without Ear Protectors', 'Without Glass', 'Without Glove', 'Without Helmet', 'Without Mask', 'Without Shoes', 'Without Vest'] # Replace with your actual class names
NUM_CLASSES = len(CLASS_NAMES)

dataset_params = {
    'data_dir': DATA_DIR,
    'train_images_dir': 'train/images',
    'train_labels_dir': 'train/labels',
    'val_images_dir': 'valid/images',
    'val_labels_dir': 'valid/labels',
    'classes': CLASS_NAMES
}

# ---- 2. Data Loaders ----
train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': CLASS_NAMES
    },
    dataloader_params={'batch_size': 8, 'num_workers': 2}
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': CLASS_NAMES
    },
    dataloader_params={'batch_size': 16, 'num_workers': 8}
)

# ---- 3. Load the Model ----
model = models.get(
    'yolo_nas_s',  # Choose 'yolo_nas_s', 'yolo_nas_m', or 'yolo_nas_l'
    num_classes=NUM_CLASSES,
    pretrained_weights="coco"
)

# ---- 4. Trainer Setup ----
trainer = Trainer(
    experiment_name="yolo_nas_ppe",
    ckpt_root_dir=os.path.join(os.getcwd(), "checkpoints")
)

# ---- 5. Training Parameters ----
train_params = {
    "max_epochs": 25,
    "batch_size": 16,
    "silent_mode": False,
    "average_best_models": True,
    "initial_lr": 5e-4,
    "optimizer": "Adam",  # Note: Fixed typo from 'optimzer'
    "metric_to_watch": 'mAP@0.50:0.95'
}

# ---- 6. Start Training ----
trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data
)

# ---- 7. (Optional) Test or Inference ----
# After training, you can perform inference using model.predict or trainer.test, check SuperGradients docs for examples.
