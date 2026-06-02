import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import config

def load_and_split_dataset(data_dir: str):
    """
    Loads images from directory and splits them into Train and Validation sets.
    Applies same partitions for all models.
    """
    print(f"Loading dataset from: {data_dir}...")
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42, # Seed ensures exact same partition for all 4 models
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_dataset.class_names

    # Performance optimization (Prefetching)
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, class_names

def get_class_weights(data_dir: str):
    """
    Computes class weights to handle extreme class imbalance in the dataset.
    """
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    labels = []
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        labels.extend([i] * num_images)
            
    classes_array = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes_array, y=labels)
    class_weights_dict = dict(zip(classes_array, weights))
    
    print("Class weights calculated successfully to mitigate imbalance.")
    return class_weights_dict