from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import config

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _collect_paths_and_labels(data_dir: str):
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro la carpeta '{data_dir}'.")

    class_names = sorted([entry.name for entry in dataset_path.iterdir() if entry.is_dir()])
    if not class_names:
        raise ValueError("No se encontraron clases dentro del dataset.")

    paths = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_dir = dataset_path / class_name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in _VALID_EXTENSIONS:
                paths.append(str(img_path))
                labels.append(idx)

    if not paths:
        raise ValueError("No se encontraron imagenes validas dentro del dataset.")

    return np.array(paths), np.array(labels), class_names

def _decode_and_resize(path, label, num_classes: int):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH))
    image = tf.cast(image, tf.float32) 
    
    label = tf.one_hot(label, num_classes)
    return image, label

def _make_dataset(paths, labels, num_classes: int, *, shuffle: bool, augment: bool):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=config.SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda path, label: _decode_and_resize(path, label, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment and config.USE_DATA_AUGMENTATION:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(config.AUGMENTATION_ROTATION),
            tf.keras.layers.RandomZoom(config.AUGMENTATION_ZOOM),
            tf.keras.layers.RandomContrast(config.AUGMENTATION_CONTRAST)
        ])
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def _count_labels(dataset):
    counts = {}
    for _, batch_labels in dataset:
        labels = np.argmax(batch_labels.numpy(), axis=1)
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return counts

def print_class_distribution(class_names, train_dataset, val_dataset, test_dataset):
    train_counts = _count_labels(train_dataset)
    val_counts = _count_labels(val_dataset)
    test_counts = _count_labels(test_dataset)

    print("\nDistribucion de clases (train/val/test):")
    for idx, name in enumerate(class_names):
        train_count = train_counts.get(idx, 0)
        val_count = val_counts.get(idx, 0)
        test_count = test_counts.get(idx, 0)
        print(f"- {name}: {train_count}/{val_count}/{test_count}")

def load_and_split_dataset(data_dir: str):
    print(f"Cargando dataset desde: {data_dir}...")

    paths, labels, class_names = _collect_paths_and_labels(data_dir)
    num_classes = len(class_names)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=config.TEST_SPLIT,
        random_state=config.SEED,
        stratify=labels
    )

    val_ratio = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_ratio,
        random_state=config.SEED,
        stratify=train_val_labels
    )

    train_dataset = _make_dataset(
        train_paths,
        train_labels,
        num_classes,
        shuffle=True,
        augment=True
    )
    val_dataset = _make_dataset(
        val_paths,
        val_labels,
        num_classes,
        shuffle=False,
        augment=False
    )
    test_dataset = _make_dataset(
        test_paths,
        test_labels,
        num_classes,
        shuffle=False,
        augment=False
    )

    return train_dataset, val_dataset, test_dataset, class_names


def get_class_weights(train_dataset):
    labels = []

    for _, batch_labels in train_dataset:
        labels.extend(np.argmax(batch_labels.numpy(), axis=1))

    classes_array = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes_array, y=labels)
    if config.MAX_CLASS_WEIGHT is not None:
        weights = np.minimum(weights, config.MAX_CLASS_WEIGHT)
    class_weights_dict = dict(zip(classes_array, weights))

    print("Pesos de clases calculados desde el dataset de entrenamiento.")
    return class_weights_dict