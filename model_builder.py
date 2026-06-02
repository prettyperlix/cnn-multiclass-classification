import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
import config

def build_custom_cnn(num_classes: int) -> tf.keras.Model:
    """Builds a custom CNN from scratch."""
    model = models.Sequential([
        layers.InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mobilenet_scratch(num_classes: int) -> tf.keras.Model:
    """Builds MobileNetV2 to be trained from scratch (weights=None)."""
    base_model = MobileNetV2(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights=None # Crucial for "from scratch"
    )
    model = models.Sequential([
        layers.InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet_transfer(num_classes: int) -> tf.keras.Model:
    """Builds ResNet50 using Transfer Learning (ImageNet weights, Frozen base)."""
    base_model = ResNet50(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Freeze convolutional layers
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_inception_transfer(num_classes: int) -> tf.keras.Model:
    """Builds InceptionV3 using Transfer Learning (ImageNet weights, Frozen base)."""
    base_model = InceptionV3(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Freeze convolutional layers
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model