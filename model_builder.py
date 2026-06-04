import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNet, ResNet50, InceptionV3
import config
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

def _loss_fn(label_smoothing=None):
    if config.USE_FOCAL_LOSS:
        def focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = config.FOCAL_ALPHA * tf.pow(1.0 - y_pred, config.FOCAL_GAMMA)
            return tf.reduce_sum(weight * cross_entropy, axis=-1)

        return focal_loss

    if label_smoothing is None:
        label_smoothing = config.LABEL_SMOOTHING

    return tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )

def build_custom_cnn(num_classes: int) -> tf.keras.Model:
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
        layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY)
        ),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
            dtype='float32'
        )
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.MOBILENET_LEARNING_RATE),
        loss=_loss_fn(label_smoothing=config.MOBILENET_LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    return model

def build_mobilenet_scratch(num_classes: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
    ], name="data_augmentation")

    base_model = MobileNet(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights=None,      
        alpha=0.75,        
        pooling="avg"      
    )

    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))
    
    x = data_augmentation(inputs)
    
    x = layers.Rescaling(scale=1./127.5, offset=-1.0)(x)
    
    x = base_model(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def build_resnet_transfer(num_classes: int) -> tf.keras.Model:
    base_model = ResNet50(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = not config.TRANSFER_FREEZE_BASE
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        layers.Lambda(resnet_preprocess),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
            dtype='float32'
        )
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss=_loss_fn(), metrics=['accuracy'])
    model.base_model = base_model
    return model

def build_inception_transfer(num_classes: int) -> tf.keras.Model:
    base_model = InceptionV3(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = not config.TRANSFER_FREEZE_BASE
    
    model = models.Sequential([
        layers.InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        layers.Lambda(inception_preprocess),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(config.WEIGHT_DECAY),
            dtype='float32'
        )
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss=_loss_fn(), metrics=['accuracy'])
    model.base_model = base_model
    return model