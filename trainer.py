import time
import tensorflow as tf
import config

def _merge_histories(primary, secondary):
    merged = tf.keras.callbacks.History()
    merged.history = {}
    for key in primary.history.keys():
        merged.history[key] = primary.history.get(key, []) + secondary.history.get(key, [])
    return merged

def train_model(model, train_data, val_data, class_weights, save_path, *, fine_tune=False):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    start_time = time.time()
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.EPOCHS,
        class_weight=class_weights, 
        callbacks=callbacks
    )

    if fine_tune and hasattr(model, "base_model"):
        base_model = model.base_model
        base_model.trainable = True
        if config.FINE_TUNE_AT is not None:
            for layer in base_model.layers[:config.FINE_TUNE_AT]:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LR),
            loss=model.loss,
            metrics=['accuracy']
        )

        fine_tune_history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=config.FINE_TUNE_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks
        )

        history = _merge_histories(history, fine_tune_history)
    
    end_time = time.time()
    training_time_seconds = end_time - start_time
    
    return history, training_time_seconds