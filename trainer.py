import time
import tensorflow as tf
import config

def train_model(model, train_data, val_data, class_weights, save_path):
    """
    Trains the model, applies Early Stopping, and tracks execution time.
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    start_time = time.time()
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.EPOCHS,
        class_weight=class_weights, 
        callbacks=callbacks
    )
    
    end_time = time.time()
    training_time_seconds = end_time - start_time
    
    return history, training_time_seconds