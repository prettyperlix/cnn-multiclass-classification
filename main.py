import tensorflow as tf
from tensorflow.keras import mixed_precision
import gc # Recolector de basura de Python

try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled: Tensor Cores are active.")
except Exception as e:
    print("Mixed precision not available.")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth enabled.")
    except RuntimeError as e:
        print(e)

from data_loader import load_and_split_dataset, get_class_weights
from model_builder import build_custom_cnn, build_mobilenet_scratch, build_resnet_transfer, build_inception_transfer
from trainer import train_model
from evaluator import plot_learning_curves, evaluate_per_class_error
import config

def main():
    print("\n" + "="*50)
    print("   DEEP LEARNING: MULTICLASS FISH CLASSIFICATION")
    print("="*50 + "\n")

    train_ds, val_ds, class_names = load_and_split_dataset(config.BASE_DIR)
    num_classes = len(class_names)
    class_weights = get_class_weights(config.BASE_DIR)

    # CORRECCIÓN: Ahora guardamos la "función constructora", no el modelo ya construido.
    # Nota que quitamos los paréntesis () después del nombre de la función.
    models_dict = {
        "Custom CNN (Scratch)": (build_custom_cnn, config.CUSTOM_CNN_PATH),
        "MobileNet (Scratch)": (build_mobilenet_scratch, config.MOBILENET_PATH),
        "ResNet50 (Transfer Learning)": (build_resnet_transfer, config.RESNET_PATH),
        "InceptionV3 (Transfer Learning)": (build_inception_transfer, config.INCEPTION_PATH)
    }

    results_time = {}

    for model_name, (builder_function, save_path) in models_dict.items():
        print(f"\n" + "-"*50)
        print(f" TRAINING: {model_name}")
        print("-"*50)
        
        # 1. Construir el modelo SOLO cuando es su turno
        model = builder_function(num_classes)
        
        # 2. Entrenar
        history, training_time = train_model(model, train_ds, val_ds, class_weights, save_path)
        results_time[model_name] = training_time
        
        # 3. Evaluar
        plot_learning_curves(history, model_name)
        evaluate_per_class_error(model, val_ds, class_names)
        
        # 4. LIMPIEZA EXTREMA DE MEMORIA (Previene que la terminal explote)
        print(f"Liberando memoria de {model_name}...")
        del model 
        del history
        tf.keras.backend.clear_session()
        gc.collect() 

    print(f"\n" + "="*50)
    print(f"   FINAL TRAINING TIME COMPARISON")
    print("="*50)
    for model_name, time_sec in results_time.items():
        print(f"{model_name:<35}: {time_sec/60:.2f} minutes")

if __name__ == "__main__":
    main()