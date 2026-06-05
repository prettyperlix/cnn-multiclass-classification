import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
import gc
import config

if config.USE_MIXED_PRECISION:
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled: Tensor Cores are active.")
    except Exception:
        print("Mixed precision not available.")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth enabled.")
    except RuntimeError as e:
        print(e)

np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

from data_loader import load_and_split_dataset, get_class_weights, print_class_distribution
from model_builder import build_custom_cnn, build_mobilenet_scratch, build_resnet_transfer, build_inception_transfer
from trainer import train_model
from evaluator import plot_learning_curves, evaluate_per_class_error

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def _best_metric(history, key):
    values = history.history.get(key, [])
    return max(values) if values else None

def _write_summary_csv(rows, csv_path):
    fieldnames = [
        "model_name",
        "is_transfer",
        "use_class_weights",
        "freeze_base",
        "fine_tune",
        "fine_tune_at",
        "train_time_sec",
        "train_time_min",
        "best_val_accuracy",
        "best_val_loss",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    print("\n" + "="*50)
    print("   DEEP LEARNING: MULTICLASS FISH CLASSIFICATION")
    print("="*50 + "\n")

    train_ds, val_ds, test_ds, class_names = load_and_split_dataset(config.BASE_DIR)
    print_class_distribution(class_names, train_ds, val_ds, test_ds)
    num_classes = len(class_names)
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = get_class_weights(train_ds)

    models_dict = {
        "Custom CNN (Scratch)": (build_custom_cnn, config.CUSTOM_CNN_PATH, False),
        "MobileNet (Scratch)": (build_mobilenet_scratch, config.MOBILENET_PATH, False),
        "ResNet50 (Transfer Learning)": (build_resnet_transfer, config.RESNET_PATH, True),
        "InceptionV3 (Transfer Learning)": (build_inception_transfer, config.INCEPTION_PATH, True)
    }

    results_summary = []

    for model_name, (builder_function, save_path, is_transfer) in models_dict.items():
        print(f"\n" + "-"*50)
        print(f" TRAINING: {model_name}")
        print("-"*50)
        
        model = builder_function(num_classes)
        
        if is_transfer:
            print(
                "Transfer learning: "
                f"freeze_base={config.TRANSFER_FREEZE_BASE}, "
                f"fine_tune={config.FINE_TUNE}, "
                f"fine_tune_at={config.FINE_TUNE_AT}, "
                f"fine_tune_lr={config.FINE_TUNE_LR}"
            )
        else:
            print("Training from scratch.")
        use_class_weights = config.USE_CLASS_WEIGHTS
        if model_name == "MobileNet (Scratch)" and not config.MOBILENET_USE_CLASS_WEIGHTS:
            use_class_weights = False

        effective_class_weights = class_weights if use_class_weights else None

        history, training_time = train_model(
            model,
            train_ds,
            val_ds,
            effective_class_weights,
            save_path,
            fine_tune=is_transfer and config.FINE_TUNE
        )
        metrics = evaluate_per_class_error(model, test_ds, class_names, model_name)

        best_val_accuracy = _best_metric(history, "val_accuracy")
        best_val_loss = _best_metric(history, "val_loss")

        results_summary.append({
            "model_name": model_name,
            "is_transfer": is_transfer,
            "use_class_weights": use_class_weights,
            "freeze_base": config.TRANSFER_FREEZE_BASE if is_transfer else None,
            "fine_tune": bool(config.FINE_TUNE) if is_transfer else False,
            "fine_tune_at": config.FINE_TUNE_AT if is_transfer and config.FINE_TUNE else None,
            "train_time_sec": round(training_time, 2),
            "train_time_min": round(training_time / 60, 2),
            "best_val_accuracy": None if best_val_accuracy is None else round(best_val_accuracy, 4),
            "best_val_loss": None if best_val_loss is None else round(best_val_loss, 4),
            "test_accuracy": round(metrics["accuracy"], 4),
            "test_macro_f1": round(metrics["macro_f1"], 4),
            "test_weighted_f1": round(metrics["weighted_f1"], 4),
        })

        plot_learning_curves(history, model_name)
        
        print(f"Liberando memoria de {model_name}...")
        del model 
        del history
        tf.keras.backend.clear_session()
        gc.collect() 

    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    _write_summary_csv(results_summary, summary_path)

    print(f"\n" + "="*50)
    print("   Comparacion de tiempos finales")
    print("="*50)
    for row in results_summary:
        print(f"{row['model_name']:<35}: {row['train_time_min']:.2f} minutes")

    print(f"\nResumen de metricas guardado en: {summary_path}")

if __name__ == "__main__":
    main()