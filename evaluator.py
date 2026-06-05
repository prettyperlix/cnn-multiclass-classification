import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import seaborn as sns
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_learning_curves(history, model_name: str):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()

    plt.tight_layout()
    
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    save_path = os.path.join(RESULTS_DIR, f"{safe_name}_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de convergencia guardado en: {save_path}")
    
    plt.close()

def evaluate_per_class_error(model, val_data, class_names, model_name: str):
    print(f"\nEvaluando error por clase de {model_name}...")
    y_true = []
    y_pred_probs = []

    for images, labels in val_data:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)

    y_pred = np.argmax(y_pred_probs, axis=1)

    labels = list(range(len(class_names)))

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=labels,
        zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    print("\nReporte de clasificacion:")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    save_path = os.path.join(RESULTS_DIR, f"{safe_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Matriz de confusión guardada en: {save_path}")

    plt.close()

    report_text_path = os.path.join(RESULTS_DIR, f"{safe_name}_classification_report.txt")
    with open(report_text_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)
    print(f"Reporte de clasificacion guardado en: {report_text_path}")

    report_json_path = os.path.join(RESULTS_DIR, f"{safe_name}_classification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as report_file:
        json.dump(report_dict, report_file, indent=2)
    print(f"Reporte de clasificacion (JSON) guardado en: {report_json_path}")

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }