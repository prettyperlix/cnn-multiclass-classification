# Informe - Clasificacion multiclase de peces

## 1) Objetivo
Disenar, entrenar y comparar CNN propia y MobileNet desde cero, y ResNet50 e InceptionV3 con transfer learning, usando las mismas particiones de entrenamiento/validacion/test.

## 2) Dataset y particiones
- Carpeta base: fish_image/
- Carga con image_dataset_from_directory y validation_split=0.3.
- Se usa subset="training" para train y subset="validation" para val/test.
- El subset de validacion se divide en 50% validacion y 50% test.
- Semilla fija para reproducibilidad.

## 3) Configuracion principal (config.py)
- Tamano imagen: 224x224
- Batch size: 32
- Learning rate: 3e-4
- Epocas: 50 (EarlyStopping con patience=5)
- Dropout CNN propia: 0.5
- Label smoothing: 0.1
- Weight decay: 1e-4
- Data augmentation: flip, rotation 0.08, zoom 0.15, contrast 0.1
- Mixed precision: True

## 4) Modelos desde cero
### 4.1 CNN propia
Arquitectura base:
- 3 bloques conv + maxpool
- Flatten + Dense(256) + Dropout + Dense softmax

Justificacion (resumen):
- Bloques conv con padding "same" para conservar informacion espacial.
- MaxPooling reduce dimensionalidad y sobreajuste.
- Dropout y weight decay como regularizacion.
- Label smoothing para mejorar calibracion y generalizacion.

### 4.2 MobileNetV2 (Scratch)
Arquitectura base:
- MobileNetV2 sin pesos preentrenados
- GlobalAveragePooling + Dropout + Dense softmax

Justificacion (resumen):
- MobileNetV2 reduce parametros y costo computacional.
- Normalizacion a rango [-1, 1] como recomienda el modelo.
- Dropout ajustado para evitar sobreajuste.

## 5) Transfer Learning
### 5.1 ResNet50
- Pesos: ImageNet
- include_top=False + GAP + Dropout + Dense softmax
- Congelamiento inicial segun TRANSFER_FREEZE_BASE
- Fine-tuning segun FINE_TUNE y FINE_TUNE_AT

Justificacion (resumen):
- ResNet50 captura caracteristicas generales; se ajusta la cabeza al dataset.
- Fine-tuning para adaptar capas profundas a la distribucion submarina.

### 5.2 InceptionV3
- Pesos: ImageNet
- include_top=False + GAP + Dropout + Dense softmax
- Congelamiento inicial segun TRANSFER_FREEZE_BASE
- Fine-tuning segun FINE_TUNE y FINE_TUNE_AT

Justificacion (resumen):
- InceptionV3 combina multiples escalas de convolucion.
- Transfer learning acelera la convergencia en datasets moderados.

## 6) Entrenamiento y validacion
- EarlyStopping con restore_best_weights=True
- ModelCheckpoint guarda el mejor modelo por val_loss
- Class weights opcionales para balance de clases

## 7) Resultados (completar luego de entrenar)
Archivos generados:
- results/*_curves.png (curvas de convergencia)
- results/*_confusion_matrix.png (errores por clase)
- results/*_classification_report.txt/json
- results/summary_metrics.csv (resumen comparativo)

Completar con las metricas de summary_metrics.csv:
| Modelo | Best val acc | Test acc | Macro F1 | Tiempo (min) |
|---|---|---|---|---|
| CNN propia | TODO | TODO | TODO | TODO |
| MobileNet | TODO | TODO | TODO | TODO |
| ResNet50 | TODO | TODO | TODO | TODO |
| InceptionV3 | TODO | TODO | TODO | TODO |

## 8) Discusion
- Comparar precision, F1 y tiempo entre modelos.
- Identificar clases con mas errores en matrices de confusion.
- Comentar si el fine-tuning mejora o empeora frente a congelamiento.
- Relacionar resultados con complejidad del modelo y regularizacion.

## 9) Conclusiones
- Resumir el mejor modelo segun accuracy/F1 y costo de entrenamiento.
- Recomendaciones para mejoras futuras (mas datos, ajuste de hiperparametros).
