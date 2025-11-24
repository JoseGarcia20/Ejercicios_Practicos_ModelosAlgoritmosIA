# CNN Perros vs Gatos (webcam + dataset Kaggle)

## Descripcion general
- App en Python que entrena y ejecuta un clasificador CNN binario (gato/perro).
- Dos flujos: 1) usa dataset Dogs vs Cats de Kaggle para entrenar y luego predecir en vivo; 2) entrenamiento rapido interactivo tomando ejemplos con la camara web.
- Los modelos se guardan en disco (`modelo_kaggle.h5` o `modelo_interactivo.h5`) y se usan para clasificar frames de la camara en tiempo real.

## Requisitos
- Python 3.10+ en Windows.
- Instalar dependencias base:
```
pip install tensorflow opencv-python numpy scikit-learn
```
- Para descargar el dataset necesitas la CLI de Kaggle configurada con tu `kaggle.json` en `%USERPROFILE%\.kaggle\`.

## Estructura esperada de carpetas
- `train/` y `test1/`: carpetas crudas del dataset Dogs vs Cats de Kaggle (sin subcarpetas).
- `data/train/cats` y `data/train/dogs`: imagenes organizadas para entrenamiento.
- `data/val/cats` y `data/val/dogs`: imagenes organizadas para validacion.
- `modelo_kaggle.h5`: modelo entrenado con dataset Kaggle (incluido si ya lo guardaste).
- `modelo_interactivo.h5`: modelo guardado desde el flujo interactivo (se genera al guardar).
- Ajusta la ruta base en `modeloEntrenado.py` y `organizarImagenesPorAnimales.py` si mueves el proyecto.

## Script de organizacion del dataset Kaggle
1) Descarga el dataset Dogs vs Cats (`train.zip` y `test1.zip`) con Kaggle CLI:
```
kaggle competitions download -c dogs-vs-cats
```
2) Extrae `train.zip` dentro de `train/` en la raiz del proyecto.
3) Ejecuta el script para crear `data/train` y `data/val` con subcarpetas `cats` y `dogs`:
```
python organizarImagenesPorAnimales.py
```
El script copia las imagenes, separa por nombre (`cat*` o `dog*`) y hace split 80/20 entre entrenamiento y validacion.

## Uso del menu principal (`python inicio.py`)
- Opcion 1: Entrenar modelo con dataset Kaggle (`entrenar_kaggle`). Usa generadores de Keras sobre `data/train` y `data/val`; pide numero de epocas (por defecto 10); guarda `modelo_kaggle.h5`.
- Opcion 2: Usar modelo Kaggle entrenado (`camara_kaggle`). Carga `modelo_kaggle.h5`, abre la camara y muestra en pantalla la probabilidad de perro (>0.5) o gato.
- Opcion 3: Entrenamiento interactivo (`entrenamiento_interactivo`). Crea un modelo en blanco y te deja capturar ejemplos manualmente con la camara.
- Opcion 4: Salir.

## Controles en el modo interactivo
- Ventana "Entrenamiento Interactivo - Perros vs Gatos".
- Teclas: `c` guarda frame como GATO (0); `d` guarda frame como PERRO (1); `t` entrena 5 epocas sobre los ejemplos guardados (minimo 4 ejemplos); `s` guarda el modelo como `modelo_interactivo.h5`; `q` cierra.
- En pantalla siempre veras la prediccion actual del modelo (aunque no este entrenado).

## Flujo del modelo Kaggle
- `modeloEntrenado.py` define la CNN baseline (3 bloques Conv2D + MaxPool, denso de 512 y salida sigmoide).
- El entrenamiento usa `ImageDataGenerator` con reescalado 1/255 en `data/train` y `data/val`.
- Durante prediccion, cada frame se reescala a 150x150 RGB y se normaliza antes de pasar al modelo.

## Notas y tips
- Si tus carpetas no estan en la misma ruta de ejemplo, actualiza `base_dir` y `ruta_base` dentro de los scripts.
- Asegurate de que la camara funcione antes de abrirla (OpenCV la usa en indice 0).
- Limpia `data/` si necesitas rehacer el split; el script de organizacion vuelve a crear la estructura.
