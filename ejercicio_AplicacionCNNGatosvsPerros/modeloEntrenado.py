# modelo_kaggle.py

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta base del dataset de Kaggle
base_dir = r"C:\Users\ASUS\Desktop\ESTUDIO\Semestres Ingenieria\Semestre 10\IA\PROGRAMAS\ejercicio_AplicacionCNNGatosvsPerros\data"

IMG_SIZE = (150, 150)
INPUT_SHAPE = IMG_SIZE + (3,)


def get_generators(batch_size=32):
    """Crea los generadores de imágenes para entrenamiento y validación."""
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(base_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_generator = val_datagen.flow_from_directory(
        directory=os.path.join(base_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary"
    )

    return train_generator, val_generator


def build_baseline_model(input_shape=INPUT_SHAPE):
    """Construye el modelo CNN baseline (desde cero)."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 0 = gato, 1 = perro
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def entrenar_kaggle(epochs=10, batch_size=32, modelo_path="modelo_kaggle.h5"):
    """
    Entrena el modelo baseline con el dataset de Kaggle y guarda el modelo.
    """
    print("\n=== Cargando generadores de Kaggle ===")
    train_gen, val_gen = get_generators(batch_size=batch_size)

    print("\n=== Construyendo modelo baseline (Kaggle) ===")
    model = build_baseline_model()
    model.summary()

    print(f"\n=== Entrenando modelo por {epochs} épocas ===")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )

    print(f"\n=== Guardando modelo entrenado en: {modelo_path} ===")
    model.save(modelo_path)
    print("Modelo guardado correctamente.")

    # Mostrar métricas finales en consola
    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    print("\n=== Resultados finales (Kaggle baseline) ===")
    print(f"Accuracy entrenamiento: {final_acc:.4f}")
    print(f"Accuracy validación:    {final_val_acc:.4f}")
    print(f"Loss entrenamiento:     {final_loss:.4f}")
    print(f"Loss validación:        {final_val_loss:.4f}")

    return model, history


def _procesar_frame_baseline(frame):
    """Procesa un frame para el modelo baseline Kaggle."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)
    return img_batch


def camara_kaggle(modelo_path="modelo_kaggle.h5"):
    """
    Abre la cámara y utiliza el modelo entrenado con Kaggle
    para clasificar cada frame como perro o gato.
    """
    if not os.path.exists(modelo_path):
        print(f"No se encontró el archivo de modelo: {modelo_path}")
        print("Primero entrena el modelo con la opción correspondiente.")
        return

    print(f"\n=== Cargando modelo desde {modelo_path} ===")
    model = tf.keras.models.load_model(modelo_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Cámara activa. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        img_batch = _procesar_frame_baseline(frame)
        pred = model.predict(img_batch, verbose=0)[0][0]

        if pred >= 0.5:
            etiqueta = f"Perro ({pred:.2f})"
        else:
            etiqueta = f"Gato ({pred:.2f})"

        cv2.putText(frame, etiqueta, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Modelo Kaggle - Perros vs Gatos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
