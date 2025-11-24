# modelo_interactivo.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (150, 150)
INPUT_SHAPE = IMG_SIZE + (3,)


def build_baseline_model(input_shape=INPUT_SHAPE):
    """Modelo CNN sencillo (igual arquitectura que el baseline)."""
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


def _procesar_frame(frame):
    """Procesa un frame para el modelo interactivo."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized.astype("float32") / 255.0
    return img_norm


def entrenamiento_interactivo(modelo_path="modelo_interactivo.h5"):
    """
    Abre la cámara y permite:
      - c: guardar frame etiquetado como GATO (0)
      - d: guardar frame etiquetado como PERRO (1)
      - t: entrenar el modelo con las imágenes guardadas
      - s: guardar el modelo en disco
      - q: salir
    """
    print("\n=== Iniciando entrenamiento interactivo con cámara ===")

    model = build_baseline_model()

    X_data = []
    y_data = []

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Controles:")
    print("  c -> guardar frame como GATO (0)")
    print("  d -> guardar frame como PERRO (1)")
    print("  t -> entrenar modelo con los ejemplos guardados")
    print("  s -> guardar modelo en disco")
    print("  q -> salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # Predicción actual (aunque no esté entrenado todavía)
        img_norm = _procesar_frame(frame)
        img_batch = np.expand_dims(img_norm, axis=0)
        pred = model.predict(img_batch, verbose=0)[0][0]

        if pred >= 0.5:
            etiqueta_pred = f"Perro ({pred:.2f})"
        else:
            etiqueta_pred = f"Gato ({pred:.2f})"

        cv2.putText(frame, etiqueta_pred, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Entrenamiento Interactivo - Perros vs Gatos", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            X_data.append(img_norm)
            y_data.append(0)
            print(f"Ejemplo guardado como GATO. Total ejemplos: {len(y_data)}")

        elif key == ord('d'):
            X_data.append(img_norm)
            y_data.append(1)
            print(f"Ejemplo guardado como PERRO. Total ejemplos: {len(y_data)}")

        elif key == ord('t'):
            if len(y_data) < 4:
                print("Muy pocos ejemplos para entrenar (mínimo 4).")
            else:
                print("Entrenando modelo con los ejemplos actuales...")
                X_array = np.array(X_data)
                y_array = np.array(y_data)
                model.fit(X_array, y_array, epochs=5, batch_size=4, verbose=1)
                print("Entrenamiento incremental finalizado.")

        elif key == ord('s'):
            model.save(modelo_path)
            print(f"Modelo interactivo guardado como: {modelo_path}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada. Fin del entrenamiento interactivo.")
