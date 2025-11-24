import os
import shutil
from sklearn.model_selection import train_test_split

# Ruta donde tienes la carpeta "train" descargada desde Kaggle
ruta_base = r"C:\Users\ASUS\Desktop\ESTUDIO\Semestres Ingenieria\Semestre 10\IA\PROGRAMAS\ejercicio_AplicacionCNNGatosvsPerros"

ruta_train = os.path.join(ruta_base, "train")

# Crear estructura de carpetas
folders = [
    "data/train/cats",
    "data/train/dogs",
    "data/val/cats",
    "data/val/dogs"
]

for f in folders:
    ruta = os.path.join(ruta_base, f)
    os.makedirs(ruta, exist_ok=True)

# Obtener lista de imágenes
imagenes = os.listdir(ruta_train)

# Separar por clases según el nombre del archivo
cats = [img for img in imagenes if img.startswith("cat")]
dogs = [img for img in imagenes if img.startswith("dog")]

# Dividir cada clase en train / val
train_cats, val_cats = train_test_split(cats, test_size=0.20, random_state=42)
train_dogs, val_dogs = train_test_split(dogs, test_size=0.20, random_state=42)

# Función para mover imágenes
def mover(lista, destino):
    for img in lista:
        origen = os.path.join(ruta_train, img)
        destino_final = os.path.join(ruta_base, destino, img)
        shutil.copy2(origen, destino_final)

# Mover gatos
mover(train_cats, "data/train/cats")
mover(val_cats, "data/val/cats")

# Mover perros
mover(train_dogs, "data/train/dogs")
mover(val_dogs, "data/val/dogs")

print("Proceso completado. Dataset organizado correctamente.")
