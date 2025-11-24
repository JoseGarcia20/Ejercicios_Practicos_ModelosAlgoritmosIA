from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np
from sklearn.neighbors import KDTree

# Descargar el dataset
data = fetch_california_housing()

# Ver DataSet en formto de pandas DataFrame
housing = fetch_california_housing(as_frame=True)
df = housing.frame

"""
# EXPLORACION INICIAL DEL DATASET DE VIVIENDAS EN CALIFORNIA
#-----------------------------------------------------------------------------------------------
# Ver las primeras filas
print("Primeras filas del dataset:")
print(df.head())
# Información general (tipos de datos, nulos, memoria)
print("\nInformación general:")
print(df.info())
# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())
# Revisar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Visualizar relacion entre variables
# Configuración estética
sns.set(style="whitegrid", palette="viridis")

# Histograma de la variable dependiente
plt.figure(figsize=(7,5))
sns.histplot(df["MedHouseVal"], bins=30, kde=True)
plt.title("Distribución del valor medio de las viviendas (MedHouseVal)")
plt.xlabel("Valor medio (en cientos de miles de dólares)")
plt.ylabel("Frecuencia")
plt.show()

# Matriz de correlación
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()
#-----------------------------------------------------------------------------------------------
"""

# REGRESION LINEAL - PREDDICIONES DE COMPRA DE VIVIENDAS
# Variables independientes y dependiente
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Agregar columna de 1s para el término independiente (bias)
X_b = np.c_[np.ones((X.shape[0], 1)), X.values]

# Aplicacion de ecuacion normal
# Calcular los coeficientes de regresión
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Mostrar los coeficientes
coeficiente = ['Intercepto'] + list(X.columns)
for name, coef in zip(coeficiente, theta):
    print(f"\n{name:12}: {coef:.4f}")

# Predicciones
# Filas de ejemplo
X_new = X.iloc[:5]
X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new.values]

# Predicciones
y_pred = X_new_b.dot(theta)

print("\nPredicciones de valor medio (en cientos de miles de USD):")
print(y_pred)

# Predicciones totales
y_pred_all = X_b.dot(theta)

# Calcular R²
r2 = r2_score(y, y_pred_all)
print(f"\nCoeficiente de determinación R²: {r2:.4f}")

# Ejemplo hipotético de un vecindario
nueva_casa = np.array([[1, 8.5, 30, 6, 1, 800, 2.5, 34.2, -118.4]])  # incluye el 1 para el bias
precio_estimado = nueva_casa.dot(theta)
print(f"\nPrecio estimado de compra: {precio_estimado[0]*100000:.2f} USD")


# Predicciones del modelo (ya las calculaste con ecuación normal)
y_pred_all = X_b.dot(theta)

# Calcular MSE
mse = mean_squared_error(y, y_pred_all)

# Calcular RMSE (raíz cuadrada del MSE)
rmse = np.sqrt(mse)

print(f"\nError cuadrático medio (MSE): {mse:.4f}")
print(f"\nRaíz del error cuadrático medio (RMSE): {rmse:.4f}")


# Ver las llaves del dataset original
#print(data.keys())
