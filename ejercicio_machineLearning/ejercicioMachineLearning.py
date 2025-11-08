import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""
Ejercicio de Machine Learning
Un concesionario de autos quiere predecir el precio de los vehiculos usados que recibe para reventa. La idea es
construir un modelo de Machine Learning, que, basado en el kilometraje y el año defabricacion del vehiculo
pueda estimar su precio.
"""

#Diccionario con informacion de carros usados
autos = {
    "auto_1": {"marca": "Toyota", "modelo": "Corolla", "año_fabricacion": 2018, "kilometraje": 45000, "precio": 62000},
    "auto_2": {"marca": "Mazda", "modelo": "3", "año_fabricacion": 2020, "kilometraje": 25000, "precio": 78000},
    "auto_3": {"marca": "Ford", "modelo": "Focus", "año_fabricacion": 2017, "kilometraje": 60000, "precio": 58000},
    "auto_4": {"marca": "Chevrolet", "modelo": "Cruze", "año_fabricacion": 2019, "kilometraje": 30000, "precio": 70000},
    "auto_5": {"marca": "Volkswagen", "modelo": "Golf", "año_fabricacion": 2016, "kilometraje": 55000, "precio": 65000},
    "auto_6": {"marca": "Honda", "modelo": "Civic", "año_fabricacion": 2015, "kilometraje": 48000, "precio": 60000},
    "auto_7": {"marca": "Nissan", "modelo": "Altima", "año_fabricacion": 2021, "kilometraje": 20000, "precio": 82000},
    "auto_8": {"marca": "Kia", "modelo": "Sportage", "año_fabricacion": 2018, "kilometraje": 42000, "precio": 68000},
    "auto_9": {"marca": "Hyundai", "modelo": "Elantra", "año_fabricacion": 2017, "kilometraje": 53000, "precio": 64000},
    "auto_10": {"marca": "Honda", "modelo": "Civic", "año_fabricacion": 2019, "kilometraje": 35000, "precio": 72000},
    "auto_11": {"marca": "Toyota", "modelo": "Camry", "año_fabricacion": 2016, "kilometraje": 50000, "precio": 68000},
    "auto_12": {"marca": "Ford", "modelo": "Fusion", "año_fabricacion": 2020, "kilometraje": 22000, "precio": 80000},
    "auto_13": {"marca": "Chevrolet", "modelo": "Malibu", "año_fabricacion": 2015, "kilometraje": 47000, "precio": 62000},
    "auto_14": {"marca": "Volkswagen", "modelo": "Jetta", "año_fabricacion": 2018, "kilometraje": 40000, "precio": 67000},
    "auto_15": {"marca": "Mazda", "modelo": "6", "año_fabricacion": 2017, "kilometraje": 58000, "precio": 63000},
    "auto_16": {"marca": "Nissan", "modelo": "Sentra", "año_fabricacion": 2019, "kilometraje": 32000, "precio": 71000},
    "auto_17": {"marca": "Kia", "modelo": "Forte", "año_fabricacion": 2016, "kilometraje": 52000, "precio": 66000},
    "auto_18": {"marca": "Hyundai", "modelo": "Sonata", "año_fabricacion": 2020, "kilometraje": 24000, "precio": 79000},
    "auto_19": {"marca": "Honda", "modelo": "Accord", "año_fabricacion": 2015, "kilometraje": 49000, "precio": 61000},
    "auto_20": {"marca": "Toyota", "modelo": "Yaris", "año_fabricacion": 2018, "kilometraje": 43000, "precio": 66000}
}

#Convertir un diccionario a un DataFrame de pandas
df_autos = pd.DataFrame.from_dict(autos, orient='index')
print(df_autos)

#Definir variables independientes (kilometraje y año de fabricacion) y variable dependiente (precio)
X = df_autos[['kilometraje', 'año_fabricacion']]
y = df_autos['precio']

#Dividir el conjunto de datos en entrenamiento y prueba
"""
Como es un dataset pequeño, usaremos el 80% de los datos para entrenar el modelo y el 20% restante para probar su desempeño.
"""
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Comprobar la division de los datos (Entramiento y Prueba)
print("Tamaño total del dataset:", len(df_autos))
print("Tamaño del conjunto de entrenamiento:", len(x_train))
print("Tamaño del conjunto de prueba:", len(x_test))

#Aplicar regresion lineal
modeloCarros = LinearRegression()

#Entrenar el modelo - Aplicamos datos de entreamiento y prueba
modeloCarros.fit(x_train, y_train)

#Revisar que tanto influye cada variable en la prediccion del precio
print("\nCoeficiente: ", modeloCarros.coef_)
print("Intersección: ", modeloCarros.intercept_)

#Realizar predicciones del conjunto de prueba (Generar precios estimados)
y_pred = modeloCarros.predict(x_test)
#print("\nPredicciones de precios para el conjunto de prueba:", y_pred)

#Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nEvaluación del modelo:")
print("Erro absooluto medicion(MAE):", mae)
print("Error cuadrático medio(RMSE):", rmse)
print("Coeficiente de Determinación (R^2):", r2)
"""El coeficiente de determinacion, se acerca a 1, lo cual indica que el modelo explica bien la variabilidad de los datos."""

#Comparacion de los valores reales vs los predichos
comparacionValores = pd.DataFrame({'Precio Real': y_test, 'Prediccion': y_pred})
print("\nComparación de valores reales vs prediccion:")
print(comparacionValores)

#Prueba con un nuevo vehiculo
nuevoCarro = pd.DataFrame({'kilometraje': [40000], 'año_fabricacion': [2019]})
prediccionPrecio = modeloCarros.predict(nuevoCarro)
print("\nLa predicción del valor comercial del nuevo carro, es de:", prediccionPrecio[0])


