#importar conjunto de datos
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Descargar el dataset
data = fetch_california_housing()


# Ver DataSet en formto de pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)


# Agregar la columna objetivo (valor medio de las casas)
df['ValorMedioCasas'] = data.target * 100000  # Convertir a dólares


# Cambiar nombres  de columnas a español
df.rename(columns={
    'MedInc': 'IngresoMedio',
    'HouseAge': 'EdadVivienda',
    'AveRooms': 'PromedioHabitaciones',
    'AveBedrms': 'PromedioDormitorios',
    'Population': 'Poblacion',
    'AveOccup': 'PromedioOcupacion',
    'Latitude': 'Latitud',
    'Longitude': 'Longitud'
}, inplace=True)


# Relacion entre variables dependientes y variables independientes
sns.set(style="whitegrid", palette="viridis")

# Histograma de la variable dependiente
plt.figure(figsize=(7,5)) #Crear nueva figura y dar tamaño
sns.histplot(df["ValorMedioCasas"], bins=30, kde=True) #Crear histograma con 30 barras y línea KDE
plt.title("Distribución del valor medio de las viviendas (ValorMedioCasas)") #Agregar título
plt.xlabel("Valor medio (en cientos de miles de dólares)") #Etiqueta eje X
plt.ylabel("Frecuencia") #Etiqueta eje Y
plt.show() #Mostrar gráfico


"""
    df.corr() - calcula la variable correlacion entre todas las variables num del dset.
    annot = true - muestra valores numericos dentro de la selda.
    fmt=".2f" - formato de numeros. 
    cmap="coolwarm" - paleta de colores
"""
# Matriz de correlación
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()

# Mostrar todas las relaciones entre variables independiente y dependiente
caracteristicas = df.columns[:-1]  # Todas las columnas excepto la última
VariableObjetivo = 'ValorMedioCasas' # Variable dependiente

for i in caracteristicas:
    plt.figure(figsize=(7,5))
    plt.scatter(df[i], df[VariableObjetivo], alpha=0.5)
    plt.title(f"Relación entre {i} y {VariableObjetivo}")
    plt.xlabel(i)
    plt.ylabel(VariableObjetivo)
    plt.show()



# Mostrar primeras filas del DataFrame
print("Primeras filas del dataset:")
print(df.head())
