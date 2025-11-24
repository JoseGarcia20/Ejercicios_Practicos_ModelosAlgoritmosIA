Se importo un DataSet el cual contenida información de valores de viviendas en diferents zonas de Estados Unidos, con información releveante de acada una de estás viviendas.

Se creo un modelo, el cual en base al dataset, pueda predecir el costo de una vivienda según una característica dada.

Exploración inicial:
Primero realizamos una exploración inicial de los datos, estudiando las filas del dataset, información general, estadísticas descriptivas, valores nulos por columna, obteniendo los siguientes resultados:

Variable objetivo: VALOR MEDIO DE LAS VIVIENDAS.
Se realizo una gráfica, donde se puede evidenciar la distribución del valor medio de las viviendas, lo cual sería la variable objetivo a estudiar en este ejercicio. Por la gráfica, podemos concluir la siguiente información:
-	No es una distribución normal, es asimétrica a la derecha con un sesgo positivo.
-	La mayoría de las viviendas tienen valores medios entre 1 a 2, los cual quiere decir, 100.000 USD a 200.000 USD.
-	Hay menos viviendas en valores costosos o altos.
-	Se muestra un pico fuerte en el extremo derecho, lo que indica un valor tope o limite artificial en el DataSet. 


Del mismo modo, se realizo una matriz de correlación entre las variables principales que tenia el DataSet. Esta es una tabla que mide la fuerza y dirección de la relación inicial entre pares de variables. Los valores que muestra van de -1 a +1, donde +1 es la correlación positiva perfecta, y -1 es la correlación negativa perfecta, y los valores que están en cero, muestra la relación lineal aparente.


Entre las variables a destacar y las importantes para nuestro estudio, obtuvimos los siguientes resultados y conclusiones.
En el intercepto, obtuvimos un valor de -36.9419 el cual es el valor base del modelo (cuando todas las variables son 0). No tiene una interpretación directa, ya que no existen casas con todos los valores en cero.
En la variable MedInc obtuvimos un valor de 0.4367 el cual es positivo y fuerte. Por cada incremento de 1 unidad en el ingreso medio del vecindario, el valor medio de las casas aumenta.
En la variable HouseAge obtuvimos un valor de 0.0094 el cual es ligeramente positivo. A mayor antigüedad promedio del vecindario, ligeramente aumenta el valor. 
En la variable Latitude (-0.4213) nos arroja un valor negativo moderado. A menor latitud (más al sur de California), mayor valor de la vivienda.

Coeficiente de determinación: 
R = 0.6062 el cual mide cuanta variabilidad del precio de las viviendas logra explicar el modelo. Nos muestra un valor razonablemente bueno para un modelo lineal aplicado a datos. Significa que, solo con las variables estructurales y geográficas, ya se está capturando más de la mitad de lo que determina el precio.
El valor del coeficiente de determinación es bueno, ya que este captura más de la mitad de variabilidad.

Predicciones obtenidas:
Predicción de valor medio:
4.13 → 413 000 USD
3.97 → 397 600 USD
3.24 → 324 000 USD
2.41 → 241 000 USD
Predicción obtenida del costo de vivienda: 407.900 USD.
En base a los resultados de la predicción obtenida por el costo total de una vivienda, podemos ver que el valor arrojado, está entre los parámetros de los valores medios, lo que nos indica que el modelo está ajustando correctamente la ecuación normal en el ejercicio presentado.
La predicción obtenida fue coherente dentro de los rangos reales del precio de las viviendas. 

Error cuadrático:
Error cuadrático medio (MSE): 0.5243 -> Este valor mide la magnitud promedio del error al cuadrado entre los valores reales de las viviendas y las predicciones del modelo. 

Raíz del error cuadrático medio (RMSE): 0.7241 -> Es el valor del error cuadrático, pero con la raíz cuadrada, lo cual nos indica el valor de error en las mismas unidades de precio de la vivienda.