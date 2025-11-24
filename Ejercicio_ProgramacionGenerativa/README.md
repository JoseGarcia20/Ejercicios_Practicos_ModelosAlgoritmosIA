# Programación Genética para aproximar la Ley de Planck

Este proyecto implementa un algoritmo de Programación Genética (PG) que busca aproximar la Ley de Planck a partir de datos experimentales sintéticos. La PG genera y evoluciona expresiones matemáticas (árboles de expresión) que intentan modelar la intensidad espectral de la radiación en función de la longitud de onda y la temperatura.

El código principal está en `ejercicio2.py` y ejecuta el flujo completo: generación de datos, evolución de la población, evaluación del mejor individuo y visualización de resultados.

## Resumen del enfoque

- Se generan datos sintéticos usando una implementación realista de la Ley de Planck con ruido.
- Se define un conjunto de operadores y terminales para construir expresiones (árboles) que el algoritmo evoluciona.
- Se evalúa cada expresión contra los datos experimentales mediante una función de fitness basada en el error cuadrático medio en espacio logarítmico.
- Se aplican selección, cruce (intercambio de subárboles), mutación y elitismo durante múltiples generaciones.
- Se reportan métricas (QME, R²), la mejor expresión encontrada y gráficos comparativos.

## Datos sintéticos y Ley de Planck

El método `generar_datos_experimentales` crea:

- Un rango de longitudes de onda en metros: `1e-7` a `3e-6` (50 puntos).
- La intensidad “real” con la Ley de Planck implementada en `ley_planck_real(λ, T)`, y añade ruido gaussiano relativo (5% del máximo) para simular mediciones experimentales. Se garantiza positividad mínima para evitar problemas numéricos.

Implementación usada para la comparación (forma específica del código):

I(λ, T) = (2 · π · h · c²) / [ λ⁵ · (exp(h·c/(λ·k·T)) − 1) ]

Nota: esta variante incluye el factor `2·π` en el numerador. Dependiendo de la convención/unidades, la forma más común aparece como `(2·h·c²)/λ⁵ · 1/(exp(hc/(λkT)) − 1)`. Aquí se usa explícitamente `2·π·h·c²`.

Constantes físicas utilizadas (SI):

- h = 6.626e-34 (Planck)
- c = 3.0e8 (velocidad de la luz)
- k = 1.381e-23 (Boltzmann)

## Representación de individuos (árboles de expresión)

- Operadores: `+`, `-`, `*`, `/`.
- Terminales: `λ` (longitud de onda), `T` (temperatura), `h`, `c`, `k`, constantes `1`, `2`, `5`, y `π`.

Importante sobre símbolos: según la codificación del archivo, algunos caracteres pueden verse como `ׯ` (λ) y `�?` (π). Conceptualmente, representan `λ` y `π` respectivamente.

Las expresiones se construyen recursivamente hasta una profundidad máxima (por defecto 3 al crear individuos aleatorios). Internamente se representan como listas anidadas, por ejemplo: `['*', '2', ['+', 'λ', 'T']]`.

## Evaluación de expresiones y robustez numérica

El método `evaluar_expresion(expr, λ)` evalúa la expresión en un valor de `λ`:

- Resuelve terminales a sus valores (p. ej., `T` a la temperatura fija del experimento).
- Aplica de forma segura los operadores, devolviendo `inf` si ocurre división por cero o si se detectan `NaN`, `Inf` o números complejos.

La función de fitness (aptitud) recorre todos los `λ` y:

1. Evalúa la expresión en cada `λ`.
2. Si hay valores inválidos (negativos, complejos, `NaN`/`Inf`, o extremadamente grandes), la aptitud se anula (retorna 0).
3. Calcula el error cuadrático medio en espacio logarítmico: `QME = mean((log1p(pred) - log1p(y_exp))^2)`.
4. Define `fitness = 1 / (1 + QME)`, acotado a `[0, 1]`.

Racional: usar `log1p` suaviza rangos amplios y evita problemas cuando la intensidad es muy pequeña.

## Bucle evolutivo

1. Población inicial: individuos aleatorios (árboles) con operadores/terminales definidos.
2. Evaluación: se calcula el fitness para cada individuo.
3. Selección: se ordena por fitness; se usa un torneo entre los mejores (top 20) para elegir padres.
4. Elitismo: el mejor individuo pasa directo a la siguiente generación.
5. Cruce (crossover): intercambio de subárboles entre dos padres para crear un hijo.
6. Mutación: dos modalidades
   - Mutación puntual de subárboles con probabilidad; o
   - Reemplazo completo por un individuo nuevo (profundidad limitada).
7. Repetición: el proceso se itera por `generaciones` (por defecto 20).

Durante la evolución, el script imprime estadísticas por generación y, cada 5 generaciones (y en la última), muestra la mejor expresión completa (original y una versión simplificada que elimina operaciones redundantes básicas como multiplicaciones por 1 o sumas con 0).

## Métricas y reporte final

Para el mejor individuo encontrado:

- Se calculan y reportan:
  - QME (Error Cuadrático Medio entre predicción y dato experimental),
  - R² (coeficiente de determinación, usando `sklearn.metrics.r2_score`).
- Se muestra una interpretación cualitativa del R².
- Se imprime la expresión final (y su versión simplificada) en forma legible y también como función `I(λ, T)`.
- Gráficas:
  - Curva de la Ley de Planck “real”,
  - Puntos experimentales con ruido,
  - Curva del “modelo evolucionado”.

## Parámetros principales

- `poblacion_size`: tamaño de la población (por defecto 50; el `main` usa 30).
- `generaciones`: número de iteraciones evolutivas (por defecto 20).
- `temperatura`: temperatura en Kelvin usada para generar los datos sintéticos (por defecto 5000 K).

Para modificar estos parámetros, cambia los argumentos al crear `ProgramaGeneticoPlanck` en el bloque `if __name__ == "__main__":`.

## Dependencias

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn (solo para `r2_score`)

Instalación rápida (ejemplos):

```bash
pip install numpy matplotlib scikit-learn
```

## Ejecución

Desde el directorio del proyecto:

```bash
python ejercicio2.py
```

El script:

- Inicializa la PG con `poblacion_size=30`, `generaciones=20`, `temperatura=5000`.
- Ejecuta la evolución y muestra progreso por generación.
- Reporta métricas finales y abre una ventana con las gráficas comparativas.

## Notas y limitaciones

- Operadores limitados: la PG usa solo `+ - * /` y constantes básicas. No hay funciones como `exp`, `log`, `pow` disponibles para el árbol, por lo que recuperar exactamente la forma analítica de Planck es difícil; el objetivo es aproximar el comportamiento.
- Robustez numérica: se filtran evaluaciones que produzcan `NaN`, `Inf` o complejos, asignando fitness 0 a expresiones inválidas. Esto estabiliza la búsqueda pero puede penalizar demasiado árboles “casi válidos”.
- Escala de valores: el uso de `log1p` en la función de costo ayuda cuando las intensidades varían en órdenes de magnitud.
- Codificación de caracteres: si ves símbolos extraños como `ׯ` (por `λ`) o `�?` (por `π`), es un tema de codificación. Conceptualmente, `ׯ` = `λ`, `�?` = `π`.

## Estructura de alto nivel del código

- `ProgramaGeneticoPlanck`
  - `generar_datos_experimentales`: crea λ, `I_real`, `I_experimental`.
  - `ley_planck_real`: implementación de referencia con recortes numéricos para evitar desbordes.
  - `crear_individuo_aleatorio`: genera árboles aleatorios.
  - `evaluar_expresion`: evalúa un árbol de forma segura.
  - `fitness`: calcula QME en log1p y mapea a [0, 1].
  - `expresion_a_string` y `expresion_a_formula_latex`: utilidades de visualización.
  - `simplificar_expresion`: simplificaciones algebraicas básicas.
  - `cruzar`, `mutar`, `copiar_expresion`: operadores genéticos.
  - `imprimir_expresion_completa`: salida formateada de la mejor expresión.
  - `ejecutar_evolucion`: bucle principal de la PG.
  - `analizar_resultados` y `graficar_resultados`: métricas y visualización final.

## Ideas para mejorar

- Ampliar el conjunto de funciones (p. ej., `exp`, `log`, potencias) con protecciones numéricas.
- Implementar división protegida y otras operaciones seguras para reducir `Inf/NaN`.
- Explorar otras funciones de fitness (por ejemplo, ponderaciones por regiones espectrales o métricas robustas).
- Ajustar hiperparámetros (población, profundidad de árboles, tasas de mutación/cruce).

