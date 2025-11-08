Ejercicio: Juego del taxi (Open AI Gym)
Este ejercicio consiste en la aplicación de un agente de Q‑Learning en el entorno discreto Taxi‑v3 para maximizar recompensas: recoger y dejar al pasajero en el destino (+20), penalización por cada paso (−1) y por acciones ilegales (−10).

Mide y muestra durante el entrenamiento: recompensa por episodio, conteo acumulado de episodios exitosos, pasos usados, ε actual y la estabilidad de Q (|ΔQ| medio por paso).
El objetivo, es dejar al agente en el tablero sin nada de conocimiento de su entorno, y a medida que vaya haciendo recorridos y haya completado episodios, este pueda entender su entorno y obtener mejores recompensas en base al entrenamiento.

Algunos parámetros del ejercicio:
- Cantidad máxima de pasos: 200
- Numero de episodios para entrenamiento: 2000
- Tasa de aprendizaje Alpha.
- Descuento Gamma.
- Debe tener instalado Gym/Matplotlib para la ejecución.

Ejecución:
Se presentará una tabla la cual muestra:
- Número del episodio.
- La recompensa recogida por el taxi.
- El número de éxitos acumulados.
- El numero de pasos alcanzadas por episodio.
- El porcentaje que falta por conocer del tablero.
- Cambio medio absoluto de Q por cada paso.

Graficas al terminar la ejecución:
- Grafica de recompensas por episodio: EN esta grafica podremos medir, si en base a los episodios que ah recorrido o ha tenido el agente, se ha podido ver un incremento en el aprendizaje, ya que en base a más aprendizaje, puede tomar mejores rutas, terminar en menor tiempo y así obtener mejores recompensas.
- Estabilidad de Q: Muestra el Q medio por episodio, y de esta manera, podemos presentar la convergencia del modelo.