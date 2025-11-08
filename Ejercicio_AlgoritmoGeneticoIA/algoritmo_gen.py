import numpy as np
import random

# Función objetivo (fitness)
def fitness(individuo):
    x, y, z = individuo
    return np.sin(x**2 + y**2) + np.cos(y**2 + z**2) + np.tanh(x**2 + z**2)


# Crear un individuo (x, y, z)
def crear_individuo():
    return [random.uniform(-5, 5) for _ in range(3)]

# Crear población inicial
def crear_poblacion(n):
    return [crear_individuo() for _ in range(n)]



# Selección por torneo
def seleccion(poblacion, k=3):
    torneo = random.sample(poblacion, k)
    torneo = sorted(torneo, key=fitness, reverse=True)
    return torneo[0]


# Cruce (crossover)
def crossover(padre1, padre2):
    punto = random.randint(1, len(padre1)-1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2


# Mutación
def mutacion(individuo, prob=0.1):
    for i in range(len(individuo)):
        if random.random() < prob:
            individuo[i] += random.uniform(-1, 1) 
    return individuo


# Algoritmo Genético
def algoritmo_genetico(generaciones=25, tam_poblacion=10):
    poblacion = crear_poblacion(tam_poblacion)

    for gen in range(generaciones):
        nueva_poblacion = []
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion(poblacion)
            padre2 = seleccion(poblacion)
            hijo1, hijo2 = crossover(padre1, padre2)
            hijo1 = mutacion(hijo1)
            hijo2 = mutacion(hijo2)
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = sorted(nueva_poblacion, key=fitness, reverse=True)[:tam_poblacion]

        mejor = poblacion[0]
        print(f"Generación {gen+1} | Mejor individuo: {mejor} | Fitness: {fitness(mejor):.4f}")

    return poblacion[0]



# Ejecutar algoritmo
mejor_solucion = algoritmo_genetico()
print("\nMejor solución encontrada:")
print(f"x={mejor_solucion[0]:.4f}, y={mejor_solucion[1]:.4f}, z={mejor_solucion[2]:.4f}")
print(f"f(x,y,z)={fitness(mejor_solucion):.4f}")
