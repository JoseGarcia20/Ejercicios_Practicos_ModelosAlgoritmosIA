import heapq
import random

def generar_laberinto(filas, columnas):
    laberinto = []
    for _ in range(filas):
        fila = []
        for _ in range(columnas):
            rand = random.random()
            if rand < 0.2:
                fila.append('#')  # NO HAY PASO - Coste infinito
            elif rand < 0.4:
                fila.append('*')  # HAY PASO PERO CON COSTO ADICIONAL - Coste 3 seg
            else:
                fila.append('.')  # PASO LIBRE - Coste 1 seg
        laberinto.append(fila)
    return laberinto

def distancia_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_estrella(laberinto, inicio, fin):
    filas, columnas = len(laberinto), len(laberinto[0])
    vecinos = [(0, 1), (0, -1), (1, 0), (-1, 0)] #Movimientos posibles que puede hacer el agente

    open_set = []
    heapq.heappush(open_set, (0, inicio))
    
    g_score = { (f, c): float('inf') for f in range(filas) for c in range(columnas) }
    g_score[inicio] = 0
    
    f_score = { (f, c): float('inf') for f in range(filas) for c in range(columnas) }
    f_score[inicio] = distancia_manhattan(inicio, fin)
    
    came_from = {}

    while open_set:
        _, actual = heapq.heappop(open_set)

        if actual == fin:
            camino = []
            while actual in came_from:
                camino.append(actual)
                actual = came_from[actual]
            camino.append(inicio)
            return camino[::-1]

        for df, dc in vecinos:
            vecino = actual[0] + df, actual[1] + dc

            if 0 <= vecino[0] < filas and 0 <= vecino[1] < columnas and laberinto[vecino[0]][vecino[1]] != '#':
                costo = 1 if laberinto[vecino[0]][vecino[1]] == '.' else 4
                
                tentative_g_score = g_score[actual] + costo
                
                if tentative_g_score < g_score[vecino]:
                    came_from[vecino] = actual
                    g_score[vecino] = tentative_g_score
                    f_score[vecino] = tentative_g_score + distancia_manhattan(vecino, fin)
                    if (f_score[vecino], vecino) not in open_set:
                        heapq.heappush(open_set, (f_score[vecino], vecino))

    return None

def imprimir_laberinto(laberinto, camino=None):
    for f, fila in enumerate(laberinto):
        for c, celda in enumerate(fila):
            if camino and (f, c) in camino:
                print('X', end=' ')
            else:
                print(celda, end=' ')
        print()

if __name__ == "__main__":
    laberinto = generar_laberinto(10, 10)
    
    print("Laberinto generado:")
    imprimir_laberinto(laberinto)

    while True:
        try:
            inicio_x = int(input("Ingrese la fila inicial (0-9): "))
            inicio_y = int(input("Ingrese la columna inicial (0-9): "))
            fin_x = int(input("Ingrese la fila final (0-9): "))
            fin_y = int(input("Ingrese la columna final (0-9): "))
            
            inicio = (inicio_x, inicio_y)
            fin = (fin_x, fin_y)

            if not (0 <= inicio_x < 10 and 0 <= inicio_y < 10 and 0 <= fin_x < 10 and 0 <= fin_y < 10):
                print("Coordenadas fuera de rango. Intente de nuevo.")
                continue

            if laberinto[inicio_x][inicio_y] == '#' or laberinto[fin_x][fin_y] == '#':
                print("La posición inicial o final no puede ser un obstáculo. Intente de nuevo.")
                continue
            
            break
        except ValueError:
            print("Entrada no válida. Ingrese números enteros.")

    camino = a_estrella(laberinto, inicio, fin)

    if camino:
        print("Camino encontrado:")
        imprimir_laberinto(laberinto, camino)
        
        tiempo_total = 0
        for paso in camino:
            if laberinto[paso[0]][paso[1]] == '*':
                tiempo_total += 4
            else:
                tiempo_total += 1
        print(f"Tiempo total del recorrido: {tiempo_total} segundos.")
    else:
        print("No se encontró un camino desde el inicio hasta el fin.")
