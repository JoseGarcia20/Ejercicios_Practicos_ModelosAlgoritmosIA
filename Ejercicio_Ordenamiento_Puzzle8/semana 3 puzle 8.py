from collections import deque

# Estado objetivo
objetivo = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]]

# Movimientos posibles: arriba, abajo, izquierda, derecha
movimientos = [(-1,0), (1,0), (0,-1), (0,1)]

# Convierte una matriz a tupla para poder usarla en sets (visitados)
def matriz_a_tupla(m):
    return tuple(tuple(fila) for fila in m)

# Busca la posición del 0 (espacio vacío)
def encontrar_vacio(estado):
    for i in range(3):
        for j in range(3):
            if estado[i][j] == 0:
                return i, j
    # Si no encuentra el 0, lanza error
    raise ValueError("El estado no contiene un 0 (espacio vacío)")

# Genera nuevos estados al mover la ficha
def generar_estados(estado):
    estados = []
    i, j = encontrar_vacio(estado)
    for di, dj in movimientos:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            nuevo = [list(fila) for fila in estado]
            nuevo[i][j], nuevo[ni][nj] = nuevo[ni][nj], nuevo[i][j]
            estados.append(nuevo)
    return estados

# Convierte un estado a string en una sola línea para imprimir
def estado_a_str(estado):
    return "[" + " | ".join(" ".join(str(x) for x in fila) for fila in estado) + "]"

# Búsqueda por amplitud con impresión de la cola
def bfs(inicial):
    cola = deque([(inicial, [])])
    visitados = set()
    visitados.add(matriz_a_tupla(inicial))

    paso = 0
    while cola:
        print(f"\nPaso {paso}: Cola actual -> {len(cola)} estados")
        for idx, (est, _) in enumerate(cola):
            print(f"  Estado {idx}: {estado_a_str(est)}")
        paso += 1

        estado, camino = cola.popleft()

        if estado == objetivo:
            return camino

        for nuevo_estado in generar_estados(estado):
            tupla_estado = matriz_a_tupla(nuevo_estado)
            if tupla_estado not in visitados:
                visitados.add(tupla_estado)
                cola.append((nuevo_estado, camino + [nuevo_estado]))
    return None

# Ejemplo de uso
estado_inicial = [[1, 2, 3],
                  [4, 0, 6],
                  [7, 5, 8]]

solucion = bfs(estado_inicial)

if solucion:
    print("\nSe encontró solución en", len(solucion), "movimientos:")
    for paso in solucion:
        for fila in paso:
            print(fila)
        print("---")
else:
    print("No se encontró solución")