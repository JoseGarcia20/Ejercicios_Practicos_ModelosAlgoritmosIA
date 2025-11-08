from collections import deque
import os, time

class EntornoLaberinto:
    def __init__(self, tamaño=6, meta=None, obstaculos=None):
        self.tamaño = tamaño
        self.estado_inicial = (0, 0)
        self.meta = meta if meta else (tamaño - 1, tamaño - 1)
        self.obstaculos = obstaculos if obstaculos else set()

    def acciones_posibles(self, estado):
        x, y = estado
        posibles = [
            ("ARRIBA", (x - 1, y)),
            ("ABAJO", (x + 1, y)),
            ("IZQUIERDA", (x, y - 1)),
            ("DERECHA", (x, y + 1))
        ]
        acciones = []
        for accion, (nx, ny) in posibles:
            if (0 <= nx < self.tamaño) and (0 <= ny < self.tamaño):
                if (nx, ny) not in self.obstaculos:
                    acciones.append((accion, (nx, ny)))
        return acciones

class AgenteBFS:
    def __init__(self, entorno):
        self.entorno = entorno
        self.estado_actual = entorno.estado_inicial
        self.camino = self.buscar_camino()

    def buscar_camino(self):
        inicio = self.entorno.estado_inicial
        meta = self.entorno.meta
        visitados = {inicio}
        cola = deque([(inicio, [])])  # (estado, camino hasta ahora)

        while cola:
            estado, camino = cola.popleft()
            if estado == meta:
                return camino  # devuelve la secuencia de pasos

            for accion, nuevo_estado in self.entorno.acciones_posibles(estado):
                if nuevo_estado not in visitados:
                    visitados.add(nuevo_estado)
                    cola.append((nuevo_estado, camino + [(accion, nuevo_estado)]))
        return None  # no hay camino

    def mostrar_laberinto(self):
        os.system("cls" if os.name == "nt" else "clear")
        for i in range(self.entorno.tamaño):
            fila = []
            for j in range(self.entorno.tamaño):
                if (i, j) == self.estado_actual:
                    fila.append("A")  # Agente
                elif (i, j) == self.entorno.meta:
                    fila.append("M")  # Meta
                elif (i, j) in self.entorno.obstaculos:
                    fila.append("X")  # Obstáculo
                else:
                    fila.append(".")  # Espacio libre
            print(" ".join(fila))
        print("\n")

    def ejecutar(self):
        if not self.camino:
            print("No existe camino hacia la meta")
            return
        pasos = 0
        for accion, nuevo_estado in self.camino:
            self.mostrar_laberinto()
            print(f"Paso {pasos}: {accion} -> {nuevo_estado}")
            self.estado_actual = nuevo_estado
            pasos += 1
            time.sleep(0.7)
        self.mostrar_laberinto()
        print(f"Meta alcanzada en {pasos} pasos")


if __name__ == "__main__":
    # Definición manual
    tamaño = 6
    meta = (5, 5)
    obstaculos = {(1,0), (3,1), (2,2), (3,4), (5,1), (4,4)}  # 👈 Aquí defines tú los obstáculos

    entorno = EntornoLaberinto(tamaño=tamaño, meta=meta, obstaculos=obstaculos)
    agente = AgenteBFS(entorno)
    agente.ejecutar()
