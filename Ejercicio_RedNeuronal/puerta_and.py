from typing import List, Tuple


class Perceptron:
    """Perceptrón binario con función de activación escalón. Aprende con la regla de aprendizaje del perceptrón. 
       Usa una tasa de aprendizaje fija y número máximo de épocas.
    """

    def __init__(self, tasa_aprendizaje: float = 0.1, epocas: int = 20):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.pesos: List[float] = []
        self.sesgo: float = 0.0

    @staticmethod
    def _escalon(x: float) -> int:
        """Función de activación escalón: retorna 1 si x >= 0, en caso contrario 0."""
        return 1 if x >= 0 else 0

    def ajustar(self, X: List[Tuple[int, int]], y: List[int]) -> None:
        """Entrena el perceptrón sobre los datos X con etiquetas y.

        X: lista de pares (x1, x2) binarios.
        y: lista de salidas esperadas (0 o 1).
        """
        # Inicialización de pesos (dos entradas) y sesgo en 0
        self.pesos = [0.0, 0.0]
        self.sesgo = 0.0

        for _ in range(self.epocas):
            errores = 0
            for (x1, x2), objetivo in zip(X, y):
                potencial = self.pesos[0] * x1 + self.pesos[1] * x2 + self.sesgo
                pred = self._escalon(potencial)
                ajuste = self.tasa_aprendizaje * (objetivo - pred)

                # Actualización de pesos y sesgo
                if ajuste != 0:
                    self.pesos[0] += ajuste * x1
                    self.pesos[1] += ajuste * x2
                    self.sesgo += ajuste
                    errores += 1

            # Si no hubo errores en una época, ya aprendió perfectamente
            if errores == 0:
                break

    def predecir(self, X: List[Tuple[int, int]]) -> List[int]:
        """Retorna predicciones (0/1) para una lista de pares binarios."""
        preds: List[int] = []
        for x1, x2 in X:
            potencial = self.pesos[0] * x1 + self.pesos[1] * x2 + self.sesgo
            preds.append(self._escalon(potencial))
        return preds

    def predecir_uno(self, x1: int, x2: int) -> int:
        """Predicción para una sola pareja (x1, x2)."""
        potencial = self.pesos[0] * x1 + self.pesos[1] * x2 + self.sesgo
        return self._escalon(potencial)


def main() -> None:
    # Datos de la puerta AND: solo 1 cuando ambos son 1
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y = [0, 0, 0, 1]

    modelo_entrenado = Perceptron(tasa_aprendizaje=0.1, epocas=20)
    modelo_entrenado.ajustar(X, y)

    print("Comparación de dos modelos para puerta AND")
    print("- Entrenado -> pesos: w1={:.3f}, w2={:.3f}; sesgo b={:.3f}".format(
        modelo_entrenado.pesos[0], modelo_entrenado.pesos[1], modelo_entrenado.sesgo
    ))
    print()

    print("Tabla de verdad y predicciones:")
    print("x1  x2  |  y_esp  y_entrenado")
    for (x1, x2), esperado in zip(X, y):
        p_ent = modelo_entrenado.predecir_uno(x1, x2)
        print(f" {x1}   {x2}   |    {esperado}        {p_ent}    ")

    # Demostración rápida: pedir una predicción al usuario y comparar (opcional)
    try:
        entrada = input("\nIngresa dos bits o Enter para salir: ")
        if entrada.strip():
            a_str, b_str = entrada.strip().split()
            a, b = int(a_str), int(b_str)
            print(f"Entrenado: AND({a}, {b}) = {modelo_entrenado.predecir_uno(a, b)}")
    except Exception:
        # Si la entrada no es válida, simplemente terminamos.
        pass


if __name__ == "__main__":
    main()
