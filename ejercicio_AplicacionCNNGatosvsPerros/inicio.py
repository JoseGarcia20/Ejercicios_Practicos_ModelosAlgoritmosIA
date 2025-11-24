from modeloEntrenado import entrenar_kaggle, camara_kaggle
from modeloSinEntrenar import entrenamiento_interactivo

def main():
    while True:
        print("\n===== MENÚ PRINCIPAL - CNN PERROS VS GATOS =====")
        print("1. Entrenar modelo con dataset de Kaggle.")
        print("2. Usar modelo Kaggle entrenado para predecir con la cámara")
        print("3. Entrenamiento interactivo con cámara")
        print("4. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            try:
                ep = input("Número de épocas (por defecto 10): ")
                ep = int(ep) if ep.strip() != "" else 10
            except ValueError:
                ep = 10

            entrenar_kaggle(epochs=ep)

        elif opcion == "2":
            camara_kaggle()

        elif opcion == "3":
            entrenamiento_interactivo()

        elif opcion == "4":
            print("Saliendo del programa...")
            break

        else:
            print("Opción inválida, intente de nuevo.")


if __name__ == "__main__":
    main()
