import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class ProgramaGeneticoPlanck:
    def __init__(self, poblacion_size=50, generaciones=20, temperatura=5000):
        self.poblacion_size = poblacion_size
        self.generaciones = generaciones
        self.temperatura = temperatura
        
        # Constantes físicas reales
        self.h = 6.626e-34  # Constante de Planck
        self.c = 3.0e8      # Velocidad de la luz
        self.k = 1.381e-23  # Constante de Boltzmann
        
        # Generar datos experimentales sintéticos
        self.generar_datos_experimentales()
        
        # Operadores disponibles
        self.operadores = ['+', '-', '*', '/']
        self.terminales = ['λ', 'T', 'h', 'c', 'k', '2', '1', '5', 'π']
        
        # Mejores individuos por generación
        self.mejor_fitness_historico = []
        self.mejor_expresion_historico = []
        self.mejor_expresion_str_historico = []
    
    def generar_datos_experimentales(self):
        """Genera datos sintéticos usando la Ley de Planck real"""
        # Rango de longitudes de onda en metros
        self.lambda_values = np.linspace(1e-7, 3e-6, 50)
        
        # Calcular intensidad usando la Ley de Planck real
        self.intensidad_real = self.ley_planck_real(self.lambda_values, self.temperatura)
        
        # Añadir ruido experimental
        ruido = np.random.normal(0, 0.05 * np.max(self.intensidad_real), len(self.intensidad_real))
        self.intensidad_experimental = self.intensidad_real + ruido
        self.intensidad_experimental = np.maximum(self.intensidad_experimental, 1e-30)
    
    def ley_planck_real(self, lambda_val, T):
        """Calcula la Ley de Planck real para comparación"""
        exponente = (self.h * self.c) / (lambda_val * self.k * T)
        exponente = np.clip(exponente, -100, 100)
        denominador = lambda_val**5 * (np.exp(exponente) - 1)
        denominador = np.maximum(denominador, 1e-100)
        return (2 * np.pi * self.h * self.c**2) / denominador
    
    def crear_individuo_aleatorio(self, profundidad_max=3):
        """Crea una expresión aleatoria"""
        if profundidad_max == 0 or random.random() < 0.5:
            return random.choice(self.terminales)
        else:
            operador = random.choice(self.operadores)
            return [operador, 
                   self.crear_individuo_aleatorio(profundidad_max-1),
                   self.crear_individuo_aleatorio(profundidad_max-1)]
    
    def evaluar_expresion(self, expresion, lambda_val):
        """Evalúa una expresión evitando números complejos"""
        try:
            if expresion == 'λ':
                return lambda_val
            elif expresion == 'T':
                return self.temperatura
            elif expresion == 'h':
                return self.h
            elif expresion == 'c':
                return self.c
            elif expresion == 'k':
                return self.k
            elif expresion == 'π':
                return np.pi
            elif expresion in ['2', '1', '5']:
                return float(expresion)
            
            if isinstance(expresion, list):
                operador = expresion[0]
                arg1 = self.evaluar_expresion(expresion[1], lambda_val)
                arg2 = self.evaluar_expresion(expresion[2], lambda_val)
                
                if np.iscomplex(arg1) or np.iscomplex(arg2):
                    return float('inf')
                if np.isinf(arg1) or np.isinf(arg2) or np.isnan(arg1) or np.isnan(arg2):
                    return float('inf')
                
                if operador == '+':
                    return arg1 + arg2
                elif operador == '-':
                    return arg1 - arg2
                elif operador == '*':
                    return arg1 * arg2
                elif operador == '/':
                    if arg2 == 0:
                        return float('inf')
                    return arg1 / arg2
            
            return float('inf')
            
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return float('inf')
    
    def fitness(self, expresion):
        """Calcula el fitness"""
        try:
            predicciones = []
            for lambda_val in self.lambda_values:
                pred = self.evaluar_expresion(expresion, lambda_val)
                
                if (np.isinf(pred) or np.isnan(pred) or 
                    np.iscomplex(pred) or pred < 0 or pred > 1e100):
                    return 0
                
                predicciones.append(float(pred))
            
            predicciones = np.array(predicciones, dtype=float)
            
            if np.any(np.iscomplex(predicciones)):
                return 0
            
            qme = np.mean((np.log1p(predicciones) - np.log1p(self.intensidad_experimental))**2)
            fitness_val = 1 / (1 + qme)
            return max(0, min(fitness_val, 1))
            
        except:
            return 0
    
    def expresion_a_string(self, expresion, parentesis_externos=True):
        """Convierte la expresión del árbol a string legible con formato mejorado"""
        if isinstance(expresion, list):
            izquierda = self.expresion_a_string(expresion[1], False)
            derecha = self.expresion_a_string(expresion[2], False)
            
            # Determinar si necesitamos paréntesis
            necesita_parentesis = parentesis_externos
            
            resultado = f"{izquierda} {expresion[0]} {derecha}"
            if necesita_parentesis:
                resultado = f"({resultado})"
            
            return resultado
        else:
            return str(expresion)
    
    def expresion_a_formula_latex(self, expresion):
        """Convierte la expresión a formato LaTeX para mejor visualización"""
        if isinstance(expresion, list):
            izquierda = self.expresion_a_formula_latex(expresion[1])
            derecha = self.expresion_a_formula_latex(expresion[2])
            
            if expresion[0] == '*':
                return f"{izquierda} \\cdot {derecha}"
            elif expresion[0] == '/':
                return f"\\frac{{{izquierda}}}{{{derecha}}}"
            else:
                return f"{izquierda} {expresion[0]} {derecha}"
        else:
            # Convertir símbolos a LaTeX
            if expresion == 'π':
                return '\\pi'
            elif expresion == 'λ':
                return '\\lambda'
            else:
                return str(expresion)
    
    def simplificar_expresion(self, expresion):
        """Intenta simplificar la expresión eliminando operaciones redundantes"""
        if not isinstance(expresion, list):
            return expresion
        
        izquierda = self.simplificar_expresion(expresion[1])
        derecha = self.simplificar_expresion(expresion[2])
        
        # Simplificar operaciones con 1 y 0
        if expresion[0] == '*':
            if izquierda == '1':
                return derecha
            if derecha == '1':
                return izquierda
            if izquierda == '0' or derecha == '0':
                return '0'
        elif expresion[0] == '+':
            if izquierda == '0':
                return derecha
            if derecha == '0':
                return izquierda
        
        return [expresion[0], izquierda, derecha]
    
    def cruzar(self, padre1, padre2):
        """Cruza dos individuos"""
        def encontrar_nodos(expresion):
            if not isinstance(expresion, list):
                return [expresion]
            nodos = [expresion]
            for subexp in expresion[1:]:
                nodos.extend(encontrar_nodos(subexp))
            return nodos
        
        nodos_padre1 = encontrar_nodos(padre1)
        nodos_padre2 = encontrar_nodos(padre2)
        
        if len(nodos_padre1) > 1 and len(nodos_padre2) > 1:
            hijo = self.copiar_expresion(padre1)
            nodo_a_reemplazar = random.choice([n for n in nodos_padre1 if n != padre1])
            reemplazo = random.choice(nodos_padre2)
            
            def reemplazar_nodo(exp, objetivo, nuevo):
                if exp == objetivo:
                    return self.copiar_expresion(nuevo)
                if isinstance(exp, list):
                    return [exp[0]] + [reemplazar_nodo(sub, objetivo, nuevo) for sub in exp[1:]]
                return exp
            
            return reemplazar_nodo(hijo, nodo_a_reemplazar, reemplazo)
        
        return padre1
    
    def mutar(self, expresion, prob_mutacion=0.2):
        """Aplica mutación"""
        if random.random() > prob_mutacion:
            return expresion
        
        if not isinstance(expresion, list):
            return random.choice(self.terminales)
        
        if random.random() < 0.5:
            return [expresion[0]] + [self.mutar(sub, prob_mutacion) for sub in expresion[1:]]
        else:
            return self.crear_individuo_aleatorio(profundidad_max=2)
    
    def copiar_expresion(self, expresion):
        """Copia profunda de una expresión"""
        if isinstance(expresion, list):
            return [expresion[0]] + [self.copiar_expresion(sub) for sub in expresion[1:]]
        return expresion
    
    def imprimir_expresion_completa(self, expresion, titulo="EXPRESIÓN MATEMÁTICA"):
        """Imprime la expresión completa de manera formateada"""
        expresion_str = self.expresion_a_string(expresion)
        expresion_simplificada = self.simplificar_expresion(expresion)
        expresion_simplificada_str = self.expresion_a_string(expresion_simplificada)
        
        print(f"\n╔{'═'*70}╗")
        print(f"║ {titulo:^68} ║")
        print(f"╠{'═'*70}╣")
        print(f"║ {'Expresión original:':<68} ║")
        print(f"║ {expresion_str:<68} ║")
        print(f"║ {'':<68} ║")
        print(f"║ {'Expresión simplificada:':<68} ║")
        print(f"║ {expresion_simplificada_str:<68} ║")
        print(f"╚{'═'*70}╝")
        
        # Mostrar también en formato de función
        print(f"\nFUNCIÓN MATEMÁTICA:")
        print(f"I(λ,T) = {expresion_simplificada_str}")
        
        return expresion_simplificada_str
    
    def ejecutar_evolucion(self):
        """Ejecuta el algoritmo evolutivo principal con impresión mejorada"""
        print("╔" + "═"*78 + "╗")
        print("║" + "PROGRAMACIÓN GENÉTICA - LEY DE PLANCK".center(78) + "║")
        print("╠" + "═"*78 + "╣")
        print(f"║ ▪ Temperatura: {self.temperatura} K".ljust(78) + "║")
        print(f"║ ▪ Población: {self.poblacion_size} individuos".ljust(78) + "║")
        print(f"║ ▪ Generaciones: {self.generaciones}".ljust(78) + "║")
        print(f"║ ▪ Puntos de datos: {len(self.lambda_values)}".ljust(78) + "║")
        print("╚" + "═"*78 + "╝")
        
        # Crear población inicial
        poblacion = [self.crear_individuo_aleatorio() for _ in range(self.poblacion_size)]
        
        for generacion in range(self.generaciones):
            # Evaluar fitness
            fitness_poblacion = []
            for ind in poblacion:
                fit = self.fitness(ind)
                fitness_poblacion.append((ind, fit))
            
            fitness_poblacion.sort(key=lambda x: x[1], reverse=True)
            mejor_individuo, mejor_fitness = fitness_poblacion[0]
            
            self.mejor_fitness_historico.append(mejor_fitness)
            self.mejor_expresion_historico.append(mejor_individuo)
            
            # Calcular métricas
            predicciones = []
            for lambda_val in self.lambda_values:
                pred = self.evaluar_expresion(mejor_individuo, lambda_val)
                if np.isreal(pred) and not (np.isinf(pred) or np.isnan(pred)):
                    predicciones.append(float(pred))
                else:
                    predicciones.append(0)
            
            predicciones = np.array(predicciones, dtype=float)
            
            if len(predicciones) > 0 and np.all(np.isfinite(predicciones)):
                qme = np.mean((predicciones - self.intensidad_experimental)**2)
                try:
                    r2 = r2_score(self.intensidad_experimental, predicciones)
                except:
                    r2 = -1
            else:
                qme = float('inf')
                r2 = -1
            
            # Mostrar progreso con formato mejorado
            print(f"\n┌{'─'*78}┐")
            print(f"│ GENERACIÓN {generacion + 1:2d}/{self.generaciones}".ljust(78) + "│")
            print(f"├{'─'*78}┤")
            print(f"│ ▪ Fitness: {mejor_fitness:.6f}".ljust(78) + "│")
            print(f"│ ▪ Error Cuántico Medio: {qme:.2e}".ljust(78) + "│")
            print(f"│ ▪ Coeficiente R²: {r2:.4f}".ljust(78) + "│")
            print(f"└{'─'*78}┘")
            
            # Mostrar expresión completa cada 5 generaciones o en la última
            if (generacion + 1) % 5 == 0 or generacion == self.generaciones - 1:
                expresion_str = self.imprimir_expresion_completa(mejor_individuo, 
                                                               f"MEJOR EXPRESIÓN - GENERACIÓN {generacion + 1}")
                self.mejor_expresion_str_historico.append(expresion_str)
            else:
                # Mostrar resumen de la expresión
                expresion_str = self.expresion_a_string(mejor_individuo)
                if len(expresion_str) > 100:
                    expresion_str = expresion_str[:97] + "..."
                print(f"Expresión: {expresion_str}")
            
            # Crear nueva población
            nueva_poblacion = []
            nueva_poblacion.append(self.copiar_expresion(mejor_individuo))
            
            while len(nueva_poblacion) < self.poblacion_size:
                participantes = random.sample(fitness_poblacion[:20], 2)
                ganador = max(participantes, key=lambda x: x[1])[0]
                
                if random.random() < 0.7 and len(nueva_poblacion) < self.poblacion_size - 1:
                    otro = max(random.sample(fitness_poblacion[:20], 2), key=lambda x: x[1])[0]
                    hijo = self.cruzar(ganador, otro)
                    hijo = self.mutar(hijo)
                    nueva_poblacion.append(hijo)
                
                nueva_poblacion.append(self.mutar(self.copiar_expresion(ganador)))
            
            poblacion = nueva_poblacion[:self.poblacion_size]
        
        return mejor_individuo
    
    def analizar_resultados(self, mejor_expresion):
        """Analiza y muestra los resultados finales con formato mejorado"""
        print("\n" + "╔" + "═"*78 + "╗")
        print("║" + "RESULTADOS FINALES - LEY DE PLANCK".center(78) + "║")
        print("╠" + "═"*78 + "╣")
        
        # Mostrar expresión final completa
        expresion_final = self.imprimir_expresion_completa(mejor_expresion, "EXPRESIÓN EVOLUCIONADA FINAL")
        
        # Calcular predicciones finales
        predicciones = []
        for lambda_val in self.lambda_values:
            pred = self.evaluar_expresion(mejor_expresion, lambda_val)
            if np.isreal(pred) and not (np.isinf(pred) or np.isnan(pred)):
                predicciones.append(float(pred))
            else:
                predicciones.append(0)
        
        predicciones = np.array(predicciones, dtype=float)
        
        # Métricas finales
        if len(predicciones) > 0 and np.all(np.isfinite(predicciones)):
            qme = np.mean((predicciones - self.intensidad_experimental)**2)
            try:
                r2 = r2_score(self.intensidad_experimental, predicciones)
            except:
                r2 = -1
        else:
            qme = float('inf')
            r2 = -1
        
        # Mostrar métricas
        print(f"\n┌{'─'*78}┐")
        print(f"│ MÉTRICAS DE EVALUACIÓN".ljust(78) + "│")
        print(f"├{'─'*78}┤")
        print(f"│ ▪ Error Cuántico Medio (QME): {qme:.2e}".ljust(78) + "│")
        print(f"│ ▪ Coeficiente de Determinación (R²): {r2:.4f}".ljust(78) + "│")
        
        # Interpretación de R²
        if r2 > 0.9:
            interpretacion = "Excelente ajuste - Modelo muy preciso ✓"
        elif r2 > 0.7:
            interpretacion = "Buen ajuste - Modelo confiable ✓"
        elif r2 > 0.5:
            interpretacion = "Ajuste moderado - Modelo aceptable"
        else:
            interpretacion = "Ajuste pobre - Se necesita más evolución"
        
        print(f"│ ▪ Interpretación: {interpretacion}".ljust(78) + "│")
        print(f"└{'─'*78}┘")
        
        
        # Comparación con Ley de Planck real
        print(f"\n┌{'─'*78}┐")
        print(f"│ COMPARACIÓN CON LEY DE PLANCK REAL".ljust(78) + "│")
        print(f"├{'─'*78}┤")
        ley_planck_str = "(2πhc²/λ⁵) * 1/(exp(hc/λkT) - 1)"
        print(f"│ ▪ Ley de Planck real: {ley_planck_str}".ljust(78) + "│")
        print(f"│ ▪ Modelo evolucionado: {expresion_final}".ljust(78) + "│")
        print(f"└{'─'*78}┘")
        
        # Gráficos
        self.graficar_resultados(predicciones, expresion_final)
    
    def graficar_resultados(self, predicciones, expresion_final):
        """Genera los gráficos de resultados"""
        fig = plt.figure(figsize=(16, 12))
        
        # Gráfico 1: Comparación de curvas
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.lambda_values * 1e9, self.intensidad_real, 'b-', 
                label='Ley de Planck Real', linewidth=2)
        ax1.plot(self.lambda_values * 1e9, self.intensidad_experimental, 'ro', 
                markersize=2, alpha=0.6, label='Datos Experimentales')
        ax1.plot(self.lambda_values * 1e9, predicciones, 'g--', 
                linewidth=2, label='Modelo Evolucionado')
        ax1.set_xlabel('Longitud de Onda (nm)')
        ax1.set_ylabel('Intensidad Espectral')
        ax1.set_title('Comparación de Curvas de Radiación')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        
        
        # Título general
        plt.suptitle(f'Programación Genética - Ley de Planck\nExpresión: {expresion_final[:100]}...', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

# Ejecutar el algoritmo
if __name__ == "__main__":
    print("Iniciando Programación Genética para descubrir la Ley de Planck...")
    
    pg = ProgramaGeneticoPlanck(poblacion_size=30, generaciones=20, temperatura=5000)
    
    mejor_modelo = pg.ejecutar_evolucion()
    
    pg.analizar_resultados(mejor_modelo)
    