import numpy as np
import pandas as pd
import time

#Parámetros del problema:

num_iteraciones=300
num_particulas=20
lb=-(1000) #límite inferior para inicializar los valores de las posiciones
ub=1000 #limite superior para inicializar los valores de las posiciones
# Actualización de las velocidades y posiciones de las partículas
w = 0.5 # Factor de inercia
cl = 2  # Peso cognitivo, para el local
cg = 2  # Peso social, para el global 

# Configuración de reinicialización
num_reinicios = 8  # Número máximo de reinicios
epsilon = 0.00001 # Umbral para reiniciar el algoritmo
reinicios = 0  # Contador de reinicios


mejor_solucion = None  # Almacena la mejor solución encontrada 
mejor_fitness = np.inf  # Almacena el valor de aptitud de la mejor solución encontrada


#Cada una de las particulas actualiza su velocidad y posicion en cada iteracion
# Funciones objetivo, cada una es una ecuación del sistema


def f1(x, y):
    return  x**2+x*y-10



def f2(x, y):
    return y+3*x*y**2-57

def aptitud(f1,f2):
    aptitud = abs(f1) + abs(f2)
    return aptitud
    
#Se tomará como función objetivo del problema la función que se deberá minimizar : abs(f1) + abs(f2)
def PSO():
    # Inicialización de posiciones y velocidades iniciales
    posiciones = pd.DataFrame(np.random.uniform(low=lb, high=ub, size=(num_particulas, 2)), columns=['x', 'y'])
    velocidades = pd.DataFrame(np.random.uniform(0, size= (num_particulas, 2)), columns=['x', 'y'])

    # Mejor posición global
    mejor_pos_global = None  # Almacena la mejor posición global encontrada
    mejor_fitness_global = np.inf  # Almacena el valor de aptitud de la mejor solución global encontrada

    # Mejor posición local por cada partícula
    mejor_pos_local = posiciones.copy()  # Almacena las mejores posiciones locales para cada partícula
    mejor_fitness_local = pd.DataFrame(np.inf, index=np.arange(num_particulas), columns=['fitness']) 
    #'mejor_fitness_local' es un DataFrame que almacena el valor de aptitud (fitness) más bajo obtenido por cada partícula hasta el momento. 
    # En otras palabras, representa el mejor valor de aptitud local alcanzado por cada partícula en su trayectoria de búsqueda
    for _ in range(num_iteraciones):
        fitness = pd.DataFrame({'f1': f1(posiciones['x'], posiciones['y']), 'f2': f2(posiciones['x'], posiciones['y'])})
        # Evaluación de las funciones objetivo para cada partícula, se evalúan las dos funciones de fitnes para cada particula(que tiene dos coordadenadas
        # y se guarda en un dataframe el valor de f1 y f2)
        # Actualización de la mejor posición global
        mejor_particula = fitness.loc[(aptitud(fitness['f1'], fitness['f2'])).idxmin()]  # Encuentra la partícula con el valor de aptitud mínimo combinado de las dos funciones objetivo
        if aptitud(mejor_particula['f1'],mejor_particula['f2']) < mejor_fitness_global: #Encuentra una mejor particula
            mejor_pos_global = posiciones.loc[mejor_particula.name]  # Actualiza la mejor posición global, aquí el .name nos devuelve el número de la partícula
            mejor_fitness_global = aptitud(mejor_particula['f1'],mejor_particula['f2'])  # Actualiza el mejor valor de aptitud global

        # Actualización de las mejores posiciones locales
        actualizar = aptitud(fitness['f1'],fitness['f2']) < mejor_fitness_local['fitness']  # Variable de valores booleanos que indica cuáles partículas tienen una mejor aptitud local
        mejor_pos_local[actualizar] = posiciones[actualizar]  # Selecciona solo las filas que hay que actualizar y cctualiza las posiciones locales de las partículas que tienen un valor de aptitud mejor
        mejor_fitness_local.loc[actualizar, 'fitness'] = aptitud(fitness.loc[actualizar, 'f1'], fitness.loc[actualizar, 'f2'])  # Actualiza los valores de aptitud local correspondientes
        ul = np.random.uniform(0, 1) #parámetro aleatorio referente a la posición local
        ug = np.random.uniform(0, 1) #parámetro aleatorio referente a la posición global       

        #Calculo de las nuevas velocidades:    
        velocidades = w * velocidades \
                    + cl * ul * (mejor_pos_local - posiciones) \
                    + cg * ug * (mejor_pos_global - posiciones)  # Actualiza las velocidades de las partículas
        # Calculo de las nuevas posiciones tras actualizar la velocidad
        posiciones += velocidades  

    return mejor_pos_global, mejor_fitness_global

inicio = time.time()

while reinicios < num_reinicios and mejor_fitness > epsilon:
    solucion_actual, fitness_actual = PSO()  # Ejecuta el algoritmo PSO con los parámetros dados
    
    if fitness_actual < mejor_fitness:
        mejor_solucion = solucion_actual  # Actualiza la mejor solución encontrada
        mejor_fitness = fitness_actual  # Actualiza el mejor valor de aptitud encontrado
    
    reinicios += 1  # Incrementa el contador de reinicios


# Registro del tiempo de finalización
fin = time.time()

# Calculo del tiempo transcurrido
tiempo_transcurrido = fin - inicio

print("Solución encontrada por el PSO:")
print(mejor_solucion)  # Imprime la mejor solución encontrada
print("Error cometido:")
print(mejor_fitness)  # Imprime el valor de aptitud de la mejor solución
# Imprimir el tiempo transcurrido en segundos
print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")




