import pandas as pd
import numpy as np
import time

#Parámetros del problema
num_iteraciones=500             #número de iteraciones permitidas
n=20                            #tamaño de la población
pc=0.2                          #porcentaje de cruzamiento
ub=1000                         #límite superior
lb=-(1000)                      #límite inferior
nc=6                            #número de "parejas" descendientes 
nm=round(0.3*n)                 #número de individuos mutados
mu=0.1                          #tasa de mutación
epsilon=0.00001                 #condición de convergencia
soluciones =[]                  #En esta lista almacenamos las 3 soluciones de la ecuación cúbica 


# Ecuación:     x^3 + 2x^2 - 5x - 6 = 0
# Soluciones:   x1 = -3 ; x2 = -1 ; x3 = 2

a = 1
b = 0
c = -(2)
d = -(5)

def fitness(df,a,b,c,d):
    fitness = abs(a*df['Valor']**3+b*df['Valor']**2+c*df['Valor']+d) #Función objetivo del problema 
    return fitness

#CREACIÓN POBLACIÓN
def creacion_poblacion():
    pob=pd.DataFrame(index=range(n)) #Fijamos que el tamaño del dataFrame sea el de la población
    pob['Individuo']=range(0, n)  #Añado una columna con el número de cada individuo
    pob['Valor']=np.random.uniform(lb, ub+1, size=len(pob)) # Columna valor generada en el DataFrame pob con valores aleatorios distribuidos en el espacio de búsqueda                                                       #uniformemente en el rango [lb, ub] para cada posición i 
    #Evaluación en la función objetivo:
    pob['Fitness']= fitness(pob,a,b,c,d) #Necesita que se haya definido la función fitness
    return pob

# SELECCIÓN--------------
def seleccion(P):
    r1=np.random.random()
    r2=np.random.random()      
    P_sel=P.copy()
    P_sel['Mayor_r1']=np.where((P['Fitness_Norm_Acum']>r1),1,0) 
    P_sel['Mayor_r2']=np.where((P['Fitness_Norm_Acum']>r2),1,0) 
    min_umbral_r1=P_sel[P_sel['Mayor_r1']==P_sel['Mayor_r1'].max()]['Fitness_Norm_Acum'].min() # Con max(), nos quedamos con todos los que tienen un 1 en esa columna 
                                                                                               # y con el .min() hacemos que sea el primero que sobrepasa el r1 (primero que cumple ser mayor que r1 en prob acum es el umbral)
                                                                                                                                                                                    
    padre_1=P_sel[P_sel['Fitness_Norm_Acum']==min_umbral_r1]
    min_umbral_r2=P_sel[P_sel['Mayor_r2']==P_sel['Mayor_r2'].max()]['Fitness_Norm_Acum'].min()    
    padre_2=P_sel[P_sel['Fitness_Norm_Acum']==min_umbral_r2]
    return padre_1, padre_2

# CRUCE--------------------
# hijo1= a * padre1 + (1 - a) * padre2 con 0 < a < 1
# hijo2= (1-a) * padre1 + a * padre2 con 0 < a < 1
def cruce(padre_1, padre_2):
    hijo_1=[(pc*padre_1['Valor'].values[0]+(1-pc)*padre_2['Valor'].values[0])]
    hijo_2=[((1-pc)*padre_1['Valor'].values[0]+pc*padre_2['Valor'].values[0])]
    return hijo_1, hijo_2

# MUTACIÓN------------------ 
def mutacion(mut):
    # Generar aleatoriamente +1 o -1
    signo = np.random.choice([-1, 1])
    r3=np.random.uniform(0,1)    
    sigma=mu*(ub-lb)
    mut['Valor']=round(mut['Valor']+signo*r3*sigma)
        
        
#RECOMBINACIÓN-------

def recombinacion(pob,popc,popm):
    pob=pd.concat([pob, popc, popm], axis=0)
    # Ordena la población   
    pob=pob.sort_values('Fitness')
    # Cogemos los n(n) primeros de la pob ordenada 
    pob=pob.iloc[0:n]
    return pob   
 
          
# LÍMITES SUPERIOR E INFERIOR---------
def limites(df, columna, lb, ub):            
    df[columna]=np.where(df[columna]<lb, lb, df[columna] )
    df[columna]=np.where(df[columna]>ub, ub, df[columna] )
    #si nos dan valores para los hijos menores que el minimo y mayores que el maximo se reemplazan por los límites para no salirnos del espacio de búsqueda


##########CREACIÓN POBLACIÓN###########################################################

def algoritmo_genetico(a,b,c,d):
    pob=creacion_poblacion()
    xMejor = pob[pob['Fitness'] == pob['Fitness'].min()] #Es un dataframe con una fila (mejor)
    resumen_iteraciones=pd.DataFrame(columns=['Numero_iteracion', 'Mejor_Fitness','Mejor_Valor'])        
    i=0  # contador de iteraciones
    aux_indice=0 # auxiliar para actualizar el índice de los individuos
    while xMejor['Fitness'].iloc[0] > epsilon and i < num_iteraciones :       
            P=pob.copy() 
            P['Fitness_Norm']=P['Fitness']/P['Fitness'].sum() #Añado la columna Aptitud normalizada
            P['Fitness_Norm_Acum']=np.cumsum(P['Fitness_Norm']) #Añado la columna suma acumulada de la aptitud normalizada
            popc=pd.DataFrame(columns=pob.columns)  #las columnas son individuo, posiciones y fitness ES UN DATAFRAME con esas tres columnas
            for j in range(0,nc): #Dos hijos descendientes
                ##########SELECCIÓN: Para elegir a los padres###############################################
                padre_1, padre_2 = seleccion(P)            
                ##########CRUCE: Creamos los hijos##########################################################
                hijo_1, hijo_2 = cruce(padre_1, padre_2)    
                n_individuo_hijo_1 = int(n + (aux_indice * 2)) 
                n_individuo_hijo_2 = int(n + (aux_indice * 2) + 1)     
                popc_it=pd.DataFrame(columns=pob.columns) #almacena a cada par de hijos
                popc_it.loc[j]=[n_individuo_hijo_1] + hijo_1 + [np.nan]
                popc_it.loc[j+2]= [n_individuo_hijo_2] + hijo_2 + [np.nan]    
                limites(popc_it, 'Valor', lb, ub) #recorre los valores de los nuevos hijos y ajusta segun los límites    
                popc_it['Fitness']= fitness(popc_it,a,b,c,d)  #calcula fitness para cada nuevo hijo 
                popc=pd.concat([popc, popc_it], axis=0) 
                aux_indice+=1 
            ######################ETAPA DE MUTACIÓN#####################################
            popm=pd.DataFrame(index=range(nm)) #creo un data frame de tantas filas como elementos se van a mutar vacío
            popm['Individuo']=np.nan
            popm['Valor']=np.nan   
            popm['Fitness']=np.nan 
            for k in range(0,nm): #determina cuantos se mutan (número, pero no cuáles)
                z=np.random.randint(0,n-1) #va del 0 al nºpoblación, será el que es mutado al azar
                p=pob.iloc[z] #selecciona la fila que quiero mutar
                p=pd.DataFrame(p) #data frame de dos columnas, una contiene los nombres 'Individuo', 'Valor'.. y la otra los datos 
                p=np.transpose(p) #trasponemos para tener el mismo formato que antes, que sean columna 'Individuo', 'Valor' y 'Fitness'
                popm.loc[k,'Individuo']=int(pob['Individuo'].iloc[z])
                xa=p[['Valor']].copy() #es un dataframe con una unica columna 'Valor', y una única fila, el valor correspondiente
                mutacion(xa) 
                popm.loc[k, 'Valor']=xa.iloc[0]['Valor'] 
                limites(popm, 'Valor', lb, ub)      
                popm['Fitness']= fitness(popm,a,b,c,d) # Relleno la columna Fitness
                    
        ########FIN ETAPA DE MUTACION################################################
        ##ETAPA DE RECOMINACIÓN: Elitismo##
            pob = recombinacion(pob,popc,popm)
        ##FIN ETAPA DE RECOMBINACIÓN##
            # Almacena la solución mejor y peor
            xMejor = pob[pob['Fitness'] == pob['Fitness'].min()] #Es un dataframe con una fila (la del #mejor#)
            # imprimo
            #print('Iteración:' , i)
            resumen_iteraciones.loc[i] = [i, xMejor['Fitness'].values[0], xMejor['Valor'].values[0]]

            i+=1    
    sol=xMejor['Valor'].values[0]
    soluciones.append(sol)
   
    # if len(soluciones) ==1:
    #     r = 0
    #     s = a
    #     t = b + s * sol
    #     u = c + t * sol 
    #     print(i)
    #     algoritmo_genetico(r,s,t,u)        

    # elif len(soluciones)==2: 
    #     x2=soluciones[1]
    #     x3=-(c/b)-(x2)
    #     print(i)
    #     soluciones.append(x3)
  
    # else: 
    #     pass   

    return soluciones,  i, resumen_iteraciones

# Registro del tiempo de inicio
inicio = time.time()

print(pc,n, nc)
soluciones,i,iter = algoritmo_genetico(a,b,c,d)        
print("x1=", soluciones[0])#, ";x2=", soluciones[1], ";x3=", soluciones[2])
print("Iteracion:",i)
print(iter)

# Registro del tiempo de finalización
fin = time.time()

# Cálculo del tiempo transcurrido
tiempo_transcurrido = fin - inicio

# Imprimir el tiempo transcurrido en segundos
print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")


