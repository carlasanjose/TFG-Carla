import numpy as np
import sympy as sym

def Jacobiano(variables, funciones):
    n = len(funciones)
    m = len(variables)
    # matriz Jacobiano inicia con ceros
    Jac = sym.zeros(n,m)
    for i in range(0,n,1):
        unafi = sym.sympify(funciones[i])
        for j in range(0,m,1):
            unavariable = variables[j]
            Jac[i,j] = sym.diff(unafi, unavariable)
    return Jac

# PROGRAMA ----------
# INGRESO
x = sym.Symbol('x')
y = sym.Symbol('y')

f1 = x**2 + x*y - 10
f2 = y + 3*x*(y**2)-57

x0 = 1.5
y0 = 3.5

epsilon = 0.00001

# PROCEDIMIENTO
funciones = [f1,f2]
variables = [x,y]
n = len(funciones)
m = len(variables)

Jac = Jacobiano(variables, funciones)

# valores iniciales
xi = x0
yi = y0

# tramo inicial, mayor que tolerancia
i = 0
tramo = epsilon*2


while (tramo > epsilon):
    J = Jac.subs([(x,xi),(y,yi)])  # Sustituye los valores actuales de xi y yi en la matriz Jacobiana (J)
    Jn = np.array(J, dtype=float)  # Convierte la matriz Jacobiana a un array NumPy de tipo float
    determinante = np.linalg.det(Jn)  # Calcula el determinante de la matriz Jacobiana
    f1i = f1.subs([(x,xi),(y,yi)])  # Sustituye los valores actuales de xi y yi en la función f1
    f2i = f2.subs([(x,xi),(y,yi)])  # Sustituye los valores actuales de xi y yi en la función f2
    numerador1 = f1i*Jn[n-1,m-1] - f2i*Jn[0,m-1]  # Calcula el numerador de la nueva xi que es la jacobiana
    xi1 = xi - numerador1/determinante  # Calcula el nuevo valor de xi
    numerador2 = f2i*Jn[0,0] - f1i*Jn[n-1,0]  # Calcula el numerador de la nueva yi que es la jacobiana
    yi1 = yi - numerador2/determinante  # Calcula el nuevo valor de yi
    tramo = np.max(np.abs([xi1-xi, yi1-yi]))  # Calcula el máximo de las diferencias absolutas entre los nuevos valores de xi y yi y los anteriores
    xi = xi1  # Actualiza el valor de xi con el nuevo valor calculado
    yi = yi1  # Actualiza el valor de yi con el nuevo valor calculado
    i = i + 1  # Incrementa el contador de iteraciones
    # print('iteración: ',itera)
    # print('Jacobiano con puntos iniciales: ')
    # print(J)
    # print('determinante: ', determinante)
    # print('puntos xi,yi:',xi,yi)
    # print('error:',tramo)
    
# SALIDA
print('Resultado: ')
print(xi,yi)
