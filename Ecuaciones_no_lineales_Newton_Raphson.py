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


while (tramo>epsilon):
    J = Jac.subs([(x,xi),(y,yi)])

    # determinante de J
    Jn = np.array(J,dtype=float)
    determinante =  np.linalg.det(Jn)

    # iteraciones
    f1i = f1.subs([(x,xi),(y,yi)])
    f2i = f2.subs([(x,xi),(y,yi)])

    numerador1 = f1i*Jn[n-1,m-1]-f2i*Jn[0,m-1]
    xi1 = xi - numerador1/determinante
    numerador2 = f2i*Jn[0,0]-f1i*Jn[n-1,0]
    yi1 = yi -numerador2/determinante
    
    tramo = np.max(np.abs([xi1-xi,yi1-yi]))
    xi = xi1
    yi = yi1

    i = i+1
    # print('iteración: ',itera)
    # print('Jacobiano con puntos iniciales: ')
    # print(J)
    # print('determinante: ', determinante)
    # print('puntos xi,yi:',xi,yi)
    # print('error:',tramo)
    
# SALIDA
print('Resultado: ')
print(xi,yi)
