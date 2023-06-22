
fun = lambda x: 1*x**3 -  2*x - 5
der = lambda x: 3*x**2 - 2


def newton(fun, der, x_n, epsilon=0.0005, steps=500):   
    for n in range(steps + 1):
        # Evaluación de la función para ver si el resultado es válido
        f_x = fun(x_n)
        if abs(f_x) < epsilon:
            print(abs(f_x))
            return x_n
        
        # Evaluación de la derivada
        d_f = der(x_n)
        if d_f == 0:
            print('Error la derivada es cero')
            return None
        
        # Estimación del siguiente punto
        x_n = x_n - f_x / d_f
    
    print('Se ha alcanzado el límite de iteraciones')
    return x_n, f_x

print(newton(fun, der, 0))