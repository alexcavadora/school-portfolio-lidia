from optFunc import AckleyFunction
import numpy as np
import matplotlib.pyplot as plt
import time 

## necesito la funcion exit()
from sys import exit

class linSearch:
    def __init__(self, func, stopCrit, stepCond, c1=1e-6, c2=0.9):
        self.func = func
        self.stopCrit = stopCrit 
        self.stepCond = stepCond
        self.wolfe = WolfeConditions(c1, c2)
        self.x_history = []

    def GradientDescentMethod(self, x):
        return -(self.func.grad(x) / np.linalg.norm(self.func.grad(x)))

    def NewtonMethod(self, x):
        B_k =  self.func.hess(x)
        return -np.linalg.inv(B_k) @ self.func.grad(x)

    def optMethod(self, op, x):
        if op == 1:
            return self.NewtonMethod(x)
        elif op == 2:
            return self.GradientDescentMethod(x)
        else:
            raise ValueError("Elección no válida")

    def solve(self, x0, condition="armijo", method=2):
        """Resuelve el problema de optimización usando la condición de Wolfe especificada."""
        # x = np.array(x0, dtype=float)
        x = x0.copy()
        self.x_history.append(x.copy())
        iterations = 0

        while True:
            grad_norm = np.linalg.norm(self.func.grad(x))
            prev_value = self.x_history[-1] if len(self.x_history) > 1 else x
            curr_value = x
            
            if self.stopCrit(grad_norm, prev_value, curr_value, iterations):
                break

            d = self.optMethod(method, x)
            
            alpha = self.stepCond(x, d, self.func, self.wolfe, condition) ## error
            x = x + alpha * d
            self.x_history.append(x.copy())
            iterations += 1
            # print(f"Iteración {iterations}, grad_norm = {grad_norm}")
        self.x_history = np.array(self.x_history)
        return x

    def BFGS(self, x0, condition="armijo", op=2):
        self.x_history.append(x0.copy())
        p_k = self.optMethod(op, x0)
        alpha = self.stepCond(x0, p_k, self.func, self.wolfe, condition)
        xk = x0 + alpha * p_k
        H0 = np.eye(len(x0))  # Inicializar H0 como la matriz identidad
        I = np.eye(len(x0))   # Matriz identidad
        iterations = 0
        while True:
            grad_norm = np.linalg.norm(self.func.grad(x0))
            prev_value = self.x_history[-1] if len(self.x_history) > 1 else x0
            curr_value = x0
            
            if self.stopCrit(grad_norm, prev_value, curr_value, iterations):
                break

            s_k = xk - x0  # Diferencia entre los puntos actual y anterior
            y_k = self.func.grad(xk) - self.func.grad(x0) # Diferencia entre gradientes
            rho_k = 1.0 / (y_k.T @ s_k)  # rho_k es un escalar

            # Actualización de Hk usando la fórmula BFGS
            

            Hk = (I - rho_k.item() * np.outer(s_k, y_k)) @ H0 @ (I - rho_k.item() * np.outer(y_k, s_k)) + rho_k.item() * np.outer(s_k, s_k)

            # Actualizar H0 y x0 para la siguiente iteración
            H0 = Hk
            x0 = xk

            # Calcular la dirección de descenso
            p_k = -H0 @ self.func.grad(x0)

            # Calcular el tamaño de paso alpha usando backtracking line search
            alpha = self.stepCond(x0, p_k, self.func, self.wolfe, condition)
            # Actualizar xk
            xk = x0 + alpha * p_k

            # Guardar el historial de x
            self.x_history.append(x0)
            iterations += 1

        # Convertir el historial a un array de NumPy
        self.x_history = np.array(self.x_history)
        # print(iter)
        return x0

    def plot2D(self):
        x1 = np.linspace(-30, 30, 100)
        x2 = np.linspace(-30, 30, 100)
        X1, X2 = np.meshgrid(x1, x2)
        f = np.zeros((100, 100))
        self.func.d = 2
        self.func.e = np.ones(2).T
        for i in range(100):
            for j in range(100):
                point = np.matrix([[X1[i, j]], [X2[i, j]]])
                f[i, j] = self.func.eval(point).item()
                # f[i, j] = self.func.eval([X1[i, j], X2[i, j]])

        fig, ax = plt.subplots()
        ax.contour(X1, X2, f, levels=np.linspace(np.min(f), np.max(f), 25))
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_aspect('equal')

        # Dibujar la trayectoria en rojo
        plt.plot(self.x_history[:, 0], self.x_history[:, 1], 'r.', label='Iteraciones')

        # Marcar el último punto con un triángulo azul
        plt.scatter(self.x_history[-1, 0], self.x_history[-1, 1], color='blue', marker='^', s=100, label='Último Punto')

        plt.legend()
        plt.show()
        self.x_history = []

class WolfeConditions:
    def __init__(self, c1=1e-4, c2=0.9):
        self.c1 = c1
        self.c2 = c2
        self.ctd = 0

    def armijo_condition(self, func, x, alpha, d):
        lhs = func.eval(x + alpha * d).item()
        rhs = func.eval(x).item() + self.c1 * alpha * np.dot(func.grad(x).T, d)
        return lhs <= rhs

    def curvature_condition(self, func, x, alpha, d):
        lhs = np.dot(func.grad(x + alpha * d).T, d)
        rhs = self.c2 * np.dot(func.grad(x).T, d)

        return lhs.item() >= rhs.item()

    def strong_wolfe_condition(self, func, x, alpha, d):
        grad_x = func.grad(x)
        grad_x_alpha = func.grad(x + alpha * d)
        lhs = abs(np.dot(grad_x_alpha.T, d))
        rhs = self.c2 * abs(np.dot(grad_x.T, d))

        return lhs.item() <= rhs.item()

# Definición de funciones externas
def stop_criterion(grad_norm, prev_value, curr_value, iterations):
    grad_tolerance = 1e-3
    value_tolerance = 1e-6
    max_iter = 1000

    grad_small = grad_norm < grad_tolerance
    diff = np.linalg.norm(curr_value - prev_value)
    value_converged = diff < value_tolerance if diff != 0 else False
    iter_exceeded = iterations > max_iter

    return grad_small or value_converged or iter_exceeded


def step_condition(x, d, func, wolfe, condition):
    alpha = 1.0
    iterations = 0
    min_alpha = 1e-10  # Valor mínimo para alpha
    max_iterations = 50  # Máximo número de iteraciones
    
    if condition == "armijo":
        while not wolfe.armijo_condition(func, x, alpha, d):
            alpha *= 0.5
            iterations += 1
            # print(f"Backtracking: alpha = {alpha}, iteración {iterations}")
            if alpha < min_alpha or iterations >= max_iterations:
                # print(f"WARNING: Backtracking terminado con alpha = {alpha}")
                break

    elif condition == "curvature":
        while not (wolfe.armijo_condition(func, x, alpha, d) and
                   wolfe.curvature_condition(func, x, alpha, d)):
            alpha *= 0.5
            iterations += 1
            if alpha < min_alpha or iterations >= max_iterations:
                # print(f"WARNING: Backtracking terminado con alpha = {alpha}")
                break
                
    elif condition == "strong":
        # print("Iniciando strong condition check...")
        while not (wolfe.armijo_condition(func, x, alpha, d) and
                   wolfe.strong_wolfe_condition(func, x, alpha, d)):
            
            alpha *= 0.5
            iterations += 1
            if alpha < min_alpha or iterations >= max_iterations:
                # print(f"WARNING: Backtracking terminado con alpha = {alpha}")
                break
    else:
        raise ValueError("Condición no soportada. Use 'armijo', 'curvature' o 'strong'.")
    
    return alpha



# Define the same initial point for all methods
np.random.seed(42)  # For reproducibility
ini = np.random.uniform(-30, 30)
x_initial = np.matrix([[ini], [ini], [ini], [ini], [ini]]) 
print(f"Punto inicial para todos los métodos: {x_initial}")
func = AckleyFunction(x_initial)
optimizer = linSearch(func, stop_criterion, step_condition, )
# Gradient Descent with different conditions
print("\n===== MÉTODO DE DESCENSO DE GRADIENTE =====")

# Gradient Descent with Armijo condition
func = AckleyFunction(x_initial.copy())
optimizer = linSearch(func, stop_criterion, step_condition)
start = time.time()
x_opt = optimizer.solve(x_initial.copy(), condition="armijo", method=2)
end = time.time()
print("\nDescenso de Gradiente con Condición Armijo:")
print(f"Punto inicial: {x_initial}")
print(f"Óptimo encontrado en: {x_opt}")
print(f"Valor de la función en el óptimo: {func.eval(x_opt)}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")
# optimizer.plot2D()

# Gradient Descent with Curvature (Wolfe) condition
func = AckleyFunction(x_initial.copy())
optimizer = linSearch(func, stop_criterion, step_condition)
start = time.time()
x_opt = optimizer.solve(x_initial.copy(), condition="curvature", method=2)
end = time.time()
print("\nDescenso de Gradiente con Condición Wolfe:")
print(f"Punto inicial: {x_initial}")
print(f"Óptimo encontrado en: {x_opt}")
print(f"Valor de la función en el óptimo: {func.eval(x_opt)}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")
# optimizer.plot2D()

# Newton's Method with different conditions
print("\n===== MÉTODO DE NEWTON =====")

# Newton's Method with Armijo condition
func = AckleyFunction(x_initial.copy())
optimizer = linSearch(func, stop_criterion, step_condition)
start = time.time()
x_opt = optimizer.solve(x_initial.copy(), condition="armijo", method=1)
end = time.time()
print("\nMétodo de Newton con Condición Armijo:")
print(f"Punto inicial: {x_initial}")
print(f"Óptimo encontrado en: {x_opt}")
print(f"Valor de la función en el óptimo: {func.eval(x_opt)}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")
# optimizer.plot2D()

# Newton's Method with Strong Wolfe condition
func = AckleyFunction(x_initial.copy())
optimizer = linSearch(func, stop_criterion, step_condition)
start = time.time()
x_opt = optimizer.solve(x_initial.copy(), condition="strong", method=1)
end = time.time()
print("\nMétodo de Newton con Condición Strong Wolfe:")
print(f"Punto inicial: {x_initial}")
print(f"Óptimo encontrado en: {x_opt}")
print(f"Valor de la función en el óptimo: {func.eval(x_opt)}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")
# optimizer.plot2D()

# BFGS Method with different conditions
print("\n===== MÉTODO BFGS =====")

# BFGS with Gradient Descent initial direction
func = AckleyFunction(x_initial.copy())
optimizer = linSearch(func, stop_criterion, step_condition)
start = time.time()
x_opt = optimizer.BFGS(x_initial.copy(), condition="armijo", op=2)
end = time.time()
print("\nBFGS con Dirección Inicial de Descenso de Gradiente:")
print(f"Punto inicial: {x_initial}")
print(f"Óptimo encontrado en: {x_opt}")
print(f"Valor de la función en el óptimo: {func.eval(x_opt)}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")
# optimizer.plot2D()

# BFGS with Newton initial direction
func = AckleyFunction(x_initial.copy())
optimizer = linSearch(func, stop_criterion, step_condition)
start = time.time()
x_opt = optimizer.BFGS(x_initial.copy(), condition="curvature", op=1)
end = time.time()
print("\nBFGS con Dirección Inicial de Newton:")
print(f"Punto inicial: {x_initial}")
print(f"Óptimo encontrado en: {x_opt}")
print(f"Valor de la función en el óptimo: {func.eval(x_opt)}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")
# optimizer.plot2D()
