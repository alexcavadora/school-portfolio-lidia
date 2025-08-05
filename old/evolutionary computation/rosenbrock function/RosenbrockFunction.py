import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Rosenbrock function and gradient
b = 10
def f(x, y):
    return (x - 1) ** 2 + b * (y - x**2) ** 2

def df(x, y):
    return np.array([2 * (x - 1) - 4 * b * (y - x**2) * x, 2 * b * (y - x**2)])

# Grid for plotting
X = np.linspace(-2, 2, 100)
Y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

F = lambda X: f(X[0], X[1])
dF = lambda X: df(X[0], X[1])

# Wolfe condition parameters
c1 = 1e-1
c2 = 0.7

# Starting point
x0 = np.array([-1.5, 2])

def wolfe_line_search(F, dF, x, s, alpha=1, beta=0.5, max_iters=20):
    fx = F(x)
    grad = dF(x)
    d = np.dot(grad, s)
    
    for _ in range(max_iters):
        if F(x + alpha * s) > fx + c1 * alpha * d:
            alpha *= beta  # Armijo condition
        elif np.dot(dF(x + alpha * s), s) < c2 * d:
            alpha /= beta  # Curvature condition
        else:
            break
    return alpha

# Gradient Descent with Wolfe Conditions
def gradient_descent():
    x = x0.copy()
    trajectory = [x.copy()]
    max_iters = 10000
    tol = 1e-6
    
    for i in range(max_iters):
        grad = dF(x)
        s = -grad
        
        alpha = wolfe_line_search(F, dF, x, s)
        x = x + alpha * s
        trajectory.append(x.copy())
        
        if np.linalg.norm(grad) < tol:
            print(f'Gradient Descent converged in {i+1} iterations')
            break
    else:
        print('Gradient Descent reached maximum iterations')
        
    return np.array(trajectory)

# Newton's Method
def newton_method():
    x = x0.copy()
    trajectory = [x.copy()]
    max_iters = 100000
    tol = 1e-6
    
    def H(x):
        return np.array([[2 - 4 * b * (x[1] - 3 * x[0]**2) + 12 * b * x[0]**2, -4 * b * x[0]],
                         [-4 * b * x[0], 2 * b]])
    
    for i in range(max_iters):
        grad = dF(x)
        hess = H(x)
        
        if np.linalg.cond(hess) > 1e12:
            print("Hessian is ill-conditioned. Stopping.")
            break
        
        s = np.linalg.solve(hess, -grad)
        alpha = wolfe_line_search(F, dF, x, s)
        x = x + alpha * s
        trajectory.append(x.copy())
        
        if np.linalg.norm(grad) < tol:
            print(f'Newton converged in {i+1} iterations')
            break
    else:
        print('Newton reached maximum iterations')
        
    return np.array(trajectory)

# BFGS Method
def bfgs_method():
    x = x0.copy()
    trajectory = [x.copy()]
    max_iters = 1000000
    tol = 1e-6
    B = np.eye(2)
    
    for i in range(max_iters):
        grad = dF(x)
        s = -B @ grad
        
        alpha = wolfe_line_search(F, dF, x, s)
        x_new = x + alpha * s
        y = dF(x_new) - grad
        s_step = alpha * s
        
        if np.dot(s_step, y) > 1e-10:
            Bs = B @ s_step
            B = B - np.outer(Bs, Bs) / np.dot(s_step, Bs) + np.outer(y, y) / np.dot(y, s_step)
        
        x = x_new
        trajectory.append(x.copy())
        
        if np.linalg.norm(grad) < tol:
            print(f'BFGS converged in {i+1} iterations')
            break
    else:
        print('BFGS reached maximum iterations')
        
    return np.array(trajectory)

# Run methods
gd_traj = gradient_descent()
nt_traj = newton_method()
bfgs_traj = bfgs_method()

# Plot results
plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='coolwarm')
plt.plot(-1.5, 2, 'b*', label='$\mathbf{x_0} = P(-1.5, 2)$')
plt.plot(gd_traj[:,0], gd_traj[:,1], 'r-', label='Gradient Descent')
plt.plot(nt_traj[:,0], nt_traj[:,1], 'g--', label="Newton's Method")
plt.plot(bfgs_traj[:,0], bfgs_traj[:,1], 'm-.', label='BFGS Method')
plt.plot(1,1, '*', label='$\\mathbf{x^*} = P(1, 1)$')
plt.legend()
plt.title("Optimization Paths on Rosenbrock Function")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# 3D Trajectory Plot
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.gist_heat_r, alpha=0.5)
ax.plot(-1.5, 2, f(-1.5,2), 'b*', label='$\mathbf{x_0} = P(-1.5, 2)$')
ax.plot(gd_traj[:,0], gd_traj[:,1], [F(p) for p in gd_traj], 'r-', label='GD')
ax.plot(nt_traj[:,0], nt_traj[:,1], [F(p) for p in nt_traj], 'g--', label='Newton')
ax.plot(bfgs_traj[:,0], bfgs_traj[:,1], [F(p) for p in bfgs_traj], 'm-.', label='BFGS')
ax.plot(1,1, 0, '*', label='$\\mathbf{x^*} = P(1, 1)$')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("f(X)")
ax.set_title("3D Trajectory Comparison")
plt.legend()
plt.show()

