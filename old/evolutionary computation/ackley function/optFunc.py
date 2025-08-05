import numpy as np

class AckleyFunction():

  def __init__(self,x):
    self.d = len(x)
    self.a = 20
    self.b = 0.2
    self.c = 2 * np.pi
    self.e = np.ones(self.d)

  def eval(self, x):
    
    return -self.a * np.exp(-self.b * np.sqrt(1.0/self.d * x.T * x )) - np.exp(1.0/self.d * self.e.T * np.cos(self.c * x) ) + self.a + np.exp(1)

  def ident(self):
    n = self.d
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

  def grad(self, x):
    raiz = np.sqrt(1.0/self.d * x.T * x)
    uno = x *self.a *self.b/raiz * np.exp(-self.b * raiz)   
    dos = self.c * np.sin(self.c * x) * np.exp(1.0/self.d)
    tres = self.e.T * np.cos(self.c * x)                                           
                                              
    return (1.0/self.d)*(uno + dos * tres)

  def hess(self, x):
    raiz = np.sqrt(1.0/self.d * x.T * x)
    ex1 = np.exp(-self.b * raiz)
    ex2 = np.exp(1.0/self.d * self.e.T * np.cos(self.c * x))
    I = self.ident()
    cx = self.c * x
    senx = np.sin(cx)

    return (1.0/self.d)*( ( x * x.T * self.a/self.d + ( I + ( x * x.T )*2 ) * ex1.item(0) ) * (-self.b/raiz.item()**2) + ( np.diag(np.cos(cx)) + senx * senx.T * 1.0/self.d) * self.c**2 * ex2.item())
  

if __name__ == "__main__":
  # Example with a 5D vector
  zero = 10
  x = np.matrix([[zero], [zero], [zero], [zero], [zero]])  # Create a 5D column vector
  X = np.array([zero, zero, zero, zero, zero])
  print(f"Initial x: {X}")
  # Initialize the Ackley function
  ackley = AckleyFunction(x)

  
  # Evaluate function at x
  value = ackley.eval(x).item()
  print(f"Function value f(x) = {value}")
  print("-" * 40)

  
  # Calculate gradient at x
  gradient = ackley.grad(x)
  print("Gradient at x:")
  print(gradient)
  print("-" * 40)
  
  # Calculate Hessian at x
  hessian = ackley.hess(x)
  print("Hessian matrix at x:")
  print(hessian)
  print("-" * 40)
