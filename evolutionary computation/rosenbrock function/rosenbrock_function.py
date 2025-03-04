class optFunc:
  def eval(self, x: float) -> float:
    pass

  def grad(self, x: float) -> float:
    pass

  def hess(self) -> float:
    pass

  def output(self, x: float, grade: int = 2) -> float:
    if (grade == 1): 
      return self.eval(x)
    elif (grade == 2):
      return self.grad(x)
    elif (grade == 3):
      return self.hess()
    else: 
      print('Grade not supported')

class optFuncSphere(optFunc):
  def eval(self, x):
    return sum(xi ** 2 for xi in x)

  def ident(self, x):
    n = len(x)
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

  def grad(self, x):
    return 2*x

  def hess(self, x):
    return 2*self.ident(x)