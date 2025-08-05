import numpy as np
import matplotlib.pyplot as plt

class FuzzyVar:
    def __init__(self, name, min_value, max_value, units, fuzzy_sets=None):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.units = units
        self.fuzzy_sets = fuzzy_sets if fuzzy_sets is not None else []

        if isinstance(self.fuzzy_sets, int):
            self.generate_fuzzy_sets(self.fuzzy_sets)
    
    def generate_fuzzy_sets(self, n_sets):
        """
        Genera automáticamente n_sets de conjuntos difusos triangulares equidistantes.
        """
        self.fuzzy_sets = []
        step = (self.max_value - self.min_value) / (n_sets - 1)
        for i in range(n_sets):
            a = self.min_value + (i - 1) * step if i > 0 else self.min_value
            b = self.min_value + i * step
            c = self.min_value + (i + 1) * step if i < n_sets - 1 else self.max_value
            color = plt.cm.viridis(i / (n_sets - 1))  # Color automático basado en cantidad de sets
            fuzzy_set = FuzzySet(f"Set_{i+1}", a, b, c, color=color)
            self.fuzzy_sets.append(fuzzy_set)

    def membership(self, x, show=False):
        membership = []
        for fuzzy_set in self.fuzzy_sets:
            membership.append((fuzzy_set.label, fuzzy_set.membership(x))) 
        
        if show:
            labels = [mem[0] for mem in membership if mem[1] != 0]
            values = [mem[1] for mem in membership if mem[1] != 0]
            print(f"Membresía de {x}{self.units} en {self.name}:")
            for label, value in zip(labels, values):
                print(f"\t{label}: {value:.2f}")
        return membership
    
    def add_fuzzy_set(self, fuzzy_set):
        """
        Agrega un nuevo conjunto difuso.
        """
        self.fuzzy_sets.append(fuzzy_set)

    def remove_fuzzy_set(self, label):
        """
        Elimina un conjunto difuso por su etiqueta.
        """
        self.fuzzy_sets = [fs for fs in self.fuzzy_sets if fs.label != label]

    def modify_fuzzy_set(self, label, a=None, b=None, c=None):
        """
        Modifica los parámetros de un conjunto difuso por su etiqueta.
        """
        for fuzzy_set in self.fuzzy_sets:
            if fuzzy_set.label == label:
                fuzzy_set.set_parameters(a, b, c)
                break
    
    def plot_fuzzy_sets(self, filename='TEST.png'):
        """
        Grafica todos los conjuntos difusos de la variable.
        """
        x_values = np.linspace(self.min_value, self.max_value, 500)
        plt.figure(figsize=(8, 6))
        for fuzzy_set in self.fuzzy_sets:
            fuzzy_set.plot_membership_function(x_values)

        plt.title(f"Funciones de Membresía - {self.name}", fontsize=16)
        plt.xlabel(f"Universo de Discurso ({self.units})", fontsize=12)
        plt.ylabel("Grado de Membresía", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper right", fontsize=10)
        plt.xlim(self.min_value, self.max_value)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(filename)
        
class FuzzySet:
    def __init__(self, label, a, b, c, color):
        self.label = label
        self.a = a  # inicio de la base de la función triangular
        self.b = b  # pico de la función triangular
        self.c = c  # final de la base de la función triangular
        self.color = color  # Color de la función de membresía
    
    def membership(self, x):
        """
        Calcula el valor de membresía para un valor 'x'.
        """
        if x <= self.a or x >= self.c:
            return 0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x < self.c:
            return (self.c - x) / (self.c - self.b)
    
    def set_parameters(self, a, b, c):
        """
        Permite modificar los parámetros de la función de membresía triangular.
        """
        self.a = a
        self.b = b
        self.c = c
    
    def get_parameters(self):
        """
        Retorna los parámetros actuales de la función de membresía.
        """
        return self.a, self.b, self.c
    
    def plot_membership_function(self, x_values):
        y_values = [self.membership(x) for x in x_values]
        plt.plot(x_values, y_values, label=self.label, color=self.color, linewidth=2)

# Ejemplo de uso:
# Creación de una variable difusa con sets automáticos
Temp = FuzzyVar("Temperatura", 0, 60, "°C", fuzzy_sets=5)
Temp.plot_fuzzy_sets("auto.png")

# Agregar un nuevo conjunto difuso manualmente
muy_caliente = FuzzySet("Muy Caliente", 45, 50, 55, color='red')
Temp.add_fuzzy_set(muy_caliente)

# Modificar un conjunto difuso existente
Temp.modify_fuzzy_set("Set_3", a=10, b=20, c=30)

# Eliminar un conjunto difuso
Temp.remove_fuzzy_set("Set_2")

Temp.membership(50, show=True)
Temp.membership(15, show=True)
# Graficar después de las modificaciones
Temp.plot_fuzzy_sets("auto_changed.png")