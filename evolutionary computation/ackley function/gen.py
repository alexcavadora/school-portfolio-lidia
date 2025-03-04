from optFunc import AckleyFunction 
import numpy as np
import struct
import random
import time 
import matplotlib.pyplot as plt

class GenAlgorithm():
    def __init__(self, func,pop_size, dim, max_gen, pcross=0.8, pmut=0.1):
        self.pop_size = pop_size
        self.func = func
        self.dim = dim
        self.max_gen = max_gen
        self.pop = [None for i in range(pop_size)]
        self.pop_fit = self.pop.copy()
        self.pop_select = self.pop.copy()
        self.offspring = self.pop.copy()
        self.pcross = pcross
        self.pmut = pmut
        self.x_history = []
        print("Genetic Algorithm initialized")

    def init_pop(self):
        n = self.pop_size
        d = self.dim

        for i in range(n):
            x_initial = np.matrix([[np.random.uniform(-30, 30)] for j in range(d) ])
            self.pop[i] = x_initial 
            # print(x_initial)
    
    def fitness(self, x):
        return self.func.eval(x).item()

    def tournament_selection(self, tournament_size=3):
        return [min(random.sample(self.pop, tournament_size), key=lambda ind: self.fitness(ind)) for _ in range(self.pop_size)]

    def fitness_eval(self):
        for i in range(0,self.pop_size):
            self.pop_fit[i] = self.fitness(self.pop[i])
        
        # Encontrar y preservar el mejor individuo
        best_idx = self.pop_fit.index(min(self.pop_fit))
        self.pop_select[0] = self.pop[best_idx].copy()        

    def select_pop(self):
        for i in range(1,self.pop_size):
            # Select two random individuals
            ii, jj = np.random.choice(self.pop_size, 2, replace=False)
            
            # Comparar fitness y seleccionar el mejor
            if self.pop_fit[ii] > self.pop_fit[jj]:
                self.pop_select[i] = self.pop[jj].copy()  # El de mayor fitness
            else:
                self.pop_select[i] = self.pop[ii].copy()  # El de mayor fitness

    def one_point_crossover(self, i, j):
        # Realizar crossover
        parent1 = self.pop_select[i]
        parent2 = self.pop_select[j]
        child1 = parent1.copy()
        child2 = parent2.copy()

        for k in range(self.dim):
            # Convertir los flotantes a representaciones binarias
            p1_bytes = struct.pack('d', parent1[k].item())  # 'd' para double (float64)
            p2_bytes = struct.pack('d', parent2[k].item())

            # Convertir los bytes a enteros
            p1_int = int.from_bytes(p1_bytes, 'big')
            p2_int = int.from_bytes(p2_bytes, 'big')

            # Seleccionar un punto de corte aleatorio
            num_bits = 8 * len(p1_bytes)  # Número de bits en un flotante (64 para double)
            cut_point = np.random.randint(1, num_bits)

            # Realizar el crossover a nivel de bit
            mask = (1 << cut_point) - 1  # Máscara para intercambiar bits
            child1_int = (p1_int & ~mask) | (p2_int & mask)
            child2_int = (p2_int & ~mask) | (p1_int & mask)

            # Convertir los enteros resultantes de nuevo a bytes
            child1_bytes = child1_int.to_bytes(8, 'big')
            child2_bytes = child2_int.to_bytes(8, 'big')

            # Convertir los bytes de nuevo a flotantes
            children1 = struct.unpack('d', child1_bytes)[0]
            children2 = struct.unpack('d', child2_bytes)[0]

            # Asignar los hijos a la descendencia
            child1[k] = children1
            child2[k] = children2
        self.offspring[i] = child1
        self.offspring[j] = child2



    def crossover(self):
        for i in range(0,self.pop_size,2):
            u = np.random.rand() # Random number between 0 and 1
            if u < self.pcross:
                # Realizar crossover de un punto
                self.one_point_crossover(i, i+1)
            else:
                # Copiar directamente los padres
                self.offspring[i] = self.pop_select[i].copy()
                self.offspring[i+1] = self.pop_select[i+1].copy()

    def mutate(self):
        # for i in range(self.pop_size):
        #     u = np.random.rand()
        #     if u < self.pmut:
        #         # Realizar mutación
        #         self.offspring[i] = self.offspring[i] + np.random.normal(0, 1, (self.dim, 1))
        for i in range(self.pop_size):
            for j in range(self.dim):
                u = np.random.rand()
                if u < self.pmut:
                    self.offspring[i][j] = self.offspring[i][j] + np.random.normal(-30, 30)
            
    def union(self):
        """
        Reemplaza la población actual con la descendencia
        """
        for i in range(self.pop_size):
            self.pop[i] = self.offspring[i].copy()

    def solve(self):
        self.init_pop()

        best_fitness = float('inf')
        best_solution = None        
        k_gen = 0
        print("Starting genetic algorithm")
        while k_gen < self.max_gen and best_fitness > 1e-6:
            # Evaluar fitness de la poblacion actual
            self.fitness_eval()

            # Guardar la mejor solución encontrada
            current_best = self.pop_fit.index(min(self.pop_fit))
            if self.pop_fit[current_best] < best_fitness:
                best_fitness = self.pop_fit[current_best]
                best_solution = self.pop[current_best].copy()

            # Guardar historial de soluciones
            self.x_history.append(self.pop[current_best].copy())
            

            # Proceso evolutivo
            self.select_pop()    # Selección
            self.crossover()     # Cruce
            self.mutate()        # Mutación
            self.union()         # Actualizar población

            # Opcional: mostrar progreso
            print(f"Generación {k_gen}: Mejor fitness = {best_fitness:2f}")
            k_gen += 1
        self.x_history = np.array(self.x_history)
        self.k_gen = k_gen
        return best_solution, best_fitness

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

            
            


n = 5 # 5D vector
Ackley = AckleyFunction(range(n)) 
GA = GenAlgorithm(Ackley, pop_size= 100, dim= n, max_gen=1000,pcross= 0.8, pmut=0.1)
start = time.time()
best_sol, best_fit = GA.solve()
end = time.time()
print("-" * 40)
print("Genetic Algorithm finished")
print("-" * 40)

print(f"Time elapsed: {end - start}")
print(f"Best solution: {best_sol}")
print(f"Best fitness: {best_fit}")
print(f'Generations: {GA.k_gen}')

GA.plot2D()