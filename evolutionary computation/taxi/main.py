import numpy as np
import random
from copy import deepcopy

# Load problem data from VRP.dat
def load_vrp_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse the data
    dimension = 0
    capacity = 0
    vehicles = 0
    demand = []
    distance_matrix = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        elif line.startswith('VEHICLES'):
            vehicles = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_SECTION'):
            i += 1
            for _ in range(dimension):
                row = list(map(int, lines[i].strip().split()))
                distance_matrix.append(row)
                i += 1
        elif line.startswith('DEMAND_SECTION'):
            i += 1
            for _ in range(dimension):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    demand.append(int(parts[1]))
                i += 1
        else:
            i += 1
    
    # Depot is the last node (45)
    depot = dimension - 1
    
    return {
        'dimension': dimension,
        'capacity': capacity,
        'vehicles': vehicles,
        'demand': demand,
        'distance_matrix': distance_matrix,
        'depot': depot
    }

# Initialize population with feasible solutions
def initialize_population(data, pop_size):
    dimension = data['dimension']
    capacity = data['capacity']
    vehicles = data['vehicles']
    demand = data['demand']
    depot = data['depot']
    
    population = []
    
    for _ in range(pop_size):
        # Create a random feasible solution
        solution = []
        remaining_customers = list(range(dimension - 1))  # Exclude depot
        random.shuffle(remaining_customers)
        
        for _ in range(vehicles):
            route = [depot]
            current_load = 0
            
            while remaining_customers and current_load < capacity:
                customer = remaining_customers[0]
                if current_load + demand[customer] <= capacity:
                    route.append(customer)
                    current_load += demand[customer]
                    remaining_customers.pop(0)
                else:
                    break
            
            route.append(depot)
            solution.append(route)
        
        # If there are remaining customers, assign them to existing routes if possible
        while remaining_customers:
            assigned = False
            for route in solution:
                current_load = sum(demand[c] for c in route[1:-1])
                customer = remaining_customers[0]
                if current_load + demand[customer] <= capacity:
                    # Insert at position that minimizes distance increase
                    best_pos = 1
                    min_increase = float('inf')
                    for i in range(1, len(route)):
                        increase = (data['distance_matrix'][route[i-1]][customer] + 
                                   data['distance_matrix'][customer][route[i]] - 
                                   data['distance_matrix'][route[i-1]][route[i]])
                        if increase < min_increase:
                            min_increase = increase
                            best_pos = i
                    
                    route.insert(best_pos, customer)
                    remaining_customers.pop(0)
                    assigned = True
                    break
            
            if not assigned:
                # Create a new route if possible
                if len(solution) < vehicles:
                    customer = remaining_customers.pop(0)
                    solution.append([depot, customer, depot])
                else:
                    # If no more vehicles, force assignment (will be penalized)
                    for route in solution:
                        current_load = sum(demand[c] for c in route[1:-1])
                        customer = remaining_customers[0]
                        if current_load + demand[customer] <= capacity * 1.5:  # Allow some overloading
                            best_pos = 1
                            min_increase = float('inf')
                            for i in range(1, len(route)):
                                increase = (data['distance_matrix'][route[i-1]][customer] + 
                                           data['distance_matrix'][customer][route[i]] - 
                                           data['distance_matrix'][route[i-1]][route[i]])
                                if increase < min_increase:
                                    min_increase = increase
                                    best_pos = i
                            
                            route.insert(best_pos, customer)
                            remaining_customers.pop(0)
                            break
        
        population.append(solution)
    
    return population

# Calculate total distance of a solution
def calculate_distance(solution, distance_matrix):
    total_distance = 0
    for route in solution:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i+1]]
    return total_distance

# Calculate fitness (inverse of distance with penalty for overload)
def calculate_fitness(solution, data):
    distance = calculate_distance(solution, data['distance_matrix'])
    capacity = data['capacity']
    demand = data['demand']
    penalty = 0
    
    for route in solution:
        route_demand = sum(demand[c] for c in route[1:-1])
        if route_demand > capacity:
            penalty += (route_demand - capacity) * 1000  # Large penalty for overload
    
    return 1 / (distance + penalty + 1)  # Add 1 to avoid division by zero

# Tournament selection
def tournament_selection(population, data, tournament_size=3):
    selected = []
    for _ in range(2):  # Select 2 parents
        contestants = random.sample(population, tournament_size)
        best = max(contestants, key=lambda x: calculate_fitness(x, data))
        selected.append(best)
    return selected

# Ordered crossover for VRP
def ordered_crossover(parent1, parent2, data):
    depot = data['depot']
    
    # Flatten parents (excluding depots)
    flat1 = [c for route in parent1 for c in route if c != depot]
    flat2 = [c for route in parent2 for c in route if c != depot]
    
    # Perform OX
    size = len(flat1)
    a, b = sorted(random.sample(range(size), 2))
    
    child1_flat = [-1] * size
    child2_flat = [-1] * size
    
    # Copy the segment between a and b
    child1_flat[a:b] = flat1[a:b]
    child2_flat[a:b] = flat2[a:b]
    
    # Fill remaining positions from parent2/parent1
    fill1 = [c for c in flat2 if c not in child1_flat[a:b]]
    fill2 = [c for c in flat1 if c not in child2_flat[a:b]]
    
    ptr1 = 0
    ptr2 = 0
    for i in list(range(0, a)) + list(range(b, size)):
        if ptr1 < len(fill1):
            child1_flat[i] = fill1[ptr1]
            ptr1 += 1
        if ptr2 < len(fill2):
            child2_flat[i] = fill2[ptr2]
            ptr2 += 1
    
    # Split back into routes
    def split_to_routes(flat, data):
        capacity = data['capacity']
        demand = data['demand']
        vehicles = data['vehicles']
        depot = data['depot']
        
        solution = []
        current_route = [depot]
        current_load = 0
        
        for customer in flat:
            if current_load + demand[customer] <= capacity:
                current_route.append(customer)
                current_load += demand[customer]
            else:
                current_route.append(depot)
                solution.append(current_route)
                current_route = [depot, customer]
                current_load = demand[customer]
        
        current_route.append(depot)
        solution.append(current_route)
        
        # If we have too many routes, merge some
        while len(solution) > vehicles:
            # Find two routes with smallest merge cost
            best_i, best_j = 0, 1
            best_cost = float('inf')
            for i in range(len(solution)):
                for j in range(i+1, len(solution)):
                    cost = (data['distance_matrix'][solution[i][-2]][solution[j][1]] + 
                           data['distance_matrix'][solution[j][-2]][solution[i][1]] - 
                           data['distance_matrix'][solution[i][-2]][solution[i][-1]] - 
                           data['distance_matrix'][solution[j][-2]][solution[j][-1]])
                    if cost < best_cost:
                        best_cost = cost
                        best_i, best_j = i, j
            
            # Merge the two routes
            merged = solution[best_i][:-1] + solution[best_j][1:]
            del solution[max(best_i, best_j)]
            del solution[min(best_i, best_j)]
            solution.append(merged)
        
        return solution
    
    child1 = split_to_routes(child1_flat, data)
    child2 = split_to_routes(child2_flat, data)
    
    return child1, child2

# Swap mutation
def swap_mutation(solution, data):
    mutated = deepcopy(solution)
    
    # Select two random routes
    route1_idx, route2_idx = random.sample(range(len(mutated)), 2)
    route1 = mutated[route1_idx]
    route2 = mutated[route2_idx]
    
    # Select random customers from each route (excluding depots)
    if len(route1) > 2 and len(route2) > 2:
        cust1_idx = random.randint(1, len(route1) - 2)
        cust2_idx = random.randint(1, len(route2) - 2)
        
        # Check capacity constraints
        route1_demand = sum(data['demand'][c] for c in route1[1:-1])
        route2_demand = sum(data['demand'][c] for c in route2[1:-1])
        
        new_route1_demand = route1_demand - data['demand'][route1[cust1_idx]] + data['demand'][route2[cust2_idx]]
        new_route2_demand = route2_demand - data['demand'][route2[cust2_idx]] + data['demand'][route1[cust1_idx]]
        
        if (new_route1_demand <= data['capacity'] and 
            new_route2_demand <= data['capacity']):
            # Perform the swap
            route1[cust1_idx], route2[cust2_idx] = route2[cust2_idx], route1[cust1_idx]
    
    return mutated

# Local search improvement (2-opt)
def two_opt(route, distance_matrix):
    improved = True
    best_route = route
    best_distance = calculate_distance([route], distance_matrix)
    
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i+1, len(route) - 1):
                if j - i == 1:
                    continue  # No point swapping adjacent
                
                # Create new route by reversing i..j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_distance = calculate_distance([new_route], distance_matrix)
                
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        
        route = best_route
    
    return best_route

def apply_local_search(solution, data):
    improved_solution = []
    for route in solution:
        improved_route = two_opt(route, data['distance_matrix'])
        improved_solution.append(improved_route)
    return improved_solution

# Main EA algorithm
def evolutionary_algorithm(data, pop_size=50, generations=100, crossover_prob=0.8, mutation_prob=0.2):
    # Initialize population
    population = initialize_population(data, pop_size)
    
    best_solution = None
    best_fitness = -float('inf')
    
    for gen in range(generations):
        # Evaluate population
        fitnesses = [calculate_fitness(ind, data) for ind in population]
        
        # Track best solution
        current_best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        print("generation: ", gen,", fitness: ", current_best_fitness)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = deepcopy(population[current_best_idx])
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best solution
        new_population.append(best_solution)
        
        while len(new_population) < pop_size:
            # Selection
            parents = tournament_selection(population, data)
            
            # Crossover
            if random.random() < crossover_prob:
                offspring1, offspring2 = ordered_crossover(parents[0], parents[1], data)
            else:
                offspring1, offspring2 = parents[0], parents[1]
            
            # Mutation
            if random.random() < mutation_prob:
                offspring1 = swap_mutation(offspring1, data)
            if random.random() < mutation_prob:
                offspring2 = swap_mutation(offspring2, data)
            
            # Apply local search to some offspring
            if random.random() < 0.3:  # 30% chance for local search
                offspring1 = apply_local_search(offspring1, data)
            if random.random() < 0.3:
                offspring2 = apply_local_search(offspring2, data)
            
            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)
        
        population = new_population
    
    # Final evaluation
    fitnesses = [calculate_fitness(ind, data) for ind in population]
    best_idx = np.argmax(fitnesses)
    best_solution = population[best_idx]
    
    return best_solution

# Main execution
if __name__ == "__main__":
    # Load the VRP data
    vrp_data = load_vrp_data("A045-03f.dat")
    
    # Run the evolutionary algorithm
    best_solution = evolutionary_algorithm(
        vrp_data, 
        pop_size=50, 
        generations=200,
        crossover_prob=0.9,
        mutation_prob=0.1
    )
    
    # Print the best solution
    print("Best Solution Found:")
    total_distance = calculate_distance(best_solution, vrp_data['distance_matrix'])
    print(f"Total Distance: {total_distance}")
    
    for i, route in enumerate(best_solution, 1):
        route_demand = sum(vrp_data['demand'][c] for c in route[1:-1])
        print(f"Vehicle {i}: Route {route} | Demand: {route_demand}/{vrp_data['capacity']} | Distance: {calculate_distance([route], vrp_data['distance_matrix'])}")