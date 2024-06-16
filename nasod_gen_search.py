import numpy as np
import random
import matplotlib.pyplot as plt
from constructer import flops_returner
import re
from surrogates_fit import winner_surro,Autoencoder


import logging
import sys
from contextlib import contextmanager

import pyfiglet

text = "NASOD-NET SEARCH BEGINS"
ascii_art = pyfiglet.figlet_format(text)
print(ascii_art)

# Suppress all INFO logging
logging.getLogger().setLevel(logging.WARNING)

@contextmanager
def suppress_stdout():
    with open('/dev/null', 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
            
            

global gen_count
gen_count = 0

# Define initial population based on the provided population
file_path = "bioex/initial_population.txt"

with open(file_path, "r") as file:
    architectures_content = file.read()
numbers = re.findall(r'\d+', architectures_content)
initial_population = [str(num) for num in numbers]

for items in initial_population:
    genome = ""
    for jtems in items:
        genome += str(jtems)
        


# Convert initial population from string to list of lists of integers
def convert_population(pop):
    return [list(map(int, list(ind))) for ind in pop]

population = convert_population(initial_population)
print("INITIAL POPULATION")
print(population)

# Fitness functions
## SURROGATE ASSISTED OBJECTIVE FUNCTION

def fitness_map(individual):
    # MEAN AVERAGE PRECISION
    print(individual)
    
    print("USING ARCHIVE WINNER SURROGATE")
    predicted_map = winner_surro(individual[:18])

    print(f"surrogate map : {predicted_map[0][0]}")     

    return predicted_map[0][0]

def fitness_flops(individual):
    # FLOPS OF THE ARCHITECTURES
    #print(individual)
    string_genome = ""
    for items in individual:
        string_genome += str(items)
    #print(string_genome)    
    flops = flops_returner(string_genome)
    print(f"flops:{flops}")
    #exit("hastalavista")
    return flops

# Non-Dominated Sorting
def non_dominated_sort(population):
    fronts = [[]]
    domination_count = [0] * len(population)
    dominated_solutions = [[] for _ in range(len(population))]

    for p in range(len(population)):
        for q in range(len(population)):
            if p != q:
                if (fitness_map(population[p]) < fitness_map(population[q]) and
                    fitness_flops(population[p]) < fitness_flops(population[q])):
                    dominated_solutions[p].append(q)
                elif (fitness_map(population[q]) < fitness_map(population[p]) and
                      fitness_flops(population[q]) < fitness_flops(population[p])):
                    domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

# Crowding Distance Calculation
def calculate_crowding_distance(front, population):
    distances = [0] * len(front)
    if len(front) == 0:
        return distances

    for m in range(2):  # number of objectives
        if m == 0:
            front.sort(key=lambda x: fitness_map(population[x]))
        else:
            front.sort(key=lambda x: fitness_flops(population[x]))

        distances[0] = distances[-1] = float('inf')
        for i in range(1, len(front)-1):
            ## remove G from flops and convert to float as its string
            print(fitness_map(population[front[i+1]]))
            print("hello")
            print(type(fitness_flops(population[front[i+1]])))
            print(fitness_flops(population[front[i+1]])[0:-1])
            distances[i] += (fitness_map(population[front[i+1]]) - fitness_map(population[front[i-1]])) if m == 0 else \
                            (float(fitness_flops(population[front[i+1]])[0:-1]) - float(fitness_flops(population[front[i-1]])[0:-1]))
    return distances

# Tournament Selection
def tournament_selection(population, fronts, crowding_distances):
    selected = []

    # Create a map from individual index to its front index
    individual_fronts = {}
    for front_index, front in enumerate(fronts):
        for individual in front:
            individual_fronts[individual] = front_index

    # Create a flat list of crowding distances for all individuals
    crowding_distances_flat = [0] * len(population)
    for front in fronts:
        front_crowding_distances = calculate_crowding_distance(front, population)
        for idx, individual in enumerate(front):
            crowding_distances_flat[individual] = front_crowding_distances[idx]

    for i in range(len(population)):
        a, b = random.sample(range(len(population)), 2)
        if (individual_fronts[a] < individual_fronts[b] or
            (individual_fronts[a] == individual_fronts[b] and crowding_distances_flat[a] > crowding_distances_flat[b])):
            selected.append(population[a])
        else:
            selected.append(population[b])
    return selected

# Crossover and Mutation
def crossover(parent1, parent2):
    # Example: single-point crossover
    point = random.randint(1, len(parent1)-1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutate(individual, mutation_rate=0.05):
    # INTEGER ENCODING #################################
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            if   i == 0:
                individual[i] = random.randint(0, 6)
            elif i == 1:
                individual[i] = random.randint(0, 5)
            elif i == 2:
                individual[i] = random.randint(0, 4)
            elif i == 3:
                individual[i] = random.randint(0, 3)
            elif i == 4:
                individual[i] = random.randint(0, 2)
            elif i == 5:
                individual[i] = random.randint(0, 1)
            elif i == 6:
                individual[i] = random.randint(0, 6)
            elif i == 7:
                individual[i] = random.randint(0, 5)
            elif i == 8:
                individual[i] = random.randint(0, 4)
            elif i == 9:
                individual[i] = random.randint(0, 3)
            elif i == 10:
                individual[i] = random.randint(0, 2)
            elif i == 11:
                individual[i] = random.randint(0, 1)
            elif i == 12:
                individual[i] = random.randint(0, 6)
            elif i == 13:
                individual[i] = random.randint(0, 5)
            elif i == 14:
                individual[i] = random.randint(0, 4)
            elif i == 15:
                individual[i] = random.randint(0, 3)
            elif i == 16:
                individual[i] = random.randint(0, 2)
            elif i == 17:
                individual[i] = random.randint(0, 1)

    return individual

# Main NSGA-II algorithm
def nsga2(population, generations, pop_size):
    for generation in range(generations):
        print(f"Generation {generation}")
        fitness_values = [(fitness_map(ind), fitness_flops(ind)) for ind in population]
        fronts = non_dominated_sort(population)

        next_population = []
        crowding_distances = []
        for front in fronts:
            if len(next_population) + len(front) <= pop_size:
                next_population.extend(front)
            else:
                crowding_distances = calculate_crowding_distance(front, population)
                sorted_front = sorted(zip(front, crowding_distances), key=lambda x: x[1], reverse=True)
                next_population.extend([x[0] for x in sorted_front[:pop_size - len(next_population)]])
                break

        if not crowding_distances:
            crowding_distances = [float('inf')] * len(population)

        mating_pool = tournament_selection(population, fronts, crowding_distances)

        offspring_population = []
        while len(offspring_population) < pop_size:
            parent1, parent2 = random.sample(mating_pool, 2)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring_population.append(mutate(offspring1))
            if len(offspring_population) < pop_size:
                offspring_population.append(mutate(offspring2))

        population = [population[i] for i in next_population] + offspring_population
        print(f"Population size: {len(population)}")
        print("offspring population")
        with open('bioex/archive.txt','a') as file:
         file.write(offspring_population)
         file.write('\n')
        for item in offspring_population:
             print(item)
             exit("hastalavista")
        print("Total Population:")
        for item in population:
             print(item)
        print("fitness values:")
        for item in fitness_values:
             print(item)


    return population, fronts

# Parameters
pop_size = len(initial_population)
generations = 2

# Run NSGA-II
final_population, final_fronts = nsga2(population, generations, pop_size)

# Convert final population back to string format
def convert_back_population(pop):
    return [''.join(map(str, ind)) for ind in pop]

final_population_str = convert_back_population(final_population)

# Print final population
print("Final Population:")
for ind in final_population_str:
    print(ind)

# Extract and print final Pareto front
final_pareto_front = final_fronts[0]
pareto_front_population = [final_population[i] for i in final_pareto_front]
pareto_front_population_str = convert_back_population(pareto_front_population)

print("\nFinal Pareto Front:")
for ind in pareto_front_population_str:
    print(ind)

# Plotting the Pareto front
pareto_fitness_values = [(fitness_map(ind), fitness_flops(ind)) for ind in pareto_front_population]

# Separate fitness values for plotting
map_values = [f[0] for f in pareto_fitness_values]
flops_values = [f[1] for f in pareto_fitness_values]

plt.figure(figsize=(10, 6))
plt.plot(map_values, flops_values, color='red', label='Pareto Front')

# Annotate points with the population strings
for i, ind in enumerate(pareto_front_population_str):
    plt.annotate(ind, (map_values[i], flops_values[i]), fontsize=8)


plt.xlabel('Fitness Map')
plt.ylabel('Fitness FLOPs')
plt.title('Pareto Front')
plt.legend()
plt.grid(True)
plt.savefig("final_pareto.jpeg")