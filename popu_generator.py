import numpy as np
import matplotlib.pyplot as plt
import os
import re

num_architectures = 100
upper_bound = [6, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1, 500, 500]
lower_bound = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100]

folder_name = 'bioex'
folder_path = os.path.join(os.getcwd(), folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_name}' created successfully at '{folder_path}'")
else:
    print(f"Folder '{folder_name}' already exists at '{folder_path}'")


def print_green(text):
    print("\033[92m" + str(text) + "\033[0m")

def generate_architectures(num_architectures):
    architectures = []
    for _ in range(num_architectures):
        architecture = []
        for i in range(len(upper_bound)):
            # Using discrete uniform distribution
            value = np.random.randint(lower_bound[i], upper_bound[i] + 1)
            architecture.append(value)
        architectures.append(architecture)
    return architectures

def calculate_distance(arch1, arch2):
    return np.sqrt(np.sum((np.array(arch1) - np.array(arch2))**2))

def calculate_complexity(architecture):
    return sum(architecture)

def select_diverse_architectures(architectures, num_architectures):
    selected_architectures = []
    remaining_architectures = architectures.copy()

    while len(selected_architectures) < num_architectures:
        if not remaining_architectures:
            break
        max_distance = 0
        best_architecture = None
        for architecture in remaining_architectures:
            min_distance = float('inf')
            for selected in selected_architectures:
                distance = calculate_distance(architecture, selected)
                min_distance = min(min_distance, distance)
            if min_distance > max_distance:
                max_distance = min_distance
                best_architecture = architecture
        selected_architectures.append(best_architecture)
        remaining_architectures.remove(best_architecture)

    return selected_architectures

all_architectures = generate_architectures(num_architectures)
selected_architectures = select_diverse_architectures(all_architectures, num_architectures)

# Calculate complexities of selected architectures
complexities = [calculate_complexity(arch) for arch in selected_architectures]

# Calculate the total complexity of selected architectures
total_complexity = sum(complexities)

# Calculate the total complexity of the minimum and maximum possible architectures
min_possible_complexity = sum(lower_bound)
max_possible_complexity = sum(upper_bound)

# Plot the complexities
plt.plot(range(1, len(complexities) + 1), complexities, marker='o', linestyle='-', color='b', label='Selected Architectures')
plt.axhline(y=min_possible_complexity, color='r', linestyle='--', label='Minimum Possible Complexity')
plt.axhline(y=max_possible_complexity, color='g', linestyle='--', label='Maximum Possible Complexity')
plt.xlabel('Architecture')
plt.ylabel('Complexity')
plt.title('Complexity of Selected Architectures')
plt.legend()
plt.savefig(os.path.join(folder_name, "initial_population.png"))


formatted_architectures = "Architectures = [\n"
for architecture in selected_architectures:
    architecture_list = [[int(digit) for digit in str(arch)] for arch in architecture]
    formatted_architectures += "   ["
    for sublist in architecture_list:
        formatted_architectures += "".join(map(str, sublist))
    formatted_architectures += "],\n"
formatted_architectures += "]"

file_path = "bioex/initial_population.txt"
with open(file_path, "a") as file:
    file.write(formatted_architectures)


file_path = "bioex/initial_population.txt"


with open(file_path, "r") as file:
    architectures_content = file.read()

numbers = re.findall(r'\d+', architectures_content)

numbers_list = [int(num) for num in numbers]

def print_red(text):
    print("\033[91m" + text + "\033[0m")


# Print the list
print(numbers_list)    




print_red(f"Total complexity of selected architectures: {total_complexity}")
print_red(f"Minimum possible complexity: {min_possible_complexity}")
print_red(f"Maximum possible complexity: {max_possible_complexity}")