import re
from trainnasod import make_archive

file_path = "bioex/initial_population.txt"
with open(file_path, "r") as file:
    architectures_content = file.read()
numbers = re.findall(r'\d+', architectures_content)
numbers_list = [str(num) for num in numbers]
red_code = "\033[91m"  # Red color
green_code = "\033[92m"  # Green color
reset_code = "\033[0m"  # Reset color
for items in numbers_list:
    print(items)
    genome1 = []
    genome2 = []
    genome3 = []
    for i in range(0,6):
       genome1.append(int(items[i]))
    for i in range(6,12):
       genome2.append(int(items[i]))
    for i in range(12,18):
       genome3.append(int(items[i]))              
    print(genome1)
    print(genome2)
    print(genome3)
    print("starting archive")
    make_archive(genome1,genome2.genome3)   
    break


