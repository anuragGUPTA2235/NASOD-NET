import re
from constructer import flops_returner
file_path = "bioex/initial_population.txt"
with open(file_path, "r") as file:
    architectures_content = file.read()
numbers = re.findall(r'\d+', architectures_content)
numbers_list = [int(num) for num in numbers]
red_code = "\033[91m"  # Red color
green_code = "\033[92m"  # Green color
reset_code = "\033[0m"  # Reset color
for items in numbers_list:
    flops = flops_returner(str(items))
    print(f"{red_code}{items}:{reset_code}{green_code}{flops}{reset_code}",end="")


