import re
from trainnasod import make_archive
import pyfiglet

text = "Archive"
ascii_art = pyfiglet.figlet_format(text)
print(ascii_art)

file_path = "/run/user/1001/projectredmark/new_nasod/bioex/initial_population.txt"
archive_file_path = "/run/user/1001/projectredmark/new_nasod/bioex/archive.txt"


with open(file_path, "r") as file:
    architectures_content = file.read()
numbers = re.findall(r'\d+', architectures_content)
numbers_list = [str(num) for num in numbers]
red_code = "\033[91m"  # Red color
green_code = "\033[92m"  # Green color
reset_code = "\033[0m"  # Reset color
print("starting archive")
print("????? ???")

for items in numbers_list:
    genome1 = []
    genome2 = []
    genome3 = []
    for i in range(0,6):
       genome1.append(int(items[i]))
    for i in range(6,12):
       genome2.append(int(items[i]))
    for i in range(12,18):
       genome3.append(int(items[i]))              
    print(f"genome1:{genome1} genome2:{genome2} genome3:{genome3}")
    make_archive(genome1,genome2,genome3)   

print("archive successfully created and saved in bioex folder")    


