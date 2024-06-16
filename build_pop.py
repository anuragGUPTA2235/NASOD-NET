import subprocess
import time
import os
def print_green(text):
    print("\033[92m" + str(text) + "\033[0m")
def print_red(text):
    print("\033[91m" + text + "\033[0m")
print_green("starting...")
time.sleep(2)
print_green("genetic neural architecture search for object detectors with surrogates")
# Replace 'ls' with the command you want to run
command = "ascii-image-converter images/dna2.jpeg -b -C"
command1 = "neofetch | pv -qL 100"
# Run the command
time.sleep(2)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
process1 = subprocess.Popen(command1.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# Get the output and error (if any)
output1, error1 = process1.communicate()
output, error = process.communicate()
# Print the output
print(output.decode())
time.sleep(2)
print()
print_green("minimum system requirements to run NASOD-NET : ")
print("\033[92m" + 'cpu' + "\033[0m",end=" ")
print_red("16 gb")
print("\033[92m" + 'gpu' + "\033[0m",end=" ")
print_red("nvidia cud 25 gb")
print("\033[92m" + 'os' + "\033[0m",end=" ")
print_red("ubuntu or debian based")
print("\033[92m" + 'cpu cores' + "\033[0m",end=" ")
print_red("8")
print("\033[92m" + 'python version' + "\033[0m",end=" ")
print_red("python3 or above")
print()
print_green("your system requirements : ")
print()
print(output1.decode())
print()
print_red("NASOD-NET may not run if minimum requirements are not fullfilled")
print_green("search space consists of 75 trillion object detectors")
time.sleep(2)
print_green("generating initial population of 100 arch")
print_green("all are distributed uniformly over the mammoth search space")
time.sleep(1)
subprocess.run(['python3', 'popu_generator.py'])
time.sleep(2)
subprocess.run(['python3', 'surrogates_fit.py'])
time.sleep(2)






