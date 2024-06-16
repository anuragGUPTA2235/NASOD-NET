import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pySOT.surrogate import RBFInterpolant, CubicKernel, TPSKernel, GPRegressor, PolyRegressor
import logging
from genome_to_surrinput import genome_surroinput,Autoencoder

file_path = '/run/user/1001/projectredmark/new_nasod/bioex/archive.txt'

def replace_nan_with_zero(data):
    """
    Function to replace NaN with 0 in the list of genomes.
    """
    for entry in data:
        if math.isnan(entry["mAP"]):
            entry["mAP"] = 0

def load_and_process_data(file_path):
    """
    Function to load data from a formatted text file, replace NaN values, and prepare xx and y arrays.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            genome1 = [int(line[9]), int(line[12]), int(line[15]), int(line[18]), int(line[21]), int(line[24])]
            genome2 = [int(line[36]), int(line[39]), int(line[42]), int(line[45]), int(line[48]), int(line[51])]
            genome3 = [int(line[63]), int(line[66]), int(line[69]), int(line[72]), int(line[75]), int(line[78])]
            mAP = float(line[85:])
            data.append({"genome1": genome1, "genome2": genome2, "genome3": genome3, "mAP": mAP})

    replace_nan_with_zero(data)

    # Prepare xx (genomes_combined) and y (mAP_values)
    genomes_combined = []
    maP_values = []

    for entry in data:
        # Combine genomes into a single string
        combined_genome = ''.join(map(str, entry['genome1'] + entry['genome2'] + entry['genome3']))
        genomes_combined.append([combined_genome])  # Wrap combined_genome in a list

        # Append mAP value
        maP_values.append([entry['mAP']])  # Wrap mAP value in a list

    return genomes_combined, np.array(maP_values)

# Function to analyze data using surrogate models
def analyze_data(xx, y,individual):
    """
    Function to analyze data using surrogate models.
    """
    #print(xx)
    encoded_genes = []
    for items in xx:
     list = []
     for jtems in items:
       #print(jtems)
       for ktems in jtems:
        list.append(int(ktems))
     #print(list)  
     
     encoded_gene = genome_surroinput(list)
     #print(encoded_gene)
     encoded_genes.append(encoded_gene)
    #print(encoded_genes)
    #print(len(encoded_genes))
    lb, ub = np.zeros(1), 100 * np.ones(1)
    #print(y)
    #print(encoded_genes)
    encoded_genes = np.array(encoded_genes)
    # Surrogate models
    rbf_cubic = RBFInterpolant(dim=1, lb=lb, ub=ub, kernel=CubicKernel(), tail=None, eta=1e-6)
    rbf_cubic.add_points(encoded_genes.reshape(-1, 1), y)

    rbf_tps = RBFInterpolant(dim=1, lb=lb, ub=ub, kernel=TPSKernel(), tail=None, eta=1e-6)
    rbf_tps.add_points(encoded_genes.reshape(-1, 1), y)

    gp = GPRegressor(dim=1, lb=lb, ub=ub)
    gp.add_points(encoded_genes.reshape(-1, 1), y)

    poly = PolyRegressor(dim=1, lb=lb, ub=ub, degree=3)
    poly.add_points(encoded_genes.reshape(-1, 1), y)

    # Plot original data and surrogate model predictions

    x = encoded_genes
    plt.figure(figsize=(14, 8))
    
    plt.plot(x, rbf_cubic.predict(x.reshape(-1, 1)), 'g', label="RBF Cubic")
    plt.plot(x, rbf_tps.predict(x.reshape(-1, 1)), 'm', label="RBF TPS")
    plt.plot(x, gp.predict(x.reshape(-1, 1)), 'y', label="GP")
    plt.plot(x, poly.predict(x.reshape(-1, 1)), 'r', label="Polynomial")
    
    plt.plot(x, y, 'k.', markersize=20, label="Data")
    plt.legend(fontsize=16)
    plt.ylabel("Mean Average Precision", fontsize=16)
    plt.xlabel("Autoencoder Encoded Genomes", fontsize=16)
    plt.savefig("/run/user/1001/projectredmark/new_nasod/bioex/surrogates_fit.png")
    

    # Print errors for each surrogate model
    print("Errors:")
    print("RBF Cubic error:\t{:0.2f}".format(np.max(np.abs(y - rbf_cubic.predict(x.reshape(-1, 1))))))
    print("RBF TPS error:\t\t{:0.2f}".format(np.max(np.abs(y - rbf_tps.predict(x.reshape(-1, 1))))))
    print("GP error:\t\t{:0.2f}".format(np.max(np.abs(y - gp.predict(x.reshape(-1, 1))))))
    print("Polynomial error:\t{:0.2f}".format(np.max(np.abs(y - poly.predict(x.reshape(-1, 1))))))

    # Determine the best surrogate model based on minimum error
    errors = {
        "RBF Cubic": np.max(np.abs(y - rbf_cubic.predict(x.reshape(-1, 1)))),
        "RBF TPS": np.max(np.abs(y - rbf_tps.predict(x.reshape(-1, 1)))),
        "GP": np.max(np.abs(y - gp.predict(x.reshape(-1, 1)))),
        "Polynomial": np.max(np.abs(y - poly.predict(x.reshape(-1, 1))))
    }

    # Find the surrogate model with the minimum error
    best_model = min(errors, key=errors.get)
    print(f"The best surrogate model is: {best_model}")

    best_model = {
        "RBF Cubic": rbf_cubic,
        "RBF TPS": rbf_tps,
        "GP": gp,
        "Polynomial": poly
    }[best_model]
     
    predicted_map = best_model.predict(genome_surroinput(individual))
    #print(predicted_map)
    return predicted_map

# Main function to run the analysis
def winner_surro(individual:list):
    # Load and preprocess data
    xx, y = load_and_process_data(file_path)
    # Analyze data using surrogate models
    predicted_map = analyze_data(xx, y,individual)
    return predicted_map
 
predicted_map = winner_surro([6,5,4,3,2,1,6,5,4,3,2,1,6,5,4,3,2,1])
print(predicted_map)


    
    
    
   
