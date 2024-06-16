import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(18, 32),  # Increased number of neurons
            nn.ReLU(),
            nn.Linear(32, 16),  # Increased number of neurons
            nn.ReLU(),
            nn.Linear(16, 8),   # Reduced dimensionality to 8
            nn.ReLU(),
            nn.Linear(8, 1)     # Compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 8),    # Reconstruct from compressed representation
            nn.ReLU(),
            nn.Linear(8, 16),   # Increased number of neurons
            nn.ReLU(),
            nn.Linear(16, 32),  # Increased number of neurons
            nn.ReLU(),
            nn.Linear(32, 18),  # Reconstruct original dimensionality
    
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

        
def read_input_data(file_path):
    input_data = []
    with open(file_path, 'r') as file:
        for line in file:
            genome_string = line.strip()[:18]  # Read the first 18 characters
            genome_tensor = torch.tensor([int(bit) for bit in genome_string], dtype=torch.float32)
            input_data.append(genome_tensor)
    return torch.stack(input_data)        



# Instantiate the autoencoder
autoencoder = Autoencoder()

autoencoder = autoencoder.cuda()
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)



# Generate random input data (18-bit genomes)
file_path = '/run/user/1001/projectredmark/new_nasod/bioex/autoencoder_genome.txt'
input_data = read_input_data(file_path)
losses = []
# Training the autoencoder
num_epochs = 20000
for epoch in range(num_epochs):
    input_data= input_data.cuda()
    # Forward pass
    outputs = autoencoder(input_data)
    loss = criterion(outputs, input_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Print progress
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the trained autoencoder
encoded_data = autoencoder.encoder(input_data)
reconstructed_data = autoencoder.decoder(encoded_data)

plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig("bioex/autoenocder_dime_red.png")

torch.save(autoencoder,"bioex/autoencoder.pth")
