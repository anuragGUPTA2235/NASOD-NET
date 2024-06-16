import torch
import torch.nn as nn

# Define the autoencoder class
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
            nn.Linear(32, 18)   # Reconstruct original dimensionality
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def genome_surroinput(genome):
# Load the trained model
 autoencoder = Autoencoder()
 autoencoder = torch.load('bioex/autoencoder.pth').cuda()


# Set the model in evaluation mode
 autoencoder.eval()

# Example inference on new data
 new_input_data = torch.tensor([genome], dtype=torch.float32).cuda()
 with torch.no_grad():
    surro_data = autoencoder.encoder(new_input_data)

 return surro_data.item()
 
