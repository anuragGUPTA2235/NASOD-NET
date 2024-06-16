from nasodmodel import YOLOv1
import torch
from torchsummary import summary

# Initialize the model
model = YOLOv1(S=7, B=2, C=91)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move the model to the device
model.to(device)

# Wrap the model with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)
# Generate a random input
input = torch.randn(1, 3, 448, 448).to(device)

# Forward pass
out = model(input)

# Print the output shape
print(out.shape)
# Print the device for each model parameter
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Device: {param.device}")
    
summary(model,(3,448,448))    
