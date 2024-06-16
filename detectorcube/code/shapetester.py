import torch
import torch.nn.functional as F

# Assuming your tensor is named tensor_input
tensor_input = torch.randn(2, 1024, 14, 14)

# Resize the tensor to the desired shape
tensor_output = F.interpolate(tensor_input, size=(7, 7), mode='bilinear', align_corners=False)

print(tensor_output.size())  # This should print torch.Size([2, 1024, 7, 7])
