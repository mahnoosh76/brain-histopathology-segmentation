import torch

model = torch.load("final_model.pth", map_location="cpu")
print("Model loaded successfully")
