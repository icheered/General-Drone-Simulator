# pytorch-triton-rocm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))