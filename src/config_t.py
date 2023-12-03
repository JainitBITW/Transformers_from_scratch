#Hyperparameters
import torch
d_model = 300
num_heads = 6
d_ff = 300
num_layers = 2
dropout = 0.3
batch_size = 16
epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = None
