import torch
from torch import nn
from torch.nn import functional as F

def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.Linear(28*28,512)
            self.ln2 = nn.Linear(512,512)
            self.ln3 = nn.Linear(512,10)
        def forward(self,x):
            x = x.view(x.size(0),-1)
            x = F.relu(self.ln1(x))
            x = F.relu(self.ln2(x))
            x = self.ln3(x)
            return x
        
    model = NeuralNetwork().to(device)
    return model