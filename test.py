import torch

class a(torch.nn.Module):
    def __init__(self):
        self.w=torch.nn.Linear(2,2)
    