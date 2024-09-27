import torch
import torch.nn as nn

class SubspaceProjectorNetwork(nn.Module):
    def __init__(self, number_of_layers=2, dim_in=1024,  dim_out=1024):
        super(SubspaceProjectorNetwork, self).__init__()
        self.dim_in = dim_in
        self.dim_out= dim_out
        self.number_of_layers = number_of_layers
        self.layers = nn.ModuleList()
        self.output_layer = nn.Linear(dim_out, dim_out)
        for i in range(number_of_layers):
            self.layers.append(nn.Linear(dim_in, dim_out))
            self.layers.append(nn.ReLU())
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
