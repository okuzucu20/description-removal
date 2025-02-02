import torch

import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, number_of_hidden_layers=1, do_zero_initialize=True, print_intermediate_norms=True):
        super(MLPNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_of_hidden_layers = number_of_hidden_layers

        layers = []
        for i in range(self.number_of_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_size, self.hidden_size))
            else:
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size, self.output_size))
        
        if do_zero_initialize:
            self.zero_initialize_weights()
        self.print_intermediate_norms = print_intermediate_norms
        
    def forward(self, x):
        print(f"Input norm: {torch.norm(x)}")
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if self.print_intermediate_norms:
                print(f"Layer {i} norm: {torch.norm(x)}")
        print(f"Output norm: {torch.norm(x)}")
        return x
    
    def zero_initialize_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data.fill_(0.0)
                layer.bias.data.fill_(0.0)
    
# Example usage
if __name__ == "__main__":
    mlp = MLPNetwork(512, 1024)
    x = torch.randn(10, 512)
    output = mlp(x)
    print(output.shape)
