import torch

import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block for MLP with:
      - First linear (normal init)
      - LayerNorm
      - ReLU
      - Second linear (zero-initialized => "zero conv" style)
      - Another LayerNorm
      - Residual connection
    """
    def __init__(self, dim, do_zero_initialize=False):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        if do_zero_initialize:
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
            
    def forward(self, x):
        x_in = x
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.ln2(x)
        return x_in + x


class MLPNetwork(nn.Module): 
    """
    An MLP with residual blocks, layer norms, and zero-initialized second linear
    in each block (similar to 'zero conv' from ControlNet).

    By default, we assume input_size == hidden_size so we can do direct residuals.

    - number_of_hidden_layers: how many ResidualBlocks to stack
    - do_zero_initialize: if True, zero-init the second linear in each block
    - print_intermediate_norms: if True, print the norm at each stage
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 number_of_hidden_layers: int = 1,
                 do_zero_initialize: bool = False,
                 print_intermediate_norms: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.do_zero_initialize = do_zero_initialize
        
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.ln_in = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, do_zero_initialize) for _ in range(number_of_hidden_layers)
        ])
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.ln_out = nn.LayerNorm(output_size)
        self.print_intermediate_norms = print_intermediate_norms
        
    def forward(self, x):
        if self.print_intermediate_norms:
            print(f"Input norm: {x.norm().item()}")
        x = self.linear_in(x)
        x = self.ln_in(x)
        x = self.activation(x)
        for i, block in enumerate(self.blocks):
            if self.print_intermediate_norms:
                print(f"Block {i} norm: {x.norm().item()}")
            x = block(x)
        x = self.linear_out(x)
        x = self.ln_out(x)
        return x

    
# Example usage
if __name__ == "__main__":
    mlp = MLPNetwork(512, 1024)
    x = torch.randn(10, 512)
    output = mlp(x)
    print(output.shape)
