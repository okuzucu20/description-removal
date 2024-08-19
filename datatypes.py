import torch
from dataclasses import dataclass

@dataclass
class TransformerNetworkOutput:
    foreground_output: torch.Tensor
    background_output: torch.Tensor
    
@dataclass
class AlphaCLIPFocusedEmbeddings:
    foreground_focused: torch.Tensor
    background_focused: torch.Tensor