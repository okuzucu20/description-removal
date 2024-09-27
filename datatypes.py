import torch
from dataclasses import dataclass
from typing import List, Optional
import PIL

@dataclass
class TransformerNetworkOutput:
    foreground_output: torch.Tensor
    background_output: torch.Tensor

@dataclass
class AlphaCLIPFocusedEmbeddings:
    foreground_focused: torch.Tensor
    background_focused: torch.Tensor

@dataclass
class ValidationBatchProcessedOutput:
    images: Optional[List[PIL.Image.Image]] = None
    masks: Optional[List[PIL.Image.Image]] = None
    fg_focused_generation: Optional[List[PIL.Image.Image]] = None
    bg_focused_generation: Optional[List[PIL.Image.Image]] = None
    fg_only_generation: Optional[List[PIL.Image.Image]] = None
    bg_only_generation: Optional[List[PIL.Image.Image]] = None
    transformer_output_plus_clipaway_generation: Optional[List[PIL.Image.Image]] = None
    default_clipaway_generation: Optional[List[PIL.Image.Image]] = None
    fg_focused_generation_inpaint: Optional[List[PIL.Image.Image]] = None
    bg_focused_generation_inpaint: Optional[List[PIL.Image.Image]] = None
    fg_only_generation_inpaint: Optional[List[PIL.Image.Image]] = None
    bg_only_generation_inpaint: Optional[List[PIL.Image.Image]] = None
    transformer_output_plus_clipaway_generation_inpaint: Optional[List[PIL.Image.Image]] = None
    default_clipaway_generation_inpaint: Optional[List[PIL.Image.Image]] = None
    alpha_clip_transformer_mean: Optional[List[PIL.Image.Image]] = None
    alpha_clip_transformer_mean_inpaint: Optional[List[PIL.Image.Image]] = None
    alpha_clip_transformer_mean_fg: Optional[List[PIL.Image.Image]] = None
    alpha_clip_transformer_mean_inpaint_fg: Optional[List[PIL.Image.Image]] = None   
    alpha_clip_transformer_mean_bg: Optional[List[PIL.Image.Image]] = None
    alpha_clip_transformer_mean_inpaint_bg: Optional[List[PIL.Image.Image]] = None
