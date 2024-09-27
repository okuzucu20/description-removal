import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from typing import Dict, Any, Union
from torchvision.transforms import ToTensor

class COCODataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, metadata_file: str, embeds_dir:str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata_file = metadata_file
        self.embeds_dir = embeds_dir
        self.embeds_dir_fg = os.path.join(embeds_dir, "fg")
        self.embeds_dir_bg = os.path.join(embeds_dir, "bg")

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.image_filenames = [item['image_id'] for item in self.metadata.values()]
        self.mask_filenames = list(self.metadata.keys())
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.mask_filenames)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        image_path = os.path.join(self.image_dir, self.image_filenames[idx] + '.jpg')
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx] + '.png')
        fg_description = self.metadata[self.mask_filenames[idx]]['fg_text']
        bg_description = self.metadata[self.mask_filenames[idx]]['bg_text']
        bg_embed = torch.load(os.path.join(self.embeds_dir_bg, self.mask_filenames[idx] + ".pt")).squeeze(0)
        fg_embed = torch.load(os.path.join(self.embeds_dir_fg, self.mask_filenames[idx] + ".pt")).squeeze(0)

        image = Image.open(image_path).resize((224, 224), Image.LANCZOS).convert('RGB')
        mask = Image.open(mask_path).resize((224, 224), Image.NEAREST).convert('L')

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return {
            "image": image,
            "mask": mask,
            "fg_description": fg_description,
            "bg_description": bg_description,
            "fg_embed": fg_embed,
            "bg_embed": bg_embed,
        }

class EvalDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.mask_filenames)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        image_id = self.mask_filenames[idx].split('_')[0] + '.jpg'
        image_path = os.path.join(self.image_dir, image_id)

        image = Image.open(image_path).resize((224, 224), Image.LANCZOS).convert('RGB')
        mask = Image.open(mask_path).resize((224, 224), Image.NEAREST).convert('L')

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return {
            "image": image,
            "mask": mask
        }

