import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from typing import Dict, Any, Union
from torchvision.transforms import ToTensor

class COCODataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, metadata_file: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata_file = metadata_file

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

        image = Image.open(image_path).resize((224, 224), Image.LANCZOS).convert('RGB')
        mask = Image.open(mask_path).resize((224, 224), Image.NEAREST).convert('L')

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return {
            "image": image,
            "mask": mask,
            "fg_description": fg_description,
            "bg_description": bg_description,
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


class LayerDiffuseDataset(Dataset):
    def __init__(self, root_dir, images_dir):
        self.combined_images_dir = os.path.join(root_dir, 'combined')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.metadata_file = os.path.join(root_dir, 'layer_diffuse_metadata.json')
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
            
        self.bg_images = os.listdir(os.path.join(images_dir, "bg"))
        self.fg_images = os.listdir(os.path.join(images_dir, "fg"))
        self.masks = [os.path.join(self.mask_dir, mask) for mask in os.listdir(self.mask_dir)]
        self.combined_images = [os.path.join(self.combined_images_dir, img) for img in os.listdir(self.combined_images_dir)]
        self.to_tensor = ToTensor()
    
        
    def __len__(self):
        return len(self.metadata)
    
    def get_fg_objects(self, fg_path1, fg_path2):
        fg_word1 = fg_path1.split('/')[-1].split('_')[0]
        
        if fg_path2 is not None:
            fg_word2 = fg_path2.split('/')[-1].split('_')[0]
            return fg_word1, fg_word2, True
        else:
            return fg_word1, None, False
    
    def __getitem__(self, idx):
        combined_image = self.combined_images[idx]
        
        bg_path = self.metadata[combined_image]['bg_path']
        fg_word1, _, _ = self.get_fg_objects(self.metadata[combined_image]['fg_path1'], self.metadata[combined_image]['fg_path2'])
        mask_path = self.metadata[combined_image]['mask_path']
        
        bg_image = Image.open(bg_path).resize((224, 224), Image.BICUBIC).convert('RGB')
        combined_image = Image.open(combined_image).resize((224, 224), Image.BICUBIC).convert('RGB')
        mask = Image.open(mask_path).resize((224, 224), Image.NEAREST).convert('L')
        
        mask = mask.point(lambda p: 255 if p > 128 else 0)
        
        bg_image = self.to_tensor(bg_image)
        combined_image = self.to_tensor(combined_image)
        mask = self.to_tensor(mask)
        
        return {
            "bg_image": bg_image,
            "combined_image": combined_image,
            "mask": mask,
            "fg_word1": fg_word1,
        }
            
if __name__ == "__main__":
    dataset = LayerDiffuseDataset(root_dir='res', images_dir='images')
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor with shape {value.shape}")
        else:
            print(f"{key}: {value}")
            
        