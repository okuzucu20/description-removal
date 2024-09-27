import argparse
import os
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
#import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from accelerate import Accelerator
from dataset import COCODataset, EvalDataset
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from transformer import TransformerNetwork
from accelerate import DistributedDataParallelKwargs
from datatypes import ValidationBatchProcessedOutput

from utils import (
    get_alpha_clip_embedding,
    initialize_alpha_clip,
    initialize_and_load_projection_block,
    initialize_and_load_ipadapater,
    get_complement_of_mask,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()

def initialize_training_setup(config, save_path):
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Initialize training setup
    transformer = TransformerNetwork(config['embed_dim'], config['num_heads'], config['num_layers']).to(config['device'])

    if save_path is not None:
        transformer.load_state_dict(torch.load(save_path))
    with torch.no_grad():
        sd = StableDiffusionPipeline.from_pretrained(config['diffusion_model'], safety_checker=None).to(config['device'])
        sd_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "botp/stable-diffusion-v1-5-inpainting",
            safety_checker=None,
            use_safetensors=False
        ).to(config['device'])
        alpha_clip, alpha_clip_preprocess, mask_transform = initialize_alpha_clip(config['alpha_clip_id'], config['alpha_vision_ckpt_pth'], config['device'], dtype)
        ip_adapter = initialize_and_load_ipadapater(sd, config['ip_image_encoder_ckpt_pth'], config['ip_adapter_ckpt_pth'], config['device'])
        ip_adapter_inpaint = initialize_and_load_ipadapater(sd_inpaint, config['ip_image_encoder_ckpt_pth'], config['ip_adapter_ckpt_pth'], config['device'])
        projection_block = initialize_and_load_projection_block(config, config['projection_block_ckpt_pth'], config['device'], dtype)
        
    return transformer, alpha_clip, alpha_clip_preprocess, mask_transform, ip_adapter, ip_adapter_inpaint, projection_block

def clipaway_projection_block(bg_embeds, fg_embeds):
    projected_embeds = []
    for i in range(bg_embeds.shape[0]):
        print(f"bg_embeds shape: {bg_embeds[i].shape}, fg_embeds shape: {fg_embeds[i].shape}")
        projected_vector_magnitude = bg_embeds[i].dot(fg_embeds[i]) / fg_embeds[i].norm()
        print(f"projected_vector_magnitude shape: {projected_vector_magnitude.shape}")
        projected_vector = projected_vector_magnitude * fg_embeds[i] / fg_embeds[i].norm()
        print(f"projected_vector shape: {projected_vector.shape}")
        projected_embeds.append(projected_vector)
    projected_embeds = torch.stack(projected_embeds)
    print(f"projected_embeds shape: {projected_embeds.shape}")
    return projected_embeds

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    ckpt = args.ckpt
    save_path = args.save_path
    transformer, alpha_clip, alpha_clip_preprocess, mask_transform, ip_adapter, ip_adapter_inpaint, projection_block = initialize_training_setup(config, save_path)
    transformer.eval()
    alpha_clip.eval()
    projection_block.eval()
    
    test_dataset = EvalDataset(config['test_data'], config['device'], config['dtype']) #TODO: needs fix
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(config['device'])
            masks = batch['mask'].to(config['device'])
            
            alpha_clip_fg_focused_embeddings = get_alpha_clip_embedding(images, masks, alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
            alpha_clip_bg_focused_embeddings = get_alpha_clip_embedding(images, get_complement_of_mask(masks), alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
            transformer_output = transformer(alpha_clip_fg_focused_embeddings, alpha_clip_bg_focused_embeddings)

            fg_only_embeddings = transformer_output.foreground_output
            bg_only_embeddings = transformer_output.background_output

            adjusted_fg_embeddings = (fg_only_embeddings + alpha_clip_fg_focused_embeddings) / 2
            adjusted_bg_embeddings = (bg_only_embeddings + alpha_clip_bg_focused_embeddings) / 2
            
            adjusted_fg_embeddings = projection_block(adjusted_fg_embeddings)
            adjusted_bg_embeddings = projection_block(adjusted_bg_embeddings)
            
            clipaway_embeds = clipaway_projection_block(adjusted_bg_embeddings, adjusted_fg_embeddings)
            
            fg_only_generation = ip_adapter.generate(
                clip_image_embeds=adjusted_fg_embeddings,
                seed=42,
                num_inference_steps=50,
            ).images
            bg_only_generation = ip_adapter.generate(
                clip_image_embeds=adjusted_bg_embeddings,
                seed=42,
                num_inference_steps=50,
            ).images
            removed_description_generation = ip_adapter_inpaint.generate(
                clip_image_embeds=clipaway_embeds,
                seed=42,
                num_inference_steps=50,
            ).images
            
            for j in range(len(fg_only_generation)):
                fg_only_generation[i].save(os.path.join(save_path, f"fg_only_{j * i + i}.png"))
