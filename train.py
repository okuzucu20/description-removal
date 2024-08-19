import argparse
import os
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from torchvision import transforms
from accelerate import Accelerator
from dataset import COCODataset, EvalDataset
from diffusers import StableDiffusionPipeline
from transformer import TransformerNetwork

from utils import (
    get_alpha_clip_embedding,
    get_unclip_text_to_image_embedding_transformer,
    initialize_alpha_clip,
    initialize_and_load_projection_block,
    initialize_and_load_ipadapater,
    get_complement_of_mask,
)

# trainer.py
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training_config.yaml")
    return parser.parse_args()

def process_batch_val(batch, config, transformer, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, ip_adapter):
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Process a batch of data
    images = batch['image'].to(config['device'], dtype=dtype)
    masks = batch['mask'].to(config['device'], dtype=dtype)

    print(f"Images shape: {images.shape}, Masks shape: {masks.shape} eval")
    with torch.no_grad():
        alpha_clip_fg_focused_embeddings = get_alpha_clip_embedding(images, masks, alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
        alpha_clip_bg_focused_embeddings = get_alpha_clip_embedding(images, get_complement_of_mask(masks), alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
        transformer_output = transformer(alpha_clip_fg_focused_embeddings, alpha_clip_bg_focused_embeddings)    

        alpha_clip_fg_focused_embeddings = projection_block(alpha_clip_fg_focused_embeddings)
        alpha_clip_bg_focused_embeddings = projection_block(alpha_clip_bg_focused_embeddings)
        fg_only_embeddings = transformer_output.foreground_output
        bg_only_embeddings = transformer_output.background_output
        
        fg_only_embeddings = projection_block(fg_only_embeddings)
        bg_only_embeddings = projection_block(bg_only_embeddings)

        fg_focused_generation = ip_adapter.generate(
            clip_image_embeds=alpha_clip_fg_focused_embeddings,
            seed=42,
            num_inference_steps=50,
        )
        bg_focused_generation = ip_adapter.generate(
            clip_image_embeds=alpha_clip_bg_focused_embeddings,
            seed=42,
            num_inference_steps=50,
        )

        fg_only_generation = ip_adapter.generate(
            clip_image_embeds=fg_only_embeddings,
            seed=42,
            num_inference_steps=50,
        )
        bg_only_generation = ip_adapter.generate(
            clip_image_embeds=bg_only_embeddings,
            seed=42,
            num_inference_steps=50,
        )

    return images, masks, fg_focused_generation, bg_focused_generation, fg_only_generation, bg_only_generation


def process_val_dataloader(val_dataloader, config, transformer, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, ip_adapter):
    # Process the validation dataloader
    images_ = []
    masks_ = []
    fg_focused = []
    bg_focused = []
    fg_only = []
    bg_only = []

    for batch in val_dataloader:
        images, masks, fg_focused_generation, bg_focused_generation, fg_only_generation, bg_only_generation = process_batch_val(batch, config, transformer, alpha_clip,
                                                                                                                                alpha_clip_preprocess, mask_transform, projection_block, ip_adapter)
        images_.extend(images)
        masks_.extend(masks)
        fg_focused.extend(fg_focused_generation)
        bg_focused.extend(bg_focused_generation)
        fg_only.extend(fg_only_generation)
        bg_only.extend(bg_only_generation)

    return images_, masks_, fg_focused, bg_focused, fg_only, bg_only


def validate(config, transformer, val_dataloader, ip_adapter, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, epoch):
    # Perform validation
    transformer.eval()
    images_, masks_, fg_focused, bg_focused, fg_only, bg_only = process_val_dataloader(val_dataloader, config, transformer, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, ip_adapter)
            
    if epoch == 1:
        wandb.log({"Images": [wandb.Image(image) for image in images_]})
        wandb.log({"Masks": [wandb.Image(mask) for mask in masks_]})
    
    wandb.log({"Foreground Focused": [wandb.Image(fg) for fg in fg_focused]})
    wandb.log({"Background Focused": [wandb.Image(bg) for bg in bg_focused]})
    wandb.log({"Foreground Only": [wandb.Image(fg) for fg in fg_only]})
    wandb.log({"Background Only": [wandb.Image(bg) for bg in bg_only]})

def training_step(config, transformer, optimizer, batch, loss, alpha_clip, alpha_clip_preprocess, mask_transform, unclip_transformer):
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    transformer.train()
    # Perform a single training step
    fg_desc = batch['fg_description']
    bg_desc = batch['bg_description']
    images = batch['image'].to(config['device'])
    masks = batch['mask'].to(config['device'])

    with torch.no_grad():
        alpha_clip_fg_focused_embeddings = get_alpha_clip_embedding(images, masks, alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
        alpha_clip_bg_focused_embeddings = get_alpha_clip_embedding(images, get_complement_of_mask(masks), alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
        fg_text_unclip = unclip_transformer.text_to_image_embedding(fg_desc) 
        bg_text_unclip = unclip_transformer.text_to_image_embedding(bg_desc) 

        # https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        positive_target_label = torch.ones(fg_text_unclip.shape[0]).to(config['device']) * -1
        negative_target_label = torch.ones(fg_text_unclip.shape[0]).to(config['device'])

    optimizer.zero_grad()
    transformer_output = transformer(alpha_clip_fg_focused_embeddings, alpha_clip_bg_focused_embeddings)
    foreground_pred = transformer_output.foreground_output
    background_pred = transformer_output.background_output

    attract_loss_fg = loss(foreground_pred, fg_text_unclip, positive_target_label)
    attract_loss_bg = loss(background_pred, bg_text_unclip, positive_target_label)
    repell_loss_fg = loss(foreground_pred, bg_text_unclip, negative_target_label)
    repell_loss_bg = loss(background_pred, fg_text_unclip, negative_target_label)
    
    return attract_loss_fg, attract_loss_bg, repell_loss_fg, repell_loss_bg

def initialize_training_setup(config):
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Initialize training setup
    transformer = TransformerNetwork(config['embed_dim'], config['num_heads'], config['num_layers']).to(config['device'])    
    
    if config["transformer_ckpt_pth"] is not None:
        transformer.load_state_dict(torch.load(config["transformer_ckpt_pth"]))
    with torch.inference_mode():
        sd = StableDiffusionPipeline.from_pretrained(config['diffusion_model'], safety_checker=None).to(config['device'])
        alpha_clip, alpha_clip_preprocess, mask_transform = initialize_alpha_clip(config['alpha_clip_id'], config['alpha_vision_ckpt_pth'], config['device'], dtype) 
        unclip_transformer = get_unclip_text_to_image_embedding_transformer(dtype, config['device'])
        ip_adapter = initialize_and_load_ipadapater(sd, config['ip_image_encoder_ckpt_pth'], config['ip_adapter_ckpt_pth'], config['device'])
        projection_block = initialize_and_load_projection_block(config, config['projection_block_ckpt_pth'], config['device'], dtype)
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return transformer, optimizer, accelerator, alpha_clip, alpha_clip_preprocess, mask_transform, unclip_transformer, ip_adapter, projection_block

def train(config):
    transformer, optimizer, accelerator, alpha_clip, alpha_clip_preprocess, mask_transform, unclip_transformer, ip_adapter, projection_block = initialize_training_setup(config)
    training_dataset = COCODataset(config['train_image_dir'], config['train_mask_dir'], config['train_metadata_file'])
    val_dataset = EvalDataset(config['val_image_dir'], config['val_mask_dir'])
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False)
    loss = torch.nn.CosineEmbeddingLoss()
    
    transformer, optimizer, training_dataloader, val_dataloader = accelerator.prepare(transformer, optimizer, training_dataloader, val_dataloader)
    for epoch in tqdm(range(config['num_epochs']), desc="Epochs"):
        attract_loss_fg_epoch = 0
        attract_loss_bg_epoch = 0
        repell_loss_fg_epoch = 0
        repell_loss_bg_epoch = 0
        total_loss_epoch = 0
        
        for batch in training_dataloader:
            attract_loss_fg, attract_loss_bg, repell_loss_fg, repell_loss_bg = training_step(config, transformer, optimizer, batch, loss, 
                                                                                            alpha_clip, alpha_clip_preprocess, mask_transform, unclip_transformer)
            total_loss = attract_loss_fg + attract_loss_bg + repell_loss_fg + repell_loss_bg

            attract_loss_bg_epoch += attract_loss_bg
            attract_loss_fg_epoch += attract_loss_fg
            repell_loss_bg_epoch += repell_loss_bg
            repell_loss_fg_epoch += repell_loss_fg
            total_loss_epoch += total_loss
            
            accelerator.backward(total_loss)
            optimizer.step()
            
            wandb.log({"attract_loss_fg": attract_loss_fg, "attract_loss_bg": attract_loss_bg, "repell_loss_fg": repell_loss_fg, "repell_loss_bg": repell_loss_bg, "total_loss": total_loss})
            
        wandb.log({"attract_loss_fg_epoch": attract_loss_fg_epoch, "attract_loss_bg_epoch": attract_loss_bg_epoch, "repell_loss_fg_epoch": repell_loss_fg_epoch,
                "repell_loss_bg_epoch": repell_loss_bg_epoch, "total_loss_epoch": total_loss_epoch})
        
        validate(config, transformer, val_dataloader, ip_adapter, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, epoch)
        
        if accelerator.is_main_process:
            torch.save(transformer.state_dict(), f"{config['output_dir']}/transformer.pth")

            
if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    os.makedirs(config['output_dir'], exist_ok=True)
    wandb_run_name = config['wandb_run_name']
    wandb.init(
        project=wandb_run_name,
    )
    wandb.run.name = wandb_run_name
    train(config)
    wandb.finish()
