import argparse
import os
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import clip
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline

from dataset import LayerDiffuseDataset, EvalDataset
from datatypes import ValidationBatchProcessedOutput
from mlp import MLPNetwork
from utils import (
    get_alpha_clip_embedding,
    initialize_alpha_clip,
    initialize_and_load_projection_block,
    initialize_and_load_ipadapater,
    get_complement_of_mask,
    get_alpha_clip_text_embedding,
    save_batch_of_tensor_to_image,
    ReconstructionLoss, 
    SoftMarginForgettingLoss,
    PCGradSingleFG
)

def load_mlp(ckpt, mlp):
    """
    Loads a saved MLP checkpoint, removing 'module.' from the keys if needed.
    """
    state_dict = torch.load(ckpt)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.removeprefix("module.")
        new_state_dict[new_key] = state_dict[key]
    mlp.load_state_dict(new_state_dict)
    return mlp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training_config.yaml")
    return parser.parse_args()

@torch.no_grad()
def clipaway_projection_block(bg_embeds: torch.Tensor, fg_embeds: torch.Tensor) -> torch.Tensor:
    """
    Projects bg_embeds away from the direction of fg_embeds for each element in the batch.
    Args:
      bg_embeds: shape [B, D]
      fg_embeds: shape [B, D]
    Returns:
      A new tensor of shape [B, D], where each bg_embeds[i] has the component
      in the fg_embeds[i] direction subtracted out.
    """
    dot = torch.sum(bg_embeds * fg_embeds, dim=-1)  # [B]
    norm_fg = fg_embeds.norm(dim=-1) + 1e-8         # [B]
    alpha = dot / norm_fg                           # [B]
    fg_direction = fg_embeds / norm_fg.unsqueeze(-1)  # [B, D]
    projected = alpha.unsqueeze(-1) * fg_direction    # [B, D]
    return bg_embeds - projected

@torch.no_grad()
def process_batch_val(batch, config, mlp, alpha_clip, alpha_clip_preprocess,
                      mask_transform, projection_block, ip_adapter,
                      ip_adapter_inpaint, epoch, iteration_count):

    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    images = batch['image'].to(config['device'], dtype=dtype)
    masks = batch['mask'].to(config['device'], dtype=dtype)

    alpha_clip_fg_focused_embeddings = get_alpha_clip_embedding(
        images, masks, alpha_clip, alpha_clip_preprocess, mask_transform,
        config['device'], dtype
    )
    alpha_clip_bg_focused_embeddings = get_alpha_clip_embedding(
        images, get_complement_of_mask(masks), alpha_clip, alpha_clip_preprocess,
        mask_transform, config['device'], dtype
    )

    mlp_output = mlp(torch.cat(
        [alpha_clip_fg_focused_embeddings, alpha_clip_bg_focused_embeddings],
        dim=-1
    ))
    original_clipaway = clipaway_projection_block(alpha_clip_bg_focused_embeddings,
                                                  alpha_clip_fg_focused_embeddings)

    fg_focused = projection_block(alpha_clip_fg_focused_embeddings)
    bg_focused = projection_block(alpha_clip_bg_focused_embeddings)
    mlp_output = projection_block(mlp_output)
    original_clipaway = projection_block(original_clipaway)

    if iteration_count == 0:
        fg_focused_generation = ip_adapter.generate(
            clip_image_embeds=fg_focused,
            seed=42,
            num_inference_steps=30,
        )
        bg_focused_generation = ip_adapter.generate(
            clip_image_embeds=bg_focused,
            seed=42,
            num_inference_steps=30,
        )
        original_clipaway_generation = ip_adapter.generate(
            clip_image_embeds=original_clipaway,
            seed=42,
            num_inference_steps=30,
        )
        fg_focused_inpaint = ip_adapter_inpaint.generate(
            clip_image_embeds=fg_focused,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        bg_focused_inpaint = ip_adapter_inpaint.generate(
            clip_image_embeds=bg_focused,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        original_clipaway_generation_inpaint = ip_adapter_inpaint.generate(
            clip_image_embeds=original_clipaway,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
    else:
        fg_focused_generation = []
        bg_focused_generation = []
        original_clipaway_generation = []
        fg_focused_inpaint = []
        bg_focused_inpaint = []
        original_clipaway_generation_inpaint = []

    mlp_output_generation = ip_adapter.generate(
        clip_image_embeds=mlp_output,
        seed=42,
        num_inference_steps=30,
    )
    mlp_output_generation_inpaint = ip_adapter_inpaint.generate(
        clip_image_embeds=mlp_output,
        seed=42,
        num_inference_steps=30,
        image=images,
        mask_image=masks
    )
        
    return {
        "images": images,
        "masks": masks,
        "fg_focused_generation": fg_focused_generation,
        "bg_focused_generation": bg_focused_generation,
        "fg_focused_inpaint": fg_focused_inpaint,
        "bg_focused_inpaint": bg_focused_inpaint,
        "original_clipaway_generation": original_clipaway_generation,
        "original_clipaway_generation_inpaint": original_clipaway_generation_inpaint,
        "mlp_output_generation": mlp_output_generation,
        "mlp_output_generation_inpaint": mlp_output_generation_inpaint,
    }

@torch.no_grad()
def process_val_dataloader(val_dataloader, config, mlp, alpha_clip, alpha_clip_preprocess,
                           mask_transform, projection_block, ip_adapter, ip_adapter_inpaint,
                           epoch, iteration_count):

    images_ = []
    masks_ = []
    fg_focused = []
    bg_focused = []
    fg_focused_inpaint = []
    bg_focused_inpaint = []
    original_clipaway = []
    original_clipaway_inpaint = []
    mlp_output = []
    mlp_output_inpaint = []

    for batch in val_dataloader:
        processed_batch = process_batch_val(
            batch, config, mlp, alpha_clip, alpha_clip_preprocess,
            mask_transform, projection_block, ip_adapter, ip_adapter_inpaint,
            epoch, iteration_count
        )

        if iteration_count == 0:
            images_.extend(processed_batch["images"])
            masks_.extend(processed_batch["masks"])
            fg_focused.extend(processed_batch["fg_focused_generation"])
            bg_focused.extend(processed_batch["bg_focused_generation"])
            fg_focused_inpaint.extend(processed_batch["fg_focused_inpaint"])
            bg_focused_inpaint.extend(processed_batch["bg_focused_inpaint"])
            original_clipaway.extend(processed_batch["original_clipaway_generation"])
            original_clipaway_inpaint.extend(processed_batch["original_clipaway_generation_inpaint"])

        mlp_output.extend(processed_batch["mlp_output_generation"])
        mlp_output_inpaint.extend(processed_batch["mlp_output_generation_inpaint"])
        
    return (images_, masks_, fg_focused, bg_focused, fg_focused_inpaint, bg_focused_inpaint,
            original_clipaway, original_clipaway_inpaint, mlp_output, mlp_output_inpaint)

@torch.no_grad()
def validate(config, mlp, val_dataloader, ip_adapter, ip_adapter_inpaint,
             alpha_clip, alpha_clip_preprocess, mask_transform, projection_block,
             epoch, logger, iteration_count):

    mlp.eval()
    (images_, masks_, fg_focused, bg_focused, fg_focused_inpaint, bg_focused_inpaint,
     original_clipaway, original_clipaway_inpaint, mlp_output, mlp_output_inpaint) = process_val_dataloader(
        val_dataloader, config, mlp, alpha_clip, alpha_clip_preprocess, mask_transform,
        projection_block, ip_adapter, ip_adapter_inpaint, epoch, iteration_count
    )

    if iteration_count == 0:
        images_grid = make_grid(images_)
        masks_grid = make_grid(masks_)
        fg_focused_grid = make_grid([transforms.ToTensor()(image) for image in fg_focused])
        bg_focused_grid = make_grid([transforms.ToTensor()(image) for image in bg_focused])
        fg_focused_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in fg_focused_inpaint])
        bg_focused_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in bg_focused_inpaint])
        original_clipaway_grid = make_grid([transforms.ToTensor()(image) for image in original_clipaway])
        original_clipaway_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in original_clipaway_inpaint])

        logger.add_image("Images", images_grid, iteration_count)
        logger.add_image("Masks", masks_grid, iteration_count)
        logger.add_image("fg_focused", fg_focused_grid, iteration_count)
        logger.add_image("bg_focused", bg_focused_grid, iteration_count)
        logger.add_image("fg_focused_inpaint", fg_focused_inpaint_grid, iteration_count)
        logger.add_image("bg_focused_inpaint", bg_focused_inpaint_grid, iteration_count)
        logger.add_image("original_clipaway", original_clipaway_grid, iteration_count)
        logger.add_image("original_clipaway_inpaint", original_clipaway_inpaint_grid, iteration_count)

    if mlp_output:
        mlp_output_grid = make_grid([transforms.ToTensor()(image) for image in mlp_output])
        logger.add_image("mlp_output", mlp_output_grid, iteration_count)
    if mlp_output_inpaint:
        mlp_output_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in mlp_output_inpaint])
        logger.add_image("mlp_output_inpaint", mlp_output_inpaint_grid, iteration_count)
        
    mlp.train()

def training_step(
    config, 
    mlp, 
    optimizer, 
    batch, 
    pcgrad, 
    alpha_clip, 
    alpha_clip_preprocess, 
    mask_transform, 
    clip_model, 
    clip_transform
):
    """
    Single-FG training step with partial PCGrad.
    """
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    mlp.train()

    # 1) Data
    bg_image       = batch["bg_image"].to(config["device"], dtype=dtype)
    combined_image = batch["combined_image"].to(config["device"], dtype=dtype)
    mask           = batch["mask"].to(config["device"], dtype=dtype)
    fg_word1       = batch["fg_word1"]  # single FG concept

    # 2) Preprocess BG image for CLIP
    preprocessed_bg_image = clip_transform(bg_image)

    # 3) No grad for alpha-CLIP + standard CLIP
    with torch.no_grad():
        alpha_clip_fg = get_alpha_clip_embedding(
            combined_image, mask, alpha_clip, alpha_clip_preprocess, 
            mask_transform, config["device"], dtype
        )
        alpha_clip_bg = get_alpha_clip_embedding(
            combined_image, get_complement_of_mask(mask), alpha_clip, 
            alpha_clip_preprocess, mask_transform, config["device"], dtype
        )

        fg_embed = get_alpha_clip_text_embedding(
            fg_word1, alpha_clip, config["device"], dtype
        )
        bg_only_image_embeds = clip_model.encode_image(preprocessed_bg_image)

    # 4) MLP forward
    #   concat FG+BG embeddings => MLP => [B, 768]
    mlp_input = torch.cat([alpha_clip_fg, alpha_clip_bg], dim=-1)
    mlp_output = mlp(mlp_input)  # shape [B, 768]

    # 5) PCGradSingleFG => sets model grad
    L_fg, L_bg = pcgrad(
        preds=mlp_output,
        fg=fg_embed,
        bg=bg_only_image_embeds
    )

    # 6) Step
    optimizer.step()
    optimizer.zero_grad()

    return L_fg, L_bg


def initialize_training_setup(config):
    """
    Creates the MLP, pipelines, alpha-CLIP, and loads everything on device.
    """
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    mlp = MLPNetwork(
        input_size=2 * 768,
        hidden_size=768,
        output_size=768,
        number_of_hidden_layers=3,
        do_zero_initialize=config['do_zero_initialize']
    ).to(config['device'])
        
    sd = StableDiffusionPipeline.from_pretrained(config['diffusion_model'], safety_checker=None).to(config['device'])
    sd_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "botp/stable-diffusion-v1-5-inpainting", safety_checker=None, use_safetensors=False
    ).to(config['device'])

    alpha_clip, alpha_clip_preprocess, mask_transform = initialize_alpha_clip(
        config['alpha_clip_id'], config['alpha_vision_ckpt_pth'], config['device'], dtype
    )
    ip_adapter = initialize_and_load_ipadapater(
        sd, config['ip_image_encoder_ckpt_pth'], config['ip_adapter_ckpt_pth'], config['device']
    )
    ip_adapter_inpaint = initialize_and_load_ipadapater(
        sd_inpaint, config['ip_image_encoder_ckpt_pth'], config['ip_adapter_ckpt_pth'], config['device']
    )
    projection_block = initialize_and_load_projection_block(
        config, config['projection_block_ckpt_pth'], config['device'], dtype
    )
    
    torch.cuda.empty_cache()
    clip_model, _ = clip.load("ViT-L/14", device=config['device'])
    clip_model = clip_model.to(config['device'], dtype=dtype)
    
    clip_model.eval()
    projection_block.eval()
    alpha_clip.eval()

    # Freeze everything except MLP
    projection_block.requires_grad_(False)
    sd.vae.requires_grad_(False)
    sd.unet.requires_grad_(False)
    sd.text_encoder.requires_grad_(False)
    sd_inpaint.vae.requires_grad_(False)
    sd_inpaint.unet.requires_grad_(False)
    sd_inpaint.text_encoder.requires_grad_(False)
    clip_model.requires_grad_(False)

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return mlp, optimizer, accelerator, alpha_clip, alpha_clip_preprocess, mask_transform, \
           ip_adapter, ip_adapter_inpaint, projection_block, clip_model 


def train(config: OmegaConf, logger: SummaryWriter):
    clip_transform = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )

    # 1) Initialize
    (mlp, optimizer, accelerator, alpha_clip, alpha_clip_preprocess, mask_transform,
     ip_adapter, ip_adapter_inpaint, projection_block, clip_model) = initialize_training_setup(config)

    # 2) Datasets
    training_dataset = LayerDiffuseDataset(config["root_dir"], config["images_dir"])
    val_dataset      = EvalDataset(config["val_image_dir"], config["val_mask_dir"])
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=config["train_batch_size"], shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["val_batch_size"], shuffle=False
    )

    # 3) Losses
    reconstruction_loss = ReconstructionLoss(clamp_value=None)   # or clamp
    forgetting_loss     = SoftMarginForgettingLoss(margin=config["margin_forgetting"])
    # single-FG PCGrad
    pcgrad = PCGradSingleFG(
        mlp,
        loss_f=forgetting_loss,
        loss_r=reconstruction_loss,
        f_coeff=config["forget_lambda"],
        r_coeff=config['recons_lambda'],
        projection_alpha=config['projection_alpha'],
    )

    # 4) Accelerator
    mlp, optimizer, training_dataloader, val_dataloader = accelerator.prepare(
        mlp, optimizer, training_dataloader, val_dataloader
    )

    iteration_count = 0

    # 5) Start Training
    for epoch in tqdm(range(config["num_epochs"]), desc="Epochs"):
        sum_fg_loss  = 0.0
        sum_bg_loss  = 0.0
        step_count   = 0

        for batch in training_dataloader:
            with accelerator.accumulate(mlp):
                L_fg, L_bg = training_step(
                    config, mlp, optimizer, batch, pcgrad,
                    alpha_clip, alpha_clip_preprocess, mask_transform, 
                    clip_model, clip_transform
                )

                L_fg_val = accelerator.gather(L_fg).mean().item()
                L_bg_val = accelerator.gather(L_bg).mean().item()

                if accelerator.is_main_process:
                    step_count += 1
                    sum_fg_loss += L_fg_val
                    sum_bg_loss += L_bg_val

                    total_loss = L_fg_val + L_bg_val
                    print(f"[Iter {iteration_count}] FG={L_fg_val:.4f}, BG={L_bg_val:.4f}, total={total_loss:.4f}")
                    logger.add_scalar("fg_loss",  L_fg_val, iteration_count)
                    logger.add_scalar("bg_loss",  L_bg_val, iteration_count)
                    logger.add_scalar("total_loss", total_loss, iteration_count)

            # Evaluate periodically
            if iteration_count % config["eval_interval"] == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    validate(
                        config, mlp, val_dataloader,
                        ip_adapter, ip_adapter_inpaint,
                        alpha_clip, alpha_clip_preprocess,
                        mask_transform, projection_block,
                        epoch, logger, iteration_count
                    )
                    model_to_save = accelerator.unwrap_model(mlp)
                    torch.save(model_to_save.state_dict(), f"{config['output_dir']}/forget_lambda={post_fix}_margin_forgetting={config['margin_forgetting']}/mlp.pth")

            iteration_count += 1

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and step_count > 0:
            logger.add_scalar("epoch_fg_loss", sum_fg_loss / step_count, epoch)
            logger.add_scalar("epoch_bg_loss", sum_bg_loss / step_count, epoch)

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    post_fix = config['forget_lambda']
    logger = SummaryWriter(config['output_dir'] + f"/forget_lambda={post_fix}_margin_forgetting={config['margin_forgetting']}_recons_lambda={config['recons_lambda']}_projection_alpha={config['projection_alpha']}")
    train(config, logger)
