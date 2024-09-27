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
from subspace_projector import SubspaceProjectorNetwork

from utils import (
    get_alpha_clip_embedding,
    initialize_alpha_clip,
    initialize_and_load_projection_block,
    initialize_and_load_ipadapater,
    get_complement_of_mask,
)

def load_transformer(ckpt, transformer):
    # generate a dict that prepends module. to the keys of the state dict
    # this is necessary because the model was saved using DataParallel
    
    state_dict = torch.load(ckpt)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.removeprefix("module.")
        new_state_dict[new_key] = state_dict[key]
    transformer.load_state_dict(new_state_dict)
    return transformer

# trainer.py
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training_config.yaml")
    return parser.parse_args()

def clipaway_projection_block(bg_embeds, fg_embeds):
    # Compute dot products for each corresponding pair of vectors in the batch
    dot_products = torch.sum(bg_embeds * fg_embeds, dim=-1, keepdim=True)
    
    # Compute the norms of the fg_embeds vectors for each vector in the batch
    fg_norms = torch.norm(fg_embeds, dim=-1, keepdim=True)
    
    # Compute the projected vector magnitudes
    projected_vector_magnitudes = dot_products / fg_norms
    
    # Compute the full projected vectors
    projected_vectors = projected_vector_magnitudes * fg_embeds / fg_norms
    
    # Subtract the projected vectors from bg_embeds
    result = bg_embeds - projected_vectors
    
    return result
    
def process_batch_val(batch, config, transformer, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, ip_adapter, ip_adapter_inpaint, epoch):
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
        transformer_output_plus_clipaway = clipaway_projection_block(transformer_output.background_output, transformer_output.foreground_output)

        transformer_output_plus_clipaway = projection_block(transformer_output_plus_clipaway)
        alpha_clip_fg_focused_embeddings = projection_block(alpha_clip_fg_focused_embeddings)
        alpha_clip_bg_focused_embeddings = projection_block(alpha_clip_bg_focused_embeddings)
        default_clipaway = clipaway_projection_block(alpha_clip_bg_focused_embeddings, alpha_clip_fg_focused_embeddings)
        fg_only_embeddings = transformer_output.foreground_output
        bg_only_embeddings = transformer_output.background_output

        fg_only_embeddings = projection_block(fg_only_embeddings)
        bg_only_embeddings = projection_block(bg_only_embeddings)

        alpha_clip_transformer_mean_embeddings_fg = (alpha_clip_fg_focused_embeddings + fg_only_embeddings) / 2
        alpha_clip_transformer_mean_embeddings_bg = (alpha_clip_bg_focused_embeddings + bg_only_embeddings) / 2
        alpha_clip_transformer_mean_embeddings = clipaway_projection_block(alpha_clip_transformer_mean_embeddings_bg, alpha_clip_transformer_mean_embeddings_fg) 

        if epoch == 0:
            fg_focused_generation = ip_adapter.generate(
                clip_image_embeds=alpha_clip_fg_focused_embeddings,
                seed=42,
                num_inference_steps=30,
            )
            bg_focused_generation = ip_adapter.generate(
                    clip_image_embeds=alpha_clip_bg_focused_embeddings,
                    seed=42,
                    num_inference_steps=30,
            )
            default_clipaway_generation = ip_adapter.generate(
                clip_image_embeds=default_clipaway,
                seed=42,
                num_inference_steps=30,
            )
            default_clipaway_generation_inpaint = ip_adapter_inpaint.generate(
                clip_image_embeds=default_clipaway,
                seed=42,
                num_inference_steps=30,
                image=images,
                mask_image=masks
            )
            fg_focused_generation_inpaint = ip_adapter_inpaint.generate(
                clip_image_embeds=alpha_clip_fg_focused_embeddings,
                seed=42,
                num_inference_steps=30,
                image=images,
                mask_image=masks
            )
            bg_focused_generation_inpaint = ip_adapter_inpaint.generate(
                clip_image_embeds=alpha_clip_bg_focused_embeddings,
                seed=42,
                 num_inference_steps=30,
                 image=images,
                 mask_image=masks
            )
        else:
             fg_focused_generation = None
             bg_focused_generation = None           
             default_clipaway_generation = None
             default_clipaway_generation_inpaint = None
             fg_focused_generation_inpaint = None
             bg_focused_generation_inpaint = None

        fg_only_generation = ip_adapter.generate(
            clip_image_embeds=fg_only_embeddings,
            seed=42,
            num_inference_steps=30,
        )
        bg_only_generation = ip_adapter.generate(
            clip_image_embeds=bg_only_embeddings,
            seed=42,
            num_inference_steps=30,
        )
        transformer_output_plus_clipaway_generation = ip_adapter.generate(
            clip_image_embeds=transformer_output_plus_clipaway,
            seed=42,
            num_inference_steps=30,
        )
        fg_only_generation_inpaint = ip_adapter_inpaint.generate(
            clip_image_embeds=fg_only_embeddings,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        bg_only_generation_inpaint = ip_adapter_inpaint.generate(
            clip_image_embeds=bg_only_embeddings,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        transformer_output_plus_clipaway_generation_inpaint = ip_adapter_inpaint.generate(
            clip_image_embeds=transformer_output_plus_clipaway,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        alpha_clip_transformer_mean = ip_adapter.generate(
            clip_image_embeds=alpha_clip_transformer_mean_embeddings ,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        alpha_clip_transformer_mean_fg = ip_adapter.generate(
            clip_image_embeds=alpha_clip_transformer_mean_embeddings_fg,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        alpha_clip_transformer_mean_bg = ip_adapter.generate(
            clip_image_embeds=alpha_clip_transformer_mean_embeddings_bg,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        alpha_clip_transformer_mean_inpaint = ip_adapter_inpaint.generate(
            clip_image_embeds=alpha_clip_transformer_mean_embeddings ,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        alpha_clip_transformer_mean_inpaint_fg = ip_adapter_inpaint.generate(
            clip_image_embeds=alpha_clip_transformer_mean_embeddings_fg,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )
        alpha_clip_transformer_mean_inpaint_bg = ip_adapter_inpaint.generate(
            clip_image_embeds=alpha_clip_transformer_mean_embeddings_bg,
            seed=42,
            num_inference_steps=30,
            image=images,
            mask_image=masks
        )

    return ValidationBatchProcessedOutput(
        images, 
        masks, 
        fg_focused_generation,
        bg_focused_generation,
        fg_only_generation,
        bg_only_generation,
        transformer_output_plus_clipaway_generation,
        default_clipaway_generation,
        fg_focused_generation_inpaint,
        bg_focused_generation_inpaint,
        fg_only_generation_inpaint,
        bg_only_generation_inpaint,
        transformer_output_plus_clipaway_generation_inpaint,
        default_clipaway_generation_inpaint,
        alpha_clip_transformer_mean, 
        alpha_clip_transformer_mean_inpaint, 
        alpha_clip_transformer_mean_fg, 
        alpha_clip_transformer_mean_inpaint_fg, 
        alpha_clip_transformer_mean_bg, 
        alpha_clip_transformer_mean_inpaint_bg, 
    )


def process_val_dataloader(val_dataloader, config, transformer, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, ip_adapter, ip_adapter_inpaint, epoch):
    # Process the validation dataloader
    images_ = []
    masks_ = []
    fg_focused = []
    bg_focused = []
    fg_only = []
    bg_only = []
    alpha_clip_transformer_mean = []
    alpha_clip_transformer_mean_inpaint = []
    alpha_clip_transformer_mean_fg = []
    alpha_clip_transformer_mean_inpaint_fg = []
    alpha_clip_transformer_mean_bg = []
    alpha_clip_transformer_mean_inpaint_bg = []
    transformer_output_plus_clipaway = []
    clipaway_default = []
    fg_focused_inpaint = []
    bg_focused_inpaint = []
    fg_only_inpaint = []
    bg_only_inpaint = []
    transformer_output_plus_clipaway_inpaint = []
    default_clipaway_inpaint = []


    for batch in val_dataloader:
        validation_batch_processed = process_batch_val(batch, config, transformer, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, ip_adapter, ip_adapter_inpaint, epoch)

        if epoch == 0:
            images_.extend(validation_batch_processed.images)
            masks_.extend(validation_batch_processed.masks)
            fg_focused.extend(validation_batch_processed.fg_focused_generation)
            bg_focused.extend(validation_batch_processed.bg_focused_generation)
            clipaway_default.extend(validation_batch_processed.default_clipaway_generation)
            fg_focused_inpaint.extend(validation_batch_processed.fg_focused_generation_inpaint)
            bg_focused_inpaint.extend(validation_batch_processed.bg_focused_generation_inpaint)
            default_clipaway_inpaint.extend(validation_batch_processed.default_clipaway_generation_inpaint)

        fg_only.extend(validation_batch_processed.fg_only_generation)
        bg_only.extend(validation_batch_processed.bg_only_generation)
        transformer_output_plus_clipaway.extend(validation_batch_processed.transformer_output_plus_clipaway_generation)
        fg_only_inpaint.extend(validation_batch_processed.fg_only_generation_inpaint)
        bg_only_inpaint.extend(validation_batch_processed.bg_only_generation_inpaint)
        transformer_output_plus_clipaway_inpaint.extend(validation_batch_processed.transformer_output_plus_clipaway_generation_inpaint)
        alpha_clip_transformer_mean.extend(validation_batch_processed.alpha_clip_transformer_mean)
        alpha_clip_transformer_mean_inpaint.extend(validation_batch_processed.alpha_clip_transformer_mean_inpaint)
        alpha_clip_transformer_mean_fg.extend(validation_batch_processed.alpha_clip_transformer_mean_fg)
        alpha_clip_transformer_mean_inpaint_fg.extend(validation_batch_processed.alpha_clip_transformer_mean_inpaint_fg)
        alpha_clip_transformer_mean_bg.extend(validation_batch_processed.alpha_clip_transformer_mean_bg)
        alpha_clip_transformer_mean_inpaint_bg.extend(validation_batch_processed.alpha_clip_transformer_mean_inpaint_bg)

    return images_, masks_, fg_focused, bg_focused, fg_only, bg_only, transformer_output_plus_clipaway, clipaway_default, fg_focused_inpaint, bg_focused_inpaint, fg_only_inpaint, bg_only_inpaint, transformer_output_plus_clipaway_inpaint, default_clipaway_inpaint, alpha_clip_transformer_mean, alpha_clip_transformer_mean_inpaint , alpha_clip_transformer_mean_fg, alpha_clip_transformer_mean_inpaint_fg, alpha_clip_transformer_mean_bg, alpha_clip_transformer_mean_inpaint_bg 


def validate(config, transformer, val_dataloader, ip_adapter, ip_adapter_inpaint, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, epoch, logger):
    # Perform validation
    transformer.eval()
    images_, masks_, fg_focused, bg_focused, fg_only, bg_only, transformer_output_plus_clipaway, clipaway_default, fg_focused_inpaint, bg_focused_inpaint, fg_only_inpaint, bg_only_inpaint, transformer_output_plus_clipaway_inpaint, default_clipaway_inpaint, alpha_clip_transformer_mean, alpha_clip_transformer_mean_inpaint, alpha_clip_transformer_mean_fg, alpha_clip_transformer_mean_inpaint_fg, alpha_clip_transformer_mean_bg, alpha_clip_transformer_mean_inpaint_bg = process_val_dataloader(val_dataloader, config, transformer, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, ip_adapter, ip_adapter_inpaint, epoch)

    if epoch == 0:
        images_grid = make_grid(images_)
        masks_grid = make_grid(masks_)
        fg_focused_grid = make_grid([transforms.ToTensor()(image) for image in fg_focused])
        bg_focused_grid = make_grid([transforms.ToTensor()(image) for image in bg_focused])
        clipaway_default_grid = make_grid([transforms.ToTensor()(image) for image in clipaway_default])
        fg_focused_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in fg_focused_inpaint])
        bg_focused_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in bg_focused_inpaint])
        clipaway_default_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in default_clipaway_inpaint])

        logger.add_image("Images", images_grid, epoch)
        logger.add_image("Masks", masks_grid, epoch)
        logger.add_image("fg_focused", fg_focused_grid, epoch)
        logger.add_image("bg_focused", bg_focused_grid, epoch)
        logger.add_image("clipaway_default", clipaway_default_grid, epoch)
        logger.add_image("fg_focused_inpaint", fg_focused_inpaint_grid, epoch)
        logger.add_image("bg_focused_inpaint", bg_focused_inpaint_grid, epoch)
        logger.add_image("clipaway_default_inpaint", clipaway_default_inpaint_grid, epoch)

    fg_only_grid = make_grid([transforms.ToTensor()(image) for image in fg_only])
    bg_only_grid = make_grid([transforms.ToTensor()(image) for image in bg_only])
    transformer_output_plus_clipaway_grid = make_grid([transforms.ToTensor()(image) for image in transformer_output_plus_clipaway])
    fg_only_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in fg_only_inpaint])
    bg_only_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in bg_only_inpaint])
    transformer_output_plus_clipaway_inpaint_grid = make_grid([transforms.ToTensor()(image) for image in transformer_output_plus_clipaway_inpaint])
    alpha_clip_transformer_mean_grid  = make_grid([transforms.ToTensor()(image) for image in alpha_clip_transformer_mean ])
    alpha_clip_transformer_mean_inpaint_grid  = make_grid([transforms.ToTensor()(image) for image in alpha_clip_transformer_mean_inpaint ])
    alpha_clip_transformer_mean_grid_fg  = make_grid([transforms.ToTensor()(image) for image in alpha_clip_transformer_mean_fg ])
    alpha_clip_transformer_mean_inpaint_grid_fg  = make_grid([transforms.ToTensor()(image) for image in alpha_clip_transformer_mean_inpaint_fg ])
    alpha_clip_transformer_mean_grid_bg  = make_grid([transforms.ToTensor()(image) for image in alpha_clip_transformer_mean_bg ])
    alpha_clip_transformer_mean_inpaint_grid_bg  = make_grid([transforms.ToTensor()(image) for image in alpha_clip_transformer_mean_inpaint_bg ])

    logger.add_image("tranformer output of fg only", fg_only_grid, epoch)
    logger.add_image("transformer output of bg only", bg_only_grid, epoch)
    logger.add_image("transformer output fg ve bg onlyde clipaway", transformer_output_plus_clipaway_grid, epoch)
    logger.add_image("transformer output fg only inpaint", fg_only_inpaint_grid, epoch)
    logger.add_image("transformer output bg only inpaint", bg_only_inpaint_grid, epoch)
    logger.add_image("transformer output fg ve bg onlyde clipaway inpaint", transformer_output_plus_clipaway_inpaint_grid, epoch)
    logger.add_image("alpha_clip ve tranformer fg ve bgde mean alıp clipaway", alpha_clip_transformer_mean_grid, epoch)
    logger.add_image("alpha_clip ve tranformer fg ve bgde mean alıp clipaway inpaint", alpha_clip_transformer_mean_inpaint_grid, epoch)
    logger.add_image("alpha_clip ve transformer fgde mean", alpha_clip_transformer_mean_grid_fg, epoch)
    logger.add_image("alpha_clip ve transforme fgde  mean inpaint", alpha_clip_transformer_mean_inpaint_grid_fg, epoch)
    logger.add_image("alpha_clip ve transformer bgde mean", alpha_clip_transformer_mean_grid_bg, epoch)
    logger.add_image("alpha_clip ve transformer bgde mean inpaint_bg", alpha_clip_transformer_mean_inpaint_grid_bg, epoch)

def training_step(config, transformer, optimizer, batch, loss, alpha_clip,
        alpha_clip_preprocess, mask_transform, use_ortho_loss,  subspace_proj, use_new_subspace_projection):
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    transformer.train()
    # Perform a single training step
    images = batch['image'].to(config['device'])
    masks = batch['mask'].to(config['device'])
    fg_embed = batch['fg_embed'].to(config['device'])
    bg_embed = batch['bg_embed'].to(config['device'])
    print(f"fg_embed shape: {fg_embed.shape}, bg_embed shape: {bg_embed.shape}")

    with torch.no_grad():
        alpha_clip_fg_focused_embeddings = get_alpha_clip_embedding(images, masks, alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
        alpha_clip_bg_focused_embeddings = get_alpha_clip_embedding(images, get_complement_of_mask(masks), alpha_clip, alpha_clip_preprocess, mask_transform, config['device'], dtype)
        print(f"alpha_clip_fg_focused_embeddings shape: {alpha_clip_fg_focused_embeddings.shape}, alpha_clip_bg_focused_embeddings shape: {alpha_clip_bg_focused_embeddings.shape}")
    optimizer.zero_grad()
    transformer_output = transformer(alpha_clip_fg_focused_embeddings, alpha_clip_bg_focused_embeddings)
    foreground_pred = transformer_output.foreground_output
    background_pred = transformer_output.background_output
    print(f"foreground_pred shape: {foreground_pred.shape}, background_pred shape: {background_pred.shape}")

    if use_new_subspace_projection:
        foreground_pred = subspace_proj(foreground_pred)
        background_pred = subspace_proj(background_pred)
        fg_embed = subspace_proj(fg_embed)
        bg_embed = subspace_proj(bg_embed)

    attract_loss_fg = loss(foreground_pred, fg_embed)
    attract_loss_bg = loss(background_pred, bg_embed)
    if use_ortho_loss:
        # make sure that each row of foregrodund pred is perpendicular to background pred
        normalized_foreground_pred = foreground_pred / torch.norm(foreground_pred, dim=-1, keepdim=True)
        normalized_background_pred = background_pred / torch.norm(background_pred, dim=-1, keepdim=True)
        
        dot_products = torch.sum(normalized_foreground_pred * normalized_background_pred, dim=-1)
        
        ortho_loss = torch.mean(dot_products ** 2)

    if use_ortho_loss:
        return attract_loss_fg, attract_loss_bg, ortho_loss
    else:
        return attract_loss_fg, attract_loss_bg

def initialize_training_setup(config, use_new_subspace_projection):
    if config["dtype"] == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Initialize training setup
    transformer = TransformerNetwork(config['embed_dim'], config['num_heads'], config['num_layers']).to(config['device'])

    if config["transformer_ckpt_pth"] is not None:
        load_transformer(config["transformer_ckpt_pth"], transformer)
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
        if use_new_subspace_projection:
            subspace_proj = SubspaceProjectorNetwork(2,768,768).to(config['device'])
            subspace_proj.requires_grad_(False)
        else:
            subspace_proj = None
        alpha_clip.requires_grad_(False)
        projection_block.requires_grad_(False)
        sd.vae.requires_grad_(False)
        sd.unet.requires_grad_(False)
        sd.text_encoder.requires_grad_(False)
        sd_inpaint.vae.requires_grad_(False)
        sd_inpaint.unet.requires_grad_(False)
        sd_inpaint.text_encoder.requires_grad_(False)


    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return transformer, optimizer, accelerator, alpha_clip, alpha_clip_preprocess, mask_transform, ip_adapter, ip_adapter_inpaint, projection_block, subspace_proj

def train(config: OmegaConf, logger: SummaryWriter):
    use_new_subspace_projection = config["use_new_subspace_projection"] == True
    transformer, optimizer, accelerator, alpha_clip, alpha_clip_preprocess, mask_transform, ip_adapter, ip_adapter_inpaint, projection_block, subspace_proj = initialize_training_setup(config, use_new_subspace_projection)
    training_dataset = COCODataset(config['train_image_dir'], config['train_mask_dir'], config['train_metadata_file'], config['embed_file'])
    val_dataset = EvalDataset(config['val_image_dir'], config['val_mask_dir'])
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False)
    use_ortho_loss = config["use_ortho_loss"] == True
    print(f"Use ortho loss is {use_ortho_loss}")
    print(f"Use new subspace projection is {use_new_subspace_projection}")
    loss = torch.nn.MSELoss()

    transformer, optimizer, training_dataloader, val_dataloader = accelerator.prepare(transformer, optimizer, training_dataloader, val_dataloader)
    iteration_count = 0
    for epoch in tqdm(range(config['num_epochs']), desc="Epochs"):
        attract_loss_fg_epoch = 0
        attract_loss_bg_epoch = 0
        ortho_loss_epoch = 0
        total_loss_epoch = 0

        for batch in training_dataloader:
            print("Starting training step")
            if use_ortho_loss:
                attract_loss_fg, attract_loss_bg, ortho_loss = training_step(config, transformer, optimizer, batch, loss,
                                                                                            alpha_clip,
                                                                                            alpha_clip_preprocess,
                                                                                            mask_transform,
                                                                                            use_ortho_loss, subspace_proj, use_new_subspace_projection)

                total_loss = attract_loss_fg + attract_loss_bg + ortho_loss * config['ortho_loss_coeff']
                ortho_loss_epoch += ortho_loss
            else:
                attract_loss_fg, attract_loss_bg  = training_step(config, transformer, optimizer, batch, loss,
                                                                                            alpha_clip,
                                                                                            alpha_clip_preprocess,
                                                                                            mask_transform,
                                                                                            use_ortho_loss, subspace_proj, use_new_subspace_projection)
                total_loss = attract_loss_fg + attract_loss_bg
            print("Losses computed")

            attract_loss_bg_epoch += attract_loss_bg
            attract_loss_fg_epoch += attract_loss_fg
            total_loss_epoch += total_loss

            accelerator.backward(total_loss)
            optimizer.step()

            # print the gpu memory usage
            print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9} Gb")

            if accelerator.is_main_process:
                logger.add_scalar("attract_loss_fg", attract_loss_fg, iteration_count)
                logger.add_scalar("attract_loss_bg", attract_loss_bg, iteration_count)
                logger.add_scalar("total_loss", total_loss, iteration_count)
                if use_ortho_loss:
                    logger.add_scalar("ortho_loss", ortho_loss, iteration_count)
                iteration_count += 1

            if iteration_count % config['eval_interval'] == 0:
                accelerator.wait_for_everyone()
                print("Starting validation")
                validate(config, transformer, val_dataloader, ip_adapter, ip_adapter_inpaint, alpha_clip, alpha_clip_preprocess, mask_transform, projection_block, epoch, logger)
                print("Validation done")
                model_to_save = accelerator.unwrap_model(transformer)
                torch.save(model_to_save.state_dict(), f"{config['output_dir']}/transformer.pth")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.add_scalar("attract_loss_fg_epoch", attract_loss_fg_epoch / len(training_dataloader), epoch)
            logger.add_scalar("attract_loss_bg_epoch", attract_loss_bg_epoch / len(training_dataloader), epoch)
            logger.add_scalar("total_loss_epoch", total_loss_epoch / len(training_dataloader), epoch)
            if use_ortho_loss:
                logger.add_scalar("ortho_loss_epoch", ortho_loss_epoch /  len(training_dataloader), epoch)



if __name__ == "__main__":
    args = parse_args()
    # os.environ["WANDB_INIT_TIMEOUT"] = "300"
    config = OmegaConf.load(args.config)
    logger = SummaryWriter(config['output_dir'])
    # os.makedirs(config['output_dir'], exist_ok=True)
    # wandb_run_name = config['wandb_run_name']
    # wandb.init(
    #     project=wandb_run_name,
    # )
    # wandb.run.name = f"CLR learning with loss function of {config['loss_fn']}"
    train(config, logger)
    # wandb.finish()
