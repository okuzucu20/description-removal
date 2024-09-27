import torch
from typing import Tuple, Union, List, Optional
import diffusers
import types
import alpha_clip
from torchvision import transforms
import torch.nn as nn
from safetensors.torch import load_model
from ip_adapter.ip_adapter import IPAdapter

@torch.no_grad()
def get_unclip_text_to_image_embedding_transformer(dtype: torch.dtype, device: torch.device) -> diffusers.UnCLIPPipeline:
    prior_pipe = diffusers.UnCLIPPipeline.from_pretrained(
        "kakaobrain/karlo-v1-alpha",
        torch_dtype=dtype,
    )
    prior_pipe.decoder = None
    prior_pipe.super_res_first = None
    prior_pipe.super_res_last = None
    prior_pipe.to(device)
    prior_pipe.text_to_image_embedding = types.MethodType(karlo_prior, prior_pipe)
    
    return prior_pipe

@torch.no_grad()
def initialize_alpha_clip(alpha_clip_id: str, alpha_vision_ckpt_pth: str, device: torch.device, dtype: torch.dtype) -> Tuple[torch.nn.Module, transforms.Compose, transforms.Compose]:
    alpha_clip_model, alpha_clip_preprocess = alpha_clip.load(alpha_clip_id, alpha_vision_ckpt_pth, device)
    image_encoder = alpha_clip_model.visual.to(device, dtype=dtype)
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
        transforms.Normalize(0.5, 0.26)
    ]) 
    return image_encoder, alpha_clip_preprocess, mask_transform

@torch.no_grad()
def karlo_prior(
    self,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    prior_num_inference_steps: int = 20,
    generator: Optional[torch.Generator] = None,
    prior_latents: Optional[torch.FloatTensor] = None,
    prior_guidance_scale: float = 4.0,
) -> torch.Tensor:
    """
    copy from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/unclip/pipeline_unclip.py#L234-L358
    """
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    device = self._execution_device

    batch_size = batch_size * num_images_per_prompt

    do_classifier_free_guidance = prior_guidance_scale > 1.0

    text_embeddings, text_encoder_hidden_states, text_mask = self._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance
    )

    # prior

    self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device)
    prior_timesteps_tensor = self.prior_scheduler.timesteps

    embedding_dim = self.prior.config.embedding_dim
    prior_latents = self.prepare_latents(
        (batch_size, embedding_dim),
        text_embeddings.dtype,
        device,
        generator,
        prior_latents,
        self.prior_scheduler,
    )

    for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents

        predicted_image_embedding = self.prior(
            latent_model_input,
            timestep=t,
            proj_embedding=text_embeddings,
            encoder_hidden_states=text_encoder_hidden_states,
            attention_mask=text_mask,
        ).predicted_image_embedding

        if do_classifier_free_guidance:
            predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
            predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                predicted_image_embedding_text - predicted_image_embedding_uncond
            )

        if i + 1 == prior_timesteps_tensor.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = prior_timesteps_tensor[i + 1]

        prior_latents = self.prior_scheduler.step(
            predicted_image_embedding,
            timestep=t,
            sample=prior_latents,
            generator=generator,
            prev_timestep=prev_timestep,
        ).prev_sample

    prior_latents = self.prior.post_process_latents(prior_latents)

    image_embeddings = prior_latents
    return image_embeddings

@torch.no_grad()
def generate_projection_layer(config):
    projection_layer = nn.ModuleList()
    
    for i in range(config.number_of_hidden_layers):
        if i < config.number_of_hidden_layers // 2:
            projection_layer.append(nn.Linear(config.alpha_clip_embed_dim, config.alpha_clip_embed_dim))
            projection_layer.append(nn.LayerNorm(config.alpha_clip_embed_dim))
        elif i == config.number_of_hidden_layers // 2:
            projection_layer.append(nn.Linear(config.alpha_clip_embed_dim, config.ip_adapter_embed_dim))
            projection_layer.append(nn.LayerNorm(config.ip_adapter_embed_dim))
        else:
            projection_layer.append(nn.Linear(config.ip_adapter_embed_dim, config.ip_adapter_embed_dim))
            projection_layer.append(nn.LayerNorm(config.ip_adapter_embed_dim))
        projection_layer.append(nn.GELU())
        
    projection_layer.append(nn.Linear(config.ip_adapter_embed_dim, config.ip_adapter_embed_dim))

    return nn.Sequential(*projection_layer)

@torch.no_grad()
def get_alpha_clip_embedding(image: torch.Tensor, mask: torch.Tensor, image_encoder: torch.nn.Module, 
                            alpha_clip_preprocess: transforms.Compose, mask_transform: transforms.Compose, 
                            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image_pil = [transforms.ToPILImage()(img) for img in image]
    mask_pil = [transforms.ToPILImage()(img) for img in mask]
    
    clip_image = [alpha_clip_preprocess(image) for image in image_pil]
    clip_image = torch.stack(clip_image).to(device, dtype=dtype)
    masks = [mask_transform(mask) for mask in mask_pil]
    masks = torch.stack(masks).to(device, dtype=dtype)
    
    return image_encoder(clip_image, masks)

@torch.no_grad()
def initialize_and_load_projection_block(config: dict, ckpt_path: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    projection_block = generate_projection_layer(config)
    load_model(projection_block, ckpt_path)
        
    return projection_block.to(device, dtype=dtype)

def initialize_and_load_ipadapater(sd_pipe: diffusers.StableDiffusionPipeline, image_encoder_path: str, ip_ckpt_path: str, device: torch.device) -> IPAdapter:
    ip_adapter = IPAdapter( sd_pipe, image_encoder_path, ip_ckpt_path, device)
    return ip_adapter

def get_complement_of_mask(mask: torch.Tensor) -> torch.Tensor:
    return 1 - mask

