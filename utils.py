import torch
from typing import Tuple, Union, List, Optional
import diffusers
import types
import alpha_clip
from torchvision import transforms
import torch.nn as nn
from safetensors.torch import load_model
from ip_adapter.ip_adapter import IPAdapter
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import random

def generate_image_from_embedding_with_ipadapter(ip_adapter: IPAdapter, embedding: torch.Tensor) -> torch.Tensor:
    return ip_adapter.generate(clip_image_embeds=embedding)

def inpaint_from_embedding_with_ipadapter(ip_adapter: IPAdapter, embedding: torch.Tensor, img: PIL.Image, mask: PIL.Image) -> torch.Tensor:
    return ip_adapter.generate(clip_image_embeds=embedding, mask_image=mask, image=img)

def save_batch_of_tensor_to_image(tensor: torch.Tensor, path: str, is_luminance: bool = False):
    if not is_luminance:
        tensor = tensor.cpu().detach().numpy()
        tensor = (tensor * 255).astype('uint8')
        tensor = tensor.transpose(0, 2, 3, 1)
        for i, img in enumerate(tensor):
            img = Image.fromarray(img)
            img.save(f"{path}_{i}.png")
    else:
        tensor = tensor.cpu().detach().numpy()
        tensor = (tensor * 255).astype('uint8')
        for i, img in enumerate(tensor):
            img = Image.fromarray(img[0])
            img.save(f"{path}_{i}.png")

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
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
        transforms.Normalize(0.5, 0.26)
    ]) 
    return alpha_clip_model, alpha_clip_preprocess, mask_transform

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
    clip_image = torch.stack(clip_image).to(device).half()
    masks = [mask_transform(mask) for mask in mask_pil]
    masks = torch.stack(masks).to(device).half()
    
    return image_encoder.visual(clip_image, masks).to(device, dtype=dtype)

def get_alpha_clip_text_embedding(text: str, model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    text = alpha_clip.tokenize(text).to(device)
    return model.encode_text(text).to(device, dtype=dtype)

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


class SDLOSS(nn.Module):
    def __init__(self, sd):
        super(SDLOSS, self).__init__()
        self.unet = sd.unet
        self.scheduler = sd.scheduler
        self.unet.requires_grad_(True)
        self.vae = sd.vae
        self.model = sd
        
    
    def img_to_latents(self, x: torch.Tensor, vae):
        if x.min() >= 0 and x.max() <= 1:
            x = 2. * x - 1.
        posterior = vae.encode(x).latent_dist
        latents = posterior.mean * 0.18215
        return latents


    def forward(
        self,
        ip_embeddings: torch.Tensor,
        clean_images: torch.Tensor,
    ):
        latents = self.img_to_latents(clean_images, self.vae)
        # Sample noise to add to the images
        noise = torch.randn(latents.shape, device=latents.device)
        bs = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bs,), device=latents.device,
            dtype=torch.int64
        )
        text_embeds = self.model.encode_prompt([""]*bs, device=latents.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
        encoder_hidden_states = torch.cat([text_embeds, ip_embeddings], dim=1)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.unet(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

##############################################################################
# PCGradTwoLoss: a simpler, more stable approach to multi-loss gradient handling.
##############################################################################

class SoftMarginForgettingLoss(nn.Module):
    """
    A margin-based approach to push two embeddings apart, but uses a continuous penalty:
       penalty = (relu(cos_sim - margin))^2
    so that even if cos_sim < margin, there's a small gradient if cos_sim drifts back up.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        # Normalize each embedding
        a_norm = F.normalize(embedding_a, p=2, dim=-1)
        b_norm = F.normalize(embedding_b, p=2, dim=-1)
        # Cosine similarity per sample
        cos_sim = F.cosine_similarity(a_norm, b_norm, dim=-1)
        # We apply a continuous margin penalty:
        # The portion above margin is penalized, and we square it for a smoother gradient
        over_margin = F.relu(cos_sim - self.margin)
        penalty = over_margin ** 2
        return penalty.mean()


class ReconstructionLoss(nn.Module):
    """
    By default, a standard MSE between embeddings. Optionally clamp diff to avoid huge outliers.
    """
    def __init__(self, clamp_value=None):
        super().__init__()
        self.clamp_value = clamp_value

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        diff = embedding_a - embedding_b
        if self.clamp_value is not None:
            diff = torch.clamp(diff, min=-self.clamp_value, max=self.clamp_value)
        return torch.mean(diff * diff)

class PCGradSingleFG(nn.Module):
    """
    A two-task PCGrad approach:
      - FG (forget object)
      - BG (reconstruct background)

    Uses partial gradient projection (alpha=0.5) to avoid fully removing
    negative dot products. The forgetting and reconstruction losses
    are provided at init. 
    """
    def __init__(self, model, loss_f, loss_r, f_coeff=1.0, r_coeff=1.0, projection_alpha=0.5):
        """
        Args:
          model: The MLP (or any nn.Module) being trained.
          loss_f: The forgetting loss (e.g., a margin-based forgetting).
          loss_r: The reconstruction loss (e.g., MSE).
          f_coeff: Weight for FG forgetting objective.
          r_coeff: Weight for BG reconstruction objective.
          projection_alpha: fraction of the negative overlap to remove in PCGrad.
        """
        super().__init__()
        self.model = model
        self.loss_f = loss_f
        self.loss_r = loss_r
        self.f_coeff = f_coeff
        self.r_coeff = r_coeff
        self.projection_alpha = projection_alpha

    def compute_grad(self, loss):
        """
        Compute gradients w.r.t model parameters in a single flattened vector.
        """
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False, retain_graph=True)
        # Flatten all param grads (skip None in case some params don't get grads)
        return torch.cat([g.reshape(-1) for g in grads if g is not None])

    def apply_combined_grad(self, combined_grad):
        """
        Unflatten combined_grad and assign back to each parameter's .grad.
        """
        offset = 0
        for param in self.model.parameters():
            numel = param.numel()
            grad_slice = combined_grad[offset:offset+numel].view(param.size())
            param.grad = grad_slice
            offset += numel

    def project_conflict(self, grad_i, grad_j):
        """
        Soft PCGrad approach: if dot(grad_i, grad_j) < 0, remove alpha fraction.
        """
        dot_val = torch.dot(grad_i, grad_j)
        if dot_val < 0:
            norm_j_sq = torch.dot(grad_j, grad_j).clamp_min(1e-12)
            alpha = self.projection_alpha
            grad_i = grad_i - alpha * (dot_val / norm_j_sq) * grad_j
        return grad_i

    def forward(self, preds, fg, bg):
        """
        preds: [B, D] => MLP output
        fg:    [B, D] => FG embedding (e.g., text or object concept)
        bg:    [B, D] => background embedding

        Returns: (L_fg, L_bg) for logging.

        Steps:
          1) Build FG loss, BG loss.
          2) PCGrad across the 2 tasks => final gradient => model.param.grad.
          3) Return losses for logging.
        """
        # 1) Compute each scalar loss
        L_fg = self.loss_f(preds, fg) * self.f_coeff
        L_bg = self.loss_r(preds, bg) * self.r_coeff

        # 2) Gather tasks
        task_losses = [L_fg, L_bg]
        grads = []
        for loss_val in task_losses:
            g = self.compute_grad(loss_val)
            grads.append(g)

        # Debug: print grad norms
        print("Grad norms => FG:", torch.norm(grads[0]), " BG:", torch.norm(grads[1]))

        # 3) PCGrad logic (2 tasks is simpler, but we'll keep a consistent pattern)
        indices = [0, 1]
        random.shuffle(indices)
        new_grads = [grads[i].clone() for i in indices]

        # For i=1, project out negative from j=0
        # If the random shuffle leads to [BG, FG], it projects BG onto FG, etc.
        for i in range(1, len(new_grads)):
            for j in range(i):
                new_grads[i] = self.project_conflict(new_grads[i], new_grads[j])

        # 4) Sum final grads
        final_grads = torch.stack(new_grads, dim=0).sum(dim=0)

        # 5) Assign to model param.grad
        with torch.no_grad():
            self.apply_combined_grad(final_grads)

        return L_fg, L_bg
