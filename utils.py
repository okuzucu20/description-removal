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
from PIL import Image

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


import torch
import torch.nn.functional as F

class RestrictedGradientLoss:
    def __init__(self, model, loss_f, loss_r, r_coeff=1.0, f_coeff=1.0):
        """
        Initializes the RestrictedGradientLoss.

        Args:
            model: The PyTorch model being trained.
            loss_f: A loss function representing L_f.
            loss_r: A loss function representing L_r.
            r_coeff: Weight for L_r.
            f_coeff: Weight for L_f.
        """
        self.model = model
        self.loss_f = loss_f
        self.loss_r = loss_r
        self.r_coeff = r_coeff
        self.f_coeff = f_coeff

    def compute_gradients(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        return torch.cat([g.view(-1) for g in grads])

    def compute_delta_star(self, grad_a, grad_b):
        dot_product = torch.dot(grad_a, grad_b)
        norm_b_squared = torch.norm(grad_b) ** 2
        delta_star = grad_a - (dot_product / norm_b_squared) * grad_b
        return delta_star

    def compute_angle(self, grad_a, grad_b):
        dot_product = torch.dot(grad_a, grad_b)
        norm_a = torch.norm(grad_a)
        norm_b = torch.norm(grad_b)
        cos_theta = dot_product / (norm_a * norm_b + 1e-8)  # Avoid division by zero
        angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # Clamp for numerical stability
        return angle.item() * 180 / 3.14159265359  # Convert to degrees

    def __call__(self, predictions, fg1, fg2, is_2_objects, bg):
        if self.f_coeff == 0:
            return torch.tensor(0.0), self.loss_r(predictions, bg) * self.r_coeff, torch.tensor(0.0), torch.tensor(0.0)
        else:
            loss_f1 = self.loss_f(predictions, fg1) * self.f_coeff

            loss_f2 = 0
            for i in range(len(is_2_objects)):
                if is_2_objects[i]:
                    loss_f2 += self.loss_f(predictions, fg2[i]) * self.f_coeff
                
            loss_f = loss_f1 + loss_f2
            
            loss_r = self.loss_r(predictions, bg) * self.r_coeff
            
            # Compute gradients
            grad_f = self.compute_gradients(loss_f)
            grad_r = self.compute_gradients(loss_r)

            # Compute the angle
            angle = self.compute_angle(grad_f, grad_r)

            if angle > 90 and self.r_coeff != 0 and self.f_coeff != 0:
                # Apply restricted gradients
                delta_f_star = self.compute_delta_star(grad_f, grad_r)
                delta_r_star = self.compute_delta_star(grad_r, grad_f)
                combined_gradient = delta_f_star + delta_r_star
                angle_after_surgery = self.compute_angle(delta_f_star, delta_r_star)
                if angle_after_surgery > 90:
                    # If the angle is still greater than 90 degrees,  zero out the gradients
                    combined_gradient = torch.zeros_like(combined_gradient)
                    angle_after_surgery = 0
            else:
                # Use direct aggregation if gradients are aligned
                combined_gradient = grad_f + grad_r
                angle_after_surgery = angle

            # Apply the gradient to the model's parameters manually
            with torch.no_grad():
                offset = 0
                for param in self.model.parameters():
                    numel = param.numel()
                    grad = combined_gradient[offset : offset + numel].view(param.size())
                    param.grad = grad
                    offset += numel
                    
            return loss_f, loss_r, angle, angle_after_surgery




class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(embedding_a, embedding_b)
    
class ForgettingLoss(nn.Module):
    def __init__(self):
        super(ForgettingLoss, self).__init__()

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        embedding_a = F.normalize(embedding_a, p=2, dim=-1)
        embedding_b = F.normalize(embedding_b, p=2, dim=-1)
        cosine_similarity = F.cosine_similarity(embedding_a, embedding_b, dim=-1)
        orthogonality_loss = cosine_similarity.abs().mean()
        return orthogonality_loss