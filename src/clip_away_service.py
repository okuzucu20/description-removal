from model.clip_away import CLIPAway
from diffusers import StableDiffusionInpaintPipeline
from typing import List
from src.models import COCODatapoint
from src.coco_dataset import COCODataset
import torch
import os
import open_clip


class CLIPAwayService:

    def __init__(self, config, img_embeddings_dir_path="image_embeds_generated", text_embeddings_dir_path="text_embeds_generated"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_away = CLIPAway(
            sd_pipe=StableDiffusionInpaintPipeline.from_pretrained(
                config.sd_model_key, safety_checker=None, torch_dtype=torch.float32),
            image_encoder_path=config.image_encoder_path,
            ip_ckpt=config.ip_adapter_ckpt_path,
            alpha_clip_path=config.alpha_clip_ckpt_pth,
            config=config,
            alpha_clip_id=config.alpha_clip_id,
            device=self.device,
            num_tokens=4
        )
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', force_custom_text=True)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.img_embeddings_dir_path = img_embeddings_dir_path
        self.text_embeddings_dir_path = text_embeddings_dir_path

    def generate_clip_away_image_embeddings_of(self, coco_data_batch: List[COCODatapoint], save_with_ids: List[int] = None):
        images, masks = COCODataset.data_batch_to_image_mask_pairs(coco_data_batch)
        complement_masks = [self.clip_away.get_complement_of_mask([mask]) for mask in masks]

        clip_image_embeds_fg = self.clip_away.get_alpha_clip_embeds(images, masks)
        clip_image_embeds_bg = self.clip_away.get_alpha_clip_embeds(images, complement_masks)

        projected_alpha_clip_embeds_fg = self.clip_away.mlp_projection_layer(clip_image_embeds_fg).detach().cpu()
        projected_alpha_clip_embeds_bg = self.clip_away.mlp_projection_layer(clip_image_embeds_bg).detach().cpu()

        if save_with_ids is not None:
            self._save_embeddings(save_with_ids, projected_alpha_clip_embeds_fg, projected_alpha_clip_embeds_bg, "image")

        return projected_alpha_clip_embeds_fg, projected_alpha_clip_embeds_bg  # [1, 1024]

    def generate_clip_text_embeddings_of(self, coco_data_batch: List[COCODatapoint], save_with_ids: List[int] = None):
        fg_objects, bg_descriptions = COCODataset.data_batch_to_fg_bg_pairs(coco_data_batch)

        fg_objects_tokenized = self.clip_tokenizer(fg_objects)
        bg_descriptions_tokenized = self.clip_tokenizer(bg_descriptions)

        with torch.no_grad():
            clip_text_embeds_fg = self.clip_model.encode_text(fg_objects_tokenized)
            clip_text_embeds_bg = self.clip_model.encode_text(bg_descriptions_tokenized)

        if save_with_ids is not None:
            self._save_embeddings(save_with_ids, clip_text_embeds_fg, clip_text_embeds_bg, "text")

        return clip_text_embeds_fg, clip_text_embeds_bg

    def _save_embeddings(self, image_ids, embeds_fg, embeds_bg, embed_type):
        if embed_type == "image":
            save_dir = self.img_embeddings_dir_path
        elif embed_type == "text":
            save_dir = self.text_embeddings_dir_path
        else:
            raise Exception("Wrong type specified for saving embeddings.")

        for image_id, embed_fg, embed_bg in zip(image_ids, embeds_fg, embeds_bg):
            torch.save(embed_fg, os.path.join(save_dir, str(image_id) + "_fg.pt"))
            torch.save(embed_bg, os.path.join(save_dir, str(image_id) + "_bg.pt"))

