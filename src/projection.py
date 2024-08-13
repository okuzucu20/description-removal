import torch
import os
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List
from matplotlib import pyplot as plt
from tqdm import tqdm
from model.clip_away import CLIPAway
from src.coco_dataset import COCODataset
from diffusers import StableDiffusionInpaintPipeline
import random


class Projector(torch.nn.Module):

    def __init__(self, depth=6):
        super().__init__()

        if depth < 1 or depth > 11:
            raise ValueError("Depth cannot be smaller than 1 and larger than 11")

        sizes = [int(2048/(2**i)) for i in range(depth)]
        self.downscale = torch.nn.Sequential(
            *[torch.nn.Linear(size, size//2) for size in sizes]
        )
        self.upscale = torch.nn.Sequential(
            *[torch.nn.Linear(size//2, size) for size in sizes[::-1]]
        )

    def forward(self, x):
        residuals = []
        inner_x = x

        for layer in self.downscale:
            residuals.append(inner_x)
            inner_x = layer(inner_x)

        for residual, layer in zip(residuals[::-1], self.upscale):
            inner_x = residual + layer(inner_x)

        return inner_x


class ProjectionDataset(Dataset):

    def __init__(self):
        self.image_ids: List[int] = []
        self.image_embeds: Dict[int, List[torch.Tensor]] = {}
        self.text_embeds: Dict[int, List[torch.Tensor]] = {}
        self.text_embeds_path = "text_embeds_generated"
        self.image_embeds_path = "image_embeds_generated"

    def load(self):
        self.image_embeds = {}
        self.text_embeds = {}

        self.image_ids: List[int] = self._extract_image_ids()
        self._load_embeddings()

        return self

    def _load_embeddings(self):
        for image_id in tqdm(self.image_ids, desc=f"Loading embeddings"):
            text_embed_bg = torch.load(os.path.join(self.text_embeds_path, f"{image_id}_bg.pt"), weights_only=True)
            text_embed_fg = torch.load(os.path.join(self.text_embeds_path, f"{image_id}_fg.pt"), weights_only=True)
            text_embed_bg, text_embed_fg = text_embed_bg.reshape((1, -1)), text_embed_fg.reshape((1, -1))
            self.text_embeds.update({image_id: [text_embed_bg, text_embed_fg]})

            image_embed_bg = torch.load(os.path.join(self.image_embeds_path, f"{image_id}_bg.pt"), weights_only=True)
            image_embed_fg = torch.load(os.path.join(self.image_embeds_path, f"{image_id}_fg.pt"), weights_only=True)
            self.image_embeds.update({image_id: [image_embed_bg, image_embed_fg]})

    def _extract_image_ids(self) -> List[int]:
        text_embeds_image_ids: List[int] = self._extract_image_ids_from_directory(self.text_embeds_path)
        image_embeds_image_ids: List[int] = self._extract_image_ids_from_directory(self.image_embeds_path)

        if set(text_embeds_image_ids) != set(image_embeds_image_ids):
            raise ValueError("Text and image embeddings are not the same size!")

        return text_embeds_image_ids

    @staticmethod
    def _extract_image_ids_from_directory(dir_path: str) -> List[int]:
        image_ids_bg = []
        image_ids_fg = []

        for filename in os.listdir(dir_path):
            image_id, desc_type = filename.split('.')[0].split('_')
            if desc_type == "bg":
                image_ids_bg.append(int(image_id))
            elif desc_type == "fg":
                image_ids_fg.append(int(image_id))
            else:
                raise ValueError("Wrong description type present in file names!")

        if set(image_ids_bg) != set(image_ids_fg):
            raise ValueError("Foreground and background embeddings are not the same size!")

        return image_ids_bg

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        key: int = self.image_ids[idx]
        x = torch.cat(self.image_embeds[key], dim=1)
        y = torch.cat(self.text_embeds[key], dim=1)
        return x, y  # image_bg+image_fg, text_bg+text_fg


class ProjectionTrain:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
                 criterion=torch.nn.MSELoss(), device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 num_epochs: int = 10):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs

    def train_and_validate(self, model_save_path=None, plot_save_path=None):

        plot_save_path = plot_save_path if plot_save_path is not None else \
            f"figures/loss_{self._get_current_index_in_directory('figures', 'loss')}.png"
        model_save_path = model_save_path if model_save_path is not None else \
            f"ckpts/projection_model_{self._get_current_index_in_directory('ckpts', 'projection_model')}.pt"

        train_losses = []
        val_losses = [self._val_epoch()]
        for epoch in tqdm(list(range(1, self.num_epochs + 1)), desc="Epochs"):

            epoch_train_losses = self._train_epoch()
            epoch_val_loss = self._val_epoch()

            train_losses += epoch_train_losses
            val_losses.append(epoch_val_loss)

        self._save_model(model_save_path)
        self._generate_and_save_plot(train_losses, val_losses, plot_save_path)

    def _train_epoch(self):
        epoch_losses = []
        self.model.train()
        for x, y in tqdm(self.train_dataloader, desc="Training"):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(x)
            loss = self.criterion(output, y)

            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())
        return epoch_losses

    def _val_epoch(self):
        epoch_loss = 0
        self.model.eval()
        for x, y in tqdm(self.val_dataloader, desc="Validation"):
            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            loss = self.criterion(output, y)

            epoch_loss += loss.detach().cpu().numpy() * x.size(0)

        # noinspection PyTypeChecker
        return epoch_loss / len(self.val_dataloader.dataset)

    @staticmethod
    def _get_current_index_in_directory(directory: str, prefix: str):
        count = 0
        for filename in os.listdir(directory):
            if filename.startswith(prefix):
                count += 1
        return count + 1

    @staticmethod
    def _generate_and_save_plot(train_losses, val_losses, save_path):
        val_x = np.linspace(0, len(val_losses)-1, len(val_losses))
        train_x = np.linspace(0, len(val_losses)-1, len(train_losses))
        plt.plot(train_x, train_losses, label='Train Loss')
        plt.plot(val_x, val_losses, label='Validation Loss')
        plt.title("Training and Validation Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.savefig(save_path)

    def _save_model(self, save_path: str):
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)


class CLIPAwayWithLearnedProjection(CLIPAway):

    def __init__(self, config, projector: Projector):
        super().__init__(image_encoder_path=config.image_encoder_path,
                         ip_ckpt=config.ip_adapter_ckpt_path,
                         alpha_clip_path=config.alpha_clip_ckpt_pth,
                         config=config,
                         alpha_clip_id=config.alpha_clip_id,
                         device="cuda" if torch.cuda.is_available() else "cpu",
                         num_tokens=4,
                         sd_pipe=StableDiffusionInpaintPipeline.from_pretrained(config.sd_model_key,
                                                                                safety_checker=None,
                                                                                torch_dtype=torch.float32))
        self.projector = projector

    def clipaway_projection_block(self, bg_embeds, fg_embeds):
        x = torch.cat([bg_embeds, fg_embeds], dim=-1)
        print("x shape:", x.shape)
        out = self.projector(x).chunk(2, dim=-1)[0]
        print("out shape:", out.shape)
        return out


class ProjectionInference:

    def __init__(self, clipaway_config, test_dataset: COCODataset):
        self.clipaway_config = clipaway_config
        self.test_dataset = test_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.projector = None
        self.clipaway = None

    def _load_projection_model(self, model_idx: int = None):
        latest_idx = ProjectionInference._get_current_index_in_directory("ckpts", "projection_model") - 1

        if model_idx is None:
            model_idx = latest_idx

        if not isinstance(model_idx, int) or model_idx < 1 or model_idx > latest_idx:
            raise ValueError("Wrong index used for loading projection model.")

        checkpoint = torch.load(f"ckpts/projection_model_{model_idx}.pt")
        self.projector = Projector()
        self.projector.load_state_dict(checkpoint["model_state_dict"])

        self.projector.to(self.device)

    def _load_clipaway(self):
        self.clipaway = CLIPAwayWithLearnedProjection(self.clipaway_config, self.projector)

    def load(self):
        self._load_projection_model()
        self._load_clipaway()
        return self

    def generate(self, count: int, seed=None):

        seed = seed if seed is not None else self.clipaway_config.seed
        latents = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(seed)).to(self.device)
        random.seed(seed)

        datapoint_indices: List[int] = random.sample(range(len(self.test_dataset)), count)
        for i in tqdm(datapoint_indices):
            datapoint, datapoint_raw = self.test_dataset[i]
            out = self.clipaway.generate(prompt=[""], scale=self.clipaway_config.scale, seed=seed,
                                         pil_image=[datapoint.image], alpha=[datapoint.segments[0].mask],
                                         strength=self.clipaway_config.strength, latents=latents)[0]
            out.save(f"image_samples_generated/{COCODataset.filepath_to_id(datapoint_raw.image_path)}.jpg")

    @staticmethod
    def _get_current_index_in_directory(directory: str, prefix: str):
        count = 0
        for filename in os.listdir(directory):
            if filename.startswith(prefix):
                count += 1
        return count + 1

