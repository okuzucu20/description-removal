import torch
import os
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List
from matplotlib import pyplot as plt
from tqdm import tqdm


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
        self.image_embeds: Dict[int, List[torch.Tensor]] = {}
        self.text_embeds: Dict[int, List[torch.Tensor]] = {}
        self.text_embeds_path = "text_embeds_generated"
        self.image_embeds_path = "image_embeds_generated"
        self.embed_keys = []

    def load(self):
        self.image_embeds = {}
        self.text_embeds = {}
        self._load_embeddings("image")
        self._load_embeddings("text")

        if len(self.image_embeds.keys()) != len(self.text_embeds.keys()):
            raise ValueError(f"Image and text embeddings are not the same amount! \n"
                             f"Image: {len(self.image_embeds.keys())} \n"
                             f"Text: {len(self.text_embeds.keys())}")

        self.embed_keys = self.image_embeds.keys()

        return self

    def _load_embeddings(self, embed_type: str):
        if embed_type == "image":
            embeds = self.image_embeds
            embeds_dir_path = self.image_embeds_path
        elif embed_type == "text":
            embeds = self.text_embeds
            embeds_dir_path = self.text_embeds_path
        else:
            raise ValueError("Wrong embedding type specified")

        for embed_file in os.listdir(embeds_dir_path):
            image_id, desc_type = embed_file.split('.')[0].split('_')
            image_id = int(image_id)

            embed = torch.load(os.path.join(embeds_dir_path, embed_file), weights_only=True)
            embed_pair = embeds.get(image_id, [None, None])
            if desc_type == "bg":
                embed_pair = [embed] + embed_pair[1:]
            elif desc_type == "fg":
                embed_pair = embed_pair[:1] + [embed]
            else:
                raise ValueError("Wrong description type present in file names!")
            embeds.update({image_id: embed_pair})

    def __len__(self):
        return len(self.image_embeds.keys())

    def __getitem__(self, idx):
        key: int = self.embed_keys[idx]
        x = torch.cat(self.image_embeds[key], dim=1)
        y = torch.cat(self.text_embeds[key], dim=1)
        return x, y  # image_bg+image_fg, text_bg+text_fg


class Trainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
                 criterion=torch.nn.MSELoss(), device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 num_epochs: int = 40):
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
        return epoch_loss / len(self.val_dataloader)

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



