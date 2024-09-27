from tqdm import tqdm
import os
from utils import get_unclip_text_to_image_embedding_transformer
import json
import torch

JSON_PATH = "final.json"
SAVE_PATH = "unclip_data"
dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
unclip_transformer = get_unclip_text_to_image_embedding_transformer(dtype, device)
with torch.inference_mode():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        for keys, values in tqdm(data.items()):
            fg_desc = values["fg_text"]
            bg_desc = values["bg_text"]
            fg_text_unclip = unclip_transformer.text_to_image_embedding(fg_desc)
            bg_text_unclip = unclip_transformer.text_to_image_embedding(bg_desc)
            torch.save(fg_text_unclip.detach().cpu(), os.path.join(SAVE_PATH, "fg", f"{keys}.pt"))
            torch.save(bg_text_unclip.detach().cpu(), os.path.join(SAVE_PATH, "bg", f"{keys}.pt"))
