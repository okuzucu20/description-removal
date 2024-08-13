from src.coco_dataset import COCODataset
from src.gpt_client import GPTClient
from src.clip_away_service import CLIPAwayService
from src.models import *
from src.projection import *
from omegaconf import OmegaConf
from typing import Tuple, List
from math import ceil
import argparse
import os
from tqdm import tqdm
from random import shuffle
from torch.utils.data import DataLoader, random_split
from torch import Generator
from torch.optim import Adam

BG_GENERATION_COUNT = 118287


def remove_foreground_and_save_results():
    coco_dataset: COCODataset = COCODataset().load(background=False)
    gpt_client: GPTClient = GPTClient()

    datapoint_indices = list(range(len(coco_dataset)))
    shuffle(datapoint_indices)

    count = 0
    for i in tqdm(datapoint_indices):

        coco_data, coco_data_raw = coco_dataset[i]
        try:
            bg_desc: str = gpt_client.remove_foreground(coco_data.caption, coco_data.segments[0].objectType)
        except Exception as e:
            break

        coco_data_raw.bg_desc = bg_desc
        coco_dataset.save_bg_description_of(coco_data_raw)
        print("Caption:", coco_data.caption, "Object:", coco_data.segments[0].objectType, "Background:", bg_desc)

        count += 1
        if count == BG_GENERATION_COUNT:
            break


def test_coco_initialization():
    coco_dataset: COCODataset = COCODataset().load(background=True)

    count = 0
    for coco_data, coco_data_raw in coco_dataset:

        s_idx = -1
        for i, s in enumerate(coco_data_raw.segments):
            if isinstance(s.segment, COCODatapointSegmentRLE):
                print(type(s.segment))
                s_idx = i
                break

        if s_idx == -1:
            continue

        image_id_str = str(COCODataset.filepath_to_id(coco_data_raw.image_path))
        caption_path = os.path.join("coco", "caption", image_id_str + ".txt")
        image_path = os.path.join("coco", "image", image_id_str + ".jpg")
        mask_path = os.path.join("coco", "mask", image_id_str + ".png")
        object_path = os.path.join("coco", "object", image_id_str + ".txt")

        with open(caption_path, 'w') as cf:
            cf.write(coco_data.caption)

        with open(object_path, 'w') as of:
            of.write(coco_data.segments[s_idx].objectType)

        coco_data.image.save(image_path)
        coco_data.segments[s_idx].mask.save(mask_path)

        count += 1
        if count == BG_GENERATION_COUNT:
            break


def generate_embeddings(config, embed_type):
    clip_away_service: CLIPAwayService = CLIPAwayService(config)
    coco_dataset: COCODataset = COCODataset().load(background=True)
    
    if embed_type == "image":
        batch_size = 1
        embedding_generator = clip_away_service.generate_clip_away_image_embeddings_of
    elif embed_type == "text":
        batch_size = 32
        embedding_generator = clip_away_service.generate_clip_text_embeddings_of
    else:
        raise Exception("Wrong type specified for generating embeddings.")

    iteration_count = ceil(len(coco_dataset)/batch_size)
    for i in tqdm(list(range(iteration_count))):
        datapoint_pairs: List[Tuple[COCODatapoint, COCODatapointRaw]] = coco_dataset[i*batch_size:(i+1)*batch_size]
        datapoint_batch: List[COCODatapoint] = [d[0] for d in datapoint_pairs]
        datapoint_raw_batch: List[COCODatapointRaw] = [d[1] for d in datapoint_pairs]

        image_ids: List[int] = COCODataset.data_raw_batch_to_image_ids(datapoint_raw_batch)
        embedding_generator(datapoint_batch, save_with_ids=image_ids)


def train_projection():
    model = Projector()

    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    dataset = ProjectionDataset().load()
    train_dataset, val_dataset = random_split(dataset, [0.75, 0.25], generator=Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=32)

    trainer = ProjectionTrain(model, optimizer, train_dataloader, val_dataloader)
    trainer.train_and_validate()


def generate_samples(config, count, seed):
    coco_dataset: COCODataset = COCODataset().load(background=False)
    inference_module = ProjectionInference(config, coco_dataset).load()
    inference_module.generate(count=count, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-foreground", dest="remove_foreground", action='store_true')
    parser.add_argument("--image-embeddings", dest="image_embeddings", action='store_true')
    parser.add_argument("--text-embeddings", dest="text_embeddings", action='store_true')
    parser.add_argument("--train-projection", dest="train_projection", action='store_true')
    parser.add_argument("--test-coco-initialization", dest="test_coco_initialization", action='store_true')
    parser.add_argument("--config", type=str, default="config/clip_away_inference.yaml")
    parser.add_argument("--generate-samples", dest="sample_count", type=int)
    parser.add_argument("-s", dest="seed", help="seed for sampling", type=int)

    options = parser.parse_args()

    if options.remove_foreground:
        remove_foreground_and_save_results()
    elif options.image_embeddings:
        generate_embeddings(OmegaConf.load(options.config), "image")
    elif options.text_embeddings:
        generate_embeddings(OmegaConf.load(options.config), "text")
    elif options.test_coco_initialization:
        test_coco_initialization()
    elif options.train_projection:
        train_projection()
    elif options.sample_count:
        config = OmegaConf.load(options.config)
        seed = options.seed if options.seed is not None else config.seed
        generate_samples(config, options.sample_count, seed)


if __name__ == '__main__':
    main()
