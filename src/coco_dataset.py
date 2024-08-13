from torch.utils.data import Dataset
from typing import Dict, Tuple
from src.models import *
from PIL import Image
import os
import json
import numpy as np
import cv2


class COCODataset(Dataset):

    def __init__(self) -> None:
        self.dataset_path = "/datasets/COCO"
        self.instance_annotations_file_path = os.path.join(self.dataset_path, "annotations/instances_train2017.json")
        self.caption_annotations_file_path = os.path.join(self.dataset_path, "annotations/captions_train2017.json")
        self.images_dir_path = os.path.join(self.dataset_path, "train2017")
        self.bg_description_dir_path = "bg_desc_generated"

        self.datapoints_raw = None

    def _load_image_paths(self) -> Dict[int, str]:
        image_paths: Dict[int, str] = {}

        for filename in os.listdir(self.images_dir_path):
            if filename.endswith('jpg'):
                image_paths[COCODataset.filepath_to_id(filename)] = os.path.join(self.images_dir_path, filename)

        return image_paths

    def _load_captions(self) -> Dict[int, str]:
        captions: Dict[int, str] = {}

        with open(self.caption_annotations_file_path, 'r') as f:
            caption_dict = json.load(f)

        for annotation in caption_dict["annotations"]:
            captions[annotation["image_id"]] = annotation["caption"]

        return captions

    def _load_segmentations(self) -> Dict[int, List[COCODatapointSegmentRaw]]:
        segmentations: Dict[int, List[COCODatapointSegmentRaw]] = {}

        with open(self.instance_annotations_file_path, 'r') as f:
            segmentation_dict = json.load(f)

        categories = dict([(category_dict["id"], category_dict["name"])
                           for category_dict in segmentation_dict["categories"]])

        for annotation in segmentation_dict["annotations"]:

            if isinstance(annotation["segmentation"], dict):
                #segment = COCODatapointSegmentRLE(**annotation["segmentation"]) TODO fix RLE masks
                continue
            else:
                segment = COCODatapointSegmentPolygons(polygons=annotation["segmentation"])

            segment_raw = COCODatapointSegmentRaw(segment=segment,
                                                  objectType=categories[annotation["category_id"]])

            image_id = annotation["image_id"]
            if image_id not in segmentations:
                segmentations[image_id] = [segment_raw]
            else:
                segmentations[image_id].append(segment_raw)

        return segmentations

    def _load_bg_descriptions(self) -> Dict[int, str]:
        bg_descriptions: Dict[int, str] = {}

        for bg_filename in os.listdir(self.bg_description_dir_path):

            if bg_filename.endswith('.txt'):
                bg_filepath: str = os.path.join(self.bg_description_dir_path, bg_filename)

                with open(bg_filepath, 'r') as bg_file:
                    bg_desc: str = bg_file.read()

                bg_desc = bg_desc if bg_desc[-1] != '\n' else bg_desc[:-1]
                bg_descriptions[COCODataset.filepath_to_id(bg_filename)] = bg_desc

        return bg_descriptions

    @staticmethod
    def filepath_to_id(fp: str) -> int:
        return int(fp.split('/')[-1].split('.')[0])

    def save_bg_description_of(self, datapoint_raw):
        img_id = COCODataset.filepath_to_id(datapoint_raw.image_path)
        with open(os.path.join(self.bg_description_dir_path, "{img_id}.txt".format(img_id=img_id)), 'w') as f:
            f.write(datapoint_raw.bg_desc)

    def save_bg_descriptions(self):
        for datapoint_raw in self.datapoints_raw:
            if datapoint_raw.bg_desc is None:
                continue
            self.save_bg_description_of(datapoint_raw)

    def load(self, use_blip=False, background=False):
        id_to_image_paths = self._load_image_paths()
        id_to_captions = self._load_captions()
        id_to_segmentations = self._load_segmentations()

        if background:
            id_to_bg_descriptions = self._load_bg_descriptions()
            image_ids = id_to_bg_descriptions.keys()
        else:
            id_to_bg_descriptions = None
            image_ids = id_to_segmentations.keys()

        self.datapoints_raw = []
        for image_id in image_ids:
            self.datapoints_raw.append(COCODatapointRaw(
                image_path=id_to_image_paths[image_id],
                caption=id_to_captions[image_id],
                segments=id_to_segmentations[image_id],
                bg_desc=id_to_bg_descriptions[image_id] if id_to_bg_descriptions is not None else None
            ))

        return self

    @staticmethod
    def polygons_to_mask(datapoint_segment: COCODatapointSegmentPolygons, size: Tuple[int]) -> Image:
        mask = np.zeros(size)

        for polygon in datapoint_segment.polygons:
            polygon_pts = np.array(list(zip(polygon[::2], polygon[1::2])), dtype=np.int32)
            cv2.fillPoly(mask, [polygon_pts], (255, 255, 255))

        mask: Image = Image.fromarray(np.uint8(mask))
        return mask

    @staticmethod
    def rle_to_mask(datapoint_segment: COCODatapointSegmentRLE) -> Image:
        mask = np.zeros(datapoint_segment.size[0] * datapoint_segment.size[1])

        position = 0
        for i, c in enumerate(datapoint_segment.counts):
            if i % 2 == 1:
                mask[position:position+c] = 255
            position += c

        mask: Image = Image.fromarray(mask.reshape(datapoint_segment.size, order='F').astype(np.uint8), mode='L')
        return mask

    @staticmethod
    def data_batch_to_image_mask_pairs(data_batch: List[COCODatapoint]):
        image_batch: List[Image] = [datapoint.image for datapoint in data_batch]
        mask_batch: List[Image] = [datapoint.segments[0].mask for datapoint in data_batch]
        return image_batch, mask_batch

    @staticmethod
    def data_batch_to_fg_bg_pairs(data_batch: List[COCODatapoint]):
        fg_batch: List[str] = [datapoint.segments[0].objectType for datapoint in data_batch]
        bg_batch: List[str] = [datapoint.bg_desc for datapoint in data_batch]
        return fg_batch, bg_batch
    
    @staticmethod
    def data_raw_batch_to_image_ids(data_raw_batch: List[COCODatapointRaw]):
        image_ids: List[int] = [COCODataset.filepath_to_id(data_raw.image_path) for data_raw in data_raw_batch]
        return image_ids

    def __len__(self):
        return len(self.datapoints_raw) if self.datapoints_raw is not None else 0

    def __getitem__(self, idx) -> Union[Tuple[COCODatapoint, COCODatapointRaw],
                                        List[Tuple[COCODatapoint, COCODatapointRaw]]]:

        if isinstance(idx, slice):
            items: List[Tuple[COCODatapoint, COCODatapointRaw]] = []
            start = idx.start if (idx.start is not None and idx.start >= 0) else 0
            stop = idx.stop if (idx.stop is not None and idx.stop <= len(self)) else len(self)
            step = idx.step if idx.step is not None else 1
            for i in range(start, stop, step):
                items.append(self.__getitem__(i))
            return items


        datapoint_raw: COCODatapointRaw = self.datapoints_raw[idx]

        image: Image = Image.open(datapoint_raw.image_path)

        segments = []
        for segment_raw in datapoint_raw.segments:

            if isinstance(segment_raw.segment, COCODatapointSegmentPolygons):
                mask: Image = COCODataset.polygons_to_mask(segment_raw.segment, image.size[::-1])
            elif isinstance(segment_raw.segment, COCODatapointSegmentRLE):
                mask: Image = COCODataset.rle_to_mask(segment_raw.segment)
            else:
                raise Exception("Unknown segment type")

            segments.append(COCODatapointSegment(
                mask=mask,
                objectType=segment_raw.objectType
            ))

        caption = datapoint_raw.caption
        bg_desc = datapoint_raw.bg_desc

        return COCODatapoint(
            image=image,
            caption=caption,
            segments=segments,
            bg_desc=bg_desc
        ), datapoint_raw


