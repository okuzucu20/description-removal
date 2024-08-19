import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
from tqdm import tqdm

# Path to COCO dataset annotations and images
ann_file = "coco/annotations/instances_train2017.json"
image_dir = "coco/train2017"

# Load COCO dataset
coco = COCO(ann_file)

# Get all image ids
img_ids = coco.getImgIds()

# Load category names
cats = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

# Prepare a list to store descriptions and masks
data = {}

save_path = "coco/training_data/train_metadata"
os.makedirs(save_path, exist_ok=True)
os.makedirs(f"{save_path}/masks", exist_ok=True)

for img_id in tqdm(img_ids, total=len(img_ids)):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    
    # Read image
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    cnt = 0
    for ann in anns:
        cnt_str = str(cnt).zfill(4)
        mask = coco.annToMask(ann)
        bbox = ann['area']
        ratio = np.sqrt(bbox / (height * width))  
        # Generate description for each mask
        Image.fromarray(mask * 255).save(f"{save_path}/masks/{str(img_id).zfill(12)}_{cnt_str}.png")
        description = {
            'image_id': str(img_id).zfill(12), 
            'fg_id': ann['category_id'],
            'fg_text': cat_id_to_name[ann['category_id']],
            "ratio": ratio
        }
        data[f"{img_id}_{cnt_str}"] = description
        cnt += 1

# Save descriptions and masks to JSON file
with open(f'{save_path}/coco_instance_masks.json', 'w') as f:
    json.dump(data, f)
