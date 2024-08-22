import transformers
import torch
from huggingface_hub import login
import json
from tqdm import tqdm

def extract_bg_desc(scene_desc, fg_object, pipeline):
    messages = [
    {"role": "system", "content": "You are an assistant that modifies a scene description. "
                                "You will be given a scene description and a foreground object. "
                                "The scene description will be provided to you after this text: 'Scene description: ' "
                                "The foreground object will be provided to you after this text: 'Foreground object: ' "
                                "You will remove the foreground object from the scene description. "
                                "After the modification, only the background information must be present in the description. "
                                "The only reply you give will be the new scene description where the foreground object is not present. "
                                "You will strictly not give any other response other than the new scene description."
                                "If there is no foreground object in the scene description, you will reply with the same scene description."
                                "If there are synonyms or similar words for the foreground object, you will remove them as well."
                                "You will not change the scene description in any other way. "
                                "You will not add any new information to the scene description. "
                                },
    {"role": "user", "content": "Scene description: Bear wanders in a forest. Foreground object: bear"},
    {"role": "assistant", "content": "A forest"},
    {"role": "user", "content": "Scene description: A person sits on a chair inside a cafeteria. Foreground object: person"},
    {"role": "assistant", "content": "A cafeteria with chairs inside"},
    {"role": "user", "content": "Scene description: People playing basketball on a court. Foreground object: basketball"},
    {"role": "assistant", "content": "A basketball court with people in it"},
    {"role": "user", "content": "Scene description: a boy on a skateboard doing a trick. Foreground object: person"},
    {"role": "assistant", "content": "A skateboard doing a trick"},
    {"role": "user", "content": "Scene description: a man running on a track. Foreground object: bicycle"},
    {"role": "assistant", "content": "a man running on a track"},
    {"role": "user", "content": f"Scene description: {scene_desc}. Foreground object: {fg_object}."}
    ] 
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]["content"]

JSON_FILE = "/kuacc/users/hpc-yekin/hpc_run/description-removal/coco_instance_masks_gt_02_extended.json"
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

login(token="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")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

new_data = {}
partition_data_index = 7
items_per_partition = (190258 // 8) + 1
partition_start = partition_data_index * items_per_partition
partition_end = (partition_data_index + 1) * items_per_partition
if partition_end > 190258:
    partition_end = 190258
with open(JSON_FILE, "r") as f:
    data = json.load(f)
    partition_data = list(data.items())
    sorted_partition_data = sorted(partition_data, key=lambda x: x[0])
    sorted_partition_data = sorted_partition_data[partition_start:partition_end]
    cnt = 0
    for key, value in tqdm(sorted_partition_data):
        if key != "count" and data[key]["ratio"] > 0.2 and data[key]["ratio"] < 0.5:
            image_id = data[key]["image_id"]  
            scene_desc = data[key]["scene_desc"]
            fg_text = data[key]["fg_text"]
            fg_id = data[key]["fg_id"]
            ratio = data[key]["ratio"]
            bg_text = extract_bg_desc(scene_desc, fg_text, pipeline)
            
            new_data[key] = {
                "image_id": image_id,
                "fg_id": fg_id,
                "fg_text": fg_text,
                "bg_text": bg_text,
                "scene_desc": scene_desc,
                "ratio": ratio
            }
            print(f"Image ID: {image_id}, FG ID: {fg_id}, FG Text: {fg_text}, BG Text: {bg_text}, Scene Desc: {scene_desc}, Ratio: {ratio}")
            cnt += 1
        
    with open(f"final_metadata_{partition_data_index}.json", "w") as f:
        json.dump(new_data, f)
        print(f"Saved {cnt} items to final_metadata.json")