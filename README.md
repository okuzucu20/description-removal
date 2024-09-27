# Description of the files
- AlphaCLIP: source code of alpha clip
- eval: image and mask pairs for evaluation data (used in training)
- ip_adapter: source code of ip adapter
- blip_description.json: blip descriptions of each image in training dataset
- **coco_instance_masks_gt_02_extended.json: unused disregard it**
- dataset.py: dataset class
- datatypes.py: dataclasses 
- final.json: final version of the generated metadata file for training
- **final_metadata.json: unused disregard it**
- generate_embeds.py: generates torch tensors of embeddings regarding fg and bg descriptions of unclip
- generate_instance_masks.py: generate instance masks from coco
- gpt_client: file that uses llama to extract bg only description
- inference.py: inference code
- install.sh: install necessary weights for setting up repository
- subspace_projector.py: network used for kernel trick (aykut hoca's idea)
- train.py: training file
- training_config.yaml: training config
- transformer.py: transformer file that is trained for our idea 
- utils.py: utility methods for initialization of networks and etc.

you can find fg and bg descriptions from final.json and I will provide you the masks from google drive.

# TODO:
- generate scene descriptions from blip3(xmem) use prompts regarding erkut hoca's idea (fg is x scene is y generate me a prompt that is not 1-mask region but background)
- generate a training metadata file from those scene descriptions 
- start training from blip3 generations with the following configs:
    - ortho_loss_coeff: 1, use_ortho_loss: True, use_new_subspace_projection: True
    - ortho_loss_coeff: 0.1, use_ortho_loss: True, use_new_subspace_projection: True
    - ortho_loss_coeff: 1, use_ortho_loss: False, use_new_subspace_projection: True
    - ortho_loss_coeff: 0.1, use_ortho_loss: False, use_new_subspace_projection: True
    - ortho_loss_coeff: 1, use_ortho_loss: True, use_new_subspace_projection: False
    - ortho_loss_coeff: 1, use_ortho_loss: True, use_new_subspace_projection: False
    - no orthogonal loss or subspace projection just mse