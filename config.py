import os

# Configurations
tag2text_dict = {
    "swin_14m": {
        "checkpoint_url": "https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth",
        "checkpoint_file": "tag2text_swin_14m.pth",
    }
}

sam_dict = {
    "default": {
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "checkpoint_file": "sam_vit_h_4b8939.pth",
    },
    "vit_h": {
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "checkpoint_file": "sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "checkpoint_file": "sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "checkpoint_file": "sam_vit_b_01ec64.pth",
    },
}

dino_dict = {
    "swinb": {
        "repo_id": "ShilongLiu/GroundingDINO",
        "config_file": "GroundingDINO_SwinB.cfg.py",
        "checkpoint_file": "groundingdino_swinb_cogcoor.pth",
    },
    "swint_ogc": {
        "repo_id": "ShilongLiu/GroundingDINO",
        "config_file": "GroundingDINO_SwinT_OGC.cfg.py",
        "checkpoint_file": "groundingdino_swint_ogc.pth",
    },
}

default_sam = "default"
default_tag2text = "swin_14m"
default_dino = "swint_ogc"

root_dir = os.path.dirname(os.path.abspath(__file__))
weight_dir = "weights"
abs_weight_dir = os.path.join(root_dir, weight_dir)

output_dir = "outputs"

iou_threshold = 0.5
box_threshold = 0.3
text_threshold = 0.25

# filter out attributes and action categories which are difficult to grounding
delete_tag_index = []
for i in range(3012, 3429):
    delete_tag_index.append(i)
