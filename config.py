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

weight_dir = "weights"
tag2text_checkpoint = "tag2text_swin_14m.pth"
tag2text_url = "https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth"
sam_checkpoint = "sam_vit_h_4b8939.pth"
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
output_dir = "outputs"

dino_config_file = "GroundingDINO_SwinB.cfg.py"
dino_repo_id = "ShilongLiu/GroundingDINO"
dino_checkpoint = "groundingdino_swinb_cogcoor.pth"

iou_threshold = 0.5
box_threshold = 0.3
text_threshold = 0.25

# filter out attributes and action categories which are difficult to grounding
delete_tag_index = []
for i in range(3012, 3429):
    delete_tag_index.append(i)
