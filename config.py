# Configurations
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
