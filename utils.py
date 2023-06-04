import random
import sys
from typing import Dict
from typing import List

import numpy as np
import supervision as sv
import torch
import torchvision
import torchvision.transforms as T
from groundingdino.models import build_model
from groundingdino.util.inference import Model as DinoModel
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from segment_anything import SamPredictor

# segment anything

sys.path.append("tag2text")

from tag2text.inference import inference as tag2text_inference


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def download_file_hf(repo_id, filename, cache_dir="./cache"):
    cache_file = hf_hub_download(
        repo_id=repo_id, filename=filename, force_filename=filename, cache_dir=cache_dir
    )
    return cache_file


def transform_image_tag2text(image_pil: Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image_pil)  # 3, h, w
    return image


def show_anns_sam(anns: List[Dict]):
    """Extracts the mask annotations from the Segment Anything model output and plots them.
    https://github.com/facebookresearch/segment-anything.

    Arguments:
      anns (List[Dict]): Segment Anything model output.

    Returns:
      (np.ndarray): Masked image.
      (np.ndarray): annotation encoding from https://github.com/LUSSeg/ImageNet-S
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann["segmentation"]
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img * 255

    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = np.uint8(full_img)
    return full_img, res


def show_anns_sv(detections: sv.Detections):
    """Extracts the mask annotations from the Supervision Detections object.
    https://roboflow.github.io/supervision/detection/core/.

    Arguments:
      anns (sv.Detections): Containing information about the detections.

    Returns:
      (np.ndarray): Masked image.
      (np.ndarray): annotation encoding from https://github.com/LUSSeg/ImageNet-S
    """
    if detections.mask is None:
        return
    full_img = None

    for i in np.flip(np.argsort(detections.area)):
        m = detections.mask[i]
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img * 255

    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = np.uint8(full_img)
    return full_img, res


def generate_tags(tag2text_model, image, specified_tags, device="cpu"):
    """Generate image tags and caption using Tag2Text model.

    Arguments:
      tag2text_model (nn.Module): Tag2Text model to use for prediction.
      image (np.ndarray): The image for calculating. Expects an
        image in HWC uint8 format, with pixel values in [0, 255].
      specified_tags(str): User input specified tags

    Returns:
      (List[str]): Predicted image tags.
      (str): Predicted image caption
    """
    image = transform_image_tag2text(image).unsqueeze(0).to(device)
    res = tag2text_inference(image, tag2text_model, specified_tags)
    tags = res[0].split(" | ")
    caption = res[2]
    return tags, caption


def detect(
    grounding_dino_model: DinoModel,
    image: np.ndarray,
    caption: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    post_process: bool = True,
):
    """Detect bounding boxes for the given image, using the input caption.

    Arguments:
      grounding_dino_model (DinoModel): The model to use for detection.
      image (np.ndarray): The image for calculating masks. Expects an
        image in HWC uint8 format, with pixel values in [0, 255].
      caption (str): Input caption contain object names to detect. To detect multiple objects, seperating each name with '.', like this: cat . dog . chair
      box_threshold (float): Box confidence threshold
      text_threshold (float): Text confidence threshold
      iou_threshold (float): IOU score threshold for post processing
      post_process (bool): If True, run NMS algorithm to remove duplicates segments.

    Returns:
      (sv.Detections): Containing information about the detections in a video frame.
      (str): Predicted phrases.
      (List[str]): Predicted classes.
    """
    detections, phrases = grounding_dino_model.predict_with_caption(
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    classes = list(map(lambda x: x.strip(), caption.split(".")))
    detections.class_id = DinoModel.phrases2classes(phrases=phrases, classes=classes)

    # NMS post process
    if post_process:
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                iou_threshold,
            )
            .numpy()
            .tolist()
        )

        phrases = [phrases[idx] for idx in nms_idx]
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

    return detections, phrases, classes


def segment(sam_model: SamPredictor, image: np.ndarray, boxes: np.ndarray):
    """Predict masks for the given input boxes, using the currently set image.

    Arguments:
      sam_model (SamPredictor): The model to use for mask prediction.
      image (np.ndarray): The image for calculating masks. Expects an
        image in HWC uint8 format, with pixel values in [0, 255].
      boxes (np.ndarray or None): A Bx4 array given a box prompt to the
        model, in XYXY format.
      return_logits (bool): If true, returns un-thresholded masks logits
        instead of a binary mask.

    Returns:
      (torch.Tensor): The output masks in BxCxHxW format, where C is the
        number of masks, and (H, W) is the original image size.
      (torch.Tensor): An array of shape BxC containing the model's
        predictions for the quality of each mask.
      (torch.Tensor): An array of shape BxCxHxW, where C is the number
        of masks and H=W=256. These low res logits can be passed to
        a subsequent iteration as mask input.
    """
    sam_model.set_image(image)
    transformed_boxes = None
    if boxes is not None:
        boxes = torch.from_numpy(boxes)

        transformed_boxes = sam_model.transform.apply_boxes_torch(
            boxes.to(sam_model.device), image.shape[:2]
        )

    masks, scores, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks = masks[:, 0, :, :]
    scores = scores[:, 0]
    return masks.cpu().numpy(), scores.cpu().numpy()


def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
