import json
import os
import sys
import tempfile

import gradio as gr
import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import Model as DinoModel
from PIL import Image
from segment_anything import build_sam
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from supervision.detection.utils import mask_to_polygons
from supervision.detection.utils import xywh_to_xyxy

# segment anything
# Grounding DINO

sys.path.append("tag2text")

from tag2text.models import tag2text
from config import *
from utils import download_file_hf, detect, segment, show_anns, generate_tags

if not os.path.exists(abs_weight_dir):
    os.makedirs(abs_weight_dir, exist_ok=True)

sam_checkpoint = os.path.join(abs_weight_dir, sam_dict[default_sam]["checkpoint_file"])
if not os.path.exists(sam_checkpoint):
    os.system(f"wget {sam_dict[default_sam]['checkpoint_url']} -O {sam_checkpoint}")

tag2text_checkpoint = os.path.join(
    abs_weight_dir, tag2text_dict[default_tag2text]["checkpoint_file"]
)
if not os.path.exists(tag2text_checkpoint):
    os.system(
        f"wget {tag2text_dict[default_tag2text]['checkpoint_url']} -O {tag2text_checkpoint}"
    )

dino_checkpoint = os.path.join(
    abs_weight_dir, dino_dict[default_dino]["checkpoint_file"]
)
dino_config_file = os.path.join(abs_weight_dir, dino_dict[default_dino]["config_file"])
if not os.path.exists(dino_checkpoint):
    dino_repo_id = dino_dict[default_dino]["repo_id"]
    download_file_hf(
        repo_id=dino_repo_id,
        filename=dino_dict[default_dino]["config_file"],
        cache_dir=weight_dir,
    )
    download_file_hf(
        repo_id=dino_repo_id,
        filename=dino_dict[default_dino]["checkpoint_file"],
        cache_dir=weight_dir,
    )

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tag2text_model = tag2text.tag2text_caption(
    pretrained=tag2text_checkpoint,
    image_size=384,
    vit="swin_b",
    delete_tag_index=delete_tag_index,
)
# threshold for tagging
# we reduce the threshold to obtain more tags
tag2text_model.threshold = 0.64
tag2text_model.to(device)
tag2text_model.eval()


sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)
sam_automask_generator = SamAutomaticMaskGenerator(sam)

grounding_dino_model = DinoModel(
    model_config_path=dino_config_file, model_checkpoint_path=dino_checkpoint
)


def process(image_path, task, prompt, box_threshold, text_threshold, iou_threshold):
    global tag2text_model, sam_predictor, sam_automask_generator, grounding_dino_model, device
    output_gallery = []
    detections = None
    metadata = {"image": {}, "annotations": []}

    try:
        # Load image
        image = Image.open(image_path)
        image_pil = image.convert("RGB")
        image = np.array(image_pil)

        # Extract image metadata
        filename = os.path.basename(image_path)
        h, w = image.shape[:2]
        metadata["image"]["file_name"] = filename
        metadata["image"]["width"] = w
        metadata["image"]["height"] = h

        # Generate tags
        if task in ["auto", "detection"] and prompt == "":
            tags, caption = generate_tags(tag2text_model, image_pil, "None", device)
            prompt = " . ".join(tags)
            print(f"Caption: {caption}")
            print(f"Tags: {tags}")

            # ToDo: Extract metadata
            metadata["image"]["caption"] = caption
            metadata["image"]["tags"] = tags

        if prompt:
            metadata["prompt"] = prompt
            print(f"Prompt: {prompt}")

        # Detect boxes
        if prompt != "":
            detections, phrases, classes = detect(
                grounding_dino_model,
                image,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                iou_threshold=iou_threshold,
                post_process=True,
            )

            # Draw boxes
            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{classes[class_id] if class_id else 'Unkown'} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in detections
            ]
            image = box_annotator.annotate(
                scene=image, detections=detections, labels=labels
            )
            output_gallery.append(image)

        # Segmentation
        if task in ["auto", "segment"]:
            if detections:
                masks, scores = segment(
                    sam_predictor, image=image, boxes=detections.xyxy
                )
                detections.mask = masks
            else:
                masks = sam_automask_generator.generate(image)
                sorted_generated_masks = sorted(
                    masks, key=lambda x: x["area"], reverse=True
                )

                xywh = np.array([mask["bbox"] for mask in sorted_generated_masks])
                mask = np.array(
                    [mask["segmentation"] for mask in sorted_generated_masks]
                )
                scores = np.array(
                    [mask["predicted_iou"] for mask in sorted_generated_masks]
                )
                detections = sv.Detections(
                    xyxy=xywh_to_xyxy(boxes_xywh=xywh), mask=mask
                )
                # opacity = 0.4
                # mask_image, _ = show_anns_sam(masks)
                # annotated_image = np.uint8(mask_image * opacity + image * (1 - opacity))

            mask_annotator = sv.MaskAnnotator()
            mask_image = np.zeros_like(image, dtype=np.uint8)
            mask_image = mask_annotator.annotate(
                mask_image, detections=detections, opacity=1
            )
            annotated_image = mask_annotator.annotate(image, detections=detections)
            output_gallery.append(mask_image)
            output_gallery.append(annotated_image)

        # ToDo: Extract metadata
        if detections:
            id = 1
            for (xyxy, mask, confidence, class_id, _), area, box_area, score in zip(
                detections, detections.area, detections.box_area, scores
            ):
                annotation = {
                    "id": id,
                    "bbox": [int(x) for x in xyxy],
                    "box_area": float(box_area),
                }
                if class_id:
                    annotation["box_confidence"] = float(confidence)
                    annotation["label"] = classes[class_id] if class_id else "Unkown"
                if mask is not None:
                    # annotation["segmentation"] = mask_to_polygons(mask)
                    annotation["area"] = int(area)
                    annotation["predicted_iou"] = float(score)
                metadata["annotations"].append(annotation)
                id += 1

        meta_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        meta_file_path = meta_file.name
        with open(meta_file_path, "w") as fp:
            json.dump(metadata, fp)

        return output_gallery, meta_file_path
    except Exception as error:
        raise gr.Error(f"global exception: {error}")


title = "Annotate Anything"

with gr.Blocks(css="style.css", title=title) as demo:
    with gr.Row(elem_classes=["container"]):
        with gr.Column(scale=1):
            input_image = gr.Image(type="filepath", label="Input")
            task = gr.Dropdown(
                ["detect", "segment", "auto"], value="auto", label="task_type"
            )
            text_prompt = gr.Textbox(label="Detection Prompt")
            with gr.Accordion("Advanced parameters", open=False):
                box_threshold = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.3,
                    step=0.05,
                    label="Box threshold",
                    info="Hash size to use for image hashing",
                )
                text_threshold = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.25,
                    step=0.05,
                    label="Text threshold",
                    info="Number of history images used to find out duplicate image",
                )
                iou_threshold = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.05,
                    label="IOU threshold",
                    info="Minimum similarity threshold (in percent) to consider 2 images to be similar",
                )
            run_button = gr.Button(label="Run")

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            ).style(preview=True, grid=2, object_fit="scale-down")
            meta_file = gr.File(label="Metadata file")

    gr.Examples(
        [
            ["examples/dog.png", "auto", ""],
            ["examples/eiffel.png", "auto", ""],
            ["examples/eiffel.png", "segment", ""],
            ["examples/girl.png", "auto", "girl . face"],
            ["examples/horse.png", "detect", "horse"],
            ["examples/horses.jpg", "auto", "horse"],
            ["examples/traffic.jpg", "auto", ""],
        ],
        [input_image, task, text_prompt],
    )
    run_button.click(
        fn=process,
        inputs=[
            input_image,
            task,
            text_prompt,
            box_threshold,
            text_threshold,
            iou_threshold,
        ],
        outputs=[gallery, meta_file],
    )

demo.queue(concurrency_count=2).launch()
