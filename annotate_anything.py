import argparse
import json
import os
import sys
import tempfile

import numpy as np
import supervision as sv
from groundingdino.util.inference import Model as DinoModel
from imutils import paths
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from tqdm import tqdm

sys.path.append("tag2text")

from tag2text.models import tag2text
from config import *
from utils import detect, segment, show_anns, generate_tags


def process(
    tag2text_model,
    grounding_dino_model,
    sam_predictor,
    sam_automask_generator,
    image_path,
    task,
    prompt,
    box_threshold,
    text_threshold,
    iou_threshold,
    device,
    output_dir=None,
    save_mask=False,
):
    detections = None
    metadata = {"image": {}, "annotations": [], "assets": {}}

    if save_mask:
        metadata["assets"]["intermediate_mask"] = []

    try:
        # Load image
        image = Image.open(image_path)
        image_pil = image.convert("RGB")
        image = np.array(image_pil)

        # Extract image metadata
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
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
            detections, _, classes = detect(
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
            box_image = box_annotator.annotate(
                scene=image, detections=detections, labels=labels
            )
            # Save detection image
            if output_dir:
                box_image_path = os.path.join(output_dir, basename + "_detect.png")
                metadata["assets"]["detection"] = box_image_path
                Image.fromarray(box_image).save(box_image_path)

        # Segmentation
        if task in ["auto", "segment"]:
            if detections:
                masks, scores = segment(
                    sam_predictor, image=image, boxes=detections.xyxy
                )
                detections.mask = masks

                mask_annotator = sv.MaskAnnotator()

                mask_image = np.zeros_like(image, dtype=np.uint8)
                mask_image = mask_annotator.annotate(
                    mask_image, detections=detections, opacity=1
                )
                annotated_image = mask_annotator.annotate(
                    box_image, detections=detections
                )
            else:
                masks = sam_automask_generator.generate(image)
                opacity = 0.3
                mask_image, res = show_anns(masks)
                annotated_image = np.uint8(mask_image * opacity + image * (1 - opacity))
                # Save annotation encoding from https://github.com/LUSSeg/ImageNet-S
                mask_enc_path = os.path.join(output_dir, basename + "_mask_enc.npy")
                np.save(mask_enc_path, res)
                metadata["assets"]["mask_enc"] = mask_enc_path

            # Save annotated image
            if output_dir:
                mask_image_path = os.path.join(output_dir, basename + "_mask.png")
                metadata["assets"]["mask"] = mask_image_path
                Image.fromarray(mask_image).save(mask_image_path)

                annotated_image_path = os.path.join(
                    output_dir, basename + "_annotate.png"
                )
                metadata["assets"]["annotate"] = annotated_image_path
                Image.fromarray(annotated_image).save(annotated_image_path)

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
                    "box_confidence": float(confidence),
                    "label": classes[class_id] if class_id else "Unkown",
                }
                if mask is not None:
                    annotation["area"] = int(area)
                    annotation["predicted_iou"] = float(score)
                metadata["annotations"].append(annotation)

                if output_dir and save_mask:
                    mask_image_path = os.path.join(
                        output_dir, f"{basename}_mask_{id}.png"
                    )
                    metadata["assets"]["intermediate_mask"].append(mask_image_path)
                    Image.fromarray(mask * 255).save(mask_image_path)

                id += 1
        else:
            id = 1
            # Auto masking
            for mask in masks:
                bbox = mask["bbox"]
                annotation = {
                    "id": id,
                    "bbox": [
                        bbox[0],
                        bbox[1],
                        bbox[0] + bbox[2],
                        bbox[1] + bbox[3],
                    ],  # Convert from XYWH to XYXY format
                    "box_area": bbox[3] * bbox[2],
                    "area": float(mask["area"]),
                    "predicted_iou": float(mask["predicted_iou"]),
                    "stability_score": float(mask["stability_score"]),
                    "crop_box": list(mask["crop_box"]),
                    "point_coords": list(mask["point_coords"]),
                }
                metadata["annotations"].append(annotation)

                if output_dir and save_mask:
                    mask_image_path = os.path.join(
                        output_dir, f"{basename}_mask_{id}.png"
                    )
                    metadata["assets"]["intermediate_mask"].append(mask_image_path)
                    Image.fromarray(mask * 255).save(mask_image_path)

                id += 1

        if output_dir:
            meta_file_path = os.path.join(output_dir, basename + "_meta.json")
            with open(meta_file_path, "w") as fp:
                json.dump(metadata, fp)
        else:
            meta_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            meta_file_path = meta_file.name

        return meta_file_path
    except Exception as error:
        raise ValueError(f"global exception: {error}")


def main(args: argparse.Namespace) -> None:
    device = args.device
    prompt = args.prompt
    task = args.task

    tag2text_model = None
    grounding_dino_model = None
    sam_predictor = None
    sam_automask_generator = None

    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    save_mask = args.save_mask

    # load model
    if task in ["auto", "detection"] and prompt == "":
        print("Loading Tag2Text model...")
        tag2text_model = tag2text.tag2text_caption(
            pretrained=args.tag2text_checkpoint,
            image_size=384,
            vit="swin_b",
            delete_tag_index=delete_tag_index,
        )
        # threshold for tagging
        # we reduce the threshold to obtain more tags
        tag2text_model.threshold = 0.64
        tag2text_model.to(device)
        tag2text_model.eval()

    if task in ["auto", "detection"] or prompt != "":
        print("Loading Grounding Dino model...")
        grounding_dino_model = DinoModel(
            model_config_path=dino_config_file, model_checkpoint_path=dino_checkpoint
        )

    if task in ["auto", "segment"]:
        print("Loading SAM...")
        sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        sam_automask_generator = SamAutomaticMaskGenerator(sam)

    if not os.path.exists(args.input):
        raise ValueError("The input directory doesn't exist!")
    elif not os.path.isdir(args.input):
        image_paths = [args.input]
    else:
        image_paths = paths.list_images(args.input)

    os.makedirs(args.output, exist_ok=True)

    with tqdm(image_paths) as pbar:
        for image_path in pbar:
            pbar.set_postfix_str(f"Processing {image_path}")
            process(
                tag2text_model=tag2text_model,
                grounding_dino_model=grounding_dino_model,
                sam_predictor=sam_predictor,
                sam_automask_generator=sam_automask_generator,
                image_path=image_path,
                task=task,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                iou_threshold=iou_threshold,
                device=device,
                output_dir=args.output,
                save_mask=save_mask,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic detection and mask generation on an input image or directory of images"
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help=(
            "Path to the directory where masks will be output. Output will be either a folder "
            "of PNGs per image or a single json with COCO-style masks."
        ),
    )

    parser.add_argument(
        "--sam-type",
        type=str,
        default="default",
        help="The type of SA model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=sam_checkpoint,
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument(
        "--tag2text-checkpoint",
        type=str,
        default=tag2text_checkpoint,
        help="The path to the Tag2Text checkpoint to use for tags and caption generation.",
    )

    parser.add_argument(
        "--dino-config",
        type=str,
        default=dino_config_file,
        help="The config file of Grounding Dino model to load",
    )

    parser.add_argument(
        "--dino-checkpoint",
        type=str,
        default=dino_checkpoint,
        help="The path to the Grounding Dino checkpoint to use for detection.",
    )
    parser.add_argument(
        "--task",
        help="Task to run",
        default="auto",
        choices=["auto", "detect", "segment"],
        type=str,
    )
    parser.add_argument(
        "--prompt",
        help="Detection prompt",
        default="",
        type=str,
    )

    parser.add_argument(
        "--box-threshold", type=float, default=0.25, help="box threshold"
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.2, help="text threshold"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="iou threshold"
    )

    parser.add_argument(
        "--save-mask",
        action="store_true",
        default=False,
        help="If True, save all intermidiate masks.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to run generation on."
    )
    args = parser.parse_args()
    main(args)
