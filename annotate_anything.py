import argparse
import functools
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import supervision as sv
from groundingdino.util.inference import Model as DinoModel
from imutils import paths
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from supervision.detection.utils import xywh_to_xyxy
from tqdm import tqdm

sys.path.append("tag2text")

from tag2text.models import tag2text
from config import *
from utils import detect, download_file_hf, segment, generate_tags, show_anns_sv


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
    kernel_size=2,
    expand_mask=False,
    device="cuda",
    output_dir=None,
    save_ann=True,
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
        orig_image = image.copy()

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
            # print(f"Caption: {caption}")
            # print(f"Tags: {tags}")

            # ToDo: Extract metadata
            metadata["image"]["caption"] = caption
            metadata["image"]["tags"] = tags

        if prompt:
            metadata["prompt"] = prompt

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

            # Save detection image
            if output_dir and save_ann:
                # Draw boxes
                box_annotator = sv.BoxAnnotator()
                labels = [
                    f"{phrases[i]} {detections.confidence[i]:0.2f}"
                    for i in range(len(phrases))
                ]
                box_image = box_annotator.annotate(
                    scene=image, detections=detections, labels=labels
                )
                box_image_path = os.path.join(output_dir, basename + "_detect.png")
                metadata["assets"]["detection"] = box_image_path
                Image.fromarray(box_image).save(box_image_path)

        # Segmentation
        if task in ["auto", "segment"]:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1)
            )
            if detections:
                masks, scores = segment(
                    sam_predictor, image=orig_image, boxes=detections.xyxy
                )
                if expand_mask:
                    masks = [
                        cv2.dilate(mask.astype(np.uint8), kernel) for mask in masks
                    ]
                else:
                    masks = [
                        cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                        for mask in masks
                    ]
                detections.mask = masks
                binary_mask = functools.reduce(
                    lambda x, y: x + y, detections.mask
                ).astype(np.bool)
            else:
                masks = sam_automask_generator.generate(orig_image)
                sorted_generated_masks = sorted(
                    masks, key=lambda x: x["area"], reverse=True
                )

                xywh = np.array([mask["bbox"] for mask in sorted_generated_masks])
                scores = np.array(
                    [mask["predicted_iou"] for mask in sorted_generated_masks]
                )
                if expand_mask:
                    mask = np.array(
                        [
                            cv2.dilate(mask["segmentation"].astype(np.uint8), kernel)
                            for mask in sorted_generated_masks
                        ]
                    )
                else:
                    mask = np.array(
                        [mask["segmentation"] for mask in sorted_generated_masks]
                    )
                detections = sv.Detections(
                    xyxy=xywh_to_xyxy(boxes_xywh=xywh), mask=mask
                )
                binary_mask = None

            # Save annotated image
            if output_dir and save_ann:
                mask_annotator = sv.MaskAnnotator()
                mask_image, res = show_anns_sv(detections)
                annotated_image = mask_annotator.annotate(image, detections=detections)

                mask_image_path = os.path.join(output_dir, basename + "_mask.png")
                metadata["assets"]["mask"] = mask_image_path
                Image.fromarray(mask_image).save(mask_image_path)

                # Save annotation encoding from https://github.com/LUSSeg/ImageNet-S
                mask_enc_path = os.path.join(output_dir, basename + "_mask_enc.npy")
                np.save(mask_enc_path, res)
                metadata["assets"]["mask_enc"] = mask_enc_path

                if binary_mask is not None:
                    cutout_image = np.expand_dims(binary_mask, axis=-1) * orig_image
                    cutout_image_path = os.path.join(
                        output_dir, basename + "_cutout.png"
                    )
                    Image.fromarray(cutout_image).save(cutout_image_path)

                annotated_image_path = os.path.join(
                    output_dir, basename + "_annotate.png"
                )
                metadata["assets"]["annotate"] = annotated_image_path
                Image.fromarray(annotated_image).save(annotated_image_path)

        # ToDo: Extract metadata
        if detections:
            i = 0
            for (xyxy, mask, confidence, _, _), area, box_area in zip(
                detections, detections.area, detections.box_area
            ):
                annotation = {
                    "id": i + 1,
                    "bbox": [int(x) for x in xyxy],
                    "box_area": float(box_area),
                }
                if confidence:
                    annotation["confidence"] = float(confidence)
                    annotation["label"] = phrases[i]
                if mask is not None:
                    # annotation["segmentation"] = mask_to_polygons(mask)
                    annotation["area"] = int(area)
                    annotation["predicted_iou"] = float(scores[i])
                metadata["annotations"].append(annotation)
                i += 1

                if output_dir and save_mask:
                    mask_image_path = os.path.join(
                        output_dir, f"{basename}_mask_{id}.png"
                    )
                    metadata["assets"]["intermediate_mask"].append(mask_image_path)
                    Image.fromarray(mask * 255).save(mask_image_path)

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
    save_ann = not args.no_save_ann
    save_mask = args.save_mask

    # load model
    if task in ["auto", "detection"] and prompt == "":
        print("Loading Tag2Text model...")
        tag2text_type = args.tag2text_type
        tag2text_checkpoint = os.path.join(
            abs_weight_dir, tag2text_dict[tag2text_type]["checkpoint_file"]
        )
        if not os.path.exists(tag2text_checkpoint):
            print(f"Downloading weights for Tag2Text {tag2text_type} model")
            os.system(
                f"wget {tag2text_dict[tag2text_type]['checkpoint_url']} -O {tag2text_checkpoint}"
            )
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

    if task in ["auto", "detection"] or prompt != "":
        print("Loading Grounding Dino model...")
        dino_type = args.dino_type
        dino_checkpoint = os.path.join(
            abs_weight_dir, dino_dict[dino_type]["checkpoint_file"]
        )
        dino_config_file = os.path.join(
            abs_weight_dir, dino_dict[dino_type]["config_file"]
        )
        if not os.path.exists(dino_checkpoint):
            print(f"Downloading weights for Grounding Dino {dino_type} model")
            dino_repo_id = dino_dict[dino_type]["repo_id"]
            download_file_hf(
                repo_id=dino_repo_id,
                filename=dino_dict[dino_type]["checkpoint_file"],
                cache_dir=weight_dir,
            )
            download_file_hf(
                repo_id=dino_repo_id,
                filename=dino_dict[dino_type]["checkpoint_file"],
                cache_dir=weight_dir,
            )
        grounding_dino_model = DinoModel(
            model_config_path=dino_config_file,
            model_checkpoint_path=dino_checkpoint,
            device=device,
        )

    if task in ["auto", "segment"]:
        print("Loading SAM...")
        sam_type = args.sam_type
        sam_checkpoint = os.path.join(
            abs_weight_dir, sam_dict[sam_type]["checkpoint_file"]
        )
        if not os.path.exists(sam_checkpoint):
            print(f"Downloading weights for SAM {sam_type}")
            os.system(
                f"wget {sam_dict[sam_type]['checkpoint_url']} -O {sam_checkpoint}"
            )
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
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
                save_ann=save_ann,
                save_mask=save_mask,
            )


if __name__ == "__main__":
    if not os.path.exists(abs_weight_dir):
        os.makedirs(abs_weight_dir, exist_ok=True)

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
        help="Path to the directory where masks will be output.",
    )

    parser.add_argument(
        "--sam-type",
        type=str,
        default=default_sam,
        choices=sam_dict.keys(),
        help="The type of SA model use for segmentation.",
    )

    parser.add_argument(
        "--tag2text-type",
        type=str,
        default=default_tag2text,
        choices=tag2text_dict.keys(),
        help="The type of Tag2Text model use for tags and caption generation.",
    )

    parser.add_argument(
        "--dino-type",
        type=str,
        default=default_dino,
        choices=dino_dict.keys(),
        help="The type of Grounding Dino model use for promptable object detection.",
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
        "--kernel-size",
        type=int,
        default=2,
        choices=range(1, 6),
        help="kernel size use for smoothing/expanding segment masks",
    )
    parser.add_argument(
        "--expand-mask",
        action="store_true",
        default=False,
        help="If True, expanding segment masks for smoother output.",
    )

    parser.add_argument(
        "--no-save-ann",
        action="store_true",
        default=False,
        help="If False, save original image with blended masks and detection boxes.",
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
