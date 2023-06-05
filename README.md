# Annotate Anything

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/binh234/annotate-anything/blob/main/notebooks/Annotate_Anything.ipynb)

![demo](images/demo.png)

![demo_seg](images/demo_seg.png)

![demo_horse](images/demo_horse.png)

## Install libraries

**Important notes**: To install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) with GPU, CUDA_HOME environment variable must be set.

```bash
pip install -r requirements.txt
```

## Download weights

```bash
python download_weights.py
```

## Run Gradio App

```bash
gradio run app.py
```

## Command-line options

```bash
python annotate_anything.py -i examples -o outputs --task segment
```

### Annotate anything

Runs automatic detection and mask generation on an input image or directory of images

| Flag                                     | Description                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `-h`, `--help`                           | Show help message and exit                                                                                                                 |
| `--input INPUT`, `-i INPUT`              | Path to either a single input image or folder of images.                                                                                   |
| `--output OUTPUT`, `-o OUTPUT`           | Path to the directory where masks will be output. Output will be either a folder of PNGs per image or a single JSON with COCO-style masks. |
| `--sam-type {default,vit_h,vit_l,vit_b}` | The type of SA model use for segmentation.                                                                                                 |
| `--tag2text-type {swin_14m}`             | The type of Tag2Text model use for tags and caption generation.                                                                            |
| `--dino-type {swinb,swint_ogc}`          | The type of Grounding Dino model use for promptable object detection.                                                                      |
| `--task {auto,detect,segment}`           | Task to run. Possible values: `auto`, `detect`, `segment`                                                                                  |
| `--prompt PROMPT`                        | Detection prompt                                                                                                                           |
| `--box-threshold BOX_THRESHOLD`          | Box threshold                                                                                                                              |
| `--text-threshold TEXT_THRESHOLD`        | Text threshold                                                                                                                             |
| `--iou-threshold IOU_THRESHOLD`          | IoU threshold                                                                                                                              |
| `--save-mask`                            | If True, save all intermediate masks.                                                                                                      |
| `--device DEVICE`                        | The device to run generation on.                                                                                                           |

### Automatic mask generation

Runs automatic mask generation on an input image or directory of images, and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, as well as pycocotools if saving in RLE format.

#### Basic settings

| Flag                           | Description                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `-h`, `--help`                 | Show help message and exit.                                                                                                                |
| `--input INPUT`, `-i INPUT`    | Path to either a single input image or folder of images.                                                                                   |
| `--output OUTPUT`, `-o OUTPUT` | Path to the directory where masks will be output. Output will be either a folder of PNGs per image or a single JSON with COCO-style masks. |
| `--model-type MODEL_TYPE`      | The type of model to load. Possible values: `default`, `vit_h`, `vit_l`, `vit_b`.                                                          |
| `--checkpoint CHECKPOINT`      | The path to the SAM checkpoint to use for mask generation.                                                                                 |
| `--device DEVICE`              | The device to run generation on.                                                                                                           |
| `--convert-to-rle`             | Save masks as COCO RLEs in a single JSON instead of as a folder of PNGs. Requires pycocotools.                                             |

#### AMG settings

| Flag                                                              | Description                                                                                                       |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--points-per-side POINTS_PER_SIDE`                               | Generate masks by sampling a grid over the image with this many points to a side.                                 |
| `--points-per-batch POINTS_PER_BATCH`                             | How many input points to process simultaneously in one batch.                                                     |
| `--pred-iou-thresh PRED_IOU_THRESH`                               | Exclude masks with a predicted score from the model that is lower than this threshold.                            |
| `--stability-score-thresh STABILITY_SCORE_THRESH`                 | Exclude masks with a stability score lower than this threshold.                                                   |
| `--stability-score-offset STABILITY_SCORE_OFFSET`                 | Larger values perturb the mask more when measuring stability score.                                               |
| `--box-nms-thresh BOX_NMS_THRESH`                                 | The overlap threshold for excluding a duplicate mask.                                                             |
| `--crop-n-layers CROP_N_LAYERS`                                   | If >0, mask generation is run on smaller crops of the image to generate more masks.                               |
| `--crop-nms-thresh CROP_NMS_THRESH`                               | The overlap threshold for excluding duplicate masks across different crops.                                       |
| `--crop-overlap-ratio CROP_OVERLAP_RATIO`                         | Larger numbers mean image crops will overlap more.                                                                |
| `--crop-n-points-downscale-factor CROP_N_POINTS_DOWNSCALE_FACTOR` | The number of points-per-side in each layer of crop is reduced by this factor.                                    |
| `--min-mask-region-area MIN_MASK_REGION_AREA`                     | Disconnected mask regions or holes with an area smaller than this value in pixels are removed by post-processing. |

## Metadata file

The metadata file will contain the following information:

```text
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
    "caption"               : str,              # Image caption
    "tags"                  : [str],            # Image tags
}

annotation {
    "id"                    : int,              # Annotation id
    "bbox"                  : [x1, y1, x2, y2],     # The box around the mask, in XYXY format
    "area"                  : int,              # The area in pixels of the mask
    "box_area"              : float,            # The area in pixels of the bounding box
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "confidence"            : float,            # A measure of the prediction confidency
    "label"                 : str,              # Predicted class for the object inside the bounding box (if exist)
}
```

## References

[Segment Anything](https://github.com/facebookresearch/segment-anything)
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
[Tag2Text](https://github.com/xinyu1205/Tag2Text)
[Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
