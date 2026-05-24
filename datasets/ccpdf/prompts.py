# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY NVIDIA CORPORATION AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Disclaimer:
For each dataset an user elects to use, the user is responsible for checking if the dataset license
is fit for the intended purpose.
"""

import json
import logging
import random
import re
from math import ceil
from pathlib import Path

import albumentations as A
import click
import cv2
import numpy as np
from packaging import version
from parallel import ProcessBound, process_generator
from PIL import Image, ImageDraw
from tqdm import tqdm

logger = logging.getLogger(__name__)

assert version.parse(A.__version__) < version.parse("1.4.0"), (
    f"albumentations version {A.__version__} detected. "
    f"This code requires albumentations < 1.4.0 (recommend 1.3.x). "
    f"Install with: pip install 'albumentations>=1.3.0,<1.4.0'"
)


_re_newlines = re.compile(r"\n\n*", re.DOTALL)
_re_fix_dots1 = re.compile(r"(?:\s*\.\s*){3,}", re.DOTALL)
_re_fix_dots2 = re.compile(r"\.{6,}", re.DOTALL)


class Erosion(A.ImageOnlyTransform):
    """Apply morphological erosion to the image."""

    def __init__(self, scale, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        if isinstance(scale, (tuple, list)):
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2)))
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(A.ImageOnlyTransform):
    """Apply morphological dilation to the image."""

    def __init__(self, scale, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        if isinstance(scale, (tuple, list)):
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2)))
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(A.ImageOnlyTransform):
    """Threshold image to create bitmap effect."""

    def __init__(
        self,
        value: int = 0,
        lower: int = 200,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img


class RandomRotateOnly90(A.RandomRotate90):
    """Rotate only by 90 or 270 degrees (not 0 or 180)."""

    def get_params(self):
        return {"factor": random.choice([1, 3])}


class LongestMaxSizeHW(A.DualTransform):
    """Resize image so that longest side doesn't exceed max while preserving aspect ratio."""

    def __init__(
        self,
        max_size_height: int | list[int] = 1024,
        max_size_width: int | list[int] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size_height = max_size_height
        self.max_size_width = max_size_width

    def apply(self, img: np.ndarray, interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        max_size_height = (
            self.max_size_height if isinstance(self.max_size_height, int) else random.choice(self.max_size_height)
        )
        max_size_width = (
            self.max_size_width if isinstance(self.max_size_width, int) else random.choice(self.max_size_width)
        )

        height, width = img.shape[:2]
        aspect_ratio = width / height

        new_height = height
        new_width = width

        if height > max_size_height:
            new_height = max_size_height
            new_width = int(new_height * aspect_ratio)

        if new_width > max_size_width:
            new_width = max_size_width
            new_height = int(new_width / aspect_ratio)

        return A.geometric.functional.resize(img, height=new_height, width=new_width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args_names(self):
        return ("max_size_height", "max_size_width", "interpolation")


def _alb_wrapper(transform):
    """Wrap albumentations transform for PIL images."""

    def f(im):
        return transform(image=np.asarray(im))["image"]

    return f


PHOTOMETRIC_TRANSFORM = _alb_wrapper(
    A.Compose(
        [
            A.OneOf([Erosion((1, 2)), Dilation((1, 2))], p=0.25),
            Bitmap(p=0.25),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, p=0.1),
                    A.Emboss(alpha=(0.1, 0.3), strength=(0.2, 0.3), p=0.25),
                ]
            ),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.25),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.25),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.25),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.25),
            A.Posterize(num_bits=4, p=0.25),
            A.InvertImg(p=0.25),
            A.PixelDropout(dropout_prob=0.05, p=0.25),
        ]
    )
)

TRAIN_AFFINE_TRANSFORM = A.Compose(
    [
        RandomRotateOnly90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit_x=(0, 0.04),
            shift_limit_y=(0, 0.03),
            scale_limit=(-0.15, 0.03),
            rotate_limit=2,
            border_mode=0,
            interpolation=2,
            value=(255, 255, 255),
            p=0.3,
        ),
        A.GridDistortion(
            distort_limit=0.05,
            border_mode=0,
            interpolation=2,
            value=(255, 255, 255),
            p=0.2,
        ),
        A.OpticalDistortion(p=0.25, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
        A.Perspective(scale=(0.01, 0.03), p=0.3, pad_val=(255, 255, 255), fit_output=True),
        LongestMaxSizeHW(
            p=1,
            max_size_height=[i for i in range(860, 3000, 16)],
            max_size_width=[i for i in range(780, 2550, 16)],
        ),
    ],
    bbox_params=A.BboxParams(format="pascal_voc"),
)


GROUNDING_PROB = 0.5
GROUNDING_FORMAT = ["list_of_tuples", "list_of_dicts"]

PROMPTS_TEXT_ONLY = [
    "Convert the text in this image into a plain text readable document. Use LaTeX to represent tables. Ignore the text in pictures but keep all captions at the end.",
    "Transcribe this document in reading order ignoring the text inside pictures. Extract all tables as LaTeX.",
    "Can you extract all visible text from the document here in reading order and output as plain text? Tables should be represented as LaTeX. Text in pictures should be ignored.",
    "Can you read the text from this document in reading order? Parse tables in latex format and skip the text inside figures or images.",
    "Fetch the text (except for text inside pictures) from the provided image in reading order - headers, the main body, footnotes, footers and captions. For tables, use latex formatting.",
]

PROMPTS_PARSE = [
    "Can you parse this document in reading order? Use LaTeX to represent tables.",
    "Extract the elements in this image in reading order. Format tables as latex.",
    "Fetch the text blocks from the provided image. Extract the text in reading order - headers, the main body, footnotes and footers, pictures, tables and captions. Extract tables and represent them as LaTeX.",
    "Parse this document. Use LaTeX to represent tables.",
    "Can you parse this document in reading order? Extract all pictures and tables at the end followed by any captions. Format tables with latex.",
]

POST_INSTRUCTIONS_BBOXES_AND_CLASSES = (
    "Ignore the text inside pictures, returning just the bounding boxes for them. "
    "Fetch the bounding box for each block along with the corresponding category from the following options: "
    "Caption, Code, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text and Title."
)

POST_INSTRUCTIONS_LIST_TUPLES = (
    "The coordinates should be normalized ranging from 0 to 1000 by the image width and height "
    "and the answer should be in the following format:\n[(x1, y1, x2, y2, category, text content), (x1, y1, x2, y2, category, text content)...]."
)

POST_INSTRUCTIONS_LIST_DICTS = (
    "The coordinates should be normalized ranging from 0 to 1000 by the image width and height.\n"
    'Your answer should be in the following format:\n[{{"bbox": [x1, y1, x2, y2], "category": category, "content": text_content)}}...].'
)


def _convert_label(label: str) -> str:
    """Convert and normalize category labels.

    Args:
        label: Original category label

    Returns:
        Normalized category label
    """
    if label == "Floating-text":
        return "Caption"
    return label


def _clean_text(text: str) -> str:
    """Clean and normalize text content.

    Args:
        text: Raw text content

    Returns:
        Cleaned text
    """
    text = re.sub(r"\n$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text, flags=re.MULTILINE)
    text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text, flags=re.MULTILINE)
    return text


def _fix_toc_dots(text: str) -> str:
    """Fix table of contents dots.

    Args:
        text: Text content

    Returns:
        Text with fixed dots
    """

    def _fix_dots(m):
        s = m.group(0)
        return s.startswith(" ") * " " + s.count(".") * "." + s.endswith(" ") * " "

    text = _re_fix_dots2.sub(".....", _re_fix_dots1.sub(_fix_dots, text))
    return text


def _normalize_bbox(bbox: list[float], image_width: float, image_height: float) -> tuple[int, int, int, int]:
    """Normalize bounding box to 0-1000 range.

    Args:
        bbox: [x1, y1, x2, y2] in original coordinates
        image_width: Width of image
        image_height: Height of image

    Returns:
        Normalized (x1, y1, x2, y2) tuple
    """
    x1, y1, x2, y2 = bbox
    x1_norm = int(x1 / image_width * 1000)
    y1_norm = int(y1 / image_height * 1000)
    x2_norm = ceil(x2 / image_width * 1000)
    y2_norm = ceil(y2 / image_height * 1000)
    return x1_norm, y1_norm, x2_norm, y2_norm


def _should_whiteout_block(text: str, category: str) -> bool:
    """Check if block should be whited out.

    Args:
        text: Block text content
        category: Block category

    Returns:
        True if block contains {EQN} or is empty (except Picture)
    """
    if "{EQN}" in text:
        return True

    if text.strip() == "" and category != "Picture":
        return True

    return not (category == "Picture" or text)


def _fix_negative_boxes(block_boxes: list[list[float]]) -> None:
    """Fix boxes with negative width/height by swapping coordinates.

    Args:
        block_boxes: List of [x1, y1, x2, y2] (modifies in place)
    """
    for bbox in block_boxes:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]


def _clamp_and_identify_out_of_bounds(bboxes: list[list], width: int, height: int, max_outside: int = 10) -> list[int]:
    """Clamp bounding boxes and identify out-of-bounds boxes.

    Args:
        bboxes: List of [x1, y1, x2, y2, text, cls] (modifies first 4 coords in place)
        width: Image width
        height: Image height
        max_outside: Maximum allowed distance outside boundaries

    Returns:
        Indices of boxes that are too far out of bounds
    """
    whiteout_idxs = []
    for idx, bbox in enumerate(bboxes):
        if (
            bbox[0] < -max_outside
            or bbox[1] < -max_outside
            or bbox[2] >= width + max_outside
            or bbox[3] >= height + max_outside
        ):
            whiteout_idxs.append(idx)
        else:
            bbox[0] = min(max(bbox[0], 0), width - 1)
            bbox[1] = min(max(bbox[1], 0), height - 1)
            bbox[2] = min(max(bbox[2], 0), width - 1)
            bbox[3] = min(max(bbox[3], 0), height - 1)
    return whiteout_idxs


def _whiteout_bbox(image: Image.Image, bbox: list[float], rng: random.Random) -> None:
    """White out a bbox region with black, white, or corner-averaged color."""

    img_draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = bbox[:4]
    x1 = max(0, min(int(x1), image.width - 1))
    y1 = max(0, min(int(y1), image.height - 1))
    x2 = max(0, min(int(x2), image.width - 1))
    y2 = max(0, min(int(y2), image.height - 1))

    if rng.random() < 0.3:
        corner_colors = (
            image.getpixel((x1, y1))[:3],
            image.getpixel((x2, y1))[:3],
            image.getpixel((x1, y2))[:3],
            image.getpixel((x2, y2))[:3],
        )
        color = tuple(sum(c) // 4 for c in zip(*corner_colors))
    else:
        color = (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)

    img_draw.rectangle([x1, y1, x2, y2], fill=color)


def _whiteout_and_remove_boxes(
    image: Image.Image, whiteout_idxs: list[int], bboxes: list[list], rng: random.Random
) -> bool:
    """White out boxes in image and remove them from list.

    Args:
        image: PIL Image to modify
        whiteout_idxs: Indices of boxes to white out
        bboxes: List of [x1, y1, x2, y2, text, cls] (modified in place)
        rng: Random generator for deterministic behavior

    Returns:
        True if any boxes were whited out
    """
    if len(whiteout_idxs) == 0:
        return False

    assert len(whiteout_idxs) == len(set(whiteout_idxs)), "duplicate indices found"

    for idx in sorted(whiteout_idxs, reverse=True):
        _whiteout_bbox(image, bboxes[idx], rng)
        bboxes.pop(idx)

    return True


def _visualize_bboxes(
    image: Image.Image,
    block_boxes: list[list[float]],
    block_classes: list[str],
    output_path: Path,
) -> None:
    """Draw bounding boxes with category labels and save."""
    CATEGORY_COLORS = {
        "Section-header": (255, 107, 107),
        "Page-footer": (78, 205, 196),
        "Table": (69, 183, 209),
        "Caption": (255, 160, 122),
        "Page-header": (152, 216, 200),
        "Picture": (247, 220, 111),
        "Text": (189, 195, 199),
        "Title": (155, 89, 182),
        "List-item": (52, 152, 219),
        "Code": (241, 196, 15),
        "Formula": (230, 126, 34),
        "Footnote": (149, 165, 166),
    }
    DEFAULT_COLOR = (149, 165, 166)

    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    try:
        from PIL import ImageFont

        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError, ImportError):
        font = None

    overlay = Image.new("RGBA", vis_image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for bbox, category in zip(block_boxes, block_classes):
        x1, y1, x2, y2 = bbox
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if font:
            text_bbox = draw.textbbox((x1, y1), category, font=font)
        else:
            text_bbox = (x1, y1, x1 + len(category) * 8, y1 + 16)

        text_bg = [
            text_bbox[0] - 2,
            text_bbox[1] - 2,
            text_bbox[2] + 2,
            text_bbox[3] + 2,
        ]
        rgba_color = color + (128,)
        overlay_draw.rectangle(text_bg, fill=rgba_color)

    vis_image = vis_image.convert("RGBA")
    vis_image = Image.alpha_composite(vis_image, overlay)
    vis_image = vis_image.convert("RGB")

    draw = ImageDraw.Draw(vis_image)
    for bbox, category in zip(block_boxes, block_classes):
        x1, y1, x2, y2 = bbox
        draw.text((x1, y1), category, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis_image.save(output_path)


def _fuse_headers_and_footers(block_classes: list[str], block_boxes: list[list[float]], block_text: list[str]) -> None:
    """Fuse multiple headers/footers into single blocks."""

    def fuse_bboxes(bboxes: list[list[float]]) -> list[float] | None:
        if len(bboxes) == 0:
            return None
        if len(bboxes) == 1:
            return bboxes[0]
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return [x1, y1, x2, y2]

    header_indices = [idx for idx, cls in enumerate(block_classes) if cls == "Page-header"]
    if len(header_indices) > 1:
        fused_bbox = fuse_bboxes([block_boxes[idx] for idx in header_indices])
        fused_text = " ".join([block_text[idx] for idx in header_indices])

        for idx in sorted(header_indices, reverse=True):
            del block_boxes[idx]
            del block_text[idx]
            del block_classes[idx]

        block_boxes.insert(0, fused_bbox)
        block_text.insert(0, fused_text)
        block_classes.insert(0, "Page-header")

    footer_indices = [idx for idx, cls in enumerate(block_classes) if cls == "Page-footer"]
    if len(footer_indices) > 1:
        fused_bbox = fuse_bboxes([block_boxes[idx] for idx in footer_indices])
        fused_text = " ".join([block_text[idx] for idx in footer_indices])

        for idx in sorted(footer_indices, reverse=True):
            del block_boxes[idx]
            del block_text[idx]
            del block_classes[idx]

        block_boxes.append(fused_bbox)
        block_text.append(fused_text)
        block_classes.append("Page-footer")


def _reorder_blocks(block_classes: list[str], block_boxes: list[list[float]], block_text: list[str]) -> None:
    """Reorder blocks: headers first, then content, then footers/pictures/tables/captions."""
    end = len(block_text) - 1

    def move_to_start(idxs: list[int]) -> None:
        for dst, idx in enumerate(idxs):
            if idx != dst:
                block_boxes.insert(dst, block_boxes.pop(idx))
                block_text.insert(dst, block_text.pop(idx))
                block_classes.insert(dst, block_classes.pop(idx))

    def move_to_end(idxs: list[int]) -> None:
        for offs, idx in enumerate(idxs):
            if idx - offs != end:
                block_boxes.append(block_boxes.pop(idx - offs))
                block_text.append(block_text.pop(idx - offs))
                block_classes.append(block_classes.pop(idx - offs))

    move_to_start([idx for idx, cls in enumerate(block_classes) if cls == "Page-header"])
    move_to_end([idx for idx, cls in enumerate(block_classes) if cls == "Footnote"])
    move_to_end([idx for idx, cls in enumerate(block_classes) if cls == "Page-footer"])
    move_to_end([idx for idx, cls in enumerate(block_classes) if cls == "Picture"])
    move_to_end([idx for idx, cls in enumerate(block_classes) if cls == "Table"])
    move_to_end([idx for idx, cls in enumerate(block_classes) if cls == "Caption"])


def _fix_content_plain(
    block_classes: list[str],
    block_text: list[str],
    block_boxes: list[list[float]],
    is_train: bool = True,
) -> None:
    """Clear Picture text and validate no empty non-Picture/Table/Formula blocks."""
    for idx, cls in enumerate(block_classes):
        if cls == "Picture":
            block_text[idx] = ""

    if is_train:
        for cls, txt, bbox in zip(block_classes, block_text, block_boxes):
            if cls not in ("Picture", "Table", "Formula") and not txt:
                assert cls in ("Picture", "Table", "Formula") or txt, (
                    f"Empty text in plaintext format block element cls={cls!r}, bbox={bbox}"
                )


def _fix_content_md(
    block_classes: list[str],
    block_text: list[str],
    block_boxes: list[list[float]],
    is_train: bool = True,
) -> None:
    """Clear Picture text and validate no empty non-Picture blocks."""
    for idx, cls in enumerate(block_classes):
        if cls == "Picture":
            block_text[idx] = ""

    if is_train:
        for cls, txt, bbox in zip(block_classes, block_text, block_boxes):
            if cls != "Picture" and not txt:
                assert cls == "Picture" or txt, f"Empty text in md format block element cls={cls!r}, bbox={bbox}"


def _apply_prompts(
    line: str,
    image_root: Path,
    output_image_dir: Path,
    data_format: str = "plain",
    visualize_dir: Path | None = None,
    line_num: int = 0,
    base_seed: int = 0,
) -> str:
    """Generate conversation format prompt with augmentation from ccpdf entry."""
    # Initialize thread-local random generators for deterministic augmentation
    # Use different seeds for different augmentation stages
    combined_seed = base_seed + line_num

    rng = random.Random(combined_seed)
    # Running multi-processing, thus within the process, the global seed is safe. Required for albumentations.
    np.random.seed(combined_seed)

    has_grounding = rng.random() <= GROUNDING_PROB
    grounding_format = rng.choice(GROUNDING_FORMAT)

    entry = json.loads(line)

    image_path = entry["image"]
    annotations = entry["ann"]
    width = entry["metadata"]["width"]
    height = entry["metadata"]["height"]

    block_classes = [_convert_label(ann["category_id"]) for ann in annotations]
    block_boxes = [
        [
            ann["bbox"][0],
            ann["bbox"][1],
            ann["bbox"][0] + ann["bbox"][2],
            ann["bbox"][1] + ann["bbox"][3],
        ]
        for ann in annotations
    ]
    if any(len(bbox) != 4 for bbox in block_boxes):
        logger.warning(f"Invalid bbox in {image_path}: {block_boxes}")
        return None
    block_text = [_clean_text(ann.get("text", "")) for ann in annotations]

    _fix_negative_boxes(block_boxes)
    _fuse_headers_and_footers(block_classes, block_boxes, block_text)
    _reorder_blocks(block_classes, block_boxes, block_text)

    # Identify blocks to white out
    whiteout_idxs = []
    for idx, (cls, text) in enumerate(zip(block_classes, block_text)):
        if _should_whiteout_block(text, cls):
            whiteout_idxs.append(idx)

    full_image_path = image_root / image_path
    image = Image.open(full_image_path)

    # Scale bboxes if actual image dimensions differ from metadata
    actual_width, actual_height = image.size
    if actual_width != width or actual_height != height:
        scale_x = actual_width / width
        scale_y = actual_height / height

        # Warn if scaling is non-proportional (>1% difference)
        scale_diff_pct = abs(scale_x - scale_y) / max(scale_x, scale_y) * 100
        if scale_diff_pct > 1.0:
            logger.warning(
                f"Non-proportional scaling for {image_path}: scale_x={scale_x:.4f}, scale_y={scale_y:.4f} (diff={scale_diff_pct:.2f}%)"
            )

        block_boxes = [
            [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y] for bbox in block_boxes
        ]
        width = actual_width
        height = actual_height

    # Create combined bboxes structure: [x1, y1, x2, y2, text, cls]
    bboxes = [list(bbox) + [text, cls] for bbox, text, cls in zip(block_boxes, block_text, block_classes)]

    # Clamp boxes and identify out-of-bounds (before augmentation only)
    clamp_whiteout_idxs = _clamp_and_identify_out_of_bounds(bboxes, width, height)
    if len(clamp_whiteout_idxs) > 0:
        if len(whiteout_idxs) > 0:
            tmp = set(whiteout_idxs)
            tmp.update(clamp_whiteout_idxs)
            whiteout_idxs = sorted(tmp)
        else:
            whiteout_idxs = clamp_whiteout_idxs

    # Visualize before augmentation if requested
    if visualize_dir:
        vis_boxes = [b[:4] for b in bboxes]
        vis_classes = [b[5] for b in bboxes]
        vis_path = visualize_dir / f"beforeaug_{image_path}"
        _visualize_bboxes(image, vis_boxes, vis_classes, vis_path)

    # White out and remove boxes (before augmentation)
    _whiteout_and_remove_boxes(image, whiteout_idxs, bboxes, rng)

    # Always apply augmentation
    transformed = TRAIN_AFFINE_TRANSFORM(
        image=np.array(image).astype(np.uint8),
        bboxes=bboxes,
    )

    if len(transformed["bboxes"]) != len(bboxes):
        logger.warning(f"Augmentation removed boxes for {image_path}: {len(bboxes)} -> {len(transformed['bboxes'])}")

    # Clamp transformed boxes to image bounds
    for i_trb in range(len(transformed["bboxes"])):
        box_trb = transformed["bboxes"][i_trb]
        new_b = [
            box_trb[0],
            box_trb[1],
            min(transformed["image"].shape[1] - 1, box_trb[2]),
            min(transformed["image"].shape[0] - 1, box_trb[3]),
            box_trb[4],
        ]
        if len(box_trb) == 6:
            new_b.append(box_trb[5])
        transformed["bboxes"][i_trb] = tuple(new_b)

    image = Image.fromarray(transformed["image"])
    image = Image.fromarray(PHOTOMETRIC_TRANSFORM(image))

    # Extract boxes, text, and classes from transformed bboxes
    block_boxes = [list(b[:4]) for b in transformed["bboxes"]]
    block_text = [b[4] for b in transformed["bboxes"]]
    block_classes = [b[5] for b in transformed["bboxes"]]

    width = image.width
    height = image.height

    # Save modified image
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_image_dir / image_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    # Fix TOC dots
    for idx in range(len(block_text)):
        block_text[idx] = _fix_toc_dots(block_text[idx]).strip()

    # Fix content based on format
    if data_format == "markdown":
        _fix_content_md(block_classes, block_text, block_boxes, is_train=True)
    else:
        _fix_content_plain(block_classes, block_text, block_boxes, is_train=True)

    # Visualize after augmentation if requested
    if visualize_dir:
        vis_path = visualize_dir / f"afteraug_{image_path}"
        _visualize_bboxes(image, block_boxes, block_classes, vis_path)

    if not has_grounding:
        content = "\n\n".join([text for text in block_text if text])
        prompt = rng.choice(PROMPTS_TEXT_ONLY)
    elif grounding_format == "list_of_tuples":
        prompt = (
            rng.choice(PROMPTS_PARSE) + " " + POST_INSTRUCTIONS_BBOXES_AND_CLASSES + " " + POST_INSTRUCTIONS_LIST_TUPLES
        )
        content_parts = []
        for text, bbox, cls in zip(block_text, block_boxes, block_classes):
            x1, y1, x2, y2 = _normalize_bbox(bbox, width, height)
            answer = f'({x1}, {y1}, {x2}, {y2}, "{cls}", {json.dumps(text, ensure_ascii=False)})'
            content_parts.append(answer)
        content = "[" + ", ".join(content_parts) + "]"
    else:
        prompt = (
            rng.choice(PROMPTS_PARSE) + " " + POST_INSTRUCTIONS_BBOXES_AND_CLASSES + " " + POST_INSTRUCTIONS_LIST_DICTS
        )
        content_parts = []
        for text, bbox, cls in zip(block_text, block_boxes, block_classes):
            x1, y1, x2, y2 = _normalize_bbox(bbox, width, height)
            content_parts.append({"bbox": [x1, y1, x2, y2], "category": cls, "content": text})
        content = json.dumps(content_parts, ensure_ascii=False, sort_keys=True, indent=4)

    result = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "metadata": {
                            "width": width,
                            "height": height,
                            "format": "PNG",
                            "mode": "RGB",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": content}]},
        ]
    }

    if "id" in entry:
        result["id"] = entry["id"]

    if "source" in entry:
        result["source"] = entry["source"]

    result["metadata"] = {
        "has_grounding": has_grounding,
        "grounding_format": grounding_format,
    }

    return json.dumps(result, ensure_ascii=False)


@click.command()
@click.argument("input_jsonl", type=click.Path(exists=True, path_type=Path))
@click.argument("output_jsonl", type=click.Path(path_type=Path))
@click.option(
    "--image-root",
    type=click.Path(exists=True, path_type=Path),
    help="Root directory for images (needed for augmentation/whiteout)",
    required=True,
)
@click.option(
    "--output-image-dir",
    type=click.Path(path_type=Path),
    help="Directory to save modified images (augmented/whited out)",
    required=True,
)
@click.option(
    "--data-format",
    type=click.Choice(["plain", "markdown"]),
    default="plain",
    help="Data format for content validation (default: plain)",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducibility",
    default=42,
)
@click.option(
    "--visualize-dir",
    type=click.Path(path_type=Path),
    help="Directory to save visualizations of augmented bounding boxes",
    default=None,
)
@click.option(
    "--pool-size",
    type=int,
    help="Number of processes to use for parallel processing",
    default=32,
)
def main(
    input_jsonl: Path,
    output_jsonl: Path,
    image_root: Path,
    output_image_dir: Path,
    data_format: str,
    seed: int | None,
    visualize_dir: Path | None,
    pool_size: int,
) -> None:
    """Generate conversation prompts from ccpdf JSONL files.

    Converts ccpdf annotations to conversation format matching release.py output.
    Empty blocks (except Picture) are whited out. Image augmentation is always applied.
    Clamping happens once before augmentation only.

    Each line is processed with a deterministic seed (base_seed + line_number) for
    reproducible, thread-safe augmentation.

    Args:
        input_jsonl: Input JSONL from ccpdf (e.g., ccpdf_nv_tables.jsonl)
        output_jsonl: Output JSONL path
        image_root: Root directory for images
        output_image_dir: Directory to save modified images
        data_format: "plain" or "markdown" for content validation
        seed: Optional base random seed (combined with line number per sample)
        visualize_dir: Optional directory to save bbox visualizations
        pool_size: Number of processes to use for parallel processing
    """
    logger.info(f"Processing {input_jsonl}")
    logger.info(f"Output: {output_jsonl}")
    logger.info(f"Data format: {data_format}")

    if visualize_dir:
        logger.info(f"Visualizations will be saved to: {visualize_dir}")
        visualize_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        logger.info(f"Using base random seed: {seed} (combined with line number per sample)")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    error_count = 0

    with open(input_jsonl, "r") as in_f:
        num_lines = sum(1 for _ in in_f)

    with open(input_jsonl, "r") as in_f, open(output_jsonl, "w") as out_f:
        for line_num, result in tqdm(
            enumerate(
                process_generator(
                    (
                        ProcessBound(
                            _apply_prompts,
                            line,
                            image_root,
                            output_image_dir,
                            data_format,
                            visualize_dir,
                            idx,
                            seed,
                        )
                        for idx, line in enumerate(in_f)
                    ),
                    pool_size=pool_size,
                    in_order=True,
                    auto_raise=False,
                )
            ),
            desc="Processing",
            unit="lines",
            total=num_lines,
        ):
            if isinstance(result, Exception):
                logger.error(f"Error processing line {line_num + 1}: {result}")
                error_count += 1
                continue
            out_f.write(result + "\n")
            processed_count += 1
    logger.info("Processing complete!")
    logger.info(f"Successfully processed: {processed_count} entries")
    logger.info(f"Errors: {error_count} entries")
    logger.info(f"Output written to: {output_jsonl}")


if __name__ == "__main__":
    main()
