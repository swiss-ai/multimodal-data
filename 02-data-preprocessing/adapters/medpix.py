import io
import json
import logging
import os
import sys
import zipfile

from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    BaseDataset,
    MultiImageTextSample,
    Sample,
    SampleMetadata,
)

DATASET_ID = "medpix"


def _load_zip_data(
    zip_path: str,
) -> tuple[list[dict], dict[str, dict], dict[str, bytes]]:
    """Load all MedPix data from the outer zip.

    Returns:
        cases: list of case dicts from Case_topic.json
        desc_by_image: {image_id -> description dict} from Descriptions.json
        image_bytes: {image_id -> raw PNG bytes} from images.zip
    """
    with zipfile.ZipFile(zip_path) as outer:
        # Load case metadata
        with outer.open("Case_topic.json") as f:
            cases = json.load(f)

        # Load per-image descriptions
        with outer.open("Descriptions.json") as f:
            descriptions = json.load(f)
        desc_by_image = {d["image"]: d for d in descriptions}

        # Load all image bytes from the nested images.zip
        with outer.open("images.zip") as raw:
            images_buf = io.BytesIO(raw.read())
        image_bytes: dict[str, bytes] = {}
        with zipfile.ZipFile(images_buf) as inner:
            for name in inner.namelist():
                if name.lower().endswith(".png"):
                    stem = os.path.splitext(os.path.basename(name))[0]
                    image_bytes[stem] = inner.read(name)

    return cases, desc_by_image, image_bytes


def _decode_image(raw: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception:
        return None


def _build_text(
    case: dict, image_ids: list[str], desc_by_image: dict[str, dict]
) -> str:
    """Build the caption text for a case, with image tokens prepended."""
    c = case.get("Case", {})
    topic = case.get("Topic", {})

    parts = []

    # Case-level narrative
    title = c.get("Title") or topic.get("Title") or ""
    if title:
        parts.append(f"Title: {title}")
    history = c.get("History", "")
    if history and history != "N/A":
        parts.append(f"History: {history}")
    findings = c.get("Findings", "")
    if findings and findings != "N/A":
        parts.append(f"Findings: {findings}")
    diagnosis = c.get("Case Diagnosis") or c.get("Differential Diagnosis") or ""
    if diagnosis and diagnosis != "N/A":
        parts.append(f"Diagnosis: {diagnosis}")
    discussion = topic.get("Disease Discussion", "")
    if discussion and discussion != "N/A":
        parts.append(f"Discussion: {discussion}")

    # Per-image captions with tokens
    for i, img_id in enumerate(image_ids, start=1):
        desc = desc_by_image.get(img_id, {})
        d = desc.get("Description", {})
        caption = d.get("Caption", "")
        modality = d.get("Modality", "")
        plane = d.get("Plane", "")
        img_meta = ", ".join(x for x in [modality, plane] if x)
        img_line = f"<|img{i}|>"
        if img_meta:
            img_line += f" [{img_meta}]"
        if caption:
            img_line += f" {caption}"
        parts.append(img_line)

    return "\n".join(parts)


class MedPixAdapter(BaseDataset):
    """Adapter for MedPix-2.0 dataset (671 cases, ~2050 images).

    Expects the path to the MedPix-2_0.zip file.
    Emits ImageTextSample for single-image cases, MultiImageTextSample for multi.
    """

    def __init__(self, dataset_id: str, zip_path: str):
        self._id = dataset_id
        self.zip_path = zip_path

    @property
    def id(self):
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0

        logger.info(f"[{self.id}] Loading MedPix data from {self.zip_path}")
        cases, desc_by_image, image_bytes = _load_zip_data(self.zip_path)
        logger.info(f"[{self.id}] Loaded {len(cases)} cases, {len(image_bytes)} images")

        batch: list[Sample] = []
        for counter, case in enumerate(cases):
            if counter < skip:
                continue

            u_id = case.get("U_id", "")
            # Combine TAC and MRI image lists
            image_ids = case.get("TAC", []) + case.get("MRI", [])
            if not image_ids:
                continue

            # Decode images, skip any that fail
            pil_images = []
            valid_ids = []
            for img_id in image_ids:
                raw = image_bytes.get(img_id)
                if raw is None:
                    logger.warning(
                        f"[{self.id}] Case {u_id}: missing bytes for {img_id}"
                    )
                    continue
                img = _decode_image(raw)
                if img is None:
                    logger.warning(
                        f"[{self.id}] Case {u_id}: failed to decode {img_id}"
                    )
                    continue
                pil_images.append(img)
                valid_ids.append(img_id)

            if not pil_images:
                continue

            text = _build_text(case, valid_ids, desc_by_image)
            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=counter,
                data={"u_id": u_id},
            )

            sample: Sample = MultiImageTextSample(images=pil_images, text=text, meta=m)

            batch.append(sample)

            if len(batch) >= batch_size:
                yield batch
                batch = []

            if counter % 100 == 0 and counter > 0:
                logger.info(f"[{self.id}] Processed {counter} cases")

        if batch:
            yield batch

        logger.info(f"[{self.id}] Finished streaming.")


if __name__ == "__main__":
    logger = logging.getLogger("medpix")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    ZIP_PATH = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/MedPix-2_0.zip"
    adapter = MedPixAdapter(dataset_id=DATASET_ID, zip_path=ZIP_PATH)

    for batch in adapter.stream(logger=logger, skip=0, batch_size=4):
        for s in batch:
            assert isinstance(s, MultiImageTextSample)
            print(f"  ({len(s.images)} imgs): {s.text[:200]}")
        break
