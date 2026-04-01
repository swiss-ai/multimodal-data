import os
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from datasets import Dataset
from faster_whisper import WhisperModel, BatchedInferencePipeline

from loaders import LOADERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [W%(worker_id)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Transcriber:
    """Wraps a faster-whisper pipeline and transcribes HuggingFace Datasets."""

    def __init__(self, model_name: str = "turbo", language: str = "en", segment_batch_size: int = 16):
        self.language = language
        self.segment_batch_size = segment_batch_size

        logger.info("Loading model '%s'...", model_name)
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        self.pipeline = BatchedInferencePipeline(model=model)

    def _transcribe_batch(self, examples: dict) -> dict:
        results = []
        for audio in examples["audio"]:
            try:
                audio_array = np.asarray(audio["array"], dtype=np.float32)
                segments, _ = self.pipeline.transcribe(
                    audio_array,
                    batch_size=self.segment_batch_size,
                    language=self.language,
                    condition_on_previous_text=False,
                )
                text = " ".join(seg.text for seg in segments).strip()
            except Exception:
                logger.exception("Transcription failed; writing empty text.")
                text = ""
            results.append(text)
        return {"text": results}

    def transcribe(self, ds: Dataset, batch_size: int = 16, desc: str = "") -> Dataset:
        return ds.map(
            self._transcribe_batch,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=False,
            desc=desc,
        )


class ShardProcessor:
    """Processes a list of input shards: load, transcribe and save."""

    def __init__(self, transcriber: Transcriber, load_fn: Callable, output_dir: str, batch_size: int = 16):
        self.transcriber = transcriber
        self.load_fn = load_fn
        self.output_dir = output_dir
        self.batch_size = batch_size

    def _out_dir_for(self, file_path: str) -> str:
        folder_name = Path(file_path).stem + "_processed"
        return os.path.join(self.output_dir, folder_name)

    @staticmethod
    def _save(ds: Dataset, out_dir: str) -> None:
        parent = Path(out_dir).parent
        with tempfile.TemporaryDirectory(dir=parent, prefix=".tmp_") as tmp:
            ds.save_to_disk(tmp)
            os.rename(tmp, out_dir)

    def process(self, file_path: str) -> None:
        filename = os.path.basename(file_path)
        out_dir = self._out_dir_for(file_path)

        if os.path.exists(out_dir):
            logger.info("Skipping %s (already processed).", filename)
            return

        logger.info("Processing %s", filename)

        ds = self.load_fn(file_path)
        ds = self.transcriber.transcribe(ds, self.batch_size, desc=f"Transcribing {filename}")

        self._save(ds, out_dir)
        logger.info("Saved %s → %s", filename, out_dir)

    def run(self, file_paths: list[str]) -> None:
        for file_path in file_paths:
            try:
                self.process(file_path)
            except Exception:
                logger.exception("Fatal error processing %s; skipping.", os.path.basename(file_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio shards with faster-whisper.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_format", type=str, default="arrow", choices=list(LOADERS.keys()))
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--whisper_model", type=str, default="turbo")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--language", type=str, default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    old_factory = logging.getLogRecordFactory()
    def record_factory(*a, **kw):
        record = old_factory(*a, **kw)
        record.worker_id = args.worker_id
        return record
    logging.setLogRecordFactory(record_factory)

    os.makedirs(args.output_dir, exist_ok=True)

    all_files, load_fn = LOADERS[args.input_format](args.input_dir)

    if not all_files:
        logger.error("No files found in %s for format '%s'", args.input_dir, args.input_format)
        return

    my_files = [f for i, f in enumerate(all_files) if i % args.num_workers == args.worker_id]
    logger.info("Assigned %d / %d files.", len(my_files), len(all_files))

    transcriber = Transcriber(model_name=args.whisper_model, language=args.language)
    processor = ShardProcessor(transcriber, load_fn, args.output_dir, args.batch_size)
    processor.run(my_files)


if __name__ == "__main__":
    main()