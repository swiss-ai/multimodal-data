#!/usr/bin/env python3
"""
HuggingFace Dataset Loader Test Script

Tests loading datasets using two methods:
- "default": load_dataset() - downloads if needed
- "builder_load": builder.as_dataset() - uses pre-downloaded datasets

Supports streaming and split slicing (e.g., "train[:100]").

Cache Configuration:
- HF_HUB_CACHE: Raw files from HuggingFace Hub (set via env var)
- --cache-dir: Processed datasets ready for use (required arg)

Usage:
  python builder_loader_test.py \
      --dataset-name "mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M" \
      --cache-dir "/capstor/.../hf_datasets_cache" \
      --method builder_load

  HF_HUB_CACHE="/capstor/.../hf_hub_cache" python builder_loader_test.py \
      --dataset-name "google/docci" \
      --cache-dir "./cache" \
      --method default \
      --split "train[:1000]"
"""

import sys
import os
import re
import argparse
import logging
from typing import Optional, Tuple, Dict, List
from datasets import load_dataset_builder, load_dataset, Dataset
from download_hf_dataset import get_configs_to_process

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


#
# Load datasets methods copied from benchmark-image-tokenizer-repo
#
def _parse_split_slice(split: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse split string that may contain slice notation.

    Args:
        split: Split string (e.g., "train", "train[:100]", "train[100:200]")

    Returns:
        Tuple of (base_split, start, end) where start/end are None if no slice

    Examples:
        "train" -> ("train", None, None)
        "train[:100]" -> ("train", None, 100)
        "train[100:200]" -> ("train", 100, 200)
        "train[100:]" -> ("train", 100, None)
    """
    # Pattern to match split[start:end] syntax
    match = re.match(r"^(\w+)(?:\[(\d*):(\d*)\])?$", split)
    if not match:
        # No slice notation or invalid format
        return split, None, None

    base_split = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)

    # Convert to int if present, otherwise None
    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None

    return base_split, start, end


def load_hf_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    method: str = "default",
    streaming: bool = False,
) -> Dataset:
    """
    Load a HuggingFace dataset using specified method.

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceM4/FineVision")
        config_name: Dataset configuration/subset name (e.g., "lrv_chart")
        split: Dataset split to load (e.g., "train", "test", "validation", "train[:100]")
        cache_dir: Cache directory for dataset files
        num_proc: Number of processes (only used with "default" method)
        method: Loading method - "default" or "builder_load"
        streaming: Whether to use streaming mode

    Returns:
        Dataset object supporting len(), .shard(), .select(), iteration

    Raises:
        ValueError: If method is not "default" or "builder_load"
        FileNotFoundError: If "builder_load" is used but dataset is not prepared
    """
    if method not in ["default", "builder_load"]:
        raise ValueError(f"Invalid method: {method}. Must be 'default' or 'builder_load'")

    if method == "default":
        config_info = f" (config: {config_name})" if config_name else ""
        logger.info(f"Loading dataset using load_dataset(): {dataset_name}{config_info}/{split}")

        if streaming:
            logger.warning(
                "Number of processes is set to 1 as streaming with multiple processes is not implemented in hf"
            )
            num_proc = None

            # Parse split to handle slice notation (not supported in streaming mode)
            base_split, start, end = _parse_split_slice(split)

            if start is not None or end is not None:
                logger.info(
                    f"Detected slice notation in split '{split}'. Loading base split '{base_split}' and applying slice via .skip()/.take()"
                )
                actual_split = base_split
            else:
                actual_split = split
        else:
            actual_split = split
            start, end = None, None

        dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=actual_split,
            cache_dir=cache_dir,
            num_proc=num_proc,
            streaming=streaming,
        )

        # Apply slicing for streaming datasets
        if streaming and (start is not None or end is not None):
            if start is not None and start > 0:
                logger.info(f"Skipping first {start} samples")
                dataset = dataset.skip(start)
            if end is not None:
                count = end - (start or 0)
                logger.info(f"Taking {count} samples")
                dataset = dataset.take(count)
            elif start is not None:
                # start specified but no end - take all remaining
                logger.info(f"Taking all samples after skipping {start}")

        return dataset

    elif method == "builder_load":
        config_info = f" (config: {config_name})" if config_name else ""
        logger.info(f"Loading dataset using builder.as_dataset(): {dataset_name}{config_info}/{split}")

        # Warn if num_proc provided (not supported)
        if num_proc is not None:
            logger.warning(
                f"num_proc parameter ({num_proc}) is ignored when using 'builder_load' method. "
                f"Dataset is already prepared."
            )

        # Warn if cache_dir not provided
        if cache_dir is None:
            logger.warning(
                "Using 'builder_load' without explicit cache_dir. " "Will use default: ~/.cache/huggingface/datasets"
            )

        builder = load_dataset_builder(dataset_name, name=config_name, cache_dir=cache_dir)

        try:
            dataset = builder.as_dataset(split=split)
        except FileNotFoundError as e:
            logger.error(
                f"Dataset '{dataset_name}' is not prepared. "
                f"When using 'builder_load', run download_and_prepare() first."
            )
            logger.error(
                f"To prepare:\n"
                f"  from datasets import load_dataset_builder\n"
                f"  builder = load_dataset_builder('{dataset_name}'"
                f"{f', name={config_name!r}' if config_name else ''}"
                f"{f', cache_dir={cache_dir!r}' if cache_dir else ''})\n"
                f"  builder.download_and_prepare()"
            )
            raise FileNotFoundError(f"Dataset not prepared. Use method='default' or prepare dataset first.") from e

    # Log dataset size (streaming datasets don't have len())
    if streaming:
        logger.info(f"Loaded streaming dataset from {split} split")
    else:
        logger.info(f"Loaded {len(dataset)} samples from {split} split")
    return dataset


def print_dataset_stats(
    dataset,
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    method: str,
    streaming: bool,
    condensed: bool = False,
):
    """
    Print dataset statistics (always condensed, with optional full details).

    Args:
        dataset: The loaded dataset (Dataset or IterableDataset)
        dataset_name: Name of the dataset
        config_name: Configuration name (if any)
        split: Split that was loaded
        method: Loading method used
        streaming: Whether streaming mode was used
        condensed: If True, show only brief stats; if False, show condensed + full details
    """
    config_display = config_name or "<default>"

    # Always print condensed info
    print(f"  Config: {config_display} | Split: {split} | Method: {method}", end="")
    if streaming:
        print(" | Type: IterableDataset")
    else:
        print(f" | Rows: {len(dataset):,}")

    # Get column info for condensed output
    try:
        if hasattr(dataset, 'features'):
            features = dataset.features
            column_names = list(features.keys())
            if len(column_names) <= 5:
                print(f"  Columns: {column_names}")
            else:
                print(f"  Columns ({len(column_names)}): {column_names[:5]} ... (truncated)")
        else:
            print("  Columns: Not available")
    except Exception as e:
        print(f"  Columns: Error - {e}")

    # If not condensed, print full details below
    if not condensed:
        print("\n" + "=" * 80)
        print("FULL DATASET DETAILS")
        print("=" * 80)

        # Schema Info - Feature Types
        print("\n🔧 Feature Types:")
        try:
            if hasattr(dataset, 'features'):
                for col_name, feature_type in features.items():
                    type_str = str(feature_type)
                    if len(type_str) > 80:
                        type_str = type_str[:77] + "..."
                    print(f"  - {col_name}: {type_str}")
            else:
                print("  Features: Not available")
        except Exception as e:
            print(f"  Error retrieving features: {e}")

        # Sample Data Preview
        print("\n📝 Sample Data Preview (first 3 samples):")
        print("-" * 80)
        try:
            samples = []
            if streaming:
                iterator = iter(dataset)
                for i in range(3):
                    try:
                        sample = next(iterator)
                        samples.append(sample)
                    except StopIteration:
                        break
            else:
                num_samples = min(3, len(dataset))
                for i in range(num_samples):
                    samples.append(dataset[i])

            if not samples:
                print("  No samples available")
            else:
                for idx, sample in enumerate(samples, 1):
                    print(f"\nSample {idx}:")
                    for key, value in sample.items():
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        print(f"  {key}: {value_str}")
                    if idx < len(samples):
                        print("-" * 40)
        except Exception as e:
            print(f"  Error retrieving samples: {e}")

        print("\n" + "=" * 80)


def print_summary_report(
    dataset_name: str,
    successful_tests: List[str],
    failed_tests: Dict[str, str],
):
    """
    Print summary report for multi-config testing.

    Args:
        dataset_name: Name of the dataset
        successful_tests: List of successfully tested config names
        failed_tests: Dictionary mapping config names to error messages
    """
    total_configs = len(successful_tests) + len(failed_tests)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Total configurations tested: {total_configs}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print()

    if successful_tests:
        print("✅ Successfully tested configs:")
        for config in successful_tests:
            print(f"  • {config}")
        print()

    if failed_tests:
        print("❌ Failed configs:")
        for config, error in failed_tests.items():
            # Truncate error message for summary
            error_short = error.split('\n')[0][:100]
            print(f"  • {config}: {error_short}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test loading HuggingFace datasets with comprehensive statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HuggingFace dataset repository name (e.g., HuggingFaceM4/FineVision)",
    )

    parser.add_argument(
        "--subset-name",
        type=str,
        default=None,
        help="Dataset configuration/subset name - single, comma-separated, or None for auto-detect (e.g., 'lrv_chart', 'config1,config2', or omit to test all)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load - supports slicing notation (e.g., 'train', 'train[:100]', 'train[100:200]') (default: train)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Datasets cache directory path where processed datasets are stored (HF datasets cache)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="builder_load",
        choices=["default", "builder_load"],
        help="Loading method: 'default' (load_dataset) or 'builder_load' (builder.as_dataset) (default: builder_load)",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode for large datasets (only with 'default' method)",
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of parallel processes for loading (only with 'default' method, auto-detect if not specified)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("HuggingFace Dataset Loader Test")
    print("=" * 80)

    # Display configuration
    print(f"\n📂 Configuration:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Config: {args.subset_name or '<auto-detect all configs>'}")
    print(f"  Split: {args.split}")
    print(f"  Method: {args.method}")
    print(f"  Streaming: {args.streaming}")
    print(f"  Cache Dir: {args.cache_dir}")
    print(f"  HF Hub Cache: {os.environ.get('HF_HUB_CACHE') or '<default>'}")
    print(f"  Num Proc: {args.num_proc or 'auto'}")
    print("=" * 80)
    print()

    # Get list of configs to test
    configs_to_test = get_configs_to_process(args.dataset_name, args.subset_name)
    total_configs = len(configs_to_test)

    # Show mode information if manually specified
    if args.subset_name is not None:
        if total_configs == 1:
            print("Mode: Single configuration test")
        else:
            print(f"Mode: Multiple configurations specified ({total_configs})")
            for i, config in enumerate(configs_to_test, 1):
                print(f"  {i}. {config or '<default>'}")
        print()

    # Warning for streaming with multiple configs
    if args.streaming and total_configs > 1:
        print("⚠ WARNING: Streaming mode with multiple configs")
        print("  Streaming datasets consume the iterator - full details cannot be shown")
        print("  Recommend using non-streaming mode for multi-config testing")
        print()

    # Test all configs
    successful_tests: List[str] = []
    failed_tests: Dict[str, str] = {}

    print("=" * 80)
    print("TESTING CONFIGURATIONS")
    print("=" * 80)
    print()

    for idx, config_name in enumerate(configs_to_test, 1):
        config_display = config_name or "<default>"

        if total_configs > 1:
            print(f"[{idx}/{total_configs}] Testing: {config_display}")

        try:
            # Load dataset
            dataset = load_hf_dataset(
                dataset_name=args.dataset_name,
                config_name=config_name,
                split=args.split,
                cache_dir=args.cache_dir,
                num_proc=args.num_proc,
                method=args.method,
                streaming=args.streaming,
            )

            # Print statistics (condensed for multi-config, full for single)
            print_dataset_stats(
                dataset=dataset,
                dataset_name=args.dataset_name,
                config_name=config_name,
                split=args.split,
                method=args.method,
                streaming=args.streaming,
                condensed=(total_configs > 1),
            )

            successful_tests.append(config_display)
            if total_configs > 1:
                print(f"  ✅ Success\n")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            failed_tests[config_display] = error_msg
            print(f"  ❌ Failed: {error_msg}\n")

    # Print summary for multi-config tests
    if total_configs > 1:
        print_summary_report(
            dataset_name=args.dataset_name,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
        )

    # Final status
    print("=" * 80)
    if len(failed_tests) == 0:
        print("✅ All tests completed successfully!")
    else:
        print(f"⚠ Tests completed with {len(failed_tests)} failure(s)")
    print("=" * 80)

    # Return exit code: 0 if all successful, 1 if any failures
    return 0 if len(failed_tests) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())