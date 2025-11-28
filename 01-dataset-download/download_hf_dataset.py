#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader

Downloads HF datasets with HTTP retry logic and automatic multi-config support.
- Auto-detects and downloads all configs when --subset-name not specified
- Skips already-cached configs automatically
- Continues on failures and saves detailed error report

NOTE: The --split argument is currently ignored. The download_and_prepare() method
downloads all splits by default. To work with specific splits, use the dataset
after downloading.

Example Usage:
    # Single config (here ex. with setting custom HF_HUB_CACHE dir)
    HF_HUB_CACHE=".test_hub_cache" python download_hf_dataset.py \
        --dataset-name "ibm-research/duorc" \
        --subset-name "ParaphraseRC" \
        --cache-dir "./test_cache"

    # Multiple configs (comma-separated)
    python download_hf_dataset.py \
        --dataset-name "ibm-research/duorc" \
        --subset-name "ParaphraseRC,SelfRC" \
        --cache-dir "./test_cache"

    # Auto-download all configs
    python download_hf_dataset.py \
        --dataset-name "HuggingFaceM4/FineVision" \
        --cache-dir "/path/to/cache"
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from datasets import load_dataset_builder, get_dataset_config_names, DownloadMode, VerificationMode
from huggingface_hub import configure_http_backend
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets to cache",
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
        help="Dataset subset/config name(s) - single name or comma-separated list (e.g., 'ParaphraseRC' or 'ParaphraseRC,SelfRC')",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train) - NOTE: Currently ignored, all splits are downloaded",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Cache directory path for storing downloaded data",
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for downloading (default: auto-detect)",
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download even if cached",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts for failed downloads (default: 5)",
    )

    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=1.0,
        help="Exponential backoff multiplier in seconds (default: 1.0)",
    )

    return parser.parse_args()


def check_hf_authentication():
    """Check if HuggingFace token is set and display authentication status."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_token:
        print("✓ HuggingFace authentication: Token detected")
        print("  Rate limit: ~1,000-2,500 requests per 5 minutes")
        return True
    else:
        print("⚠ HuggingFace authentication: No token detected")
        print("  Rate limit: ~500 requests per 5 minutes (reduced)")
        print("  Tip: Set HF_TOKEN environment variable to increase rate limit")
        return False


def setup_http_retry_backend(
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    timeout: int = 900,
):
    """
    Configure HTTP backend with retry logic for HuggingFace libraries.

    This configures the underlying HTTP session used by huggingface_hub and datasets
    libraries. All subsequent HTTP requests will automatically retry on failures with
    exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        backoff_factor: Exponential backoff multiplier in seconds (default: 1.0)
            Sleep time = backoff_factor * (2 ** (retry_number - 1))
        timeout: Request timeout in seconds (default: 900 = 15 minutes)

    Retries on:
        - Connection errors (network failures)
        - Read/timeout errors
        - HTTP 500, 502, 503, 504 (server errors)
        - HTTP 429 (rate limiting) - respects Retry-After header
    """
    # Configure retry with urllib3.Retry
    retries = Retry(
        total=max_retries,              # Total number of retries
        connect=max_retries,             # Connection errors
        read=max_retries,                # Read errors
        status=max_retries,              # Status code errors
        backoff_factor=backoff_factor,   # Exponential backoff
        status_forcelist=(500, 502, 503, 504, 429),  # HTTP codes to retry
        raise_on_status=False,           # Don't raise exception, return response
        respect_retry_after_header=True  # Respect server's Retry-After header
    )

    class TimeoutHTTPAdapter(HTTPAdapter):
        """HTTPAdapter with default timeout for all requests."""

        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.pop("timeout", timeout)
            super().__init__(*args, **kwargs)

        def send(self, request, **kwargs):
            """Override send to apply default timeout."""
            kwargs["timeout"] = kwargs.get("timeout", self.timeout)
            return super().send(request, **kwargs)

    def backend_factory() -> Session:
        """Factory function to create HTTP session with retry logic."""
        session = Session()

        # Create adapter with retry configuration and timeout
        adapter = TimeoutHTTPAdapter(max_retries=retries, timeout=timeout)

        # Mount adapter for both HTTP and HTTPS
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    # Configure huggingface_hub to use this backend
    configure_http_backend(backend_factory=backend_factory)

    # Calculate example backoff times
    backoff_times = [backoff_factor * (2 ** i) for i in range(min(max_retries, 5))]
    backoff_str = ", ".join([f"{t:.1f}s" for t in backoff_times])

    print(f"✓ HTTP retry configured: {max_retries} retries, {backoff_factor}s backoff")
    print(f"  Retry delays: {backoff_str}")
    print(f"  Retries on: connection errors, timeouts, HTTP 429/500/502/503/504")


def download_single_config(
    dataset_name: str,
    config_name: Optional[str],
    cache_dir: str,
    num_proc: Optional[int],
    download_mode: DownloadMode,
    verification_mode: VerificationMode,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Download a single dataset configuration.

    Args:
        dataset_name: HuggingFace dataset repository name
        config_name: Configuration/subset name (None for default)
        cache_dir: Cache directory path
        num_proc: Number of processes for downloading
        download_mode: Download mode (REUSE_CACHE_IF_EXISTS or FORCE_REDOWNLOAD)
        verification_mode: Verification mode for checksums

    Returns:
        Tuple of (success: bool, error_message: Optional[str], info: Optional[dict])
        - success: True if download succeeded, False otherwise
        - error_message: Error message if failed, None if succeeded
        - info: Dictionary with builder info (splits, features) if succeeded, None otherwise
    """
    try:
        # Initialize dataset builder
        builder = load_dataset_builder(
            dataset_name,
            name=config_name,
            cache_dir=cache_dir,
        )
        print(f"  Builder loaded: {builder.info.builder_name}")

        # Download and prepare dataset (resumable and cached)
        builder.download_and_prepare(
            download_mode=download_mode,
            num_proc=num_proc,
            verification_mode=verification_mode,
        )

        # Extract info for reporting
        info = {
            'splits': list(builder.info.splits.keys()) if builder.info.splits else [],
            'features': str(builder.info.features) if builder.info.features else 'N/A',
        }

        return True, None, info

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, error_msg, None


def save_failure_report(failures: Dict[str, str], dataset_name: str) -> str:
    """
    Save failure report to file.

    Args:
        failures: Dictionary mapping config names to error messages
        dataset_name: Name of the dataset

    Returns:
        Path to the saved failure report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"download_failures_{dataset_name.replace('/', '_')}_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Download Failure Report: {dataset_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total failed configurations: {len(failures)}\n\n")

        for config_name, error_msg in failures.items():
            f.write(f"Config: {config_name}\n")
            f.write(f"Error: {error_msg}\n")
            f.write("-" * 80 + "\n")

    return filename


def main():
    args = parse_args()

    print("=" * 80)
    print("Downloading dataset via HuggingFace DatasetBuilder")
    print("=" * 80)

    check_hf_authentication()

    # Configure HTTP backend with retry logic BEFORE any downloads
    setup_http_retry_backend(
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        timeout=900,
    )

    download_mode = (
        DownloadMode.FORCE_REDOWNLOAD
        if args.force_redownload
        else DownloadMode.REUSE_CACHE_IF_EXISTS
    )

    # All checks only useful to test deterioration over time
    # (hf hub files have hash as filename so could run check directly on it manually)
    verification_mode = VerificationMode.BASIC_CHECKS

    print(f"Dataset:                {args.dataset_name}")
    print(f"Subset:                 {args.subset_name or '<auto-detect all configs>'}")
    print(f"HF datasets Cache dir:  {args.cache_dir}")
    print(f"HF Hub Cache dir:       {os.environ.get('HF_HUB_CACHE') or '<default>'}")
    print(f"Num processes:          {args.num_proc or 'auto'}")
    print(f"Download mode:          {download_mode}")
    print(f"Max retries:            {args.max_retries}")
    print(f"Backoff factor:         {args.backoff_factor}")

    # Warn if split argument was provided
    if args.split != "train":
        print()
        print("⚠ WARNING: --split argument is currently ignored")
        print("  download_and_prepare() downloads all splits by default")
        print(f"  Requested split '{args.split}' will not be used")

    print()

    if args.subset_name is not None:
        # Parse comma-separated config names
        configs_to_download = [s.strip() for s in args.subset_name.split(',') if s.strip()]

        if len(configs_to_download) == 1:
            print(f"Mode: Single configuration download")
        else:
            print(f"Mode: Multiple configurations specified ({len(configs_to_download)})")
            for i, config in enumerate(configs_to_download, 1):
                print(f"  {i}. {config}")
        print()
    else:
        # No config specified - attempt to enumerate all configs
        print("Mode: Auto-detecting configurations...")
        try:
            all_configs = get_dataset_config_names(args.dataset_name)
            if all_configs:
                configs_to_download = all_configs
                print(f"Found {len(all_configs)} configurations:")
                for i, config in enumerate(all_configs, 1):
                    print(f"  {i}. {config}")
                print()
            else:
                # No configs found - use default (None)
                configs_to_download = [None]
                print("No configurations found - using default config")
                print()
        except Exception as e:
            # Failed to enumerate - use default (None)
            configs_to_download = [None]
            print(f"Could not enumerate configs ({type(e).__name__})")
            print("Proceeding with default config")
            print()

    # Download all configs
    total_configs = len(configs_to_download)
    successful_downloads: List[str] = []
    failed_downloads: Dict[str, str] = {}

    for idx, config_name in enumerate(configs_to_download, 1):
        config_display = config_name or "<default>"

        if total_configs > 1:
            print("=" * 80)
            print(f"Processing configuration {idx}/{total_configs}: {config_display}")
            print("=" * 80)

        success, error_msg, info = download_single_config(
            dataset_name=args.dataset_name,
            config_name=config_name,
            cache_dir=args.cache_dir,
            num_proc=args.num_proc,
            download_mode=download_mode,
            verification_mode=verification_mode,
        )

        if success:
            print(f"  ✅ Success: {config_display}")
            if info:
                print(f"  Splits: {info['splits']}")
                print(f"  Features: {info['features']}")
            successful_downloads.append(config_display)
        else:
            print(f"  ❌ Failed: {config_display}")
            print(f"  Error: {error_msg}")
            failed_downloads[config_display] = error_msg

        print()

    # Print summary
    print("=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Total configurations: {total_configs}")
    print(f"Successful: {len(successful_downloads)}")
    print(f"Failed: {len(failed_downloads)}")
    print()

    if successful_downloads:
        print("✅ Successfully downloaded/cached:")
        for config in successful_downloads:
            print(f"  • {config}")
        print()

    if failed_downloads:
        print("❌ Failed configurations:")
        for config, error in failed_downloads.items():
            # Truncate error message for summary
            error_short = error.split('\n')[0][:100]
            print(f"  • {config}: {error_short}")
        print()

        # Save detailed failure report
        failure_file = save_failure_report(failed_downloads, args.dataset_name)
        print(f"Detailed failure report saved to: {failure_file}")
        print()

    print(f"Cache location: {args.cache_dir}")
    print("=" * 80)

    # Return exit code: 0 if all successful, 1 if any failures
    return 0 if len(failed_downloads) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())