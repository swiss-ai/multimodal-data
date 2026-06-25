#!/usr/bin/env python3
"""
Fast parallel checksum verification for HuggingFace Hub cache files.
Uses memory-mapped I/O for better performance on large files.
"""

import hashlib
import os
import mmap
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List
import argparse
from tqdm import tqdm
import multiprocessing as mp


def compute_sha256_mmap(filepath: Path) -> str:
    """Compute SHA256 hash using memory-mapped file for better performance."""
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        # For small files, just read directly
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return sha256_hash.hexdigest()
        
        if file_size < 1024 * 1024:  # Less than 1MB
            sha256_hash.update(f.read())
        else:
            # Use memory-mapped file for larger files
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                sha256_hash.update(mmapped)
    
    return sha256_hash.hexdigest()


def verify_blob_batch(blob_paths: List[Path]) -> List[Tuple[str, bool, str]]:
    """
    Verify a batch of blob files in a single process.
    Assumes HF hub structure: filename is sha256 hash
    
    Args:
        blob_paths: List of paths to verify
        
    Returns:
        List of tuples (filename, is_valid, message)
    """
    results = []
    
    for blob_path in blob_paths:
        try:
            expected_hash = blob_path.name
            
            # Skip if it doesn't look like a hash
            if len(expected_hash) != 64 or not all(c in '0123456789abcdef' for c in expected_hash):
                results.append((str(blob_path), None, "Not a hash filename"))
                continue

            actual_hash = compute_sha256_mmap(blob_path)

            is_valid = actual_hash == expected_hash
            message = "OK" if is_valid else f"MISMATCH"
            
            results.append((str(blob_path), is_valid, message))
            
        except Exception as e:
            results.append((str(blob_path), False, f"ERROR: {str(e)}"))
    
    return results


def chunk_list(lst: List, n: int) -> List[List]:
    """Split a list into n roughly equal chunks."""
    chunk_size = len(lst) // n + (1 if len(lst) % n else 0)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def find_cache_blobs(cache_dir: Path, dataset_name: str = None) -> List[Path]:
    """Find all blob files in the HuggingFace cache, including nested subsets."""
    blobs = []

    if dataset_name:
        cache_name = dataset_name.replace("/", "--")
        blob_dirs = list(cache_dir.glob(f"hub/datasets--{cache_name}/**/blobs"))
    else:
        blob_dirs = list(cache_dir.glob("hub/datasets--*/**/blobs"))

    for blob_dir in blob_dirs:
        if blob_dir.is_dir():
            blobs.extend([f for f in blob_dir.iterdir() if f.is_file()])

    return blobs


def verify_cache_optimized(cache_dir: Path, dataset_name: str = None, 
                          max_workers: int = None, batch_size: int = 10) -> dict:
    """
    Verify all blobs using optimized parallel processing (ProcessPoolExecutor) with batching.
    
    Args:
        cache_dir: Root cache directory
        dataset_name: Optional dataset name to filter by
        max_workers: Number of parallel workers (default: CPU count)
        batch_size: Number of files per batch (higher = less overhead, but less granular progress)
        
    Returns:
        Dictionary with verification results
    """
    print(f"Scanning cache directory: {cache_dir}")
    
    blobs = find_cache_blobs(cache_dir, dataset_name)
    
    if not blobs:
        print("No blobs found in cache!")
        return {"total": 0, "valid": 0, "invalid": 0, "errors": 0}
    
    print(f"Found {len(blobs)} blob files to verify")
    
    if max_workers is None:
        max_workers = os.cpu_count()
    
    print(f"Using {max_workers} parallel workers with batch size {batch_size}")
    
    # Split blobs into batches
    batches = chunk_list(blobs, max_workers * batch_size)
    
    results = {
        "total": len(blobs),
        "valid": 0,
        "invalid": 0,
        "errors": 0,
        "skipped": 0,
        "details": []
    }
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(verify_blob_batch, batch): batch for batch in batches}
        
        with tqdm(total=len(blobs), desc="Verifying blobs", unit="files") as pbar:
            for future in as_completed(future_to_batch):
                batch_results = future.result()
                
                for filepath, is_valid, message in batch_results:
                    if is_valid is None:
                        results["skipped"] += 1
                    elif is_valid:
                        results["valid"] += 1
                    elif "ERROR" in message:
                        results["errors"] += 1
                        results["details"].append((filepath, message))
                    else:
                        results["invalid"] += 1
                        results["details"].append((filepath, message))
                    
                    pbar.update(1)
    
    return results


def list_datasets(cache_dir: Path):
    """List all datasets in the cache."""
    dataset_dirs = list(cache_dir.glob("hub/datasets--*"))
    
    if not dataset_dirs:
        print("No datasets found in cache!")
        return
    
    print(f"\nDatasets in cache ({len(dataset_dirs)}):")
    print("="*60)
    
    for dataset_dir in sorted(dataset_dirs):
        # Convert cache name back to dataset name
        cache_name = dataset_dir.name.replace("datasets--", "")
        dataset_name = cache_name.replace("--", "/", 1)

        # Count blobs across all subsets recursively
        blob_dirs = list(dataset_dir.glob("**/blobs"))
        blob_count = 0
        for blob_dir in blob_dirs:
            if blob_dir.is_dir():
                blob_count += len([f for f in blob_dir.iterdir() if f.is_file()])
        print(f"  {dataset_name}: {blob_count} files")


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel checksum verification for HuggingFace Hub cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all datasets
  %(prog)s
  
  # Verify a specific dataset
  %(prog)s --dataset username/dataset-name
  
  # Use more workers for faster verification
  %(prog)s --workers 16
  
  # List all datasets in cache
  %(prog)s --list
        """
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
        help="HuggingFace cache directory (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to verify (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Files per batch (default: 10)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all datasets in cache and exit"
    )
    
    args = parser.parse_args()
    
    if not args.cache_dir.exists():
        print(f"Error: Cache directory does not exist: {args.cache_dir}")
        return 1
    
    # Handle --list
    if args.list:
        list_datasets(args.cache_dir)
        return 0
    
    # Run verification
    results = verify_cache_optimized(
        args.cache_dir, 
        args.dataset, 
        args.workers,
        args.batch_size
    )
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total files:    {results['total']}")
    print(f"Valid:          {results['valid']} ✓")
    print(f"Invalid:        {results['invalid']} ✗")
    print(f"Errors:         {results['errors']} ⚠")
    print(f"Skipped:        {results['skipped']}")
    
    if results["details"]:
        print("\n" + "="*60)
        print("ISSUES FOUND")
        print("="*60)
        for filepath, message in results["details"]:
            print(f"\n{filepath}")
            print(f"  → {message}")
    
    if results["invalid"] > 0 or results["errors"] > 0:
        print("\n⚠ Verification FAILED - issues detected!")
        return 1
    else:
        print("\n✓ All files verified successfully!")
        return 0


if __name__ == "__main__":
    # For Windows compatibility
    mp.freeze_support()
    exit(main())