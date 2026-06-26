"""
Test script to load and verify one WebDataset shard created by preprocess.py
"""

import tarfile
import json
import io
import soundfile as sf
import numpy as np
from pathlib import Path

# Path to the WebDataset shards
WEBDATASET_PATH = "/capstor/store/cscs/swissai/infra01/audio-datasets/clean/peoples_speech_merged_webdataset"

def load_and_test_one_shard(shard_path, num_samples_to_check=5):
    """
    Load one WebDataset shard and verify the data.
    
    Args:
        shard_path: Path to a single tar shard file
        num_samples_to_check: Number of samples to inspect in detail
    """
    print(f"Loading shard: {shard_path}")
    
    samples = {}  # Group files by sample key
    
    with tarfile.open(shard_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Parse filename: {sample_id}.{extension}
                name = member.name
                if '.' in name:
                    sample_key = name.rsplit('.', 1)[0]
                    ext = name.rsplit('.', 1)[1]
                    
                    if sample_key not in samples:
                        samples[sample_key] = {}
                    
                    # Read file content
                    f = tar.extractfile(member)
                    if f:
                        samples[sample_key][ext] = f.read()
    
    print(f"Found {len(samples)} samples in shard")
    
    print("\n" + "="*80)
    print(f"Checking first {num_samples_to_check} samples:")
    print("="*80)
    
    total_duration_ms = 0
    
    for i, (sample_key, sample_data) in enumerate(samples.items()):
        if i >= num_samples_to_check:
            break
        
        print(f"\n--- Sample {i + 1} ---")
        print(f"Key: {sample_key}")
        
        # Load and check JSON metadata
        if 'json' in sample_data:
            metadata = json.loads(sample_data['json'].decode('utf-8'))
            print(f"Metadata:")
            print(f"  ID: {metadata.get('id', 'N/A')}")
            print(f"  Duration: {metadata.get('duration_ms', 'N/A')} ms ({metadata.get('duration_ms', 0)/1000:.2f} sec)")
            print(f"  Sample rate: {metadata.get('sample_rate', 'N/A')} Hz")
            print(f"  Num samples merged: {metadata.get('num_samples_merged', 'N/A')}")
            total_duration_ms += metadata.get('duration_ms', 0)
            
            # Show original IDs
            original_ids = metadata.get('original_ids', [])
            if original_ids:
                print(f"  Original IDs merged: {len(original_ids)}")
                for j, oid in enumerate(original_ids[:3]):
                    print(f"    {j+1}. {oid}")
                if len(original_ids) > 3:
                    print(f"    ... and {len(original_ids) - 3} more")
        
        # Load and check text
        if 'txt' in sample_data:
            text = sample_data['txt'].decode('utf-8')
            print(f"Text (first 150 chars): {text}...")
        
        # Load and check audio
        if 'wav' in sample_data:
            audio_buffer = io.BytesIO(sample_data['wav'])
            audio_array, sr = sf.read(audio_buffer)
            print(f"Audio: shape={audio_array.shape}, sample_rate={sr}, duration={len(audio_array)/sr:.2f}s")
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Total samples in shard: {len(samples)}")
    print(f"Samples checked in detail: {min(num_samples_to_check, len(samples))}")
    
    return samples


if __name__ == "__main__":
    print("="*80)
    print("WebDataset Verification Script (Single Shard)")
    print("="*80)
    print(f"\nWebDataset path: {WEBDATASET_PATH}")
    
    # Find all tar files
    tar_files = sorted(Path(WEBDATASET_PATH).glob("*.tar"))
    
    if not tar_files:
        print(f"No tar files found in {WEBDATASET_PATH}")
    else:
        print(f"Found {len(tar_files)} shard files:")
        for tar_file in tar_files:
            print(f"  - {tar_file.name}")
        
        # Test first shard only
        print("\n" + "="*80)
        print("Testing first shard only:")
        print("="*80)
        samples = load_and_test_one_shard(tar_files[0], num_samples_to_check=5)
    
    print("\n" + "="*80)
    print("Verification complete!")
    print("="*80)
