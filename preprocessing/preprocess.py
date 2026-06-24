
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import os
import numpy as np
from pathlib import Path
import tarfile
import json
import io
import soundfile as sf

cache_path = "/capstor/store/cscs/swissai/infra01/audio-datasets/peoples_speech_cache/MLCommons___peoples_speech"

dataset_name = "MLCommons/peoples_speech"

dataset = load_dataset(
    dataset_name, 'clean',
    cache_dir="/capstor/store/cscs/swissai/infra01/audio-datasets/peoples_speech_cache"
)

print(f"Dataset loaded: {dataset}")
print(f"Available splits: {list(dataset.keys())}")

# Get train split
train_split = dataset['train'] if 'train' in dataset else dataset

# Function to save dataset in WebDataset format
def save_to_webdataset(dataset_split, output_path, samples_per_shard=1000, split_name="train"):
    """
    Save a HuggingFace dataset split to WebDataset format.
    
    Args:
        dataset_split: The dataset split to save
        output_path: Directory where WebDataset shards will be saved
        samples_per_shard: Number of samples per tar file (shard)
        split_name: Name of the split (e.g., "train", "test")
    """
    os.makedirs(output_path, exist_ok=True)
    
    total_samples = len(dataset_split)
    num_shards = (total_samples + samples_per_shard - 1) // samples_per_shard
    
    print(f"\nSaving {total_samples} samples to WebDataset format...")
    print(f"Output directory: {output_path}")
    print(f"Samples per shard: {samples_per_shard}")
    print(f"Number of shards: {num_shards}")
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, total_samples)
        
        # Create shard filename (WebDataset format: split_name-000000.tar)
        shard_filename = f"{split_name}-{shard_idx:06d}.tar"
        shard_path = os.path.join(output_path, shard_filename)
        
        with tarfile.open(shard_path, 'w') as tar:
            for i in range(start_idx, end_idx):
                sample = dataset_split[i]
                sample_id = str(sample['id'])
                
                # Get audio data
                audio_array = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']
                
                # Convert audio array to WAV format in memory
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio_array, sample_rate, format='WAV')
                wav_buffer.seek(0)
                
                # Add audio file to tar (WebDataset format: key.wav)
                audio_info = tarfile.TarInfo(name=f"{sample_id}.wav")
                audio_info.size = len(wav_buffer.getvalue())
                tar.addfile(audio_info, wav_buffer)
                
                # Create metadata dictionary
                metadata = {
                    'id': sample_id,
                    'text': sample['text'],
                    'duration_ms': sample['duration_ms'],
                    'sample_rate': sample_rate,
                    'audio_shape': list(audio_array.shape) if hasattr(audio_array, 'shape') else None
                }
                
                # Add metadata as JSON file (WebDataset format: key.json)
                metadata_json = json.dumps(metadata).encode('utf-8')
                metadata_info = tarfile.TarInfo(name=f"{sample_id}.json")
                metadata_info.size = len(metadata_json)
                tar.addfile(metadata_info, io.BytesIO(metadata_json))
                
                # Add text file (optional, for easy text access)
                text_bytes = sample['text'].encode('utf-8')
                text_info = tarfile.TarInfo(name=f"{sample_id}.txt")
                text_info.size = len(text_bytes)
                tar.addfile(text_info, io.BytesIO(text_bytes))
        
        if (shard_idx + 1) % 10 == 0 or shard_idx == num_shards - 1:
            print(f"  Created shard {shard_idx + 1}/{num_shards}: {shard_filename} ({end_idx - start_idx} samples)")
    
    print(f"\nWebDataset saved successfully to: {output_path}")
    print(f"Total shards created: {num_shards}")
    print(f"\nTo load the WebDataset, use:")
    print(f"  import webdataset as wds")
    print(f"  dataset = wds.WebDataset('{output_path}/{split_name}-{{000000..{num_shards-1:06d}}}.tar')")

# Configuration for WebDataset export
OUTPUT_WEBDATASET_DIR = "/capstor/store/cscs/swissai/infra01/audio-datasets/clean"
SAMPLES_PER_SHARD = 10000  # 10k samples per shard

class IncrementalWebDatasetWriter:
    """Writer that saves samples incrementally to WebDataset format"""
    def __init__(self, output_path, samples_per_shard=10000, split_name="train", max_shards=None):
        self.output_path = output_path
        self.samples_per_shard = samples_per_shard
        self.split_name = split_name
        self.max_shards = max_shards  # Limit number of shards
        self.current_shard_idx = 0
        self.current_shard_count = 0
        self.current_tar = None
        self.total_samples = 0
        self.limit_reached = False
        os.makedirs(output_path, exist_ok=True)
        
    def _open_new_shard(self):
        """Open a new shard file"""
        if self.current_tar is not None:
            self.current_tar.close()
        
        shard_filename = f"{self.split_name}-{self.current_shard_idx:06d}.tar"
        shard_path = os.path.join(self.output_path, shard_filename)
        self.current_tar = tarfile.open(shard_path, 'w')
        self.current_shard_count = 0
        print(f"  Creating shard {self.current_shard_idx + 1}: {shard_filename}")
    
    def save_sample(self, merged_sample):
        """Save a merged sample to the current shard. Returns False if limit reached."""
        # Check if we've hit the shard limit
        if self.limit_reached:
            return False
            
        if self.current_tar is None or self.current_shard_count >= self.samples_per_shard:
            if self.current_tar is not None:
                self.current_tar.close()
                print(f"  Completed shard {self.current_shard_idx + 1} with {self.current_shard_count} samples")
                self.current_shard_idx += 1  # Only increment after closing a completed shard
            
            # Check if we've reached max shards
            if self.max_shards is not None and self.current_shard_idx >= self.max_shards:
                self.limit_reached = True
                print(f"\n  Reached max shard limit ({self.max_shards} shards). Stopping.")
                return False
            
            self._open_new_shard()
        
        sample_id = str(merged_sample['id'])
        audio_array = merged_sample['audio']['array']
        sample_rate = merged_sample['audio']['sampling_rate']
        
        # Convert audio array to WAV format in memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_array, sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        # Add audio file to tar
        audio_info = tarfile.TarInfo(name=f"{sample_id}.wav")
        audio_info.size = len(wav_buffer.getvalue())
        self.current_tar.addfile(audio_info, wav_buffer)
        
        # Create metadata dictionary
        metadata = {
            'id': sample_id,
            'text': merged_sample['text'],
            'duration_ms': merged_sample['duration_ms'],
            'sample_rate': sample_rate,
            'audio_shape': list(audio_array.shape) if hasattr(audio_array, 'shape') else None,
            'num_samples_merged': merged_sample.get('num_samples_merged', 1),
            'original_ids': merged_sample.get('original_ids', [])
        }
        
        # Add metadata as JSON file
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_info = tarfile.TarInfo(name=f"{sample_id}.json")
        metadata_info.size = len(metadata_json)
        self.current_tar.addfile(metadata_info, io.BytesIO(metadata_json))
        
        # Add text file
        text_bytes = merged_sample['text'].encode('utf-8')
        text_info = tarfile.TarInfo(name=f"{sample_id}.txt")
        text_info.size = len(text_bytes)
        self.current_tar.addfile(text_info, io.BytesIO(text_bytes))
        
        self.current_shard_count += 1
        self.total_samples += 1
    
    def close(self):
        """Close the current shard"""
        if self.current_tar is not None:
            self.current_tar.close()
            print(f"  Completed shard {self.current_shard_idx + 1} with {self.current_shard_count} samples")
            print(f"\nWebDataset saved successfully to: {self.output_path}")
            print(f"Total shards created: {self.current_shard_idx + 1}")
            print(f"Total samples saved: {self.total_samples}")

# WebDataset saving will be done after merging (commented out for now)
# Uncomment after merging to save merged dataset in WebDataset format
# print("\n" + "="*80)
# print("Saving dataset in WebDataset format...")
# print("="*80)
# save_to_webdataset(
#     train_split,
#     OUTPUT_WEBDATASET_DIR,
#     samples_per_shard=SAMPLES_PER_SHARD,
#     split_name="train"
# )

# Configuration
MIN_DURATION_SEC = 60  # 1 minute
MAX_DURATION_SEC = 120  # 2 minutes
OUTPUT_DIR = "/iopsstor/scratch/cscs/aditikhandelwal/speechtokens/merged_dataset"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

import re
import random

def extract_base_id_and_sequence(sample_id):
    """
    Extract base ID and sequence number from sample ID.
    Example: "07282016HFUUforum_SLASH_07-28-2016_HFUUforum_DOT_mp3_00035.flac"
    Returns: ("07282016HFUUforum_SLASH_07-28-2016_HFUUforum_DOT_mp3", 35)
    """
    sample_id_str = str(sample_id)
    # Match pattern like _00035 or _mp3_00035 before .flac
    # Look for underscore followed by digits before the file extension
    match = re.search(r'(.+?)_(\d+)(?:\.flac|\.wav|\.mp3)?$', sample_id_str)
    if match:
        base_id = match.group(1)
        sequence_num = int(match.group(2))
        return base_id, sequence_num
    # Fallback: if no pattern matches, use the whole ID as base
    return sample_id_str, None

def merge_consecutive_samples(dataset_split, min_duration=MIN_DURATION_SEC, max_duration=MAX_DURATION_SEC, 
                             save_callback=None, save_batch_size=100):
    """
    Merge consecutive samples with the same base ID to create chunks with duration 
    between min_duration and max_duration, with diverse lengths.
    
    Only merges samples that:
    1. Share the same base ID (e.g., same prefix before _00035)
    2. Have consecutive sequence numbers (e.g., _00035, _00036, _00037)
    3. Don't exceed max_duration when combined
    
    Each chunk gets a random target duration between min_duration and max_duration
    to create diversity in the output lengths.
    
    Args:
        dataset_split: The dataset split to process
        min_duration: Minimum duration in seconds (default: 60)
        max_duration: Maximum duration in seconds (default: 120 = 2min)
        save_callback: Optional callback function to save samples incrementally (sample, count)
        save_batch_size: Batch size for saving (default: 100)
    
    Returns:
        Total count of merged samples created
    """
    merged_samples_batch = []  # Small batch buffer
    merged_count = 0
    
    def get_random_target_duration():
        """Get a random target duration between min and max for diversity."""
        return random.uniform(min_duration, max_duration)
    
    current_chunk = {
        'audio_arrays': [],
        'texts': [],
        'ids': [],
        'total_duration_ms': 0,
        'sample_rate': None,
        'base_id': None,
        'last_sequence': None,
        'target_duration': get_random_target_duration()  # Random target for this chunk
    }
    
    stop_processing = False
    
    def save_merged_sample(merged_sample):
        """Save a merged sample incrementally. Returns False if should stop."""
        nonlocal merged_count, merged_samples_batch, stop_processing
        merged_count += 1
        if save_callback:
            result = save_callback(merged_sample, merged_count)
            if result is False:
                stop_processing = True
                return False
        else:
            # If no callback, keep in batch (will be returned for display)
            merged_samples_batch.append(merged_sample)
        return True
    
    total_samples = len(dataset_split)
    print(f"Processing {total_samples} samples...")
    
    for idx, sample in enumerate(dataset_split):
        # Show progress every 10k samples or at start/end
        if idx == 0 or (idx + 1) % 10000 == 0 or (idx + 1) == total_samples:
            progress_pct = ((idx + 1) / total_samples) * 100
            print(f"  Processed {idx + 1}/{total_samples} samples ({progress_pct:.1f}%) - Created {merged_count} merged samples so far...")
        sample_id = str(sample['id'])
        base_id, sequence_num = extract_base_id_and_sequence(sample_id)
        duration_sec = sample['duration_ms'] / 1000.0
        current_duration_sec = current_chunk['total_duration_ms'] / 1000.0
        
        # Check if we should start a new chunk BEFORE adding this sample:
        # 1. Different base ID
        # 2. Not consecutive sequence number  
        # 3. Adding this sample would exceed max_duration (and we have at least min_duration)
        should_start_new = False
        
        if current_chunk['base_id'] is not None:
            if base_id != current_chunk['base_id']:
                # Different base ID - start new chunk
                should_start_new = True
            elif sequence_num is not None and current_chunk['last_sequence'] is not None:
                if sequence_num != current_chunk['last_sequence'] + 1:
                    # Not consecutive - start new chunk
                    should_start_new = True
            # Check against the random target duration for this chunk (for diversity)
            # Also ensure we don't exceed absolute max_duration
            target = current_chunk.get('target_duration', max_duration)
            if (current_duration_sec >= target and current_duration_sec >= min_duration) or \
               (current_duration_sec + duration_sec > max_duration and current_duration_sec >= min_duration):
                # Either reached random target or would exceed max - start new chunk
                should_start_new = True
        
        if should_start_new and current_chunk['audio_arrays']:
            # Save current chunk if it meets minimum duration
            if current_chunk['total_duration_ms'] / 1000.0 >= min_duration:
                merged_sample = create_merged_sample(current_chunk)
                save_merged_sample(merged_sample)
                if stop_processing:
                    break
            
            # Start new chunk (this allows continuing to merge remaining samples
            # from the same base_id that exceed the max_duration limit)
            current_chunk = {
                'audio_arrays': [],
                'texts': [],
                'ids': [],
                'total_duration_ms': 0,
                'sample_rate': None,
                'base_id': None,
                'last_sequence': None,
                'target_duration': get_random_target_duration()  # New random target for diversity
            }
        
        # Add current sample to chunk (even if we just started a new chunk,
        # this ensures we continue merging consecutive samples from the same base_id)
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        current_chunk['audio_arrays'].append(audio_array)
        current_chunk['texts'].append(sample['text'])
        current_chunk['ids'].append(sample['id'])
        current_chunk['total_duration_ms'] += sample['duration_ms']
        current_chunk['base_id'] = base_id
        current_chunk['last_sequence'] = sequence_num
        if current_chunk['sample_rate'] is None:
            current_chunk['sample_rate'] = sample_rate
        
        # After adding, check if we've reached the target duration or exceeded max_duration
        # If so, finalize this chunk (will start new one on next iteration)
        new_duration_sec = current_chunk['total_duration_ms'] / 1000.0
        target = current_chunk.get('target_duration', max_duration)
        if new_duration_sec >= target and new_duration_sec >= min_duration:
            # Save this chunk since it's at or above max_duration
            merged_sample = create_merged_sample(current_chunk)
            save_merged_sample(merged_sample)
            if stop_processing:
                break
            # Reset for next chunk with new random target
            current_chunk = {
                'audio_arrays': [],
                'texts': [],
                'ids': [],
                'total_duration_ms': 0,
                'sample_rate': None,
                'base_id': None,
                'last_sequence': None,
                'target_duration': get_random_target_duration()  # New random target for diversity
            }
    
    # Handle remaining samples in the last chunk (only if not stopped)
    if not stop_processing and current_chunk['audio_arrays'] and current_chunk['total_duration_ms'] / 1000.0 >= min_duration:
        merged_sample = create_merged_sample(current_chunk)
        save_merged_sample(merged_sample)
    
    print(f"  Completed! Processed {total_samples} samples, created {merged_count} merged samples.")
    
    return merged_count, merged_samples_batch  # Return count and any remaining batch

def create_merged_sample(chunk):
    """
    Create a merged sample from a chunk of consecutive samples.
    
    Args:
        chunk: Dictionary containing audio_arrays, texts, ids, total_duration_ms, and sample_rate
    
    Returns:
        Dictionary with merged audio, text, and metadata
    """
    # Concatenate audio arrays
    merged_audio = np.concatenate(chunk['audio_arrays'])
    
    # Combine texts with space separator
    merged_text = ' '.join(chunk['texts'])
    
    # Create combined ID as base_id_{first_idx}_{last_idx}
    first_id = str(chunk['ids'][0])
    last_id = str(chunk['ids'][-1])
    first_base_id, first_seq = extract_base_id_and_sequence(first_id)
    last_base_id, last_seq = extract_base_id_and_sequence(last_id)
    
    # Use the base_id from the first sample and append first and last sequence numbers
    if first_seq is not None and last_seq is not None:
        combined_id = f"{first_base_id}_{first_seq:05d}_{last_seq:05d}"
    else:
        # Fallback if sequence numbers can't be extracted
        combined_id = f"{first_base_id}_{first_id}_{last_id}"
    
    # Get sample rate (should be the same for all samples)
    sample_rate = chunk['sample_rate'] if chunk['sample_rate'] is not None else 16000
    
    return {
        'id': combined_id,
        'audio': {
            'array': merged_audio,
            'sampling_rate': sample_rate
        },
        'text': merged_text,
        'duration_ms': chunk['total_duration_ms'],
        'num_samples_merged': len(chunk['audio_arrays']),
        'original_ids': chunk['ids']  # Store original IDs for reference
    }

# Process the dataset
print("\nMerging consecutive samples...")

# First, process a small subset to show an example
print("\n" + "="*80)
print("Processing small subset first to show example...")
print("="*80)
subset_size = min(1000, len(train_split))
subset = train_split.select(range(subset_size))
merged_count_subset, merged_subset = merge_consecutive_samples(subset)

print(f"\nCreated {merged_count_subset} merged samples from {subset_size} original samples (subset)")

# Print example merged samples from the subset
if merged_subset:
    num_examples = min(2, len(merged_subset))
    for example_idx in range(num_examples):
        print("\n" + "="*80)
        print(f"Example merged sample {example_idx + 1} (from subset):")
        print("="*80)
        example = merged_subset[example_idx]
        print(f"Combined ID: {example['id']}")
        print(f"Number of samples merged: {example['num_samples_merged']}")
        print(f"Total duration: {example['duration_ms']} ms ({example['duration_ms']/1000:.2f} seconds)")
        print(f"Audio shape: {example['audio']['array'].shape}")
        print(f"Sample rate: {example['audio']['sampling_rate']} Hz")
        print(f"\nFull merged text:")
        print(f"  {example['text']}")
        print(f"\nOriginal sample IDs that were merged:")
        original_ids = example.get('original_ids', [])
        for i, orig_id in enumerate(original_ids):
            if i < 10:  # Show first 10 IDs
                print(f"  {i+1}. {orig_id}")
        if len(original_ids) > 10:
            print(f"  ... and {len(original_ids) - 10} more")
        print("="*80)

# Now process the full dataset with incremental saving
print("\n" + "="*80)
print("Processing full dataset and saving incrementally to WebDataset...")
print("="*80)

# Create incremental WebDataset writer
webdataset_output_path = os.path.join(OUTPUT_WEBDATASET_DIR, "peoples_speech_merged_webdataset")

# Estimate total merged samples to calculate samples per shard for ~20 shards
# Based on earlier runs: ~10% of original samples become merged samples
estimated_merged = int(len(train_split) * 0.105)  # ~158k merged from 1.5M
TARGET_SHARDS = 20
samples_per_shard = max(1000, (estimated_merged // TARGET_SHARDS) + 1)  # Round up
print(f"Estimated ~{estimated_merged} merged samples, targeting {TARGET_SHARDS} shards")
print(f"Using {samples_per_shard} samples per shard")

writer = IncrementalWebDatasetWriter(
    webdataset_output_path,
    samples_per_shard=samples_per_shard,
    split_name="train",
    max_shards=None  # No limit - save all samples
)

# Create callback to save samples incrementally
def save_callback(merged_sample, count):
    return writer.save_sample(merged_sample)  # Returns False when limit reached

# Process and save incrementally (no memory accumulation)
merged_count, _ = merge_consecutive_samples(
    train_split,
    save_callback=save_callback
)

# Close the writer
writer.close()

print(f"\nCreated {merged_count} merged samples from {len(train_split)} original samples")
print(f"\nMerged dataset saved incrementally to: {webdataset_output_path}")
print("No additional dataset format needed - data is already in WebDataset format!")
