# SFT Text Tokenization

Tokenization pipeline for supervised fine-tuning with per-turn loss masking and sequence packing.


## Design

<details>
<summary><b>Loss Mask Building</b></summary>

Uses Numba JIT-compiled functions to scan tokenized sequences and identify assistant turn boundaries via special tokens (`sft_assistant_begin_sequence`, `sft_eot_token`).

| Style | Start Token | End Token | Notes |
|-------|-------------|-----------|-------|
| `llama3` | `<\|start_header_id\|>assistant<\|end_header_id\|>` | `<\|eot_id\|>` | Shared EOT for all roles |
| `apertus` | `<\|assistant_start\|>` | `<\|assistant_end\|>` | Distinct end token per role |

</details>

<details>
<summary><b>Per-Turn Training Control</b></summary>

Each assistant turn can specify `"train": false` to exclude it from loss computation:

```json
{"messages": [
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "content": "Let me think...", "train": false},
  {"role": "user", "content": "Yes?"},
  {"role": "assistant", "content": "The answer is 4."}
]}
```

> **Result:** Only "The answer is 4." contributes to loss. User messages are never trained.

</details>

<details>
<summary><b>Packing & Padding</b></summary>

Uses Megatron-Bridge [packing utilities](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/953aabf75c0500180dc14a6a76cf9e7e7c4baec7/src/megatron/bridge/data/datasets/packing_utils.py).

#### Packed `.npy` Format

| Field | Description |
|:------|:------------|
| `input_ids` | Concatenated tokens, variable length (up to `pack_size`) |
| `loss_mask` | Same length, [shifted by 1](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/953aabf75c0500180dc14a6a76cf9e7e7c4baec7/src/megatron/bridge/data/datasets/packing_utils.py#L229) (`[1:] + [False]`) |
| `seq_start_id` | Start positions of each sample, e.g., `[0, 50]` |

> **Note:** No padding in stored files - padding happens at batch collation.

#### Data Flow

```
Storage (.npy)
├── input_ids:    [tok tok tok ... tok tok tok]  (96 tokens)
├── loss_mask:    [0 0 1 1 1 ... 0 0 1 1 1 0]    (96 values, pre-shifted)
└── seq_start_id: [0, 50]                         (2 samples)
         ↓
Load (__getitem__) → append len(input_ids) to get boundaries
         ↓
├── seq_boundaries: [0, 50, 96]
│   └── Sample 1: [0:50], Sample 2: [50:96]
         ↓
Collate (batch) → pad to max_length, build cu_seqlens
         ↓
├── input_ids:  [sample1][sample2][EOS EOS EOS ...]  (4096 tokens)
├── loss_mask:  [mask1  ][mask2  ][0   0   0   ...]  (4096 values)
└── cu_seqlens: [0, 49, 95, 4096]                    (attention boundaries)
```

<sub>See: [`__getitem__`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/953aabf75c0500180dc14a6a76cf9e7e7c4baec7/src/megatron/bridge/data/datasets/sft.py#L782) | [`collate_fn`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/953aabf75c0500180dc14a6a76cf9e7e7c4baec7/src/megatron/bridge/data/datasets/sft.py#L842)</sub>

</details>


## Usage

```bash
# From local JSONL
python tokenize_sft.py \
  --input data.jsonl \
  --output ./output/tokenized \
  --tokenizer /path/to/tokenizer \
  --style llama3 \
  --pack --pack_size 4096

# From HuggingFace Hub
python tokenize_sft.py \
  --input HuggingFaceM4/FineVision \
  --hf_dataset \
  --hf_subset text_openhermes_2_5 \
  --hf_split train \
  --output ./output/tokenized \
  --tokenizer /path/to/tokenizer \
  --style llama3 \
  --pack --pack_size 4096
```

## Input Formats

| Format | Structure |
|:-------|:----------|
| `messages` | `[{"role": "user/assistant", "content": "..."}]` |
| `conversations` | `[{"from": "User/Assistant", "value": "..."}]` |
| `texts` | `[{"user": "...", "assistant": "..."}]` *(FineVision)* |

## Requirements

Tokenizer must have `sft_assistant_begin_sequence` and `sft_eot_token` in config (set by `create_instruct.py`).

## Troubleshooting

<details>
<summary><code>IndexError: list index out of range</code> during packing</summary>

<br>

This error from [`create_hist`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/953aabf75c0500180dc14a6a76cf9e7e7c4baec7/src/megatron/bridge/data/datasets/packing_utils.py#L103) means a sample exceeds `max_seq_length`. The histogram has size `max_seq_length` and crashes when indexing beyond it.

> **Why?** Chat datasets (`GPTSFTChatDataset`, `SFTChatDataset`) do not truncate during tokenization unlike base `GPTSFTDataset`. Truncation only happens at runtime in `collate_fn`.

**Solution:** Filter or truncate long samples before packing.

</details>

## Tests

```bash
pytest tests/ -v
```

## TODO

- [x] Implement per-turn loss masking (`train: true/false`)
- [x] Test loss mask correctness for llama3 and apertus styles
- [x] Integrate Megatron-Bridge packing with custom `chat_preprocess`
- [ ] Test local JSONL tokenization end-to-end
- [ ] Add pre-packing filter for samples exceeding `max_seq_length`
- [ ] Multi-threaded tokenization
- [ ] Convert to Megatron-LM bin/idx format
