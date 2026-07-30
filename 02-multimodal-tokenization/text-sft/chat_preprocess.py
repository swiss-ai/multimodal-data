"""Efficient SFT chat preprocessing with per-turn masking and packing for Megatron-Bridge."""
import json
import logging
from pathlib import Path

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


@njit(cache=True)
def _build_mask_llama3(input_ids, train_flags, assistant_header, eot):
    """
    Mask builder for Llama3-style tokenizers.

    Template format:
        <|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>

    Note:
        Returns mask of length N (same as input_ids). Caller (collate_fn) shifts
        with [1:] to get length N-1, aligned with labels for loss computation.

    Args:
        input_ids: Token IDs (length N)
        train_flags: Per-turn training flags for assistant turns
        assistant_header: 3-token array [128006, 78191, 128007] for
            <|start_header_id|>assistant<|end_header_id|>
        eot: End-of-turn token ID (128009 for <|eot_id|>)

    Returns:
        mask: Boolean array of length N, True for tokens to train on
    """
    n = len(input_ids)
    mask = np.zeros(n, dtype=np.bool_)
    turn_idx, in_assistant, should_train = 0, False, True
    h0, h1, h2 = assistant_header[0], assistant_header[1], assistant_header[2]

    for i in range(2, n):
        # Mask first (before state changes)
        if in_assistant and should_train:
            mask[i] = True
        # Detect header at h2 (so content starts from next token)
        if input_ids[i-2] == h0 and input_ids[i-1] == h1 and input_ids[i] == h2:
            in_assistant = True
            should_train = train_flags[turn_idx] if turn_idx < len(train_flags) else True
        # EOT check
        if input_ids[i] == eot:
            if in_assistant:
                turn_idx += 1
            in_assistant = False
    return mask


@njit(cache=True)
def _build_mask_apertus(input_ids, train_flags, assistant_start, assistant_end):
    """
    Mask builder for Apertus-style tokenizers.

    Template format:
        <|user_start|>{user_msg}<|user_end|><|assistant_start|>{assistant_msg}<|assistant_end|>

    Note:
        Returns mask of length N (same as input_ids). Caller (collate_fn) shifts
        with [1:] to get length N-1, aligned with labels for loss computation.

    Args:
        input_ids: Token IDs (length N)
        train_flags: Per-turn training flags for assistant turns
        assistant_start: Single token ID for <|assistant_start|> (e.g., 67)
        assistant_end: Single token ID for <|assistant_end|> (e.g., 68)

    Returns:
        mask: Boolean array of length N, True for tokens to train on
    """
    n = len(input_ids)
    mask = np.zeros(n, dtype=np.bool_)
    turn_idx, in_assistant, should_train = 0, False, True

    for i in range(n):
        # Mask first (before state changes)
        if in_assistant and should_train:
            mask[i] = True
        # Detect assistant_start
        if input_ids[i] == assistant_start:
            in_assistant = True
            should_train = train_flags[turn_idx] if turn_idx < len(train_flags) else True
        # Detect assistant_end
        if input_ids[i] == assistant_end:
            if in_assistant:
                turn_idx += 1
            in_assistant = False
    return mask


def _ensure_special_tokens(input_ids, mask, bos_id, eos_id, add_bos, add_eos):
    """Handle BOS/EOS tokens: BOS never trained, EOS always trained."""
    # BOS: prepend if needed, never train on it
    if bos_id is not None:
        if add_bos and (len(input_ids) == 0 or input_ids[0] != bos_id):
            input_ids = np.concatenate([[bos_id], input_ids])
            mask = np.concatenate([[False], mask])
        elif len(input_ids) > 0 and input_ids[0] == bos_id:
            mask[0] = False

    # EOS: append if needed, train on it
    if add_eos and eos_id is not None and (len(input_ids) == 0 or input_ids[-1] != eos_id):
        input_ids = np.concatenate([input_ids, [eos_id]])
        mask = np.concatenate([mask, [True]])

    return input_ids, mask


def chat_preprocess(
    source: dict,
    tokenizer,
    style: str = "apertus",
    add_bos: bool = True,
    add_eos: bool = True,
) -> dict:
    """
    Preprocess chat messages with per-turn masking support.

    Args:
        source: {"messages": [{"role": "user/assistant", "content": "...", "train": bool}]}
        tokenizer: HuggingFace tokenizer with apply_chat_template
        style: "apertus" or "llama3" - determines mask building strategy
        add_bos: Add BOS token if not present (default: False)
        add_eos: Add EOS token if not present (default: True)

    Returns:
        dict with:
            input_ids: Token IDs (length N)
            loss_mask: Boolean mask (length N), caller shifts with [1:] to align with labels
            context_ids: Context portion of input_ids
            answer_ids: Answer portion of input_ids

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/apertus_tokenizer")
        >>> result = chat_preprocess(
        ...     {"messages": [
        ...         {"role": "user", "content": "What is 2+2?"},
        ...         {"role": "assistant", "content": "2+2 is 5", "train": False},  # mask this
        ...         {"role": "user", "content": "That's wrong"},
        ...         {"role": "assistant", "content": "Sorry, 2+2=4"},  # train on this
        ...     ]},
        ...     tokenizer,
        ...     style="apertus"
        ... )
        >>> result["loss_mask"].sum()  # only second assistant turn
        10
    """
    # Handle different message formats: messages, conversations, texts (FineVision)
    if "messages" in source:
        messages = source["messages"]
    elif "conversations" in source:
        messages = source["conversations"]
    elif "texts" in source:
        # FineVision format: {"texts": [{"user": ..., "assistant": ...}]}
        messages = []
        for turn in source["texts"]:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"], "train": turn.get("train", True)})
    else:
        messages = []

    # Normalize format
    chat = [
        {"role": m.get("role", m.get("from", "")).lower(), "content": m.get("content", m.get("value", ""))}
        for m in messages
    ]

    # Get train flags for assistant turns
    train_flags = np.array(
        [m.get("train", True) for m in messages if m.get("role", m.get("from", "")).lower() == "assistant"],
        dtype=np.bool_
    )

    # Tokenize
    input_ids = np.array(
        tokenizer.apply_chat_template(chat, tokenize=True, tools=source.get("tools")),
        dtype=np.int64
    )

    # Get SFT tokens from tokenizer config (set by create_instruct.py)
    # Check attribute first, then init_kwargs (HuggingFace stores custom config fields there)
    assistant_begin = tokenizer.init_kwargs.get('sft_assistant_begin_sequence')
    eot = tokenizer.init_kwargs.get('sft_eot_token')

    if assistant_begin is None or eot is None:
        raise ValueError(
            "Tokenizer missing sft_assistant_begin_sequence or sft_eot_token. "
            "Use omni_tokenizer/create_instruct.py to create the tokenizer."
        )

    # Build mask with JIT-compiled function
    if style == "apertus":
        mask = _build_mask_apertus(input_ids, train_flags, assistant_begin[0], eot[0])
    elif style == "llama3":
        mask = _build_mask_llama3(input_ids, train_flags, assistant_begin, eot[0])
    else:
        raise ValueError(f"Unknown style: {style}. Use 'apertus' or 'llama3'")

    # Handle BOS/EOS tokens
    input_ids, mask = _ensure_special_tokens(
        input_ids, mask,
        getattr(tokenizer, 'bos_token_id', None),
        getattr(tokenizer, 'eos_token_id', None),
        add_bos, add_eos
    )

    # Context/answer split at last non-masked position
    zero_indices = np.where(~mask)[0]
    last_zero = zero_indices[-1] + 1 if len(zero_indices) > 0 else len(mask)

    return {
        "input_ids": input_ids,
        "loss_mask": mask,
        "context_ids": input_ids[:last_zero],
        "answer_ids": input_ids[last_zero:],
    }


# Import here to avoid circular dependency
from megatron.bridge.data.datasets.sft import GPTSFTChatDataset


class SFTChatDataset(GPTSFTChatDataset):
    """
    SFT Chat Dataset using efficient Numba-JIT mask builders.

    Inherits from GPTSFTChatDataset, overrides _process_example to use
    chat_preprocess with _build_mask_apertus or _build_mask_llama3.

    Usage:
        from chat_preprocess import SFTChatDataset

        # From JSONL file
        dataset = SFTChatDataset(
            file_path="training.jsonl",
            tokenizer=tokenizer,
            max_seq_length=8096,
            style="apertus",
        )

        # From HuggingFace Hub dataset
        dataset = SFTChatDataset(
            file_path="HuggingFaceM4/FineVision",
            tokenizer=tokenizer,
            max_seq_length=8096,
            style="llama3",
            hf_dataset=True,
            hf_dataset_subset="text_openhermes_2_5",
            hf_dataset_split="train",
        )
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        style: str = "apertus",
        add_bos: bool = False,
        add_eos: bool = True,
        hf_dataset_subset: str = None,
        hf_dataset_split: str = "train",
        **kwargs,
    ):
        self.style = style
        self._add_bos = add_bos
        self._add_eos = add_eos
        self._hf_dataset_subset = hf_dataset_subset
        self._hf_dataset_split = hf_dataset_split

        # Wrap HuggingFace tokenizer with Megatron-compatible wrapper
        from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer
        wrapped_tokenizer = HuggingFaceTokenizer(
            tokenizer_path=tokenizer.name_or_path,
            chat_template=tokenizer.chat_template,
        )

        # Pass use_hf_tokenizer_chat_template=False to skip parent's validation
        # We override _process_example and handle chat template ourselves
        super().__init__(
            file_path=file_path,
            tokenizer=wrapped_tokenizer,
            use_hf_tokenizer_chat_template=False,
            add_bos=add_bos,
            add_eos=add_eos,
            **kwargs,
        )

    def _load_dataset(self):
        """Load dataset from JSONL file or HuggingFace Hub."""
        from datasets import load_dataset
        from megatron.bridge.data.datasets.utils import _JSONLMemMapDataset

        if self.hf_dataset:
            # Check if file_path is a local file or HuggingFace Hub dataset
            if Path(self.file_path).exists():
                # Local JSONL file
                self.indexed_dataset = load_dataset(
                    "json",
                    data_files=self.file_path,
                    cache_dir=self.index_mapping_dir,
                    num_proc=self.memmap_workers,
                    split="train",
                )
            else:
                # HuggingFace Hub dataset (e.g., HuggingFaceM4/FineVision)
                self.indexed_dataset = load_dataset(
                    self.file_path,
                    name=self._hf_dataset_subset,
                    cache_dir=self.index_mapping_dir,
                    split=self._hf_dataset_split,
                )
        else:
            # Use memory-mapped JSONL loader
            self.indexed_dataset = _JSONLMemMapDataset(
                dataset_paths=[self.file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=self.index_mapping_dir,
                workers=self.memmap_workers,
            )

    def _process_example(self, example):
        """Override to use our chat_preprocess with Numba-JIT mask builders."""
        result = chat_preprocess(
            source=example,
            tokenizer=self.tokenizer.tokenizer,  # Access underlying HF tokenizer
            style=self.style,
            add_bos=self._add_bos,
            add_eos=self._add_eos,
        )

        # Add metadata for compatibility
        result["metadata"] = {
            k: v for k, v in example.items()
            if k not in ["conversations", "messages"]
        }

        return result


def prepare_packed_sft_data(
    input_path: Path,
    output_path: Path,
    output_metadata_path: Path,
    packed_sequence_size: int,
    tokenizer,
    max_seq_length: int,
    style: str = "apertus",
    seed: int = 0,
    packing_algorithm: str = "first_fit_shuffle",
    add_bos: bool = False,
    add_eos: bool = True,
    **dataset_kwargs,
):
    """
    Prepare packed sequence data using SFTChatDataset with Numba-JIT mask builders.

    Uses Megatron-Bridge's packing utilities (create_hist, create_packing_strategy,
    fill_packing_strategy) with our custom SFTChatDataset for tokenization.

    Args:
        input_path: Path to input JSONL dataset file.
        output_path: Path to save packed .npy file.
        output_metadata_path: Path to save packing metadata JSON.
        packed_sequence_size: Maximum size for each packed sequence.
        tokenizer: HuggingFace tokenizer with sft_assistant_begin_sequence and sft_eot_token.
        max_seq_length: Maximum sequence length for truncation.
        style: "apertus" or "llama3" - determines mask building strategy.
        seed: Random seed for shuffling.
        packing_algorithm: "first_fit_shuffle" or "first_fit_decreasing".
        add_bos: Add BOS token if not present.
        add_eos: Add EOS token if not present.
        **dataset_kwargs: Additional kwargs passed to SFTChatDataset.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/apertus_instruct_tokenizer")
        >>> prepare_packed_sft_data(
        ...     input_path=Path("train.jsonl"),
        ...     output_path=Path("train_packed.npy"),
        ...     output_metadata_path=Path("packing_metadata.json"),
        ...     packed_sequence_size=8192,
        ...     tokenizer=tokenizer,
        ...     max_seq_length=8192,
        ...     style="apertus",
        ... )
    """
    from megatron.bridge.data.datasets.packing_utils import (
        create_hist,
        create_packing_strategy,
        fill_packing_strategy,
    )

    logger.info(f"Preparing packed sequence from {input_path} using SFTChatDataset")

    # Create dataset using our SFTChatDataset with Numba-JIT mask builders
    dataset = SFTChatDataset(
        file_path=str(input_path),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        style=style,
        add_bos=add_bos,
        add_eos=add_eos,
        seed=seed,
        is_test=True,  # Disable shuffling for deterministic packing
        **dataset_kwargs,
    )

    # Tokenize all examples
    logger.info(f"Tokenizing {len(dataset)} examples...")
    tokenized_data = np.array([dataset[i] for i in range(len(dataset))])

    # Use Megatron-Bridge packing utilities
    sequences, histogram = create_hist(tokenized_data, max_seq_length)
    assignments, packing_metadata = create_packing_strategy(
        histogram, packed_sequence_size, packing_algorithm
    )
    output_data = fill_packing_strategy(
        assignments, sequences, packed_sequence_size, tokenizer.eos_token_id
    )

    # Save packed data
    np.save(output_path, output_data)
    logger.info(f"Saved packed data to {output_path}")

    # Save/append packing metadata
    if output_metadata_path is not None:
        try:
            with open(output_metadata_path, "r") as f:
                packing_metadata_file = json.load(f)
                assert isinstance(packing_metadata_file, list)
        except FileNotFoundError:
            packing_metadata_file = []

        packing_metadata_file.append(packing_metadata)
        with open(output_metadata_path, "w") as f:
            json.dump(packing_metadata_file, f, indent=2)
        logger.info(f"Saved packing metadata to {output_metadata_path}")

    return output_data, packing_metadata
