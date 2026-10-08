#!/usr/bin/env python3
"""
Pytest tests for chat_preprocess loss mask correctness.

Run with: pytest test_chat_preprocess.py -v
"""

import pytest
from transformers import AutoTokenizer
from chat_preprocess import chat_preprocess


@pytest.fixture
def llama3_tokenizer():
    return AutoTokenizer.from_pretrained(
        '/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer'
    )


@pytest.fixture
def apertus_tokenizer():
    return AutoTokenizer.from_pretrained(
        '/capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_instruct_tokenizer'
    )


def decode_masked_tokens(tokenizer, input_ids, loss_mask, mask_value: bool):
    """Decode tokens where loss_mask == mask_value."""
    tokens = [tokenizer.decode([input_ids[i]]) for i in range(len(input_ids)) if loss_mask[i] == mask_value]
    return "".join(tokens)


class TestLossMaskLlama3:
    """Test loss mask for llama3 style tokenizer."""

    def test_all_train(self, llama3_tokenizer):
        """All assistant turns should be trained when no train flag specified."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "The answer is 6."},
        ]

        result = chat_preprocess({"messages": messages}, llama3_tokenizer, style="llama3")
        trained_text = decode_masked_tokens(llama3_tokenizer, result["input_ids"], result["loss_mask"], True)

        assert "The answer is 4." in trained_text, "First answer should be trained"
        assert "The answer is 6." in trained_text, "Second answer should be trained"

    def test_one_train_false(self, llama3_tokenizer):
        """First assistant turn with train=False should not be trained."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4.", "train": False},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "The answer is 6."},
        ]

        result = chat_preprocess({"messages": messages}, llama3_tokenizer, style="llama3")
        trained_text = decode_masked_tokens(llama3_tokenizer, result["input_ids"], result["loss_mask"], True)
        not_trained_text = decode_masked_tokens(llama3_tokenizer, result["input_ids"], result["loss_mask"], False)

        assert "The answer is 4." in not_trained_text, "First answer (train=False) should NOT be trained"
        assert "The answer is 6." in trained_text, "Second answer should be trained"

    def test_multi_train_false(self, llama3_tokenizer):
        """Multiple assistant turns with train=False should not be trained."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4.", "train": False},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "The answer is 6.", "train": False},
            {"role": "user", "content": "What is 4+4?"},
            {"role": "assistant", "content": "The answer is 8."},
        ]

        result = chat_preprocess({"messages": messages}, llama3_tokenizer, style="llama3")
        trained_text = decode_masked_tokens(llama3_tokenizer, result["input_ids"], result["loss_mask"], True)
        not_trained_text = decode_masked_tokens(llama3_tokenizer, result["input_ids"], result["loss_mask"], False)

        print("Trained Text:", trained_text)
        print("Not Trained Text:", not_trained_text)

        assert "The answer is 4." in not_trained_text, "First answer (train=False) should NOT be trained"
        assert "The answer is 6." in not_trained_text, "Second answer (train=False) should NOT be trained"
        assert "The answer is 8." in trained_text, "Third answer should be trained"

    def test_none_train(self, llama3_tokenizer):
        """All assistant turns with train=False - minimal training."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4.", "train": False},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "The answer is 6.", "train": False},
        ]

        result = chat_preprocess({"messages": messages}, llama3_tokenizer, style="llama3")
        not_trained_text = decode_masked_tokens(llama3_tokenizer, result["input_ids"], result["loss_mask"], False)

        assert "The answer is 4." in not_trained_text, "First answer should NOT be trained"
        assert "The answer is 6." in not_trained_text, "Second answer should NOT be trained"

    def test_user_never_trained(self, llama3_tokenizer):
        """User messages should never be trained regardless of train flags."""
        messages = [
            {"role": "user", "content": "USERQUESTION1"},
            {"role": "assistant", "content": "Answer1"},
            {"role": "user", "content": "USERQUESTION2"},
            {"role": "assistant", "content": "Answer2", "train": False},
        ]

        result = chat_preprocess({"messages": messages}, llama3_tokenizer, style="llama3")
        trained_text = decode_masked_tokens(llama3_tokenizer, result["input_ids"], result["loss_mask"], True)

        assert "USERQUESTION1" not in trained_text, "User message should not be trained"
        assert "USERQUESTION2" not in trained_text, "User message should not be trained"


class TestLossMaskApertus:
    """Test loss mask for apertus style tokenizer."""

    def test_all_train(self, apertus_tokenizer):
        """All assistant turns should be trained when no train flag specified."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result = chat_preprocess({"messages": messages}, apertus_tokenizer, style="apertus")
        trained_text = decode_masked_tokens(apertus_tokenizer, result["input_ids"], result["loss_mask"], True)

        assert "The answer is 4." in trained_text, "Answer should be trained"

    def test_one_train_false(self, apertus_tokenizer):
        """First assistant turn with train=False should not be trained."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4.", "train": False},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "The answer is 6."},
        ]

        result = chat_preprocess({"messages": messages}, apertus_tokenizer, style="apertus")
        trained_text = decode_masked_tokens(apertus_tokenizer, result["input_ids"], result["loss_mask"], True)
        not_trained_text = decode_masked_tokens(apertus_tokenizer, result["input_ids"], result["loss_mask"], False)

        print("Trained Text:", trained_text)
        print("Not Trained Text:", not_trained_text)

        assert "The answer is 4." in not_trained_text, "First answer (train=False) should NOT be trained"
        assert "The answer is 6." in trained_text, "Second answer should be trained"

    def test_user_never_trained(self, apertus_tokenizer):
        """User messages should never be trained."""
        messages = [
            {"role": "user", "content": "USERQUESTION"},
            {"role": "assistant", "content": "Answer"},
        ]

        result = chat_preprocess({"messages": messages}, apertus_tokenizer, style="apertus")
        trained_text = decode_masked_tokens(apertus_tokenizer, result["input_ids"], result["loss_mask"], True)

        assert "USERQUESTION" not in trained_text, "User message should not be trained"


class TestPacking:
    """Test packing functionality - focus on token lengths and seq_start_id correctness."""

    @pytest.fixture
    def two_multiturn_jsonl(self, tmp_path):
        """Create JSONL with 2 multi-turn conversation samples."""
        import json
        data = [
            # Sample 0: 2 turns
            {"messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "That equals 6."},
            ]},
            # Sample 1: 3 turns
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am doing well."},
                {"role": "user", "content": "Goodbye"},
                {"role": "assistant", "content": "Bye!"},
            ]},
        ]
        jsonl_path = tmp_path / "multiturn.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return jsonl_path

    def test_seq_start_id_two_multiturn_samples(self, llama3_tokenizer, two_multiturn_jsonl, tmp_path):
        """seq_start_id should correctly mark boundary between 2 multi-turn samples."""
        import json
        from chat_preprocess import prepare_packed_sft_data, chat_preprocess

        output_path = tmp_path / "packed.npy"
        metadata_path = tmp_path / "metadata.json"

        # Get individual sample lengths (raw token count, not adjusted)
        with open(two_multiturn_jsonl) as f:
            samples = [json.loads(line) for line in f]

        len_sample_0 = len(chat_preprocess(samples[0], llama3_tokenizer, style="llama3")["input_ids"])
        len_sample_1 = len(chat_preprocess(samples[1], llama3_tokenizer, style="llama3")["input_ids"])
        print(f"Sample 0 (2-turn) raw length: {len_sample_0}")
        print(f"Sample 1 (3-turn) raw length: {len_sample_1}")

        # Pack
        output_data, _ = prepare_packed_sft_data(
            input_path=two_multiturn_jsonl,
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=4096,
            tokenizer=llama3_tokenizer,
            max_seq_length=4096,
            style="llama3",
            seed=42,
        )

        # Both samples should pack into 1 sequence
        assert len(output_data) == 1, f"Expected 1 packed sequence, got {len(output_data)}"

        packed_seq = output_data[0]
        seq_start_ids = packed_seq["seq_start_id"]
        print(f"seq_start_id: {seq_start_ids}")
        print(f"Total packed length: {len(packed_seq['input_ids'])}")

        # Should have 2 start positions for 2 samples
        assert len(seq_start_ids) == 2, f"Expected 2 seq_start_ids, got {len(seq_start_ids)}"
        assert seq_start_ids[0] == 0, "First sample should start at 0"
        # Second boundary should be at first sample's raw length (packing may shuffle order)
        assert seq_start_ids[1] == len_sample_0 or seq_start_ids[1] == len_sample_1, \
            f"Second sample start should be {len_sample_0} or {len_sample_1}, got {seq_start_ids[1]}"

    def test_packed_total_length_equals_sum(self, llama3_tokenizer, two_multiturn_jsonl, tmp_path):
        """Total packed tokens should equal sum of individual multi-turn sample tokens."""
        import json
        from chat_preprocess import prepare_packed_sft_data, chat_preprocess

        output_path = tmp_path / "packed.npy"
        metadata_path = tmp_path / "metadata.json"

        # Get individual sample lengths (raw token count)
        with open(two_multiturn_jsonl) as f:
            samples = [json.loads(line) for line in f]

        expected_total = sum(
            len(chat_preprocess(s, llama3_tokenizer, style="llama3")["input_ids"])
            for s in samples
        )

        output_data, _ = prepare_packed_sft_data(
            input_path=two_multiturn_jsonl,
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=4096,
            tokenizer=llama3_tokenizer,
            max_seq_length=4096,
            style="llama3",
        )

        actual_total = sum(len(seq["input_ids"]) for seq in output_data)
        assert actual_total == expected_total, \
            f"Packed total ({actual_total}) != sum of samples ({expected_total})"

    def test_loss_mask_length_equals_input_ids(self, llama3_tokenizer, two_multiturn_jsonl, tmp_path):
        """loss_mask length should equal input_ids length in packed output."""
        from chat_preprocess import prepare_packed_sft_data

        output_path = tmp_path / "packed.npy"
        metadata_path = tmp_path / "metadata.json"

        output_data, _ = prepare_packed_sft_data(
            input_path=two_multiturn_jsonl,
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=4096,
            tokenizer=llama3_tokenizer,
            max_seq_length=4096,
            style="llama3",
        )

        for seq in output_data:
            assert len(seq["input_ids"]) == len(seq["loss_mask"]), \
                f"input_ids ({len(seq['input_ids'])}) != loss_mask ({len(seq['loss_mask'])})"

    def test_long_sample_raises_index_error(self, llama3_tokenizer, tmp_path):
        """Samples exceeding max_seq_length should raise IndexError from create_hist."""
        import json
        from chat_preprocess import prepare_packed_sft_data, chat_preprocess

        # Create a sample that exceeds max_seq_length
        long_content = "This is a long response. " * 50  # ~300+ tokens

        data = [
            {"messages": [
                {"role": "user", "content": "Tell me a story"},
                {"role": "assistant", "content": long_content},
            ]},
        ]

        # Verify sample length exceeds our pack_size
        sample_len = len(chat_preprocess(data[0], llama3_tokenizer, style="llama3")["input_ids"])
        pack_size = 100
        assert sample_len > pack_size, f"Sample ({sample_len}) should exceed pack_size ({pack_size})"

        jsonl_path = tmp_path / "long_sample.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        output_path = tmp_path / "packed.npy"
        metadata_path = tmp_path / "metadata.json"

        # Should raise IndexError because sample exceeds max_seq_length
        with pytest.raises(IndexError):
            prepare_packed_sft_data(
                input_path=jsonl_path,
                output_path=output_path,
                output_metadata_path=metadata_path,
                packed_sequence_size=pack_size,
                tokenizer=llama3_tokenizer,
                max_seq_length=pack_size,
                style="llama3",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
