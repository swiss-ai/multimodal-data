from __future__ import annotations

import json

from sft_recaption.config import GENERATION_PROMPT_VERSION, JUDGE_PROMPT_VERSION

RSRCC_COT_PROMPT_VERSION = "gemma4_rsrcc_cot_v1"

# Only the reasoning is generated; the answer comes verbatim from source annotations.
RSRCC_COT_SCHEMA = {
    "type": "object",
    "required": ["reasoning"],
    "properties": {
        "reasoning": {"type": "string"},
    },
    "additionalProperties": False,
}


GENERATION_SCHEMA = {
    "type": "object",
    "required": ["reasoning_qa"],
    "properties": {
        "reasoning_qa": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "required": ["question", "response"],
                "properties": {
                    "question": {"type": "string"},
                    "response": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["groundedness", "specificity", "quality", "keep"],
    "properties": {
        "groundedness": {"type": "number"},
        "specificity": {"type": "number"},
        "quality": {"type": "number"},
        "keep": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "additionalProperties": False,
}


def build_generation_instruction() -> str:
    schema_text = json.dumps(GENERATION_SCHEMA, ensure_ascii=False)
    return (
        f"Prompt version: {GENERATION_PROMPT_VERSION}.\n"
        "You are generating image-grounded multimodal SFT training examples for scientific figures and document images.\n"
        "Use only what is visibly present in the provided image.\n"
        "Ignore any hidden source annotations, paper text outside the image, or prior QA.\n"
        "Return exactly one JSON object matching this schema and nothing else:\n"
        f"{schema_text}\n"
        "Requirements:\n"
        "- reasoning_qa: exactly 1 high-quality QA pair that requires visual reasoning, not just literal restatement.\n"
        "- Good reasoning types include comparison, trend interpretation, counting, spatial relation, legend reading, axis reading, or linking multiple visual elements.\n"
        "- reasoning_qa.response: the full assistant reply as one string.\n"
        "- Include the model's native reasoning markup inline in reasoning_qa.response, for example <think>...</think>, followed by the final answer.\n"
        "- Keep the reasoning and the final answer in the same response string. Do not split them into separate JSON fields.\n"
        "- The final answer after the reasoning markup should be concise and factual.\n"
        "- Return valid JSON. Escape every backslash as \\\\ and every newline inside reasoning_qa.response as \\n.\n"
        "- Questions must be answerable from the image alone.\n"
        "- Avoid trivial questions that only ask what object is shown.\n"
        "- Do not mention the source dataset, annotations, or hidden context.\n"
        "- Do not use markdown."
    )


def build_rsrcc_cot_instruction(question: str, answer: str) -> str:
    """Prompt the model to generate ONLY chain-of-thought reasoning.

    The final answer is taken verbatim from source annotations and will be
    appended at export time.  The model must NOT restate the answer inside
    the reasoning field.
    """
    schema_text = json.dumps(RSRCC_COT_SCHEMA, ensure_ascii=False)
    return (
        f"Prompt version: {RSRCC_COT_PROMPT_VERSION}.\n"
        "You are generating chain-of-thought reasoning for a satellite change detection training example.\n"
        "You are given two satellite images: the FIRST image shows a location BEFORE a change, "
        "and the SECOND image shows the same location AFTER the change.\n\n"
        f"Question:\n{question}\n\n"
        f"Correct answer: {answer}\n\n"
        "Your task: output ONLY the reasoning that leads to this answer — "
        "the answer itself will be appended separately, so do NOT restate it.\n"
        "Write as if you are genuinely examining both images side by side for the first time: "
        "let observations flow and build on each other naturally.\n\n"
        "Requirements:\n"
        "- Output reasoning text only; do not include the final answer.\n"
        "- Do NOT use numbered or bulleted steps ('1.', '2.', '-', 'First,', 'Second,', 'Then,').\n"
        "- Write as continuous prose: each observation leads organically to the next.\n"
        "- Reference specific visual details you observe when comparing the before and after images.\n"
        "- Return valid JSON. Escape every backslash as \\\\ and every newline as \\n.\n"
        f"Return exactly one JSON object matching this schema and nothing else:\n{schema_text}"
    )


def build_judge_instruction(user_text: str, assistant_text: str, task_type: str) -> str:
    schema_text = json.dumps(JUDGE_SCHEMA, ensure_ascii=False)
    return (
        f"Prompt version: {JUDGE_PROMPT_VERSION}.\n"
        "Evaluate whether this image-grounded training example is suitable for multimodal SFT.\n"
        f"Task type: {task_type}\n"
        f"User message:\n{user_text}\n\n"
        f"Assistant message:\n{assistant_text}\n\n"
        "Score each field from 0 to 5.\n"
        "- groundedness: is the answer supported by the visible content?\n"
        "- specificity: is it concrete and non-generic?\n"
        "- quality: is it clear, helpful, and well-written?\n"
        "Set keep=true only when the example is clearly usable.\n"
        "Return exactly one JSON object matching this schema and nothing else:\n"
        f"{schema_text}"
    )
