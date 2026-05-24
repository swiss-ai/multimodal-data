from __future__ import annotations

import json

from sft_recaption.config import GENERATION_PROMPT_VERSION, JUDGE_PROMPT_VERSION

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
