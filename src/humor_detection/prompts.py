from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

ZERO_SHOT_TEMPLATE = """You are a humor classifier. Classify if the
following text is humorous or not. Respond
with ONLY a single digit: \"0\" (not humorous)
or \"1\" (humorous).

Text: {text}
Classification:"""

FEW_SHOT_TEMPLATE = """You are a humor classifier. Classify if the
following text is humorous or not.
Respond with ONLY a single digit: \"0\" (not
humorous) or \"1\" (humorous).
Do not include any other text in your response.

Here are some examples:

{examples}

Now classify this text:

Text: {text}

Classification:"""


@dataclass(frozen=True)
class FewShotExample:

    text: str
    label: int


def format_examples(examples: Sequence[FewShotExample]) -> str:
    blocks: list[str] = []
    for example in examples:
        if example.label not in (0, 1):
            raise ValueError("Few-shot labels must be 0 or 1")
        blocks.append(f"Text: {example.text}\nClassification: {example.label}")
    return "\n\n".join(blocks)


def build_zero_shot_prompt(text: str) -> str:
    return ZERO_SHOT_TEMPLATE.format(text=text)


def build_few_shot_prompt(text: str, examples: Sequence[FewShotExample]) -> str:
    return FEW_SHOT_TEMPLATE.format(text=text, examples=format_examples(examples))
