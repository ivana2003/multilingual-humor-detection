from humor_detection.prompts import FewShotExample, build_few_shot_prompt, build_zero_shot_prompt


def test_zero_prompt_exact_appendix_template():
    assert build_zero_shot_prompt("example") == '''You are a humor classifier. Classify if the
following text is humorous or not. Respond
with ONLY a single digit: "0" (not humorous)
or "1" (humorous).

Text: example
Classification:'''


def test_few_prompt_exact_appendix_template_and_example_serialization():
    assert build_few_shot_prompt(
        "target", [FewShotExample("negative", 0), FewShotExample("positive", 1)]
    ) == '''You are a humor classifier. Classify if the
following text is humorous or not.
Respond with ONLY a single digit: "0" (not
humorous) or "1" (humorous).
Do not include any other text in your response.

Here are some examples:

Text: negative
Classification: 0

Text: positive
Classification: 1

Now classify this text:

Text: target

Classification:'''
