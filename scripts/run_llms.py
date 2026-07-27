
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from humor_detection.data import load_normalized_dataset
from humor_detection.io import atomic_write_json, append_jsonl, load_yaml, project_root, read_jsonl, sha256_file
from humor_detection.metrics import binary_classification_metrics
from humor_detection.output_parser import parse_binary_output
from humor_detection.prompts import FewShotExample, build_few_shot_prompt, build_zero_shot_prompt
from humor_detection.reproducibility import run_metadata, set_seed
from humor_detection.splitting import split_frames, validate_split_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["en", "es", "pt", "zh", "hi_en"])
    parser.add_argument("--backend", required=True, choices=["openai", "qwen"])
    parser.add_argument("--shot", required=True, choices=["zero", "few"])
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--authorize-paid-api", action="store_true", help="Required for OpenAI requests")
    parser.add_argument("--limit", type=int, help="Debug-only cap on test records; not a paper result")
    return parser.parse_args()


def prompt_hash(prompt: str, model_id: str, record_id: str) -> str:
    """Key one record/model/prompt combination for safe resumption."""
    payload = model_id + "\0" + record_id + "\0" + prompt
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_parts(root: Path, dataset: str):
    frame = load_normalized_dataset(dataset, root)
    path = root / "data" / "splits" / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    splits = json.loads(path.read_text(encoding="utf-8"))["splits"]
    validate_split_ids(splits)
    return split_frames(frame, splits), path


def choose_few_shot_examples(train, seed: int, per_class: int) -> list[FewShotExample]:
    """Choose exactly five examples per class exclusively from training data."""
    selected = []
    for label in (0, 1):
        candidates = train.loc[train["label"] == label]
        if len(candidates) < per_class:
            raise ValueError(f"Need {per_class} training examples for class {label}, found {len(candidates)}")
        sample = candidates.sample(n=per_class, random_state=seed + label)
        selected.extend(FewShotExample(text=str(row.text), label=int(row.label)) for row in sample.itertuples())
    return selected


def openai_generator(model_id: str, authorize: bool) -> Callable[[str], str]:
    if not authorize:
        raise PermissionError("OpenAI execution requires --authorize-paid-api")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install requirements-openai.txt before using the OpenAI backend") from exc
    client = OpenAI()

    def generate(prompt: str) -> str:
        response = client.responses.create(
            model=model_id,
            input=prompt,
            temperature=0.0,
            max_output_tokens=10,
        )
        return response.output_text

    return generate


def qwen_generator(model_id: str) -> Callable[[str], str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-qwen.txt before using the Qwen backend") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("The paper-aligned Qwen run requires one CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return generate


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config = load_yaml(root / "configs" / "llms.yaml")
    seed = int(config["seed"])
    set_seed(seed)
    parts, manifest_path = load_parts(root, args.dataset)
    test = parts["test"].copy()
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        test = test.head(args.limit)
    examples = []
    if args.shot == "few":
        examples = choose_few_shot_examples(
            parts["train"], seed, int(config["few_shot"]["examples_per_class"])
        )
    backend_config = config["models"][args.backend]
    model_id = backend_config["model_id"]
    destination = args.output_dir or root / "results" / "reproduced" / "llms" / args.backend / args.dataset / args.shot
    destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / "predictions.jsonl"
    cached = {record["cache_key"]: record for record in read_jsonl(cache_path)}

    pending = []
    for row in test.itertuples(index=False):
        prompt = build_zero_shot_prompt(str(row.text)) if args.shot == "zero" else build_few_shot_prompt(str(row.text), examples)
        key = prompt_hash(prompt, model_id, str(row.id))
        if key not in cached:
            pending.append((row, prompt, key))
    print(json.dumps({"test_records": len(test), "cached": len(test) - len(pending), "requests_required": len(pending)}, indent=2))
    generator = None
    if pending:
        if args.backend == "openai":
            generator = openai_generator(model_id, args.authorize_paid_api)
        else:
            generator = qwen_generator(model_id)

    for row, prompt, key in pending:
        assert generator is not None
        raw = generator(prompt)
        parsed = parse_binary_output(raw)
        record = {
            "cache_key": key,
            "id": str(row.id),
            "gold": int(row.label),
            "prediction": parsed.label,
            "parser_failure": parsed.parser_failure,
            "raw_response": raw,
            "model_id": model_id,
            "dataset": args.dataset,
            "shot": args.shot,
        }
        append_jsonl(cache_path, record)
        cached[key] = record

    ordered_records = []
    for row in test.itertuples(index=False):
        prompt = build_zero_shot_prompt(str(row.text)) if args.shot == "zero" else build_few_shot_prompt(str(row.text), examples)
        ordered_records.append(cached[prompt_hash(prompt, model_id, str(row.id))])
    gold = [int(record["gold"]) for record in ordered_records]
    predictions = [int(record["prediction"]) for record in ordered_records]
    metrics = binary_classification_metrics(gold, predictions)
    result = run_metadata(
        family="llm",
        dataset=args.dataset,
        model=backend_config["paper_name"],
        model_id=model_id,
        shot=args.shot,
        split="held_out_test" if args.limit is None else "debug_subset_not_paper_result",
        split_manifest={"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256_file(manifest_path)},
        hyperparameters={
            "temperature": backend_config.get("temperature"),
            "reported_temperature": backend_config.get("reported_temperature"),
            "decoding": backend_config.get("decoding"),
            "max_tokens": backend_config.get("max_output_tokens", backend_config.get("max_new_tokens")),
            "few_shot_examples_per_class": int(config["few_shot"]["examples_per_class"]) if args.shot == "few" else 0,
        },
        metrics={"test": metrics},
        parser_failures=sum(bool(record["parser_failure"]) for record in ordered_records),
        predictions_file=cache_path.name,
    )
    atomic_write_json(destination / "result.json", result)
    print(destination / "result.json")


if __name__ == "__main__":
    main()
