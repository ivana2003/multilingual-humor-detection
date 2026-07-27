#!/usr/bin/env python3
"""Run static repository checks that do not download data or models."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

from humor_detection.io import load_yaml, project_root

SECRET_PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_assignment": re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*=\s*['\"][^'\"\n]{8,}['\"]"),
}
ABSOLUTE_PERSONAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".sh", ".sbatch", ".csv", ".cff", ""}
EXCLUDED_MODEL_PATTERNS = {
    "Phi-3": re.compile(r"(?i)phi[-_ ]?3"),
    "GPT-Neo": re.compile(r"(?i)gpt[-_ ]?neo"),
    "OPT": re.compile(r"(?i)(?:^|[^a-z])opt[-_ ]?(?:125m|350m|1\.3b|2\.7b|6\.7b|13b|30b|66b)"),
    "Llama": re.compile(r"(?i)llama"),
    "ColBERT": re.compile(r"(?i)colbert"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    return parser.parse_args()


def main() -> None:
    root = args.root.resolve() if (args := parse_args()) else project_root()
    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            problems.append(f"Python syntax: {path.relative_to(root)}: {exc}")
    for path in sorted((root / "configs").glob("*.yaml")):
        try:
            load_yaml(path)
        except Exception as exc:
            problems.append(f"YAML: {path.relative_to(root)}: {exc}")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root)
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"Potential {name}: {rel}")
        if ABSOLUTE_PERSONAL_PATH.search(text):
            problems.append(f"Absolute personal path: {rel}")
    for area in [root / "src", root / "scripts", root / "configs", root / "slurm"]:
        for path in sorted(area.rglob("*")):
            if not path.is_file() or path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for name, pattern in EXCLUDED_MODEL_PATTERNS.items():
                if pattern.search(text):
                    problems.append(f"Stale excluded model reference ({name}): {path.relative_to(root)}")

    reported = root / "results" / "paper" / "table2_reported.csv"
    reproduced = root / "results" / "reproduced"
    if not reported.exists():
        problems.append("Missing results/paper/table2_reported.csv")
    if reported.parent == reproduced or reproduced in reported.parents:
        problems.append("Reported results are not separated from reproduced results")
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    for requirement in [".env", "logs/", "checkpoints/", "results/reproduced/*", "data/raw/*"]:
        if requirement not in ignored:
            problems.append(f".gitignore is missing required pattern: {requirement}")
    if problems:
        raise SystemExit("Repository validation failed:\n- " + "\n- ".join(problems))
    print("Repository validation passed")


if __name__ == "__main__":
    main()
