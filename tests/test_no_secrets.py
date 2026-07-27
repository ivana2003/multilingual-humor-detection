import re
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*=\s*['\"][^'\"\n]{8,}['\"]"),
]
PERSONAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")


def test_repository_contains_no_recognizable_secrets_or_personal_paths():
    root = Path(__file__).resolve().parents[1]
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(str(path.relative_to(root)))
        if PERSONAL_PATH.search(text):
            findings.append(str(path.relative_to(root)))
    assert not findings, sorted(set(findings))
