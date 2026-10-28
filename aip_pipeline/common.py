from __future__ import annotations

import hashlib, json, re, shutil
from datetime import datetime, timezone
from pathlib import Path

TOKEN_RE = re.compile(r"[a-zA-Z0-9À-ỹ]+")

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def digest(value: str | bytes) -> str:
    if isinstance(value, str): value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()

def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def tokens(text: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(text or "") if len(x) > 1]

def reset_dir(path: Path) -> None:
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True)
