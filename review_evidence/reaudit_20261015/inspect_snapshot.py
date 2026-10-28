"""Capture the current review identity and non-domain archive metadata."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION
    from damage_engine.utils import source_tree_hash
    import numpy as np
    import torch
    data = ROOT / "raw_data/dataset_5000_loitering_impact_fragment_only_v1"
    formal = data / "formal_dataset"
    manifest = json.loads((formal / "manifest.json").read_text(encoding="utf-8"))
    source_files = {
        p.relative_to(ROOT).as_posix(): sha(p)
        for directory in ("damage_engine", "tools", "tests")
        for p in sorted((ROOT / directory).rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts
    }
    for name in ("README.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt"):
        source_files[name] = sha(ROOT / name)
    shard = next(iter(sorted((formal / "shards").iterdir())))
    with np.load(shard / "tensors.npz", allow_pickle=False) as arrays:
        fields = {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)}
                  for key in arrays.files}
    report = {
        "review_date": "2026-10-15", "timezone": "Asia/Shanghai",
        "scope": "generic software integrity, availability semantics and evidence governance; no simulation or model training",
        "python": sys.executable, "python_version": sys.version,
        "dependencies": {name: version(name) for name in
                         ("numpy", "torch", "pyarrow", "pytest", "jsonschema", "scipy", "damage-dataset-engine")},
        "torch_cuda_available": torch.cuda.is_available(),
        "source_version": ENGINE_VERSION, "source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
        "engine_code_hash": ENGINE_CODE_HASH,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain": subprocess.check_output(["git", "-c", "core.quotepath=false", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"),
        "source_files": source_files,
        "archive": {"root": str(data), "manifest_hash": sha(formal / "manifest.json"),
                    "dataset_id": manifest["dataset_id"], "schema_version": manifest["schema_version"],
                    "tensor_version": manifest["tensor_contract"]["version"],
                    "engine": manifest["engine"], "label_admission": manifest.get("label_admission"),
                    "n_shards": len(list((formal / "shards").iterdir())),
                    "quality_summary": manifest.get("quality_summary"),
                    "first_shard_tensor_fields": fields},
    }
    (OUT / "snapshot.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_version": ENGINE_VERSION, "source_tree_hash": report["source_tree_hash"],
                      "engine_code_hash": ENGINE_CODE_HASH, "archive_id": manifest["dataset_id"],
                      "archive_version": report["archive"]["tensor_version"],
                      "same_declared_engine": manifest["engine"]["code_hash"] == ENGINE_CODE_HASH,
                      "shard_count": report["archive"]["n_shards"],
                      "dependencies": report["dependencies"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
