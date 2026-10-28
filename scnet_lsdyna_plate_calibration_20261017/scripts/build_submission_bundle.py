from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
ZIP_PATH = ARTIFACT_DIR / "scnet_lsdyna_plate_calibration_p0_20261017.zip"
MANIFEST_PATH = ARTIFACT_DIR / "submission_bundle_manifest.json"
FIXED_ZIP_TIME = (2026, 10, 17, 0, 0, 0)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"artifacts", "results"}:
            continue
        if relative.parts[0].startswith("manual_upload_"):
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def zip_write_bytes(archive: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    files = included_files()
    entries = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        entries.append({"path": relative, "size_bytes": len(data), "sha256": sha256_bytes(data)})
    embedded_manifest = {
        "schema_version": "hf-submission-bundle@1.0.0",
        "status": "prepared_not_submitted",
        "purpose": "SCNet LS-DYNA P0 numerical qualification and low-level plate-response reference",
        "evidence_ceiling": "high_fidelity_synthetic_reference_candidate",
        "files": entries,
    }
    manifest_bytes = (json.dumps(embedded_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(ZIP_PATH, "w") as archive:
        for path in files:
            zip_write_bytes(archive, path.relative_to(ROOT).as_posix(), path.read_bytes())
        zip_write_bytes(archive, "submission_bundle_manifest.json", manifest_bytes)
    external_manifest = dict(embedded_manifest)
    external_manifest["bundle"] = {
        "path": ZIP_PATH.name,
        "size_bytes": ZIP_PATH.stat().st_size,
        "sha256": sha256_file(ZIP_PATH),
    }
    MANIFEST_PATH.write_text(
        json.dumps(external_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(external_manifest["bundle"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
