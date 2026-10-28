from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from validate_p0_r05 import validate_candidate


REVISION = "R05_capture_ledger_preregistered"
FIXED_ZIP_TIME = (2026, 10, 21, 0, 0, 0)
WHITELIST = (
    "UPLOAD_README.txt",
    "jc4340_reference.json",
    "manifest.json",
    "p0_case_part_map.csv",
    "p0_cases_r05.json",
    "p0_velocity_mesh_calibration.k",
    "p1_preregistered_cases.csv",
    "static_validation.json",
)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def write_zip_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build_bundle(package_dir: Path, output_zip: Path, external_manifest: Path) -> dict:
    package_dir = package_dir.resolve()
    output_zip = output_zip.resolve()
    external_manifest = external_manifest.resolve()
    if output_zip == external_manifest:
        raise ValueError("output ZIP and external manifest paths must be different")
    for label, path in (("output ZIP", output_zip), ("external manifest", external_manifest)):
        if path.is_relative_to(package_dir):
            raise ValueError(f"{label} path must be outside the package directory")
    if output_zip.exists() or external_manifest.exists():
        raise FileExistsError("refusing to overwrite an existing R05 bundle or bundle manifest")
    actual = sorted(path.name for path in package_dir.iterdir() if path.is_file())
    if actual != sorted(WHITELIST):
        raise AssertionError(f"candidate package files differ from the frozen whitelist: {actual}")

    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    stored_validation = json.loads(
        (package_dir / "static_validation.json").read_text(encoding="utf-8")
    )
    validation = validate_candidate(package_dir)
    if manifest["package_revision"] != REVISION or validation["package_revision"] != REVISION:
        raise AssertionError("candidate revision mismatch")
    if validation["overall_status"] != "pass":
        raise AssertionError("candidate static validation is not PASS")
    if stored_validation != validation:
        raise AssertionError("candidate static validation report is stale")

    entries = []
    payloads: dict[str, bytes] = {}
    for name in WHITELIST:
        data = (package_dir / name).read_bytes()
        payloads[name] = data
        entries.append({"path": name, "size_bytes": len(data), "sha256": sha256_bytes(data)})
    embedded = {
        "schema_version": "hf-p0-candidate-bundle@1.0.0",
        "package_revision": REVISION,
        "status": "prepared_not_submitted_not_solver_validated",
        "file_policy": "explicit_whitelist",
        "files": entries,
        "prohibited_inference": "Bundle creation and static PASS do not establish solver, numerical, or physical acceptance.",
    }
    embedded_bytes = (json.dumps(embedded, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "x") as archive:
        for name in WHITELIST:
            write_zip_entry(archive, name, payloads[name])
        write_zip_entry(archive, "candidate_bundle_manifest.json", embedded_bytes)
    external = dict(embedded)
    external["bundle"] = {
        "path": output_zip.name,
        "size_bytes": output_zip.stat().st_size,
        "sha256": sha256_bytes(output_zip.read_bytes()),
    }
    external_manifest.write_text(
        json.dumps(external, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return external


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the explicit-whitelist P0 R05 candidate bundle")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bundle(args.package_dir, args.output_zip, args.external_manifest)
    print(json.dumps(result["bundle"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
