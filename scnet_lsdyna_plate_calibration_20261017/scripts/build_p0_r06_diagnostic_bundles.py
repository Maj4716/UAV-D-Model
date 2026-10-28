from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

from generate_p0_r06_diagnostics import DEFAULT_OUTPUT, strict_json
from validate_p0_r06_diagnostics import job_base_files, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = ROOT / "artifacts" / "r06_diagnostics_20261023"
FIXED_ZIP_TIME = (2026, 10, 23, 0, 0, 0)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def write_zip_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def finalize_staging_directory(staging: Path, output_dir: Path) -> None:
    """Atomically publish a completed bundle directory, tolerating short Windows scans."""
    for attempt in range(8):
        try:
            staging.replace(output_dir)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 7:
                raise
            time.sleep(0.05 * (2**attempt))


def build_bundles(batch_dir: Path, output_dir: Path) -> dict:
    batch_dir = batch_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == batch_dir or output_dir.is_relative_to(batch_dir):
        raise ValueError("bundle output directory must be outside the candidate batch")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle directory: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"bundle staging directory already exists: {staging}")

    validation = validate_batch(batch_dir, write_reports=False)
    if validation["overall_status"] != "pass":
        raise AssertionError("R06 batch static validation is not PASS")
    stored_readiness = strict_json(batch_dir / "R06_SUBMISSION_READINESS.json")
    if stored_readiness != validation:
        raise AssertionError("R06 stored submission-readiness report is stale")
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    design = strict_json(batch_dir / batch_manifest["design"]["path"])
    revision = design["package_revision"]
    wave_by_job: dict[str, int] = {}
    for wave_number, wave_name in enumerate(("wave_0", "wave_1", "wave_2")):
        for job_id in design["submission_sequence"].get(wave_name, []):
            wave_by_job[job_id] = wave_number
    if "wave_0" not in design["submission_sequence"]:
        wave_by_job = {
            job_id: wave_number
            for wave_number, wave_name in enumerate(("wave_1", "wave_2"), start=1)
            for job_id in design["submission_sequence"].get(wave_name, [])
        }
    staging.mkdir(parents=True)
    bundle_entries: list[dict] = []
    reports = validation["checks"]["per_job_static_validation"]["jobs"]

    for experiment in design["paired_experiments"]:
        for job in experiment["jobs"]:
            job_id = job["job_id"]
            job_dir = batch_dir / job["directory"]
            manifest = strict_json(job_dir / "manifest.json")
            stored_static = strict_json(job_dir / "static_validation.json")
            if stored_static != reports[job_id]:
                raise AssertionError(f"{job_id}: stored static-validation report is stale")
            deck_name = manifest["deck"]["path"]
            whitelist = sorted(job_base_files(manifest) | {deck_name, "static_validation.json"})
            actual = sorted(path.name for path in job_dir.iterdir() if path.is_file())
            if actual != whitelist:
                raise AssertionError(f"{job_id}: files differ from explicit whitelist: {actual}")

            payloads: dict[str, bytes] = {}
            files: list[dict] = []
            for name in whitelist:
                data = (job_dir / name).read_bytes()
                payloads[name] = data
                files.append({"path": name, "size_bytes": len(data), "sha256": sha256_bytes(data)})
            embedded = {
                "schema_version": "hf-p0-r06-diagnostic-bundle@1.0.0",
                "package_revision": revision,
                "job_id": job_id,
                "experiment_id": experiment["experiment_id"],
                "role": job["role"],
                "status": "prepared_not_submitted_not_solver_validated",
                "head_deck": deck_name,
                "file_policy": "explicit_whitelist_one_head_deck",
                "files": files,
                "prohibited_inference": "Bundle and static PASS do not establish parser/runtime, numerical response, physical validity, or P0/P1 admission.",
            }
            embedded_bytes = (json.dumps(embedded, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            zip_path = staging / f"uav_hf_{job_id}.zip"
            with zipfile.ZipFile(zip_path, "x") as archive:
                for name in whitelist:
                    write_zip_entry(archive, name, payloads[name])
                write_zip_entry(archive, "candidate_bundle_manifest.json", embedded_bytes)
            external = dict(embedded)
            external["bundle"] = {
                "path": zip_path.name,
                "size_bytes": zip_path.stat().st_size,
                "sha256": sha256_bytes(zip_path.read_bytes()),
            }
            manifest_path = staging / f"uav_hf_{job_id}.bundle_manifest.json"
            manifest_path.write_text(
                json.dumps(external, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            bundle_entries.append(
                {
                    "wave": wave_by_job[job_id],
                    "job_id": job_id,
                    "experiment_id": experiment["experiment_id"],
                    "role": job["role"],
                    "zip": external["bundle"],
                    "external_manifest": {
                        "path": manifest_path.name,
                        "size_bytes": manifest_path.stat().st_size,
                        "sha256": sha256_bytes(manifest_path.read_bytes()),
                    },
                }
            )

    release = {
        "schema_version": "hf-p0-r06-diagnostic-release@1.0.0",
        "package_revision": revision,
        "status": "prepared_not_submitted_not_solver_validated",
        "submission_policy": "Follow the frozen design wave order and do not bypass any parser/runtime/output gate.",
        "bundle_count": len(bundle_entries),
        "bundles": sorted(bundle_entries, key=lambda item: (item["wave"], item["job_id"])),
        "source_readiness": {
            "path": f"{batch_dir.name}/R06_SUBMISSION_READINESS.json",
            "sha256": sha256_bytes((batch_dir / "R06_SUBMISSION_READINESS.json").read_bytes()),
            "overall_status": validation["overall_status"],
        },
        "admission": validation["admission"],
    }
    (staging / "R06_RELEASE_MANIFEST.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    finalize_staging_directory(staging, output_dir)
    return release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic explicit-whitelist R06 diagnostic bundles")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = build_bundles(args.batch_dir, args.output_dir)
    print(json.dumps({"status": release["status"], "bundle_count": release["bundle_count"]}, indent=2))


if __name__ == "__main__":
    main()
