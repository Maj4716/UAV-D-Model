from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from build_p0_r06_diagnostic_bundles import DEFAULT_BUNDLE_DIR
from generate_p0_r06_diagnostics import DEFAULT_OUTPUT, strict_json
from validate_p0_r06_diagnostics import sha256, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R05_RETURN = ROOT / "05"
DEFAULT_R05_RUN_MANIFEST = ROOT / "analysis" / "72252362_r05_stage_ab_20261023" / "run_artifact_manifest.json"
DEFAULT_REPORT_DIR = ROOT / "analysis" / "r06_submission_readiness_20261023"
EXPECTED_R05_FULL_INVENTORY = {"file_count": 120, "total_bytes": 3606867357}
EXPECTED_R05_PACKAGE_HASHES = {
    "p0_velocity_mesh_calibration.k": "sha256:6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c",
    "p0_cases_r05.json": "sha256:b1d2027d57fc6d9cd1ee3f932ea044fd16a112344d8f2b6f508071f45191a06f",
    "p0_case_part_map.csv": "sha256:b05b59249e49c3e9d15904f475c2f9478bdff1638fb797d1b552298490aa5295",
    "manifest.json": "sha256:e178b6c28832519aaa2befdc969dcd93d2141ea4073ad740b52b17968d92ba44",
    "static_validation.json": "sha256:491753050734ba46ded5460221e35a70e37bbccd179b6b172d753bfc18c8b17e",
}


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def verify_zip(zip_path: Path, external_manifest_path: Path) -> dict:
    external = strict_json(external_manifest_path)
    bundle = external["bundle"]
    if bundle["path"] != zip_path.name:
        fail(f"external manifest names the wrong ZIP: {external_manifest_path.name}")
    if bundle["size_bytes"] != zip_path.stat().st_size or bundle["sha256"] != sha256(zip_path):
        fail(f"ZIP identity mismatch: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            fail(f"duplicate ZIP member names: {zip_path.name}")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                fail(f"unsafe ZIP member {name!r} in {zip_path.name}")
        if "candidate_bundle_manifest.json" not in names:
            fail(f"missing embedded manifest: {zip_path.name}")
        embedded = json.loads(
            archive.read("candidate_bundle_manifest.json").decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value in {zip_path.name}: {value}")
            ),
        )
        external_without_bundle = {key: value for key, value in external.items() if key != "bundle"}
        if embedded != external_without_bundle:
            fail(f"embedded/external manifest mismatch: {zip_path.name}")
        expected_names = sorted(entry["path"] for entry in embedded["files"]) + ["candidate_bundle_manifest.json"]
        if sorted(names) != sorted(expected_names):
            fail(f"ZIP members differ from embedded whitelist: {zip_path.name}")
        if len([name for name in names if name.endswith(".k")]) != 1:
            fail(f"ZIP must contain exactly one head deck: {zip_path.name}")
        if embedded["head_deck"] not in names:
            fail(f"embedded head deck is absent: {zip_path.name}")
        for entry in embedded["files"]:
            data = archive.read(entry["path"])
            if len(data) != entry["size_bytes"] or sha256_bytes(data) != entry["sha256"]:
                fail(f"ZIP member identity mismatch: {zip_path.name}/{entry['path']}")
    return {
        "status": "pass",
        "job_id": external["job_id"],
        "zip": zip_path.name,
        "size_bytes": zip_path.stat().st_size,
        "sha256": sha256(zip_path),
        "member_count": len(names),
        "head_deck": external["head_deck"],
    }


def verify_r05_immutability(return_dir: Path, manifest_path: Path) -> dict:
    manifest = strict_json(manifest_path)
    if manifest["run_id"] != "72252362" or manifest["file_count"] != 75:
        fail("R05 run manifest identity differs from frozen Stage A evidence")
    verified_bytes = 0
    for entry in manifest["files"]:
        path = return_dir / entry["path"]
        if not path.is_file():
            fail(f"R05 frozen evidence file is missing: {entry['path']}")
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            fail(f"R05 frozen evidence identity changed: {entry['path']}")
        verified_bytes += path.stat().st_size
    if verified_bytes != manifest["total_bytes"]:
        fail("R05 frozen evidence byte total differs from manifest")
    all_files = [path for path in return_dir.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in all_files)
    if len(all_files) != EXPECTED_R05_FULL_INVENTORY["file_count"] or total_bytes != EXPECTED_R05_FULL_INVENTORY["total_bytes"]:
        fail("R05 full return inventory count or byte total changed")
    return {
        "status": "pass",
        "run_id": "72252362",
        "hash_verified_file_count": manifest["file_count"],
        "hash_verified_total_bytes": verified_bytes,
        "full_inventory_file_count": len(all_files),
        "full_inventory_total_bytes": total_bytes,
    }


def verify_release(
    *,
    batch_dir: Path,
    bundle_dir: Path,
    r05_return_dir: Path,
    r05_run_manifest: Path,
) -> dict:
    readiness = validate_batch(batch_dir, write_reports=False)
    if readiness != strict_json(batch_dir / "R06_SUBMISSION_READINESS.json"):
        fail("stored R06 readiness report is stale")
    release = strict_json(bundle_dir / "R06_RELEASE_MANIFEST.json")
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    design = strict_json(batch_dir / batch_manifest["design"]["path"])
    revision = design["package_revision"]
    expected_bundle_count = int(batch_manifest["job_count"])
    if (
        release["package_revision"] != revision
        or release["bundle_count"] != expected_bundle_count
        or len(release["bundles"]) != expected_bundle_count
    ):
        fail("R06 release manifest identity or bundle count mismatch")
    if release["source_readiness"]["sha256"] != sha256(batch_dir / "R06_SUBMISSION_READINESS.json"):
        fail("release does not bind the current R06 readiness report")
    zip_reports = []
    for entry in release["bundles"]:
        zip_path = bundle_dir / entry["zip"]["path"]
        external_path = bundle_dir / entry["external_manifest"]["path"]
        if sha256(zip_path) != entry["zip"]["sha256"] or zip_path.stat().st_size != entry["zip"]["size_bytes"]:
            fail(f"release ZIP binding mismatch: {zip_path.name}")
        if sha256(external_path) != entry["external_manifest"]["sha256"] or external_path.stat().st_size != entry["external_manifest"]["size_bytes"]:
            fail(f"release external-manifest binding mismatch: {external_path.name}")
        zip_reports.append(verify_zip(zip_path, external_path))
    expected_files = {"R06_RELEASE_MANIFEST.json"}
    for entry in release["bundles"]:
        expected_files.add(entry["zip"]["path"])
        expected_files.add(entry["external_manifest"]["path"])
    actual_files = {path.name for path in bundle_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        fail(f"release directory contains files outside the explicit manifest: {sorted(actual_files - expected_files)}")

    historical_package = ROOT / "manual_upload_20261021_r05_candidate"
    historical_hashes = {}
    for name, expected in EXPECTED_R05_PACKAGE_HASHES.items():
        actual = sha256(historical_package / name)
        if actual != expected:
            fail(f"historical R05 candidate changed: {name}")
        historical_hashes[name] = actual
    r05 = verify_r05_immutability(r05_return_dir, r05_run_manifest)
    return {
        "schema_version": "hf-p0-r06-release-verification@1.0.0",
        "package_revision": revision,
        "overall_status": "pass",
        "lifecycle_status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "checks": {
            "fresh_static_readiness": {"status": "pass", "sha256": sha256(batch_dir / "R06_SUBMISSION_READINESS.json")},
            "explicit_release_file_set": {"status": "pass", "file_count": len(actual_files)},
            "zip_internal_external_identity": {"status": "pass", "bundles": zip_reports},
            "historical_r05_candidate_immutable": {"status": "pass", "files": historical_hashes},
            "historical_r05_return_immutable": r05,
        },
        "submission_order": design["submission_sequence"],
        "local_solver_precheck": {
            "status": "unavailable",
            "reason": "No local LS-DYNA executable was found; target parser/runtime acceptance is an external submission gate.",
        },
        "admission": readiness["admission"],
        "claim_boundary": "PASS means the R06 diagnostic release is internally consistent and ready for user submission. It does not mean solver, numerical, response, or physical acceptance.",
    }


def write_verification(report: dict, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite release-verification directory: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"release-verification staging directory exists: {staging}")
    staging.mkdir(parents=True)
    report_path = staging / "r06_release_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(output_dir)
    return output_dir / report_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently verify the R06 diagnostic release")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--r05-return-dir", type=Path, default=DEFAULT_R05_RETURN)
    parser.add_argument("--r05-run-manifest", type=Path, default=DEFAULT_R05_RUN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_release(
        batch_dir=args.batch_dir.resolve(),
        bundle_dir=args.bundle_dir.resolve(),
        r05_return_dir=args.r05_return_dir.resolve(),
        r05_run_manifest=args.r05_run_manifest.resolve(),
    )
    path = write_verification(report, args.output_dir.resolve())
    print(json.dumps({"overall_status": report["overall_status"], "report": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
