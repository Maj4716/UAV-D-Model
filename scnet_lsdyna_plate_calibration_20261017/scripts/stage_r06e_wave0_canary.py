from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import sha256
from verify_p0_r06_release import verify_zip


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "artifacts" / "r06e_parser_repair_20261025"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06e_wave0_canary_20261025"
REVISION = "R06_keyword_parser_repair_20261025"
JOB_ID = "r06e_0800_ctrl"


def stage_wave0(release_dir: Path, output_dir: Path) -> dict:
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Wave 0 staging directory: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"Wave 0 staging directory already exists: {staging}")

    release_path = release_dir / "R06_RELEASE_MANIFEST.json"
    release = strict_json(release_path)
    if release["package_revision"] != REVISION:
        raise AssertionError("release revision is not the frozen R06e parser repair")
    matches = [entry for entry in release["bundles"] if entry["job_id"] == JOB_ID]
    if len(matches) != 1 or matches[0]["wave"] != 0:
        raise AssertionError("release does not contain exactly one Wave 0 canary")
    entry = matches[0]
    zip_path = release_dir / entry["zip"]["path"]
    external_path = release_dir / entry["external_manifest"]["path"]
    zip_report = verify_zip(zip_path, external_path)

    staging.mkdir(parents=True)
    staged_zip = staging / zip_path.name
    staged_external = staging / external_path.name
    staged_zip.write_bytes(zip_path.read_bytes())
    staged_external.write_bytes(external_path.read_bytes())
    if sha256(staged_zip) != entry["zip"]["sha256"]:
        raise AssertionError("staged Wave 0 ZIP hash differs")
    if sha256(staged_external) != entry["external_manifest"]["sha256"]:
        raise AssertionError("staged Wave 0 external manifest hash differs")

    with zipfile.ZipFile(staged_zip) as archive:
        readme = archive.read("UPLOAD_README.txt").decode("utf-8")
        manifest = json.loads(archive.read("candidate_bundle_manifest.json"))
    if manifest["job_id"] != JOB_ID or manifest["head_deck"] != "r06e_0800_ctrl.k":
        raise AssertionError("Wave 0 embedded job/head-deck identity differs")

    instructions = """# 只提交这个 Wave 0 作业

本目录是 R06e 的唯一下一步提交入口。当前只上传：

- `uav_hf_r06e_0800_ctrl.zip`

在平台中选择 `r06e_0800_ctrl.k` 作为 head/input deck。不要同时提交 2000 m/s control，也不要提交两个 treatment。

## 通过条件

1. R15.0.2 不再出现 KEY+246、KEY+450、KEY+133，也不再对 INTFOR timing card 给出 KEY+457。
2. 作业越过 cycle 0 并正常终止。
3. 回传包含可读的 `d3part*` 与 `intfor*`，同时保留 `mes*`、solver stdout/stderr、job `.out`、`glstat`、`matsum`、`sleout`、`nodout`、`ncforc`、`binout*`、`d3plot*`、Slurm 脚本和资源记录。
4. 只有上述条件和 run-integrity 门全部通过，才提交 `r06e_2000_ctrl`。

## SCNet 参数

以下内容逐字来自 ZIP 内 `UPLOAD_README.txt`：

```text
""" + readme + """```
"""
    instruction_path = staging / "SUBMIT_ONLY_THIS_JOB.md"
    instruction_path.write_text(instructions, encoding="utf-8", newline="\n")
    wave_manifest = {
        "schema_version": "hf-p0-r06e-wave0-stage@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_only": zip_path.name,
        "head_deck": "r06e_0800_ctrl.k",
        "source_release_manifest": {
            "path": str(release_path),
            "sha256": sha256(release_path),
        },
        "zip": entry["zip"],
        "external_manifest": entry["external_manifest"],
        "instructions": {
            "path": instruction_path.name,
            "sha256": sha256(instruction_path),
        },
        "zip_verification": zip_report,
        "prohibited": [
            "submitting another R06e job before Wave 0 passes",
            "reusing either withdrawn R06d ZIP",
            "treating static PASS as target-solver acceptance",
            "using this diagnostic as a P0/P1 calibration or training label",
        ],
    }
    (staging / "WAVE0_MANIFEST.json").write_text(
        json.dumps(wave_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    finalize_staging_directory(staging, output_dir)
    return wave_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the sole R06e Wave 0 canary")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = stage_wave0(args.release_dir, args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "upload_only": report["upload_only"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
