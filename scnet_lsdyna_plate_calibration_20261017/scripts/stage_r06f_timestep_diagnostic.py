from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import comparable_blocks, grouped_blocks, sha256, split_fields
from verify_p0_r06_release import verify_zip


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "artifacts" / "r06f_timestep_sensitivity_20261028"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06f_tssfac04_submission_20261028"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72617857_r06f_0800_ctrl_20261028"
    / "r06f_return_audit.json"
)
BASELINE_DECK = (
    ROOT
    / "artifacts"
    / "r06f_wave0_canary_20261027"
    / "uav_hf_r06f_0800_ctrl"
    / "r06f_0800_ctrl.k"
)
BASELINE_DECK_SHA256 = "sha256:111911562b703b8af1e96810b08edd8c31fbeea739ddddd2921b8bcd07c517af"
REVISION = "R06_timestep_sensitivity_20261028"
JOB_ID = "r06f_0800_tssfac04"
HEAD_DECK = "r06f_0800_tssfac04.k"


def _verify_authorization_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72617857"
        or decision["resource_identity"] != "pass"
        or decision["frozen_wave0_repeatability"] != "pass"
        or decision["release_r06f_0800_tssfac04"] is not True
    ):
        raise AssertionError("Job 72617857 audit does not release the timestep diagnostic")


def _verify_single_factor_deck(deck: str) -> None:
    if sha256(BASELINE_DECK) != BASELINE_DECK_SHA256:
        raise AssertionError("released Job 72617857 control deck identity changed")
    baseline = BASELINE_DECK.read_text(encoding="ascii")
    ignored = {"*TITLE", "*CONTROL_TIMESTEP"}
    if comparable_blocks(deck, ignored) != comparable_blocks(baseline, ignored):
        raise AssertionError("staged deck differs from the control outside TITLE/TSSFAC")
    candidate_card = split_fields(grouped_blocks(deck)["*CONTROL_TIMESTEP"][0][0])
    baseline_card = split_fields(grouped_blocks(baseline)["*CONTROL_TIMESTEP"][0][0])
    differences = [
        index + 1
        for index, (control, treatment) in enumerate(
            zip(baseline_card, candidate_card, strict=True)
        )
        if control != treatment
    ]
    if (
        differences != [2]
        or baseline_card[1] != "8.000000000e-01"
        or candidate_card[1] != "4.000000000e-01"
        or candidate_card[4] != "0.000000000e+00"
    ):
        raise AssertionError("staged deck must change only TSSFAC 0.8 -> 0.4 with DT2MS=0")


def stage_timestep(release_dir: Path, output_dir: Path, audit_path: Path) -> dict:
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    audit_path = audit_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite timestep staging directory: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"timestep staging directory already exists: {staging}")

    audit = strict_json(audit_path)
    _verify_authorization_audit(audit)
    release_path = release_dir / "R06_RELEASE_MANIFEST.json"
    release = strict_json(release_path)
    if release["package_revision"] != REVISION or release["bundle_count"] != 1:
        raise AssertionError("release is not the one-job frozen TSSFAC diagnostic")
    matches = [entry for entry in release["bundles"] if entry["job_id"] == JOB_ID]
    if len(matches) != 1 or matches[0]["wave"] != 0:
        raise AssertionError("release does not contain exactly one Wave 0 timestep job")
    entry = matches[0]
    zip_path = release_dir / entry["zip"]["path"]
    external_path = release_dir / entry["external_manifest"]["path"]
    zip_report = verify_zip(zip_path, external_path)

    with zipfile.ZipFile(zip_path) as archive:
        readme = archive.read("UPLOAD_README.txt").decode("utf-8")
        embedded = json.loads(archive.read("candidate_bundle_manifest.json"))
        deck = archive.read(HEAD_DECK).decode("ascii")
    if (
        embedded["package_revision"] != REVISION
        or embedded["job_id"] != JOB_ID
        or embedded["head_deck"] != HEAD_DECK
    ):
        raise AssertionError("embedded revision/job/head-deck identity differs")
    _verify_single_factor_deck(deck)

    staging.mkdir(parents=True)
    staged_zip = staging / zip_path.name
    staged_external = staging / external_path.name
    staged_zip.write_bytes(zip_path.read_bytes())
    staged_external.write_bytes(external_path.read_bytes())
    if sha256(staged_zip) != entry["zip"]["sha256"]:
        raise AssertionError("staged timestep ZIP hash differs")
    if sha256(staged_external) != entry["external_manifest"]["sha256"]:
        raise AssertionError("staged external manifest hash differs")

    instructions = f"""# 只提交这个 R06f TSSFAC=0.4 作业

本目录是本轮唯一提交入口。只上传：

- `{zip_path.name}`

解压后在平台中选择 `{HEAD_DECK}` 作为 head/input deck。必须使用新的工作目录，
不要复用 Job 72617857 或任何 R06d/R06e/R06f 控制作业目录，也不要同时提交
2000 m/s 或 treatment。

## SCNet 参数

```text
job_name=uav_hf_r06f_0800_ts04
work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06f_0800_ts04
software=ls-dyna_R15_mmp_d
nodes=1
ranks=32
walltime=04:00:00
head_memory=512MB
child_memory=256MB
command_line_parameters=<empty>
auto_fill_previous_parameters=off
head_deck={HEAD_DECK}
```

## 开跑前资源握手

必须看到 `-N 1`、`--ntasks-per-node=32`、32 total cores、`mpirun -np 32`；
求解器启动后还应出现 `MPP execution with 32 procs`。任一不符就停止该作业。

## 必须完整下载的回传

- submitted `.k`、solver `o=`/`d3hsp`、`mes*`；
- `glstat`、`matsum`、`sleout`、`nodout`、`binout*`；
- 完整 `d3plot*`、`d3part*`；
- stdout/stderr、Slurm 脚本和资源记录；
- 若产生独立 `ncforc*` 或意外 `intfor*`，也一并保留。

## 本轮因子和门禁

- 唯一因子：`CONTROL_TIMESTEP.TSSFAC` 从 0.8 改为 0.4；
- `DT2MS=0`，不得启用质量缩放；
- 基线：Job 72617857；
- `qM/qP/qE` 绝对差分别不超过 0.01；`qV/v0` 相对差不超过 0.05；
- 还必须通过正常终止、运行完整性、D3PART/HVAR6、NCFORC 和能量检查。

本包静态验证通过，但尚未经过平台 LS-DYNA 求解。它不能用于 P0/P1、引擎拟合、
物理有效性结论或参考数据集标签。

## ZIP 内原始说明

```text
{readme}```
"""
    instruction_path = staging / "SUBMIT_ONLY_THIS_JOB.md"
    instruction_path.write_text(instructions, encoding="utf-8", newline="\n")
    wave_manifest = {
        "schema_version": "hf-p0-r06f-timestep-stage@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_only": zip_path.name,
        "head_deck": HEAD_DECK,
        "parent_control": {
            "platform_job_id": "72617857",
            "deck_sha256": BASELINE_DECK_SHA256,
            "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        },
        "single_factor": {
            "field": "CONTROL_TIMESTEP.TSSFAC",
            "control": 0.8,
            "treatment": 0.4,
            "dt2ms": 0.0,
        },
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
        "next_gate_if_return_passes": "release_only_r06f_2000_control",
        "prohibited": [
            "submitting any other job in parallel with this timestep diagnostic",
            "using nodes/ranks other than one node and 32 total MPP ranks",
            "enabling mass scaling or changing any factor other than TSSFAC",
            "treating static PASS as solver, numerical, response, physical, P0/P1, engine, or dataset acceptance",
        ],
    }
    manifest_path = staging / "WAVE0_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(wave_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    expected = {
        staged_zip.name,
        staged_external.name,
        instruction_path.name,
        manifest_path.name,
    }
    actual = {path.name for path in staging.iterdir()}
    if actual != expected or not all(path.is_file() for path in staging.iterdir()):
        raise AssertionError("timestep staging must contain exactly four ordinary files")
    verify_zip(staged_zip, staged_external)
    finalize_staging_directory(staging, output_dir)
    return wave_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the sole R06f TSSFAC=0.4 job")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = stage_timestep(args.release_dir, args.output_dir, args.audit)
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
