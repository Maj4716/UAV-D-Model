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
DEFAULT_RELEASE = ROOT / "artifacts" / "r06g_timestep_refinement_20261028_v2"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06g_tssfac02_submission_20261028_v2"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72630855_r06f_0800_tssfac04_20261028"
    / "r06f_tssfac04_return_audit.json"
)
BASELINE_DECK = (
    ROOT
    / "manual_upload_20261028_r06f_timestep_sensitivity"
    / "job_01_v0800_tssfac04"
    / "r06f_0800_tssfac04.k"
)
BASELINE_DECK_SHA256 = "sha256:afb0f34112ccfdfcfd95bf442c9b45e2cb1590fdad8d76e98fbbc3fb5d6baee7"
REVISION = "R06_timestep_refinement_20261028"
JOB_ID = "r06g_0800_tssfac02"
HEAD_DECK = "r06g_0800_tssfac02.k"


def _verify_authorization_audit(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72630855"
        or audit["source_evidence"]["deck_sha256"] != BASELINE_DECK_SHA256
        or decision["solver_parser_runtime"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["timestep_sensitivity_gate"] != "fail"
        or decision["release_r06g_0800_tssfac02"] is not True
    ):
        raise AssertionError("Job 72630855 audit does not authorize the TSSFAC=0.2 refinement")


def _verify_single_factor(deck: str) -> None:
    if sha256(BASELINE_DECK) != BASELINE_DECK_SHA256:
        raise AssertionError("immutable TSSFAC=0.4 parent deck hash differs")
    baseline = BASELINE_DECK.read_text(encoding="ascii")
    if comparable_blocks(deck, {"*TITLE", "*CONTROL_TIMESTEP"}) != comparable_blocks(
        baseline, {"*TITLE", "*CONTROL_TIMESTEP"}
    ):
        raise AssertionError("refinement deck differs from TSSFAC=0.4 parent outside TITLE/TSSFAC")
    candidate_timestep = split_fields(grouped_blocks(deck)["*CONTROL_TIMESTEP"][0][0])
    baseline_timestep = split_fields(grouped_blocks(baseline)["*CONTROL_TIMESTEP"][0][0])
    differences = [
        index + 1
        for index, (control, treatment) in enumerate(
            zip(baseline_timestep, candidate_timestep, strict=True)
        )
        if control != treatment
    ]
    if (
        differences != [2]
        or baseline_timestep[1] != "4.000000000e-01"
        or candidate_timestep[1] != "2.000000000e-01"
        or candidate_timestep[4] != "0.000000000e+00"
    ):
        raise AssertionError("refinement deck must change only TSSFAC 0.4 -> 0.2 with DT2MS=0")


def stage_refinement(release_dir: Path, output_dir: Path, audit_path: Path) -> dict:
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    audit_path = audit_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite refinement staging directory: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"refinement staging directory already exists: {staging}")

    audit = strict_json(audit_path)
    _verify_authorization_audit(audit)
    release_path = release_dir / "R06_RELEASE_MANIFEST.json"
    release = strict_json(release_path)
    if release["package_revision"] != REVISION or release["bundle_count"] != 1:
        raise AssertionError("release is not the one-job frozen TSSFAC=0.2 refinement")
    matches = [entry for entry in release["bundles"] if entry["job_id"] == JOB_ID]
    if len(matches) != 1 or matches[0]["wave"] != 0:
        raise AssertionError("release does not contain exactly one Wave 0 refinement job")
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
    _verify_single_factor(deck)

    staging.mkdir(parents=True)
    staged_zip = staging / zip_path.name
    staged_external = staging / external_path.name
    staged_zip.write_bytes(zip_path.read_bytes())
    staged_external.write_bytes(external_path.read_bytes())
    if sha256(staged_zip) != entry["zip"]["sha256"]:
        raise AssertionError("staged refinement ZIP hash differs")
    if sha256(staged_external) != entry["external_manifest"]["sha256"]:
        raise AssertionError("staged refinement external manifest hash differs")

    instructions = f"""# 只提交这个 R06g TSSFAC=0.2 细化作业

本目录是本轮唯一提交入口。只上传：

- `{zip_path.name}`

解压后在平台中选择 `{HEAD_DECK}` 作为 head/input deck。必须使用新的工作目录，
不要复用 Job 72630855、Job 72617857 或其他 R06 目录，也不要同时提交 2000 m/s
或 treatment。

## SCNet 参数

```text
job_name=uav_hf_r06g_0800_ts02
work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06g_0800_ts02
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

- 唯一因子：`CONTROL_TIMESTEP.TSSFAC` 从 0.4 改为 0.2；
- `DT2MS=0`，不得启用质量缩放；
- 外部控制：Job 72630855；
- `qM/qP/qE` 绝对差分别不超过 0.01；
- deleted-upstream mass fraction 绝对差不超过 0.01；
- qV 仅在两边均可用时比较 5% 相对差；若两边均为零确认穿越质量，qV 才可记为不适用；
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
        "schema_version": "hf-p0-r06g-timestep-refinement-stage@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_only": zip_path.name,
        "head_deck": HEAD_DECK,
        "parent_control": {
            "platform_job_id": "72630855",
            "deck_sha256": BASELINE_DECK_SHA256,
            "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        },
        "single_factor": {
            "field": "CONTROL_TIMESTEP.TSSFAC",
            "control": 0.4,
            "treatment": 0.2,
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
        "next_gate_if_return_passes": "audit_local_timestep_convergence_before_any_2000_control",
        "prohibited": [
            "submitting any other job in parallel with this refinement diagnostic",
            "using nodes/ranks other than one node and 32 total MPP ranks",
            "enabling mass scaling or changing any factor other than TSSFAC",
            "treating static PASS as solver, convergence, physical, P0/P1, engine, or dataset acceptance",
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
        raise AssertionError("refinement staging must contain exactly four ordinary files")
    verify_zip(staged_zip, staged_external)
    finalize_staging_directory(staging, output_dir)
    return wave_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the sole R06g TSSFAC=0.2 job")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = stage_refinement(args.release_dir, args.output_dir, args.audit)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output_dir.resolve()),
                "upload_only": manifest["upload_only"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
