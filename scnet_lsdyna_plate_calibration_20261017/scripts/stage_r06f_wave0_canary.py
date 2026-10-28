from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import sha256
from verify_p0_r06_release import verify_zip


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "artifacts" / "r06f_measurement_contract_20261027"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06f_wave0_canary_20261027"
REVISION = "R06_measurement_contract_repair_20261027"
JOB_ID = "r06f_0800_ctrl"
HEAD_DECK = "r06f_0800_ctrl.k"


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
    if release["package_revision"] != REVISION or release["bundle_count"] != 1:
        raise AssertionError("release is not the one-job frozen R06f measurement repair")
    matches = [entry for entry in release["bundles"] if entry["job_id"] == JOB_ID]
    if len(matches) != 1 or matches[0]["wave"] != 0:
        raise AssertionError("release does not contain exactly one Wave 0 recanary")
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
        raise AssertionError("Wave 0 embedded revision/job/head-deck identity differs")
    if "*DATABASE_BINARY_INTFOR_FILE" in deck or "*DATABASE_EXTENT_INTFOR" in deck:
        raise AssertionError("R06f deck unexpectedly requests unsupported binary INTFOR")
    if "*DATABASE_EXTENT_D3PART\n6,0,3,0,1,1,1,1\n0,0,0,0,0,0,0,0" not in deck:
        raise AssertionError("R06f deck does not contain the frozen D3PART NEIPH=6 cards")
    if "*DATABASE_NCFORC" not in deck:
        raise AssertionError("R06f deck lost the best-effort NCFORC request")

    staging.mkdir(parents=True)
    staged_zip = staging / zip_path.name
    staged_external = staging / external_path.name
    staged_zip.write_bytes(zip_path.read_bytes())
    staged_external.write_bytes(external_path.read_bytes())
    if sha256(staged_zip) != entry["zip"]["sha256"]:
        raise AssertionError("staged Wave 0 ZIP hash differs")
    if sha256(staged_external) != entry["external_manifest"]["sha256"]:
        raise AssertionError("staged Wave 0 external manifest hash differs")

    instructions = f"""# 只提交这个 R06f Wave 0 作业

本目录是本轮唯一提交入口。只上传：

- `{zip_path.name}`

解压后在平台中选择 `{HEAD_DECK}` 作为 head/input deck。请使用新的工作目录，
不要复用 R06e 工作目录，也不要同时提交 TSSFAC=0.4、2000 m/s 或 treatment。

## SCNet 参数

```text
job_name=uav_hf_r06f_0800_c
work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06f_0800_c
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

## 必须完整下载的回传

- submitted `.k`、solver `o=`/`d3hsp`、`mes*`；
- `glstat`、`matsum`、`sleout`、`nodout`、`binout*`；
- 完整 `d3plot*`、`d3part*`；
- stdout/stderr、Slurm 脚本和资源记录；
- 若意外产生 `intfor*` 或 `ncforc*`，也一并保留。

`intfor*` 不再是强制回传：R15.0.2 已用 Warning 20340 证明侵蚀接触不提供
binary INTFOR。`NCFORC` 仍被 best-effort 请求，但缺失不阻断本轮，且绝不能填成 0。

## 本轮通过条件

1. 无 KEY 错误，越过 cycle 0，32 ranks 正常结束；
2. run-integrity 通过，且无质量缩放；
3. D3PART 捕获账本可读；完整 D3PLOT HVAR6 可读并正确使用 alive mask；
4. 相对 Job 72397333，门前诊断 `qM/qP/qE` 绝对差均不超过 0.01，`qV` 相对差不超过 0.05；
5. missing、sentinel 和删除后归零均按 null/reason 处理，不伪装为有效零值。

即使本轮通过，下一步也只允许生成并提交 `800 m/s, TSSFAC=0.4` 单因素诊断，
不会直接释放 2000 m/s 或 treatment，也不会把本作业写入 P0/P1、引擎拟合或参考训练标签。

## ZIP 内原始说明

```text
{readme}```
"""
    instruction_path = staging / "SUBMIT_ONLY_THIS_JOB.md"
    instruction_path.write_text(instructions, encoding="utf-8", newline="\n")
    wave_manifest = {
        "schema_version": "hf-p0-r06f-wave0-stage@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_only": zip_path.name,
        "head_deck": HEAD_DECK,
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
        "next_gate_if_return_passes": "generate_and_submit_only_r06f_0800_tssfac04",
        "prohibited": [
            "submitting any R06d or R06e ZIP",
            "submitting R06f TSSFAC=0.4 before this recanary return is audited",
            "submitting R06f 2000 m/s or either treatment before the numerical gates pass",
            "treating static PASS as target-solver, response, physical, P0/P1, engine, or dataset acceptance",
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
        raise AssertionError("R06f staging must contain exactly four ordinary files and no directory")
    verify_zip(staged_zip, staged_external)
    staging.replace(output_dir)
    return wave_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the sole R06f Wave 0 recanary")
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
