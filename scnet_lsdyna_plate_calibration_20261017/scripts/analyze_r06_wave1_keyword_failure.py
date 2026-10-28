from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis" / "r06d_wave1_keyword_failure_20261025"
DEFAULT_CASES = (
    {
        "job_id": "r06d_0800_ctrl",
        "platform_job_id": "72269229",
        "zip": Path(r"D:\Downloads\uav_hf_r06d_0800_ctrl.zip"),
        "released_deck": ROOT
        / "manual_upload_20261023_r06_diagnostics"
        / "job_01_v0800_control"
        / "r06d_0800_ctrl.k",
    },
    {
        "job_id": "r06d_2000_ctrl",
        "platform_job_id": "72269336",
        "zip": Path(r"D:\Downloads\uav_hf_r06d_2000_ctrl.zip"),
        "released_deck": ROOT
        / "manual_upload_20261023_r06_diagnostics"
        / "job_03_v2000_control"
        / "r06d_2000_ctrl.k",
    },
)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def line_numbers(text: str, pattern: str) -> list[int]:
    expression = re.compile(pattern, re.IGNORECASE)
    return [
        index
        for index, line in enumerate(text.splitlines(), start=1)
        if expression.search(line)
    ]


def member_by_basename(names: list[str], basename: str) -> str:
    matches = [name for name in names if PurePosixPath(name).name == basename]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {basename!r}, found {matches}")
    return matches[0]


def member_matching(names: list[str], pattern: str) -> str:
    expression = re.compile(pattern)
    matches = [name for name in names if expression.fullmatch(PurePosixPath(name).name)]
    if len(matches) != 1:
        raise AssertionError(f"expected one member matching {pattern!r}, found {matches}")
    return matches[0]


def audit_case(case: dict) -> dict:
    zip_path: Path = case["zip"]
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        unsafe = []
        symlinks = []
        encrypted = []
        for info in infos:
            posix = PurePosixPath(info.filename)
            if posix.is_absolute() or ".." in posix.parts or "\\" in info.filename:
                unsafe.append(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                symlinks.append(info.filename)
            if info.flag_bits & 0x1:
                encrypted.append(info.filename)
        bad_crc_member = archive.testzip()
        if duplicates or unsafe or symlinks or encrypted or bad_crc_member:
            raise AssertionError(
                f"unsafe or corrupt return ZIP {zip_path}: "
                f"duplicates={duplicates}, unsafe={unsafe}, symlinks={symlinks}, "
                f"encrypted={encrypted}, bad_crc_member={bad_crc_member}"
            )

        deck_name = member_by_basename(names, f"{case['job_id']}.k")
        manifest_name = member_by_basename(names, "manifest.json")
        mes_name = member_by_basename(names, "mes0000")
        stdout_name = member_matching(names, rf"std\.out\.{case['platform_job_id']}")
        stderr_name = member_matching(names, rf"std\.err\.{case['platform_job_id']}")
        job_out_name = member_matching(names, r"job_.*\.out")
        deck_bytes = archive.read(deck_name)
        deck_text = deck_bytes.decode("ascii")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        mes_text = archive.read(mes_name).decode("utf-8", errors="replace")
        stdout_text = archive.read(stdout_name).decode("utf-8", errors="replace")
        stderr_bytes = archive.read(stderr_name)
        job_out_text = archive.read(job_out_name).decode("utf-8", errors="replace")

        deck_hash = sha256_bytes(deck_bytes)
        released_hash = sha256_file(case["released_deck"])
        if deck_hash != manifest["deck"]["sha256"] or deck_hash != released_hash:
            raise AssertionError(f"{case['job_id']}: returned/released/manifest deck mismatch")

        deck_lines = deck_text.splitlines()
        extent_index = deck_lines.index("*DATABASE_EXTENT_BINARY")
        d3part_index = deck_lines.index("*DATABASE_BINARY_D3PART")
        intfor_index = deck_lines.index("*DATABASE_BINARY_INTFOR_FILE")
        extent_data = deck_lines[extent_index + 1 : d3part_index]
        intfor_timing_fields = deck_lines[intfor_index + 2].split(",")
        if len(extent_data) != 1:
            raise AssertionError(f"{case['job_id']}: expected the submitted one-card EXTENT_BINARY defect")
        if len(intfor_timing_fields) != 5:
            raise AssertionError(f"{case['job_id']}: expected the submitted five-field INTFOR defect")

        error_10246 = line_numbers(mes_text, r"Error 10246")
        error_10450 = line_numbers(mes_text, r"Error 10450")
        warning_10457 = line_numbers(mes_text, r"Warning 10457")
        error_10133 = line_numbers(mes_text, r"Error 10133")
        if not all((error_10246, error_10450, warning_10457, error_10133)):
            raise AssertionError(f"{case['job_id']}: expected R15 keyword-error chain is incomplete")
        if not line_numbers(mes_text, r"Beginning of keyword reader"):
            raise AssertionError(f"{case['job_id']}: keyword-reader start not found")
        if not line_numbers(stdout_text, r"checked\s+out.*[\"']dyna[\"']"):
            raise AssertionError(f"{case['job_id']}: successful dyna license checkout not found")

        required_output_prefixes = (
            "d3plot",
            "d3part",
            "intfor",
            "glstat",
            "matsum",
            "sleout",
            "nodout",
            "ncforc",
            "binout",
        )
        present_outputs = {
            prefix: sorted(
                name
                for name in names
                if PurePosixPath(name).name.lower().startswith(prefix)
            )
            for prefix in required_output_prefixes
        }
        if any(present_outputs.values()):
            raise AssertionError(
                f"{case['job_id']}: expected an early parser failure without solver outputs"
            )

        return {
            "job_id": case["job_id"],
            "platform_job_id": case["platform_job_id"],
            "zip": {
                "path": str(zip_path),
                "size_bytes": zip_path.stat().st_size,
                "sha256": sha256_file(zip_path),
                "member_count": len(infos),
                "uncompressed_bytes": sum(info.file_size for info in infos),
                "crc": "pass",
                "path_safety": "pass",
            },
            "deck": {
                "member": deck_name,
                "sha256": deck_hash,
                "manifest_match": True,
                "released_package_match": True,
                "extent_binary_card_count": len(extent_data),
                "intfor_timing_field_count": len(intfor_timing_fields),
            },
            "execution": {
                "target_solver": "LS-DYNA R15.0.2 MPP double precision",
                "license_checkout": "pass",
                "keyword_reader_started": True,
                "time_integration_started": False,
                "normal_termination": False,
                "stderr_empty": not stderr_bytes,
            },
            "diagnostics": {
                "mes0000_member": mes_name,
                "error_10246_lines": error_10246,
                "error_10450_lines": error_10450,
                "warning_10457_lines": warning_10457,
                "error_10133_lines": error_10133,
                "job_out_cycle_zero_lines": line_numbers(job_out_text, r"cycle\s*=\s*0"),
                "required_solver_outputs": present_outputs,
            },
            "root_cause": (
                "DATABASE_EXTENT_BINARY Card 2 is absent, so the R15 reader "
                "consumes DATABASE_BINARY_D3PART as data; the following D3PART "
                "line then produces a cascading keyword-command error."
            ),
            "independent_compatibility_warning": (
                "DATABASE_BINARY_INTFOR_FILE timing Card 1 has five fields but "
                "the returned R15.0.2 parser accepts four."
            ),
        }


def write_reports(cases: list[dict], output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite failure analysis: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"failure-analysis staging directory exists: {staging}")
    staging.mkdir(parents=True)
    report = {
        "schema_version": "hf-p0-r06d-wave1-keyword-failure@1.0.0",
        "analysis_date": "2026-10-25",
        "overall_status": "root_cause_confirmed",
        "failure_stage": "target_R15_keyword_reader_before_time_integration",
        "common_cause": (
            "Both controls contain the same incomplete DATABASE_EXTENT_BINARY "
            "block and the same over-wide INTFOR timing card."
        ),
        "cases": cases,
        "repair_contract": {
            "database_extent_binary_card1": [6, 0, 3, 0, 1, 1, 1, 1],
            "database_extent_binary_card2": [0, 0, 0, 1, 1, 1, 2, 1],
            "intfor_file_timing_card": [0.00005, 0, 0, 0],
            "intfor_file_ioopt_card": [0],
            "d3part_five_field_card": [0.00005, 0, 0, 0, 900001],
        },
        "excluded_causes": [
            "upload or return deck mutation",
            "license checkout failure",
            "scheduler launch failure",
            "MPI rank-count mismatch",
            "memory exhaustion",
            "numerical instability after time integration",
        ],
        "claim_boundary": (
            "The returns establish a parser-contract failure only. They contain "
            "no numerical or physical response evidence and cannot qualify P0, "
            "P1, engine fitting, or dataset labels."
        ),
    }
    json_path = staging / "failure_diagnosis.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    worklog = [
        "# R06d Wave 1 关键词失败审计工作日志",
        "",
        "- 对两份下载 ZIP 做只读清单、CRC、路径安全、哈希和 manifest 绑定检查；未修改原始 ZIP。",
        "- 两份回传 deck 均与本地发布版逐字节一致，排除平台改写和误上传。",
        "- 两份作业均成功启动 R15.0.2、取得许可证并进入 keyword reader。",
        "- 首个致命错误均为 `*DATABASE_EXTENT_BINARY` 缺 Card 2；D3PART 报错是解析错位的级联结果。",
        "- `*DATABASE_BINARY_INTFOR_FILE` 的五字段 timing card 另触发 `KEY+457`，目标 parser 只接受四字段。",
        "- 两份回传均无 d3plot/d3part/intfor/glstat 等积分结果，因此不做数值或物理解释。",
        "",
        "## 哈希绑定",
        "",
    ]
    for case in cases:
        worklog.extend(
            [
                f"- `{case['job_id']}` / Job `{case['platform_job_id']}`：",
                f"  - ZIP `{case['zip']['sha256']}`；{case['zip']['member_count']} 个成员；CRC PASS。",
                f"  - deck `{case['deck']['sha256']}`；manifest 与本地发布版一致。",
            ]
        )
    worklog.extend(
        [
            "",
            "## 修复边界",
            "",
            "修复只允许改共享数据库输出卡片的序列化；网格、材料、EOS、接触、时间步、终止时间、速度、边界条件和两个预注册单因素均保持不变。",
            "",
        ]
    )
    (staging / "worklog.md").write_text("\n".join(worklog), encoding="utf-8")
    plan = """# R06e 下一次提交计划

1. 只提交 `r06e_0800_ctrl`（Wave 0），验证 R15.0.2 关键词解析和完整运行。
2. Wave 0 必须同时满足：无 KEY 错误、修复卡片无 excess-field warning、cycle > 0、正常终止、D3PART 与 INTFOR 均可读。
3. Wave 0 通过后只提交 `r06e_2000_ctrl`；两个 control 都通过 run-integrity 与输出可读性后，才进入 treatment。
4. 最后提交 `r06e_0800_tend008` 与 `r06e_2000_psfail_plate`，分别只与自己的 control 配对解释。
5. 任一阶段失败即停止；不得把失败或 unavailable 响应写成 0，也不得把诊断作业加入参考训练分布。
"""
    (staging / "next_submission_plan.md").write_text(plan, encoding="utf-8")
    staging.replace(output_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the two returned R06d Wave 1 ZIPs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [audit_case(case) for case in DEFAULT_CASES]
    report = write_reports(cases, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "case_count": len(report["cases"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
