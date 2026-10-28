"""Generic, local-only ZIP intake audit; never executes returned content."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import stat
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile

WORKSPACE = Path(__file__).resolve().parents[3]
PACKAGE = WORKSPACE / "scnet_lsdyna_plate_calibration_20261017"
ANALYSIS = Path(__file__).resolve().parent
ARCHIVE = WORKSPACE / "uav_hf_p0_plate_r04.zip"
DESTINATION = PACKAGE / "results" / "72095164_R15_mpp_d"
EXPECTED_DECK_SHA = "312f2ce1c42dd070097dad1036d1d670790704c55c5b534a3b845a0709709e08"
EXPECTED_DECK_BYTES = 20_604_806


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with (ANALYSIS / "process.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(name: str, value: object) -> None:
    (ANALYSIS / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def safe_member(info: zipfile.ZipInfo, seen: set[str]) -> None:
    name = info.filename
    posix = PurePosixPath(name.replace("\\", "/"))
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.drive or windows.root or ".." in posix.parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    if not posix.parts or any(":" in part or part.endswith((" ", ".")) for part in posix.parts):
        raise ValueError(f"Ambiguous Windows archive path: {name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
    if any(part.split(".")[0].upper() in reserved for part in posix.parts):
        raise ValueError(f"Reserved archive path: {name!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
        raise ValueError(f"Special archive entry: {name!r}")
    if info.flag_bits & 1:
        raise ValueError(f"Encrypted archive entry: {name!r}")
    key = str(posix).casefold()
    if key in seen:
        raise ValueError(f"Duplicate archive path: {name!r}")
    seen.add(key)
    candidate = (DESTINATION / Path(*posix.parts)).resolve()
    if not candidate.is_relative_to(DESTINATION.resolve()):
        raise ValueError(f"Archive path escapes destination: {name!r}")


def main() -> None:
    log("Scope: archive integrity, immutable intake, identity binding, and runtime evidence inventory only.")
    log("No physical-response extraction, model changes, remote submission, or returned-script execution.")
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite existing intake: {DESTINATION}")
    archive_identity = {
        "path": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
        "sha256": sha256(ARCHIVE), "audit_time": now(),
    }
    save_json("archive_identity.json", archive_identity)
    log(f"ZIP SHA256 {archive_identity['sha256']}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = archive.infolist()
        seen: set[str] = set()
        for info in entries:
            safe_member(info, seen)
        log(f"Path/type/duplicate/encryption checks passed: {len(entries)} entries.")
        bad = archive.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"CRC failure: {bad}")
        log("Full archive decompression and CRC check passed before extraction.")
        DESTINATION.mkdir(parents=True, exist_ok=False)
        inventory = []
        for info in entries:
            relative = PurePosixPath(info.filename.replace("\\", "/"))
            target = DESTINATION.joinpath(*relative.parts)
            item = {"name": info.filename, "bytes": info.file_size,
                    "compressed_bytes": info.compress_size, "crc32": f"{info.CRC:08x}",
                    "zip_timestamp": list(info.date_time), "directory": info.is_dir()}
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                item["sha256"] = None
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                with archive.open(info) as source, target.open("xb") as output:
                    while chunk := source.read(4 * 1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != info.file_size:
                    raise ValueError(f"Extracted length mismatch: {info.filename}")
                item["sha256"] = digest.hexdigest()
            inventory.append(item)
            if len(inventory) % 20 == 0:
                log(f"Extracted and hashed {len(inventory)}/{len(entries)} entries.")
    save_json("file_inventory.json", inventory)
    with (ANALYSIS / "file_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "bytes", "compressed_bytes", "crc32", "sha256", "directory", "zip_timestamp"])
        writer.writeheader()
        writer.writerows(inventory)
    deck = DESTINATION / "p0_velocity_mesh_calibration.k"
    deck_sha = sha256(deck)
    lines = deck.read_text(encoding="utf-8").splitlines()
    baseline = PACKAGE / "manual_upload_20261021_r04"
    identity = {
        "deck_bytes": deck.stat().st_size,
        "deck_sha256": deck_sha,
        "expected_deck_bytes": EXPECTED_DECK_BYTES,
        "expected_deck_sha256": EXPECTED_DECK_SHA,
        "deck_size_match": deck.stat().st_size == EXPECTED_DECK_BYTES,
        "deck_hash_match": deck_sha == EXPECTED_DECK_SHA,
        "deck_first_line": lines[0], "deck_line_14": lines[13],
        "baseline_comparisons": {},
    }
    for name in ["p0_velocity_mesh_calibration.k", "p0_case_part_map.csv", "manifest.json", "jc4340_reference.json", "static_validation.json"]:
        actual = DESTINATION / name
        expected = baseline / name
        identity["baseline_comparisons"][name] = {
            "returned_sha256": sha256(actual) if actual.exists() else None,
            "baseline_sha256": sha256(expected) if expected.exists() else None,
            "byte_identical": actual.exists() and expected.exists() and sha256(actual) == sha256(expected),
        }
    save_json("input_identity.json", identity)
    names = {item["name"] for item in inventory}
    expected_plots = ["d3plot", *[f"d3plot{i:02d}" for i in range(1, 11)]]
    expected_mes = [f"mes{i:04d}" for i in range(32)]
    availability = {
        "d3plot_expected_names": expected_plots,
        "d3plot_missing_names": [name for name in expected_plots if name not in names],
        "rank_logs_expected": 32,
        "rank_logs_missing_names": [name for name in expected_mes if name not in names],
        "ascii_database_presence": {name: name in names for name in ["d3hsp", "glstat", "matsum", "sleout", "nodout", "rcforc", "elout"]},
        "binout_presence": "binout0000" in names,
        "restart_file_count": sum(bool(re.fullmatch(r"d3dump\d+\.\d+", name)) for name in names),
        "standalone_log_presence": {name: name in names for name in ["status.out", "std.out.72095164", "std.err.72095164", "job_uav_hf_p0_plate_r04_72095164.log", "job_uav_hf_p0_plate_r04_72095164.out"]},
        "warning": "Filename presence and ZIP CRC do not establish binary database readability or numerical validity.",
    }
    save_json("availability.json", availability)
    save_json("audit_environment.json", {
        "time": now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "script_sha256": sha256(Path(__file__)),
        "temp": os.environ.get("TEMP"), "tmp": os.environ.get("TMP"),
    })
    log(f"Intake complete. Deck identity match: {identity['deck_hash_match']}.")
    print(json.dumps({"archive": archive_identity, "entries": len(inventory),
                      "uncompressed_bytes": sum(item["bytes"] for item in inventory),
                      "destination": str(DESTINATION), "identity": identity,
                      "availability": availability}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
