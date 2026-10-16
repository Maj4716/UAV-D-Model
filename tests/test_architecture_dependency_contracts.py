from __future__ import annotations

import ast
from pathlib import Path

import damage_engine.sharded_dataset as sharded_dataset
from damage_engine.input_identity import CapturedCaseInputs, CapturedCaseSource
from damage_engine.sampling_checkpoint import (
    _ReferenceFormalCaseSource,
    _ReferenceFormalFrozenInputs,
)
from damage_engine.sharded_dataset import ShardedDatasetWriter


PACKAGE_ROOT = Path(__file__).parents[1] / "damage_engine"


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _local_import_graph() -> dict[str, set[str]]:
    paths = sorted(PACKAGE_ROOT.rglob("*.py"))
    modules = {_module_name(path): path for path in paths}
    graph = {name: set() for name in modules}
    for source, path in modules.items():
        package = source if path.name == "__init__.py" else source.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package.split(".")
                    if node.level > len(prefix):
                        continue
                    base = prefix[: len(prefix) - node.level + 1]
                    if node.module:
                        base.extend(node.module.split("."))
                        candidates.append(".".join(base))
                    else:
                        candidates.extend(
                            ".".join([*base, alias.name]) for alias in node.names
                        )
                elif node.module:
                    candidates.append(node.module)
            for candidate in candidates:
                if candidate in modules and candidate != source:
                    graph[source].add(candidate)
    return graph


def _strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(tuple(sorted(component)))

    for node in graph:
        if node not in indices:
            visit(node)
    return components


def test_damage_engine_local_import_graph_has_no_cycles() -> None:
    graph = _local_import_graph()
    cycles = [component for component in _strongly_connected_components(graph) if len(component) > 1]
    assert cycles == []
    assert (
        "damage_engine.sampling_checkpoint"
        not in graph["damage_engine.sharded_dataset"]
    )
    assert (
        "damage_engine.sharded_dataset"
        not in graph["damage_engine.sampling_checkpoint"]
    )


def test_cli_uses_only_public_sampling_checkpoint_bind_api() -> None:
    path = PACKAGE_ROOT / "cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sampling_checkpoint":
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    assert not {name for name in imported if name.startswith("_")}
    assert "bind_adaptive_round_archive" in imported
    assert "bind_adaptive_round_archive" in called


def test_reference_capture_implements_neutral_input_contracts() -> None:
    assert issubclass(_ReferenceFormalCaseSource, CapturedCaseSource)
    assert issubclass(_ReferenceFormalFrozenInputs, CapturedCaseInputs)


def test_writer_private_integrity_methods_delegate_to_public_contracts(
    monkeypatch, tmp_path: Path,
) -> None:
    writer = ShardedDatasetWriter(tmp_path / "dataset", shard_size=7)
    checkpoint = {"sentinel": "checkpoint"}
    completed = {2: {"sentinel": "record"}}
    record = {"sentinel": "shard"}
    final_directory = tmp_path / "part-000002"
    validated = {"sentinel": "validated"}
    verified = {"sentinel": "verified"}
    calls: list[tuple] = []

    def validate(value, *, shard_size):
        calls.append(("checkpoint", value, shard_size))
        return validated

    def inventory(directory, records):
        calls.append(("inventory", directory, records))

    def verify(value, directory):
        calls.append(("shard", value, directory))
        return verified

    monkeypatch.setattr(sharded_dataset, "validate_sharded_checkpoint", validate)
    monkeypatch.setattr(
        sharded_dataset, "validate_committed_shard_inventory", inventory
    )
    monkeypatch.setattr(sharded_dataset, "verify_committed_shard", verify)

    assert writer._validate_checkpoint_structure(checkpoint) is validated
    assert writer._inventory_committed_directories(tmp_path, completed) is None
    assert writer._verify_completed_shard(record, final_directory) is verified
    assert calls == [
        ("checkpoint", checkpoint, 7),
        ("inventory", tmp_path, completed),
        ("shard", record, final_directory),
    ]
