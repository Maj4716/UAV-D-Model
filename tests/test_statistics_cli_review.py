"""Regression cases contain only artificial records, Boolean events and numbers."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from damage_engine import cli
from damage_engine.damage_tree import compile_damage_tree
from damage_engine.errors import DamageTreeError, InputValidationError
from damage_engine.utils import confidence_interval, parameter_value, statistics_dict


def test_one_observation_has_mean_but_no_estimable_variance():
    mean, std, low, high = confidence_interval(np.array([0.2]))
    assert mean == 0.2
    assert all(math.isnan(value) for value in (std, low, high))
    stats = statistics_dict([0.2], probability=True)
    assert stats["sample_count"] == 1
    assert stats["uncertainty_estimable"] is False
    assert stats["stddev"] is None and stats["ci95_lower"] is None and stats["ci95_upper"] is None
    assert json.loads(json.dumps(stats, allow_nan=False))["mean"] == 0.2


def test_empty_statistics_are_explicitly_unavailable():
    stats = statistics_dict([])
    assert stats["mean"] is None
    assert stats["sample_count"] == 0 and not stats["uncertainty_estimable"]
    json.dumps(stats, allow_nan=False)


def test_constant_repeated_series_keeps_empirical_zero_dispersion():
    stats = statistics_dict([0.2] * 32)
    assert stats["uncertainty_estimable"]
    assert stats["stddev"] == pytest.approx(0)
    assert stats["coverage_status"] == "fixed_sample_approximation_not_optional_stopping_validated"


def test_interval_bounds_do_not_authorize_uniform_sampling():
    parameter = {"value": 2.0, "uncertainty": {"kind": "interval", "parameters": {"minimum": 1.0, "maximum": 3.0}}}
    assert parameter_value(parameter) == 2.0
    with pytest.raises(InputValidationError, match="bounds only"):
        parameter_value(parameter, np.random.default_rng(7))


def boolean_tree(*, double_not=False):
    nodes = [
        {"id": "A", "node_type": "leaf", "inputs": [], "component_id": "toy", "mechanism": "combined", "damage_state": "event"},
        {"id": "not_A", "node_type": "not", "inputs": ["A"]},
    ]
    right = "not_A"
    if double_not:
        nodes.append({"id": "A_again", "node_type": "not", "inputs": ["not_A"]})
        right = "A_again"
    nodes.append({"id": "out", "node_type": "or", "inputs": ["A", right], "dependency_model_id": "fully_dependent"})
    return {"components": [{"id": "toy", "damage_states": ["event"]}], "damage_tree": {"version": "toy-boolean-v1", "nodes": nodes, "outputs": ["out"]}}


def test_complementary_inputs_cannot_declare_positive_dependence():
    with pytest.raises(DamageTreeError, match="opposed event polarity"):
        compile_damage_tree(boolean_tree())


def test_double_negation_has_no_polarity_conflict():
    assert compile_damage_tree(boolean_tree(double_not=True)).outputs == ("out",)


@pytest.mark.parametrize("text,suffix", [("", ".jsonl"), (" \n\n", ".jsonl"), ("[]", ".json")])
def test_empty_case_input_is_rejected(tmp_path, text, suffix):
    path = tmp_path / ("empty" + suffix)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InputValidationError, match="non-empty"):
        cli._load_cases(path)


@pytest.mark.parametrize("text,suffix", [("null", ".json"), ("[{} , 42]", ".json"), ("{}\n[]\n", ".jsonl")])
def test_nonobject_case_record_is_rejected(tmp_path, text, suffix):
    path = tmp_path / ("invalid" + suffix)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InputValidationError, match="must be an object"):
        cli._load_cases(path)


def test_jsonl_error_records_real_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{}\n\n{"id":\n', encoding="utf-8")
    with pytest.raises(InputValidationError) as error:
        cli._load_cases(path)
    assert error.value.code == "REF_INPUT_INVALID"
    assert error.value.context["line"] == 3
    assert error.value.context["path"] == str(path.resolve())


def test_single_case_command_refuses_truncation_before_engine(tmp_path, monkeypatch):
    path = tmp_path / "multiple.json"
    path.write_text('[{"id":"first"},{"id":"second"}]', encoding="utf-8")
    monkeypatch.setattr(cli, "_engine", lambda args: pytest.fail("must reject before loading engine"))
    with pytest.raises(InputValidationError, match="exactly one"):
        cli.command_simulate(SimpleNamespace(case=str(path)))


def test_cli_empty_validation_reports_refusal_before_engine(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert cli.main(["validate", "--target", "unused", "--physics", "unused", "--case", str(path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["error"]["code"] == "REF_INPUT_INVALID"


def test_valid_one_case_and_bom_are_accepted(tmp_path):
    path = tmp_path / "one.jsonl"
    path.write_text('{"id":"toy"}\n', encoding="utf-8-sig")
    assert cli._load_single_case(path) == {"id": "toy"}


def test_invalid_utf8_jsonl_has_record_location(tmp_path):
    path = tmp_path / "bad-encoding.jsonl"
    path.write_bytes(b'{}\n\xff\n')
    with pytest.raises(InputValidationError, match="UTF-8") as error:
        cli._load_cases(path)
    assert error.value.context["line"] == 2


def test_cli_validation_freezes_relative_source_from_case_directory(tmp_path, monkeypatch):
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    (case_dir / "source.json").write_text("[]", encoding="utf-8")
    path = case_dir / "cases.json"
    path.write_text(json.dumps({"warhead": {"fragment_source": {"fragment_list_ref": "source.json"}}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source.json").write_text('["wrong cwd source"]', encoding="utf-8")
    monkeypatch.setattr(cli, "_engine", lambda _: SimpleNamespace(target_data={}, registry={}))
    observed = []
    def inspect(target, case, registry):
        observed.append(case)
        from pathlib import Path
        assert Path(case["warhead"]["fragment_source"]["fragment_list_ref"]).read_text() == "[]"
        assert case.input_identity["locations"][0]["resolved_path"] == str(case_dir / "source.json")
    monkeypatch.setattr(cli, "validate_inputs", inspect)
    assert cli.command_validate(SimpleNamespace(case=str(path))) == 0
    assert len(observed) == 1


def test_cli_simulation_entry_freezes_inputs_without_domain_execution(tmp_path, monkeypatch, capsys):
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    (case_dir / "source.json").write_text("[]", encoding="utf-8")
    path = case_dir / "one.json"
    path.write_text(json.dumps({"warhead": {"fragment_source": {"fragment_list_ref": "source.json"}}}), encoding="utf-8")
    def inspect(case, **kwargs):
        from pathlib import Path
        assert hasattr(case, "input_identity")
        assert Path(case["warhead"]["fragment_source"]["fragment_list_ref"]).read_text() == "[]"
        return SimpleNamespace(status="ok", to_dict=lambda: {"status": "ok", "scope": "artificial metadata stub"})
    monkeypatch.setattr(cli, "_engine", lambda _: SimpleNamespace(run=inspect))
    assert cli.command_simulate(SimpleNamespace(case=str(path), record_refusal=False, output=None)) == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "artificial metadata stub"


def test_invalid_generation_numeric_argument_refused_before_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_engine", lambda _: pytest.fail("invalid worker count must fail before engine"))
    assert cli.main(["generate", "--target", "unused", "--physics", "unused", "--cases", "unused", "--output-directory", str(tmp_path), "--workers", "0"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "REF_INPUT_INVALID"


def test_deprecated_generate_record_refusal_flag_emits_lifecycle_warning(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cli,
        "_engine",
        lambda _: (_ for _ in ()).throw(RuntimeError("stop after warning")),
    )
    arguments = SimpleNamespace(
        workers=1,
        chunksize=None,
        shard_size=None,
        split_seed=0,
        split_ratios=(0.8, 0.1, 0.1),
        resume=False,
        record_refusal=True,
    )
    with pytest.warns(FutureWarning, match="deprecated compatibility no-op"):
        with pytest.raises(RuntimeError, match="stop after warning"):
            cli.command_generate(arguments)


def test_monolithic_writer_cannot_mutate_under_another_writer_lock(tmp_path, monkeypatch):
    import damage_engine.dataset as writer_module
    from damage_engine.input_identity import single_writer_lock
    output = tmp_path / "archive"
    monkeypatch.setattr(writer_module, "validate_precomputed_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(writer_module.DatasetWriter, "_write_frozen", lambda *args, **kwargs: pytest.fail("competing writer cannot enter publication"))
    result = SimpleNamespace(to_dict=lambda: {"scope": "opaque artificial payload"})
    with single_writer_lock(output):
        lock = tmp_path / ".archive.writer.lock"
        original = lock.read_bytes()
        with pytest.raises(InputValidationError, match="single-writer lock"):
            writer_module.DatasetWriter(output).write(None, [{"case_id": "toy"}], [result])
        assert lock.read_bytes() == original
        assert not output.exists()
