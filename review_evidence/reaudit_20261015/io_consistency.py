"""Read existing archive representations only; never construct/run an engine.

Output deliberately contains counts, identities, hashes and mismatch IDs, never
physical inputs, labels, predictions or replicate values. Arrow tables are read
one shard at a time; raw result JSONL.gz is streamed one record at a time.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))
ARCHIVE = ROOT / "raw_data/dataset_5000_loitering_impact_fragment_only_v1"
OUT = Path(__file__).parent
NUMERIC = ("mean", "stddev", "ci95_lower", "ci95_upper", "sample_count", "uncertainty_estimable")
FAMILIES = ("component", "component_frechet", "component_aux", "armor", "system")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(root.rglob("*")) if p.is_file()}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


class Audit:
    def __init__(self):
        self.checks = Counter()
        self.scalar_counts = Counter()
        self.mismatch_counts = Counter()
        self.mismatch_ids = defaultdict(set)
        self.details = []

    def check(self, category, condition, case_id="archive"):
        self.checks[category] += 1
        if not bool(condition):
            self.mismatch_counts[category] += 1
            self.mismatch_ids[category].add(str(case_id))

    def array(self, category, actual, expected, case_id, *, f32=False, numeric=False):
        actual = np.asarray(actual)
        expected = np.asarray(expected, dtype=np.float32 if f32 else np.float64 if numeric else None)
        self.checks[category] += 1
        self.scalar_counts[category] += int(expected.size)
        if actual.shape != expected.shape:
            self.mismatch_counts[category] += max(int(expected.size), 1)
            self.mismatch_ids[category].add(str(case_id))
            return
        if f32 or numeric:
            okay = np.isclose(actual.astype(np.float64), expected.astype(np.float64),
                              rtol=2*np.finfo(np.float32).eps if f32 else 1e-12,
                              atol=np.finfo(np.float32).tiny if f32 else 0, equal_nan=True)
        else:
            okay = actual == expected
        failed = int(np.count_nonzero(~okay))
        if failed:
            self.mismatch_counts[category] += failed
            self.mismatch_ids[category].add(str(case_id))


class Table:
    """Arrow-backed numeric columns; per-case indexes honor semantic IDs."""
    def __init__(self, path, keys, audit, all_ids):
        self.name = path.stem
        self.keys = tuple(keys)
        schema = pq.read_schema(path)
        columns = ["case_id", *keys, "available", *NUMERIC]
        if self.name == "labels":
            # Arrow nested-leaf projection flattens names (e.g. many 'mean'
            # columns), so retain each named struct to avoid key collisions.
            columns += [f"frechet_union_{b}_bound" for b in ("lower", "upper")]
        selected = []
        for column in columns:
            if column.split(".")[0] in schema.names and column not in selected:
                selected.append(column)
        self.table = pq.read_table(path, columns=selected)
        runs = pc.run_end_encode(self.table["case_id"].combine_chunks())
        self.ranges = defaultdict(list)
        start = 0
        for end, cid in zip(runs.run_ends.to_pylist(), runs.values.to_pylist()):
            self.ranges[cid].append((start, end-start))
            start = end
        audit.check("parquet.case_id_set", set(self.ranges) == set(all_ids), self.name)
        self.audit = audit

    def group(self, case_id, expected_keys):
        pieces = [self.table.slice(start, length) for start, length in self.ranges.get(case_id, ())]
        self.audit.check("parquet.case_group_present", bool(pieces), case_id)
        if not pieces:
            return {}
        table = pieces[0] if len(pieces) == 1 else pa.concat_tables(pieces)
        columns = table.to_pydict()
        keys = list(zip(*(columns[name] for name in self.keys)))
        mapping = {key: i for i, key in enumerate(keys)}
        self.audit.check("parquet.semantic_keys_unique", len(keys) == len(mapping), case_id)
        self.audit.check("parquet.semantic_key_set", set(mapping) == set(expected_keys), case_id)
        if any(key not in mapping for key in expected_keys):
            return {}
        indexes = [mapping[key] for key in expected_keys]
        return {name: [values[i] for i in indexes] for name, values in columns.items()}


def numbers(stats, field):
    return np.asarray([s.get(field) if s is not None else None for s in stats], dtype=np.float64)


def compare_parquet(audit, category, group, stats, case_id):
    if not group:
        return
    for field in NUMERIC:
        audit.check("parquet.statistic_field_present", field in group, case_id)
        if field in group:
            audit.array(category+"."+field, np.asarray(group[field], dtype=np.float64), numbers(stats, field), case_id, numeric=True)


def compare_npz(audit, arrays, family, index, stats, availability, case_id):
    shape = arrays[family+"_y"][index].shape
    availability = np.asarray(availability, dtype=bool).reshape(shape)
    mean = numbers(stats, "mean").reshape(shape)
    stddev = numbers(stats, "stddev").reshape(shape)
    lower = numbers(stats, "ci95_lower").reshape(shape)
    upper = numbers(stats, "ci95_upper").reshape(shape)
    counts = numbers(stats, "sample_count").reshape(shape)
    counts = np.where(availability, counts, 0).astype(np.int64)
    uncertainty = availability & (counts >= 2) & np.isfinite(stddev) & np.isfinite(lower) & np.isfinite(upper)
    for field, expected in (("mean", mean), ("stddev", np.where(uncertainty, stddev, np.nan)),
                            ("ci95_lower", np.where(uncertainty, lower, np.nan)),
                            ("ci95_upper", np.where(uncertainty, upper, np.nan))):
        actual = (arrays[family+"_y"][index] if field == "mean" else
                  arrays[family+"_y_stddev"][index] if field == "stddev" else
                  arrays[family+"_y_ci95"][index, ..., 0 if field == "ci95_lower" else 1])
        audit.array("npz."+family+"."+field, actual, expected, case_id, f32=True)
    audit.array("npz."+family+".availability", arrays[family+"_label_mask"][index], availability, case_id)
    audit.array("npz."+family+".sample_count", arrays[family+"_sample_count"][index], counts, case_id)
    audit.array("npz."+family+".uncertainty_mask", arrays[family+"_uncertainty_mask"][index], uncertainty, case_id)


def main():
    from damage_engine.engine import ENGINE_CODE_HASH
    from damage_engine import __version__
    from damage_engine.input_identity import case_identity_hash, _canonical_subtree_cache
    from damage_engine.sampling_checkpoint import _iter_hydrated_formal_cases
    from damage_engine.sampling_spec import SamplingSpec
    from damage_engine.label_spec import LabelSpec
    from damage_engine.split_registry import SplitRegistry
    from damage_engine.parquet_schema import table_schema
    from damage_engine.utils import content_hash

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before = inventory(ARCHIVE)
    source_before = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT/"damage_engine").rglob("*.py"))}
    source_before.update({p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT/"schemas").rglob("*.json"))})
    save("io_inventory_before.json", {"archive": str(ARCHIVE), "files": before, "source_hashes": source_before, "review_script_sha256": digest(__file__)})
    audit = Audit()
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "numpy_version": np.__version__, "pyarrow_version": pa.__version__,
              "scope": "existing archive bytes, schemas, semantic IDs and numeric representations; no simulation, regeneration, engine construction or training",
              "numeric_values_emitted": False,
              "comparison_tolerance": "NPZ: JSON expected cast to float32 then 2*eps32 relative tolerance; Parquet: 1e-12 relative; masks/counts exact",
              "not_checked": ["independent physical validation", "CI empirical coverage", "model training", "full numerical feature recomputation"],
              "shards": []}
    total_results, trace_count, count_contract_presence = 0, 0, Counter()
    executed, successful, refused = set(), set(), set()
    try:
        directory = ARCHIVE/"formal_dataset"
        manifest = read_json(directory/"manifest.json")
        checkpoint = read_json(directory/"checkpoint.json")
        sampling_checkpoint = read_json(ARCHIVE/"sampling_checkpoint.json")
        frozen = read_json(ARCHIVE/"plan/frozen_hashes.json")
        registry = read_json(ARCHIVE/"plan/split_registry.json")
        assignments = {x["lineage_id"]: x for x in registry["assignments"]}
        definition = {k: registry[k] for k in ("schema_version", "registry_id", "assignment_phase", "assignments")}
        audit.check("registry.definition_hash", content_hash(definition) == registry["definition_hash"])
        report["engine_identity"] = {"archive_version": manifest["engine"]["version"], "archive_code_hash": manifest["engine"]["code_hash"],
                                     "current_version": __version__, "current_code_hash": ENGINE_CODE_HASH,
                                     "same_code_identity": manifest["engine"]["code_hash"] == ENGINE_CODE_HASH,
                                     "boundary": "agreement among stored representations verifies the historical archive contract; it is not execution equivalence with the current source"}
        report["manifest_schema_version"] = manifest["schema_version"]
        report["tensor_contract_version"] = manifest["tensor_contract"]["version"]
        audit.check("root.manifest_hash", digest(directory/"manifest.json") == checkpoint["root_manifest_hash"])
        audit.check("root.dataset_id", manifest["dataset_id"] == checkpoint["dataset_id"])
        audit.check("root.complete", checkpoint["status"] == "complete" and not (directory/".dataset.incomplete").exists())
        for part in manifest["partitions"]:
            relative = "formal_dataset/"+part["path_glob"]
            audit.check("root.partition_inventory", relative in before, part["name"])
            if relative in before:
                audit.check("root.partition_hash", before[relative]["sha256"] == part["content_hash"], part["name"])
        for item in sampling_checkpoint["committed_files"]:
            relative = item["path"]
            audit.check("sampling.committed_inventory", relative in before, relative)
            if relative in before:
                audit.check("sampling.committed_hash", before[relative]["sha256"] == item["content_hash"], relative)
        round_dir = ARCHIVE/"rounds/round-000"
        round_manifest = read_json(round_dir/"round_manifest.json")
        audit.check("round.candidates_hash", digest(round_dir/"candidates.parquet") == round_manifest["candidate_ledger_hash"])
        audit.check("round.formal_cases_hash", digest(round_dir/"formal_cases.jsonl.gz") == round_manifest["formal_cases_hash"])
        candidate_rows = pq.read_table(round_dir/"candidates.parquet").to_pylist()
        candidates = {x["candidate_id"]: x for x in candidate_rows}
        audit.check("candidate.ids_unique", len(candidates) == len(candidate_rows))
        selected = {x["candidate_id"] for x in candidate_rows if x["selected"]}
        formal = {}
        parsed_registry = SplitRegistry.from_file(ARCHIVE/"plan/split_registry.json")
        parsed_sampling = SamplingSpec.from_file(ARCHIVE/"plan/sampling_spec.json")
        parsed_labels = LabelSpec.from_file(ARCHIVE/"plan/label_spec.json")
        canonical_cache = None
        for record in _iter_hydrated_formal_cases(round_dir/"formal_cases.jsonl.gz", parsed_registry, parsed_sampling,
                                                  parsed_labels, expected_reference="../../plan/split_registry.json"):
            cid = record["case_id"]
            envelope = record["metadata"]["sampling"]
            audit.check("formal.case_id_unique", cid not in formal, cid)
            if canonical_cache is None:
                canonical_cache = _canonical_subtree_cache([v for k,v in envelope.items() if k.endswith("_document")])
            identity = case_identity_hash(record, _canonical_subtrees=canonical_cache)
            formal[cid] = {"identity_hash": identity, "analysis": record["analysis"], "sampling": envelope}
            row = candidates[envelope["candidate_id"]]
            audit.check("formal.candidate_case_hash", row["formal_case_hash"] == identity, cid)
            assignment = assignments[envelope["lineage_id"]]
            for name in ("role", "epistemic_group_id"):
                audit.check("formal.registry_"+name, assignment[name] == envelope[name], cid)
            audit.check("formal.registry_split", assignment["split"] == envelope["frozen_split"], cid)
            for name in ("engine_code_hash", "target_hash", "physics_registry_hash", "label_spec_hash", "sampling_spec_hash", "split_registry_hash"):
                audit.check("formal.frozen_identity", envelope[name] == frozen[name], cid)
        audit.check("formal.selected_set", {v["sampling"]["candidate_id"] for v in formal.values()} == selected)
        audit.check("round.selected_count", len(selected) == round_manifest["selected_count"])
        seen_case_ids = set()
        for shard_index, shard in enumerate(sorted((directory/"shards").iterdir())):
            shard_started = time.perf_counter()
            sm = read_json(shard/"manifest.json")
            audit.check("shard.engine_identity", sm["engine"]["code_hash"] == manifest["engine"]["code_hash"], shard.name)
            completed = checkpoint["completed_shards"][shard_index]
            audit.check("shard.manifest_hash", digest(shard/"manifest.json") == completed["manifest_hash"], shard.name)
            with np.load(shard/"tensors.npz", allow_pickle=False) as handle:
                a = {key: handle[key] for key in handle.files}
            ids = a["case_ids"].tolist()
            input_ids = a["input_case_ids"].tolist()
            audit.array("shard.checkpoint_case_order", input_ids, completed["case_ids"], shard.name)
            cases = {x["case_id"]: x for x in pq.read_table(shard/"cases.parquet").to_pylist()}
            splits = {x["case_id"]: x for x in pq.read_table(shard/"splits.parquet").to_pylist()}
            audit.check("shard.case_sets", set(cases) == set(ids) == set(splits), shard.name)
            for part in sm["partitions"]:
                actual = before["formal_dataset/"+shard.relative_to(directory).as_posix()+"/"+part["path_glob"]]["sha256"]
                audit.check("shard.partition_hash", actual == part["content_hash"], shard.name+"/"+part["name"])
                if part["format"] == "parquet":
                    schema = pq.read_schema(shard/part["path_glob"])
                    audit.check("parquet.explicit_schema", schema.equals(table_schema(part["name"]), check_metadata=True), shard.name+"/"+part["name"])
                    audit.check("parquet.manifest_row_count", pq.ParquetFile(shard/part["path_glob"]).metadata.num_rows == part["row_count"], shard.name+"/"+part["name"])
            tables = {"labels": Table(shard/"labels.parquet", ("component_id", "mechanism", "damage_state"), audit, ids),
                      "component_aux": Table(shard/"component_aux.parquet", ("component_id", "metric"), audit, ids),
                      "armor_labels": Table(shard/"armor_labels.parquet", ("armor_id", "metric"), audit, ids),
                      "tree_outputs": Table(shard/"tree_outputs.parquet", ("evaluation_standard_id", "output_node"), audit, ids)}
            result_path = shard/next(x["path_glob"] for x in sm["partitions"] if x["name"] == "audits")
            local_results = 0
            valid_index = {cid: i for i, cid in enumerate(ids)}
            for result_index, r in enumerate(jsonl(result_path)):
                cid = r["case_id"]
                total_results += 1
                local_results += 1
                audit.check("result.case_unique", cid not in seen_case_ids, cid)
                seen_case_ids.add(cid)
                record = formal[cid]
                envelope = record["sampling"]
                candidate_id = envelope["candidate_id"]
                executed.add(candidate_id)
                audit.check("result.input_order", cid == input_ids[result_index], cid)
                audit.check("result.input_status", r["status"] == str(a["input_case_status"][result_index]), cid)
                is_success = r["status"] == "ok"
                audit.check("result.input_valid_mask", is_success == bool(a["input_case_valid_mask"][result_index]), cid)
                (successful if is_success else refused).add(candidate_id)
                audit.check("result.case_hash", r["provenance"]["case_hash"] == record["identity_hash"], cid)
                for field in ("engine_code_hash", "target_hash", "physics_registry_hash"):
                    audit.check("result.provenance_identity", r["provenance"][field] == envelope[field], cid)
                trace_count += int("replicate_trace" in r)
                for key in ("successes", "trials", "success_count", "trial_count"):
                    count_contract_presence[key] += int(key in r)
                if not is_success:
                    continue
                i = valid_index[cid]
                audit.check("result.replicate_count", r["replicate_count"] == r["provenance"]["replicate_count"] == int(a["replicate_counts"][i]), cid)
                for field, expected in (("case_hashes", record["identity_hash"]), ("aleatory_seeds", record["analysis"]["aleatory_seed"]),
                                        ("epistemic_sample_ids", record["analysis"]["epistemic_sample_id"]),
                                        ("sampling_candidate_ids", candidate_id), ("sampling_lineage_ids", envelope["lineage_id"]),
                                        ("sampling_epistemic_group_ids", envelope["epistemic_group_id"]), ("split_names", envelope["frozen_split"]),
                                        ("sampling_selection_probability", envelope["selection_probability"]), ("sampling_design_weight_raw", envelope["design_weight_raw"]),
                                        ("sampling_weight_mask", envelope["weight_mask"])):
                    audit.check("npz.case_sampling_identity", a[field][i].item() == expected, cid)
                cr = cases[cid]
                audit.check("parquet.case_hash", cr["case_hash"] == record["identity_hash"], cid)
                audit.check("parquet.case_split", cr["split"] == splits[cid]["split"] == envelope["frozen_split"], cid)
                for key, value in cr["sampling"].items():
                    audit.check("parquet.case_sampling_identity", value == envelope.get(key), cid)
                components, mechanisms, states = (a[k].tolist() for k in ("component_ids", "mechanism_names", "state_names"))
                component_stats = [r["components"][c]["damage_states"][m][s] for c in components for m in mechanisms for s in states]
                component_keys = [(c,m,s) for c in components for m in mechanisms for s in states]
                group = tables["labels"].group(cid, component_keys)
                compare_parquet(audit, "parquet.component", group, component_stats, cid)
                compare_npz(audit, a, "component", i, component_stats, group["available"], cid)
                bounds = a["frechet_bound_names"].tolist()
                frechet_stats = [r["components"][c]["damage_states"]["combined"][s][f"frechet_union_{b}_bound"] for c in components for s in states for b in bounds]
                combined_positions = [component_keys.index((c,"combined",s)) for c in components for s in states]
                parquet_bounds = [group[f"frechet_union_{b}_bound"][position] for position in combined_positions for b in bounds]
                compare_parquet(audit, "parquet.component_frechet", {name: [x[name] for x in parquet_bounds] for name in NUMERIC}, frechet_stats, cid)
                compare_npz(audit, a, "component_frechet", i, frechet_stats,
                            [set(("fragment","blast")).issubset(r["mechanisms"])]*len(frechet_stats), cid)
                aux_names = a["component_aux_label_names"].tolist()
                auxiliary = []
                for c in components:
                    for name in aux_names:
                        mechanism, key = name.split("_",1)
                        auxiliary.append(r["components"][c][mechanism+"_metrics"][key])
                group = tables["component_aux"].group(cid, [(c,n) for c in components for n in aux_names])
                compare_parquet(audit, "parquet.component_aux", group, auxiliary, cid)
                compare_npz(audit, a, "component_aux", i, auxiliary, group["available"], cid)
                armor_ids, armor_names = a["armor_ids"].tolist(), a["armor_label_names"].tolist()
                armor_stats = []
                for armor_id in armor_ids:
                    for name in armor_names:
                        if name.startswith("post_"):
                            stats = r["armor"][armor_id][name]
                        else:
                            key = {"pre_integrity":"pre_blast_integrity", "post_integrity":"post_blast_integrity", "failure_probability":"blast_failure_probability"}.get(name[6:], name[6:])
                            stats = r["armor"][armor_id]["blast_metrics"][key]
                        armor_stats.append(stats)
                group = tables["armor_labels"].group(cid, [(aid,n) for aid in armor_ids for n in armor_names])
                compare_parquet(audit, "parquet.armor", group, armor_stats, cid)
                compare_npz(audit, a, "armor", i, armor_stats, group["available"], cid)
                outputs = a["system_output_names"].tolist()
                standards = record["analysis"]["evaluation_standard_ids"]
                tree_keys = [(s,o) for s in standards for o in r["damage_tree"]["evaluation_standards"][s]]
                group = tables["tree_outputs"].group(cid, tree_keys)
                tree_stats = [r["damage_tree"]["outputs"][o] for _,o in tree_keys]
                compare_parquet(audit, "parquet.system", group, tree_stats, cid)
                availability = {o: bool(v) for (_,o),v in zip(tree_keys,group["available"])}
                system_stats = [r["damage_tree"]["outputs"][o] for o in outputs]
                compare_npz(audit, a, "system", i, system_stats, [availability.get(o,False) for o in outputs], cid)
                audit.array("npz.system.source_availability", a["system_source_available_mask"][i], [availability.get(o,False) for o in outputs], cid)
                for family_stats in (component_stats, frechet_stats, auxiliary, armor_stats, system_stats):
                    audit.check("result.stat_sample_count", all(s["sample_count"] == r["replicate_count"] for s in family_stats), cid)
                if local_results % 250 == 0:
                    print(json.dumps({"progress_shard":shard_index,"result_count":total_results,"mismatch_categories":len(audit.mismatch_counts)}), flush=True)
            audit.check("shard.result_count", local_results == len(input_ids), shard.name)
            report["shards"].append({"index":shard_index,"input_case_count":len(input_ids),"valid_case_count":len(ids),"result_count":local_results,
                                     "seconds":round(time.perf_counter()-shard_started,3)})
            del tables, a
        audit.check("global.result_formal_case_set", seen_case_ids == set(formal))
        actual_sets = {"candidate_frame":set(candidates),"selected":selected,"executed":executed,"successful":successful,"refused":refused}
        for name, actual in actual_sets.items():
            summary = sampling_checkpoint["archive_summary"]["candidate_sets"][name]
            audit.check("summary."+name+"_count", len(actual) == summary["count"])
            audit.check("summary."+name+"_set", actual == set(summary["candidate_ids"]))
        for name, actual in (("executed",executed),("successful",successful)):
            # This is an outcome-blind ledger frozen before execution. Null
            # status is unknown, not false; completion sets are checked above.
            known = [x for x in candidate_rows if type(x.get(name)) is bool]
            for row in known:
                audit.check("candidate.known_"+name+"_status", row[name] == (row["candidate_id"] in actual), row["candidate_id"])
            report["candidate_ledger_"+name+"_status_counts"] = dict(Counter(type(x.get(name)).__name__+":"+str(x.get(name)) for x in candidate_rows))
        for purpose, summary in sampling_checkpoint["archive_summary"]["candidate_sets"]["admitted_by_purpose"].items():
            declared = set(summary["candidate_ids"])
            audit.check("summary.purpose_count", len(declared) == summary["count"] == manifest["quality_summary"]["purpose_admission_counts"][purpose], purpose)
            audit.check("summary.purpose_success_subset", declared.issubset(successful), purpose)
        audit.check("root.success_count", len(successful) == manifest["quality_summary"]["execution_success_count"])
        audit.check("root.refusal_count", len(refused) == manifest["quality_summary"]["n_refused"])
        report["candidate_sets"] = {name:{"count":len(values),"candidate_ids_hash":content_hash(sorted(values))} for name,values in actual_sets.items()}
        report["formal_case_count"] = len(formal)
    except Exception as exc:
        report["fatal_error"] = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
    after = inventory(ARCHIVE)
    source_after = {name:digest(ROOT/name) for name in source_before}
    audit.check("archive.all_files_unchanged", before == after)
    audit.check("source.snapshot_unchanged", source_before == source_after)
    save("io_inventory_after.json", {"archive":str(ARCHIVE),"files":after,"source_hashes":source_after,"archive_unchanged":before==after,"source_unchanged":source_before==source_after})
    report.update({"result_count":total_results,"replicate_trace_record_count":trace_count,"top_level_success_trial_field_presence_counts":dict(count_contract_presence),
                   "checks":dict(sorted(audit.checks.items())),"scalar_comparisons":dict(sorted(audit.scalar_counts.items())),
                   "mismatch_counts":dict(sorted(audit.mismatch_counts.items())),"mismatch_case_ids":{k:sorted(v) for k,v in sorted(audit.mismatch_ids.items())},
                   "archive_unchanged":before==after,"source_unchanged":source_before==source_after,"seconds":round(time.perf_counter()-started,3)})
    if not trace_count:
        report["not_checked"].append("No archived replicate_trace: aggregate-to-replicate reconstruction and per-replicate joint alignment cannot be checked")
    if not any(count_contract_presence.values()):
        report["not_checked"].append("No top-level success/trial counting contract: no Bernoulli success count is inferred from aggregate probabilities")
    report["not_checked"].append("Domain input coverage, boundary accumulation, collinearity, near-duplicate geometry, split shift and sampling optimization were outside the permitted generic archive-integrity scope")
    report["all_executed_checks_passed"] = not audit.mismatch_counts and "fatal_error" not in report
    save("io_consistency.json", report)
    print(json.dumps({"done":True,"result_count":total_results,"check_groups":sum(audit.checks.values()),"scalar_comparisons":sum(audit.scalar_counts.values()),
                      "mismatch_counts":report["mismatch_counts"],"fatal_error_type":report.get("fatal_error",{}).get("type"),"archive_unchanged":before==after},ensure_ascii=False),flush=True)
    return 0 if report["all_executed_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
