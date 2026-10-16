"""Typed candidate ledger columns; missing optional values never infer null type."""
from .parquet_schema import validate_rows


def candidate_table_schema():
    import pyarrow as pa
    strings = {
        "candidate_id", "lineage_id", "epistemic_group_id", "learning_input_hash",
        "stream", "role", "stratum", "design_group_id", "split_group", "split",
        "p0_id", "p0_hash", "formal_case_hash", "target_hash", "tree_hash",
        "label_spec_hash", "registry_hash", "engine_code_hash", "split_registry_hash",
    }
    integers = {"round", "selection_seed", "screen_seed", "screen_count", "formal_seed", "formal_count"}
    numbers = {"score_rarity", "score_uncertainty", "score_coverage", "score_total", "inclusion_probability", "design_weight_raw"}
    fields = [pa.field(name, pa.string(), nullable=False) for name in sorted(strings)]
    fields += [pa.field(name, pa.int64(), nullable=False) for name in sorted(integers)]
    fields += [pa.field(name, pa.float64(), nullable=False) for name in sorted(numbers)]
    fields += [pa.field(name, pa.bool_(), nullable=False) for name in ("selected", "weight_mask")]
    # Parquet writes the standard nested child name ``element``.  Declare it
    # explicitly so a read-back schema preserves the complete contract.
    fields += [pa.field("latent", pa.list_(pa.field("element", pa.float64())), nullable=False)]
    fields += [pa.field(name, pa.string()) for name in ("duplicate_of_candidate_id", "acquisition_model_hash", "screening_case_hash", "admission_purpose")]
    fields += [pa.field(name, pa.float64()) for name in ("random_draw", "uncertainty")]
    fields += [pa.field("replicate_count", pa.int64())]
    fields += [pa.field(name, pa.bool_()) for name in ("executed", "successful", "admitted")]
    fields += [pa.field("refusal", pa.struct([pa.field("code", pa.string()), pa.field("message", pa.string()), pa.field("context_json", pa.string())]))]
    return pa.schema(fields, metadata={b"table_schema_version": b"sampling-candidates@2.0.0", b"table_name": b"sampling_candidates"})


def candidate_table(rows):
    import pyarrow as pa
    schema = candidate_table_schema()
    validate_rows(rows, schema)
    return pa.Table.from_pylist(rows, schema=schema)


def adaptive_candidate_table_schema():
    """Explicit schema for a frozen general-proposal candidate ledger.

    The adaptive ledger is intentionally separate from the reference-plan
    schema above: every density, hierarchy probability and score component is
    materialized, so an inferred first-row schema cannot silently drop fields.
    """

    import pyarrow as pa

    strings = {
        "candidate_id", "lineage_id", "epistemic_group_id",
        "round_lineage_id", "parent_round_lineage_id", "split", "role",
        "stream", "stratum", "p0_id", "p0_hash", "acquisition_model_hash",
    }
    numbers = {
        "p0_density", "adaptive_density", "proposal_density",
        "p0_hierarchy_probability", "proposal_hierarchy_probability",
        "score_rarity", "score_uncertainty", "score_coverage", "score_total",
        "inclusion_probability", "design_weight_raw", "random_draw",
    }
    fields = [pa.field(name, pa.string(), nullable=False) for name in sorted(strings)]
    fields += [
        pa.field("round", pa.int64(), nullable=False),
        pa.field("selection_seed", pa.int64(), nullable=False),
    ]
    fields += [pa.field(name, pa.float64(), nullable=False) for name in sorted(numbers)]
    fields += [
        pa.field("selected", pa.bool_(), nullable=False),
        pa.field("weight_mask", pa.bool_(), nullable=False),
        pa.field("candidate_case_hash", pa.string()),
        pa.field("learning_input_hash", pa.string()),
    ]
    return pa.schema(
        fields,
        metadata={
            b"table_schema_version": b"adaptive-sampling-candidates@1.1.0",
            b"table_name": b"adaptive_sampling_candidates",
        },
    )


def adaptive_candidate_table(rows):
    import pyarrow as pa

    schema = adaptive_candidate_table_schema()
    validate_rows(rows, schema)
    return pa.Table.from_pylist(rows, schema=schema)
