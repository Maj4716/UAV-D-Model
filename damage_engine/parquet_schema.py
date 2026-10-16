"""Explicit archive table schemas; unknown fields fail instead of disappearing.

Arrow's first-row inference is deliberately never used by the archive writer.
Optional statistics remain typed even when every row is null or a table is empty.
"""
from __future__ import annotations

from collections.abc import Mapping

from .errors import InputValidationError

PARQUET_SCHEMA_VERSION = "1.0.0"
TABLE_NAMES = ("cases", "components", "armor", "armor_labels", "labels",
               "component_aux", "edges", "splits", "tree_outputs")


def table_schema(name: str):
    import pyarrow as pa

    string, number, integer, boolean = pa.string(), pa.float64(), pa.int64(), pa.bool_()
    def fields(pairs, *, required=False):
        return [pa.field(key, dtype, nullable=not required) for key, dtype in pairs]

    statistics = fields([
        ("mean", number), ("stddev", number), ("ci95_lower", number), ("ci95_upper", number),
        ("sample_count", integer), ("uncertainty_estimable", boolean),
        ("insufficient_reason", string), ("estimand", string), ("stddev_estimand", string),
        ("conditioning", string), ("random_level", string), ("interval_type", string),
        ("interval_method", string), ("probability_bounds_clipped", boolean),
        ("coverage_status", string),
    ])
    bounds = fields([(f"frechet_union_{bound}_bound", pa.struct(statistics))
                     for bound in ("lower", "upper")])
    features = [("features", pa.list_(pa.field("element", number))),
                ("feature_mask", pa.list_(pa.field("element", boolean)))]
    models = [("material_id", string), ("fragment_response_model_id", string),
              ("blast_response_model_id", string)]
    sampling = pa.struct(fields([
        ("candidate_id", string), ("lineage_id", string), ("round", integer),
        ("stream", string), ("role", string), ("stratum", string),
        ("selection_probability", number), ("design_weight_raw", number),
        ("weight_mask", boolean), ("epistemic_group_id", string), ("frozen_split", string),
        ("sampling_spec_hash", string), ("label_spec_hash", string), ("split_registry_hash", string),
    ]))
    layouts = {
        "cases": fields([("case_id", string), ("target_id", string), *features,
                         ("replicate_count", integer), ("aleatory_seed", integer),
                         ("epistemic_sample_id", string), ("group_id", string),
                         ("group_source", string), ("split", string), ("case_hash", string)], required=True)
                 + fields([("sampling", sampling)]),
        "components": fields([("case_id", string), ("component_id", string),
                              ("component_name", string), ("component_type", string),
                              *features, *models], required=True),
        "armor": fields([("case_id", string), ("armor_id", string), ("armor_name", string),
                         *features, *models], required=True),
        "armor_labels": fields([("case_id", string), ("armor_id", string), ("metric", string),
                                ("unit", string), ("available", boolean)], required=True) + statistics,
        "labels": fields([("case_id", string), ("component_id", string), ("mechanism", string),
                          ("damage_state", string), ("available", boolean)], required=True) + statistics + bounds,
        "component_aux": fields([("case_id", string), ("component_id", string),
                                 ("mechanism", string), ("metric", string), ("unit", string),
                                 ("available", boolean)], required=True) + statistics,
        "edges": fields([("source_id", string), ("target_id", string), ("source_type", string),
                         ("target_type", string), ("edge_type", string)], required=True),
        "splits": fields([("case_id", string), ("group_id", string), ("group_source", string),
                          ("leakage_key", string), ("split", string)], required=True),
        "tree_outputs": fields([("case_id", string), ("evaluation_standard_id", string),
                                ("output_node", string)], required=True)
                        + fields([("available", boolean)]) + statistics,
    }
    if name not in layouts:
        raise InputValidationError("unknown archive Parquet table", context={"table": name})
    return pa.schema(layouts[name], metadata={
        b"table_schema_version": PARQUET_SCHEMA_VERSION.encode(), b"table_name": name.encode(),
    })


def validate_rows(rows: list[dict], schema) -> None:
    """Check every struct at every row before Arrow can silently drop a key."""
    import pyarrow as pa

    def check(value, dtype, location):
        if value is None:
            return
        if pa.types.is_struct(dtype):
            if not isinstance(value, Mapping):
                raise InputValidationError("Parquet struct must be an object", context={"field": location})
            expected = {field.name for field in dtype}
            unknown = set(value) - expected
            missing = [field.name for field in dtype if not field.nullable and value.get(field.name) is None]
            if unknown or missing:
                raise InputValidationError("Parquet row has unknown or missing required fields",
                                           context={"field": location, "unknown": sorted(unknown), "missing": missing})
            for field in dtype:
                check(value.get(field.name), field.type, f"{location}.{field.name}")
        elif pa.types.is_list(dtype):
            if not isinstance(value, (list, tuple)):
                raise InputValidationError("Parquet list must be an array", context={"field": location})
            for index, item in enumerate(value):
                check(item, dtype.value_type, f"{location}[{index}]")

    dtype = pa.struct(list(schema))
    for index, row in enumerate(rows):
        check(row, dtype, f"row[{index}]")
