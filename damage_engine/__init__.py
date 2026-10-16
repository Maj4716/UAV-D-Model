"""Traceable fragment/blast damage simulator for ML dataset production."""

from .engine import DamageEngine, EngineResult
from .errors import DamageEngineError, InputValidationError, OutOfDomainError
from .case_factory import CaseFactory
from .contact_contract import (
    S_CONTACT_CONTRACT_HASH,
    S_CONTACT_CONTRACT_ID,
    S_CONTACT_ESTIMAND,
    audit_s_contact,
    assert_contact_conditioned_estimand,
    validate_outcome_blind_selection,
)
from .dataset_policy import DatasetUsePolicy
from .label_spec import LabelSpec
from .normalization import (
    apply_training_normalization,
    load_training_normalization_artifact,
    normalize_target,
    save_training_normalization_artifact,
    validate_training_normalization_artifact,
)
from .ordinal_projection import (
    OrdinalProjectionConfig,
    OrdinalProjectionResult,
    aggregate_ordinal_projection,
    compile_ordinal_projection,
    nested_cumulative_probabilities,
    ordinal_ranked_probability_score,
    project_ordinal_replicates,
)
from .registry import PhysicsRegistry
from .sampling_spec import SamplingSpec
from .system_learning import (
    SYSTEM_LEARNING_REQUIRED_SOURCE_HASH_FIELDS,
    assert_trainable_tasks,
    build_system_learning_contract,
    evaluate_system_probabilities,
    fit_reference_prevalence_baseline,
    is_sha256_identity,
    masked_brier_from_logits,
    masked_soft_bce_with_logits,
    system_learning_archive_binding,
)
from .training_view import (
    TrainingViewDataset,
    fit_training_view_artifact,
    transform_case_physics,
)

__all__ = [
    "DamageEngine",
    "EngineResult",
    "PhysicsRegistry",
    "CaseFactory",
    "S_CONTACT_CONTRACT_HASH",
    "S_CONTACT_CONTRACT_ID",
    "S_CONTACT_ESTIMAND",
    "audit_s_contact",
    "assert_contact_conditioned_estimand",
    "validate_outcome_blind_selection",
    "DatasetUsePolicy",
    "LabelSpec",
    "OrdinalProjectionConfig",
    "OrdinalProjectionResult",
    "aggregate_ordinal_projection",
    "compile_ordinal_projection",
    "nested_cumulative_probabilities",
    "ordinal_ranked_probability_score",
    "project_ordinal_replicates",
    "SamplingSpec",
    "SYSTEM_LEARNING_REQUIRED_SOURCE_HASH_FIELDS",
    "assert_trainable_tasks",
    "build_system_learning_contract",
    "evaluate_system_probabilities",
    "fit_reference_prevalence_baseline",
    "is_sha256_identity",
    "masked_brier_from_logits",
    "masked_soft_bce_with_logits",
    "system_learning_archive_binding",
    "DamageEngineError",
    "InputValidationError",
    "OutOfDomainError",
    "normalize_target",
    "apply_training_normalization",
    "load_training_normalization_artifact",
    "save_training_normalization_artifact",
    "validate_training_normalization_artifact",
    "TrainingViewDataset",
    "fit_training_view_artifact",
    "transform_case_physics",
]

__version__ = "0.7.0.dev3"
SCHEMA_VERSION = "1.0.0"
