"""Run sealed-test-safe multi-fidelity learning ablations for component p_hit.

The four arms use the same teacher subsets, seeds, epochs and validation rows:

* ``teacher_only``: the frozen learner input only;
* ``proxy_feature``: append 67 FastSamplingProxy hit-probability features;
* ``teacher_minus_proxy_residual``: learn the probability residual;
* ``proxy_pretrain_then_teacher``: pretrain on an outcome-blind Sobol proxy pool,
  then fine-tune with the same teacher subset as the other arms.

The strict dataset-v2 package contains masked placeholders for test labels.  This
tool requires that contract, never evaluates the test rows and reports a fixed
``test_open_count=0``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.contact_contract import audit_s_contact  # noqa: E402
from damage_engine.continuous_direction_sampling import (  # noqa: E402
    apply_continuous_direction_manifold,
)
from damage_engine.dataset_v2 import (  # noqa: E402
    load_dataset_v2_core,
    validate_dataset_v2,
)
from damage_engine.design_sequences import generate_unit_design  # noqa: E402
from damage_engine.distribution_metrics import weighted_calibration  # noqa: E402
from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.errors import DamageEngineError, InputValidationError  # noqa: E402
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.impact import resolve_impact_detonation  # noqa: E402
from damage_engine.learning_input_identity import (  # noqa: E402
    LOITERING_IMPACT_FIRST_ARMOR_STATE_MODEL_INPUT_DEFINITION,
)
from damage_engine.munition_profiles import apply_munition_profile  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.sampling_spec import SamplingSpec  # noqa: E402
from damage_engine.utils import content_hash, file_hash  # noqa: E402


REPORT_VERSION = "proxy-learning-ablation@1.0.0"
ARMS = (
    "teacher_only",
    "proxy_feature",
    "teacher_minus_proxy_residual",
    "proxy_pretrain_then_teacher",
)
_DEVELOPMENT_SPLITS = frozenset({"train", "validation"})


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError("PyTorch is required for proxy learning ablation") from exc
    return torch


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            "proxy learning JSON input is missing or unreadable",
            context={"path": str(path)},
        ) from exc
    if not isinstance(value, dict):
        raise InputValidationError("proxy learning JSON input must be an object")
    return value


def _model_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return "sha256:" + digest.hexdigest()


def _network_class(feature_count: int, component_count: int, hidden_size: int):
    torch = _torch()

    class PHitNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(feature_count, hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_size, component_count),
            )

        def forward(self, values):
            return self.layers(values)

    return PHitNetwork


def _weighted_bce(logits, target, mask, case_weight):
    torch = _torch()
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    weight = case_weight[:, None] * mask.to(dtype=raw.dtype)
    denominator = torch.sum(weight)
    if float(denominator.detach().cpu()) <= 0.0:
        raise InputValidationError("teacher subset contains no identified p_hit labels")
    return torch.sum(raw * weight) / denominator


def _weighted_residual_mse(raw_residual, target_residual, mask, case_weight):
    torch = _torch()
    predicted_residual = torch.tanh(raw_residual)
    raw = (predicted_residual - target_residual) ** 2
    weight = case_weight[:, None] * mask.to(dtype=raw.dtype)
    denominator = torch.sum(weight)
    if float(denominator.detach().cpu()) <= 0.0:
        raise InputValidationError("teacher subset contains no identified p_hit labels")
    return torch.sum(raw * weight) / denominator


def _train_probability_model(
    *,
    x: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    case_weight: np.ndarray,
    component_count: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
    initial_model: Any | None = None,
) -> Any:
    torch = _torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = initial_model or _network_class(
        x.shape[1], component_count, hidden_size
    )()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    safe_target = np.where(mask, target, 0.0)
    target_t = torch.as_tensor(safe_target, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    weight_t = torch.as_tensor(case_weight, dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_bce(model(x_t), target_t, mask_t, weight_t)
        loss.backward()
        optimizer.step()
    return model


def _train_residual_model(
    *,
    x: np.ndarray,
    proxy: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    case_weight: np.ndarray,
    component_count: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> Any:
    torch = _torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = _network_class(x.shape[1], component_count, hidden_size)().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    safe_residual = np.where(mask, target - proxy, 0.0)
    residual_t = torch.as_tensor(safe_residual, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    weight_t = torch.as_tensor(case_weight, dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_residual_mse(
            model(x_t), residual_t, mask_t, weight_t
        )
        loss.backward()
        optimizer.step()
    return model


def _predict_probability(model: Any, x: np.ndarray, *, device: str) -> np.ndarray:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        values = torch.as_tensor(x, dtype=torch.float32, device=device)
        return torch.sigmoid(model(values)).cpu().numpy().astype(np.float64)


def _predict_residual_probability(
    model: Any, x: np.ndarray, proxy: np.ndarray, *, device: str
) -> np.ndarray:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        values = torch.as_tensor(x, dtype=torch.float32, device=device)
        residual = torch.tanh(model(values)).cpu().numpy().astype(np.float64)
    return np.clip(proxy + residual, 0.0, 1.0)


def _validation_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    case_weight: np.ndarray,
) -> dict[str, Any]:
    pair_weight = np.repeat(case_weight[:, None], observed.shape[1], axis=1)
    if not np.any(mask):
        raise InputValidationError("validation contains no identified p_hit labels")
    calibration = weighted_calibration(
        predicted[mask], observed[mask], weight=pair_weight[mask]
    )
    per_component: list[dict[str, Any]] = []
    for column in range(observed.shape[1]):
        selected = mask[:, column]
        if not np.any(selected):
            per_component.append(
                {"observation_count": 0, "brier": None, "ece": None}
            )
            continue
        result = weighted_calibration(
            predicted[selected, column],
            observed[selected, column],
            weight=case_weight[selected],
        )
        per_component.append(
            {
                "observation_count": int(np.sum(selected)),
                "brier": result["brier"],
                "ece": result["ece"],
            }
        )
    return {
        "identified_case_component_count": int(np.sum(mask)),
        "brier": calibration["brier"],
        "ece": calibration["ece"],
        "calibration_bins": calibration["bins"],
        "per_component": per_component,
    }


def _development_array_hash(
    *,
    x: np.ndarray,
    proxy: np.ndarray,
    teacher: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    splits: np.ndarray,
) -> str:
    selected = np.isin(splits.astype(str), sorted(_DEVELOPMENT_SPLITS))
    digest = hashlib.sha256()
    for value in (
        x[selected].astype(np.float64),
        proxy[selected].astype(np.float64),
        teacher[selected].astype(np.float64),
        mask[selected].astype(np.bool_),
        weights[selected].astype(np.float64),
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update("\n".join(splits[selected].astype(str)).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def run_proxy_learning_ablation_arrays(
    *,
    x: Any,
    proxy_hit_probability: Any,
    teacher_hit_probability: Any,
    label_mask: Any,
    case_weight: Any,
    splits: Sequence[str],
    proxy_pretrain_x: Any,
    proxy_pretrain_hit_probability: Any,
    fractions: Sequence[float] = (0.25, 0.50, 1.0),
    seeds: Sequence[int] = (17, 29, 43),
    epochs: int = 120,
    pretrain_epochs: int = 60,
    hidden_size: int = 96,
    learning_rate: float = 3.0e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare all four arms without reading or evaluating test labels."""

    x_array = np.asarray(x, dtype=np.float64)
    proxy = np.asarray(proxy_hit_probability, dtype=np.float64)
    teacher = np.asarray(teacher_hit_probability, dtype=np.float64)
    mask = np.asarray(label_mask, dtype=bool)
    weights = np.asarray(case_weight, dtype=np.float64)
    split_array = np.asarray([str(value) for value in splits])
    pretrain_x = np.asarray(proxy_pretrain_x, dtype=np.float64)
    pretrain_proxy = np.asarray(
        proxy_pretrain_hit_probability, dtype=np.float64
    )
    if (
        x_array.ndim != 2
        or proxy.ndim != 2
        or teacher.shape != proxy.shape
        or mask.shape != proxy.shape
        or x_array.shape[0] != proxy.shape[0]
        or weights.shape != (x_array.shape[0],)
        or split_array.shape != (x_array.shape[0],)
        or pretrain_x.ndim != 2
        or pretrain_x.shape[1] != x_array.shape[1]
        or pretrain_proxy.shape != (pretrain_x.shape[0], proxy.shape[1])
        or not np.isfinite(x_array).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isfinite(pretrain_x).all()
        or not np.isfinite(pretrain_proxy).all()
        or np.any((pretrain_proxy < 0.0) | (pretrain_proxy > 1.0))
    ):
        raise InputValidationError("proxy learning arrays are invalid or misaligned")
    train = np.flatnonzero(split_array == "train")
    validation = np.flatnonzero(split_array == "validation")
    test = np.flatnonzero(split_array == "test")
    if not train.size or not validation.size or not test.size:
        raise InputValidationError(
            "proxy learning requires train, validation and sealed test identities"
        )
    development = np.concatenate((train, validation))
    if (
        not np.isfinite(proxy[development]).all()
        or np.any((proxy[development] < 0.0) | (proxy[development] > 1.0))
        or not np.isfinite(teacher[development][mask[development]]).all()
        or np.any((teacher[development][mask[development]] < 0.0))
        or np.any((teacher[development][mask[development]] > 1.0))
    ):
        raise InputValidationError("development proxy/teacher targets are invalid")
    if not np.all(mask[test] == 0):
        raise InputValidationError("sealed test label masks must all be false")
    if epochs < 1 or pretrain_epochs < 1 or hidden_size < 1 or learning_rate <= 0.0:
        raise InputValidationError("proxy learning hyperparameters are invalid")
    fraction_values = sorted(set(float(value) for value in fractions))
    seed_values = list(dict.fromkeys(int(value) for value in seeds))
    if (
        not fraction_values
        or not seed_values
        or any(value <= 0.0 or value > 1.0 for value in fraction_values)
        or 1.0 not in fraction_values
    ):
        raise InputValidationError("fractions require unique values in (0,1] including 1")

    runs: list[dict[str, Any]] = []
    component_count = proxy.shape[1]
    for seed in seed_values:
        generator = np.random.default_rng(seed)
        permutation = generator.permutation(train)
        for fraction in fraction_values:
            subset_count = max(1, int(math.ceil(train.size * fraction)))
            subset = np.sort(permutation[:subset_count])
            mean = np.mean(x_array[subset], axis=0)
            scale = np.std(x_array[subset], axis=0)
            scale[scale < 1.0e-8] = 1.0
            train_x = (x_array[subset] - mean) / scale
            validation_x = (x_array[validation] - mean) / scale
            proxy_pool_x = (pretrain_x - mean) / scale
            train_target = teacher[subset]
            train_mask = mask[subset]
            train_weight = weights[subset]

            for arm in ARMS:
                arm_seed = int(seed)
                if arm == "proxy_feature":
                    arm_train_x = np.concatenate((train_x, proxy[subset]), axis=1)
                    arm_validation_x = np.concatenate(
                        (validation_x, proxy[validation]), axis=1
                    )
                    model = _train_probability_model(
                        x=arm_train_x,
                        target=train_target,
                        mask=train_mask,
                        case_weight=train_weight,
                        component_count=component_count,
                        hidden_size=hidden_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        seed=arm_seed,
                        device=device,
                    )
                    predicted = _predict_probability(
                        model, arm_validation_x, device=device
                    )
                elif arm == "teacher_minus_proxy_residual":
                    model = _train_residual_model(
                        x=train_x,
                        proxy=proxy[subset],
                        target=train_target,
                        mask=train_mask,
                        case_weight=train_weight,
                        component_count=component_count,
                        hidden_size=hidden_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        seed=arm_seed,
                        device=device,
                    )
                    predicted = _predict_residual_probability(
                        model,
                        validation_x,
                        proxy[validation],
                        device=device,
                    )
                elif arm == "proxy_pretrain_then_teacher":
                    model = _train_probability_model(
                        x=proxy_pool_x,
                        target=pretrain_proxy,
                        mask=np.ones_like(pretrain_proxy, dtype=bool),
                        case_weight=np.ones(pretrain_proxy.shape[0], dtype=np.float64),
                        component_count=component_count,
                        hidden_size=hidden_size,
                        epochs=pretrain_epochs,
                        learning_rate=learning_rate,
                        seed=arm_seed,
                        device=device,
                    )
                    model = _train_probability_model(
                        x=train_x,
                        target=train_target,
                        mask=train_mask,
                        case_weight=train_weight,
                        component_count=component_count,
                        hidden_size=hidden_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        seed=arm_seed,
                        device=device,
                        initial_model=model,
                    )
                    predicted = _predict_probability(
                        model, validation_x, device=device
                    )
                else:
                    model = _train_probability_model(
                        x=train_x,
                        target=train_target,
                        mask=train_mask,
                        case_weight=train_weight,
                        component_count=component_count,
                        hidden_size=hidden_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        seed=arm_seed,
                        device=device,
                    )
                    predicted = _predict_probability(
                        model, validation_x, device=device
                    )
                runs.append(
                    {
                        "arm": arm,
                        "seed": int(seed),
                        "train_fraction": float(fraction),
                        "teacher_train_case_count": int(subset.size),
                        "proxy_pretrain_case_count": (
                            int(pretrain_x.shape[0])
                            if arm == "proxy_pretrain_then_teacher"
                            else 0
                        ),
                        "teacher_epochs": int(epochs),
                        "proxy_pretrain_epochs": (
                            int(pretrain_epochs)
                            if arm == "proxy_pretrain_then_teacher"
                            else 0
                        ),
                        "validation_metrics": _validation_metrics(
                            predicted,
                            teacher[validation],
                            mask[validation],
                            weights[validation],
                        ),
                        "model_hash": _model_hash(model),
                        "teacher_subset_hash": content_hash(
                            {"indices": subset.astype(int).tolist()}
                        ),
                    }
                )

    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        selected = [
            item
            for item in runs
            if item["arm"] == arm
            and math.isclose(float(item["train_fraction"]), 1.0)
        ]
        brier = np.asarray(
            [item["validation_metrics"]["brier"] for item in selected],
            dtype=np.float64,
        )
        ece = np.asarray(
            [item["validation_metrics"]["ece"] for item in selected],
            dtype=np.float64,
        )
        arm_summary[arm] = {
            "full_fraction_seed_count": len(selected),
            "validation_brier_mean": float(np.mean(brier)),
            "validation_brier_std": float(np.std(brier)),
            "validation_ece_mean": float(np.mean(ece)),
            "validation_ece_std": float(np.std(ece)),
            "model_hashes": [item["model_hash"] for item in selected],
        }
    selected_arm = min(
        ARMS,
        key=lambda name: (
            arm_summary[name]["validation_brier_mean"],
            ARMS.index(name),
        ),
    )
    return {
        "schema_version": REPORT_VERSION,
        "arms": list(ARMS),
        "teacher_budget_contract": {
            "fractions": fraction_values,
            "seeds": seed_values,
            "teacher_epochs_per_run": int(epochs),
            "same_teacher_subset_within_fraction_and_seed": True,
        },
        "proxy_pretraining_contract": {
            "case_count": int(pretrain_x.shape[0]),
            "epochs": int(pretrain_epochs),
            "teacher_labels_used": False,
        },
        "split_counts": {
            "train": int(train.size),
            "validation": int(validation.size),
            "test_sealed_unread": int(test.size),
        },
        "test_access_policy": {
            "true_labels_present_in_development_package": False,
            "open_count": 0,
            "metrics_computed": False,
            "selection_uses_test": False,
        },
        "development_array_hash": _development_array_hash(
            x=x_array,
            proxy=proxy,
            teacher=teacher,
            mask=mask,
            weights=weights,
            splits=split_array,
        ),
        "runs": runs,
        "full_fraction_summary": arm_summary,
        "selection_scope": "validation_only_full_teacher_fraction_mean_brier",
        "selected_arm": selected_arm,
    }


def _pilot_dataset_directory(pilot: Path) -> Path:
    nested = pilot / "formal_dataset"
    if (nested / "manifest.json").is_file():
        return nested
    if (pilot / "manifest.json").is_file():
        return pilot
    raise InputValidationError("pilot lacks a completed formal dataset manifest")


def _formal_case_map(pilot: Path) -> dict[str, dict[str, Any]]:
    paths = sorted((pilot / "rounds").glob("round-*/formal_cases.jsonl.gz"))
    if not paths:
        raise InputValidationError("pilot lacks a formal-case ledger")
    cases: dict[str, dict[str, Any]] = {}
    try:
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    case = json.loads(line)
                    case_id = str(case.get("case_id", ""))
                    if not case_id or case_id in cases:
                        raise ValueError("formal case IDs must be nonempty and unique")
                    cases[case_id] = case
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise InputValidationError("formal-case ledger is unreadable") from exc
    return cases


def _proxy_hit_vector(
    proxy: FastSamplingProxy,
    case: Mapping[str, Any],
    component_ids: Sequence[str],
) -> tuple[np.ndarray, str]:
    result = proxy.evaluate(dict(case))
    components = result.get("component_metrics", {})
    if set(components) != set(component_ids):
        raise InputValidationError("proxy component vocabulary differs from dataset-v2")
    expected_hits = np.asarray(
        [float(components[value]["expected_hits"]) for value in component_ids],
        dtype=np.float64,
    )
    return -np.expm1(-np.maximum(expected_hits, 0.0)), str(
        result.get("proxy_domain_status", "unresolved")
    )


def _build_proxy_pool_case(
    *,
    base_case: Mapping[str, Any],
    sampling_spec: SamplingSpec,
    engine: DamageEngine,
    latent: np.ndarray,
    index: int,
) -> dict[str, Any]:
    independent_width = sampling_spec.independent_latent_dimension
    profile_variable = next(
        value
        for value in sampling_spec.variables
        if value["name"] == "munition_profile"
    )
    profiles = list(profile_variable["values"])
    profile_index = min(int(float(latent[0]) * len(profiles)), len(profiles) - 1)
    case = deepcopy(dict(base_case))
    case["case_id"] = f"proxy-pretrain-{index:07d}"
    case["munition"]["profile_id"] = str(profiles[profile_index])
    case["target_ref"]["content_hash"] = engine.target_hash
    manifold = sampling_spec.contact_manifold
    if manifold is None:
        raise InputValidationError("proxy pretraining requires a contact manifold")
    apply_continuous_direction_manifold(
        case,
        manifold,
        latent[independent_width:],
        engine.compiled_target,
        target_hash=engine.target_hash,
    )
    apply_munition_profile(case)
    case = resolve_impact_detonation(case, engine.compiled_target)
    conditioning = sampling_spec.contact_conditioning
    if conditioning is None:
        raise InputValidationError("proxy pretraining requires S-CONTACT conditioning")
    admission = audit_s_contact(
        case,
        engine.compiled_target,
        data_role="model_train_reference",
        estimand=str(conditioning["estimand"]),
        selection_declaration=conditioning["selection_declaration"],
    )
    case.setdefault("metadata", {})["proxy_pretrain_s_contact"] = admission
    return case


def build_outcome_blind_proxy_pool(
    *,
    engine: DamageEngine,
    sampling_spec: SamplingSpec,
    base_case: Mapping[str, Any],
    component_ids: Sequence[str],
    count: int,
    design_seed: int,
    proxy_source_seed: int,
    azimuths_per_ring: int,
    layer_quadrature_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build an independent Sobol proxy-only pool with no teacher outcomes."""

    if count < 1:
        raise InputValidationError("proxy pretraining pool size must be positive")
    attempt_count = 1 << int(math.ceil(math.log2(max(8, 4 * count))))
    design = generate_unit_design(
        attempt_count,
        sampling_spec.latent_dimension,
        seed=int(design_seed),
        method="scrambled_sobol",
    )
    proxy = FastSamplingProxy(
        engine,
        source_seed=proxy_source_seed,
        azimuths_per_ring=azimuths_per_ring,
        layer_quadrature_samples=layer_quadrature_samples,
    )
    x_rows: list[list[float]] = []
    p_rows: list[np.ndarray] = []
    rejections: Counter[str] = Counter()
    domain_status: Counter[str] = Counter()
    for index, latent in enumerate(design.points):
        if len(x_rows) >= count:
            break
        try:
            case = _build_proxy_pool_case(
                base_case=base_case,
                sampling_spec=sampling_spec,
                engine=engine,
                latent=np.asarray(latent, dtype=np.float64),
                index=index,
            )
            probability, status = _proxy_hit_vector(proxy, case, component_ids)
            if status != "complete_proxy_domain":
                rejections[status] += 1
                continue
            x_rows.append(
                LOITERING_IMPACT_FIRST_ARMOR_STATE_MODEL_INPUT_DEFINITION.vector(case)
            )
            p_rows.append(probability)
            domain_status[status] += 1
        except DamageEngineError as exc:
            rejections[str(exc.context.get("reason_code", exc.code))] += 1
        except (KeyError, TypeError, ValueError) as exc:
            rejections[type(exc).__name__] += 1
    if len(x_rows) < count:
        raise InputValidationError(
            "proxy Sobol attempt frame did not yield the requested complete pool",
            context={
                "requested": count,
                "admitted": len(x_rows),
                "attempted": attempt_count,
                "rejections": dict(rejections),
            },
        )
    x = np.asarray(x_rows, dtype=np.float64)
    p = np.asarray(p_rows, dtype=np.float64)
    return x, p, {
        "design": "independent_scrambled_sobol",
        "design_seed": int(design_seed),
        "attempt_count": int(attempt_count),
        "admitted_count": int(x.shape[0]),
        "teacher_labels_used": False,
        "rejection_counts": dict(sorted(rejections.items())),
        "proxy_domain_status_counts": dict(sorted(domain_status.items())),
        "pool_hash": content_hash(
            {"x": x.tolist(), "proxy_hit_probability": p.tolist()}
        ),
    }


def run_proxy_learning_ablation(
    *,
    dataset_directory: str | Path,
    pilot_directory: str | Path,
    engine: DamageEngine,
    proxy_pool_size: int = 512,
    proxy_pool_design_seed: int = 2026112022,
    proxy_source_seed: int = 2026111401,
    azimuths_per_ring: int = 4,
    layer_quadrature_samples: int = 16,
    fractions: Sequence[float] = (0.25, 0.50, 1.0),
    seeds: Sequence[int] = (17, 29, 43),
    epochs: int = 120,
    pretrain_epochs: int = 60,
    hidden_size: int = 96,
    learning_rate: float = 3.0e-3,
    device: str = "cpu",
    x1_gate_status: str = "failed",
) -> dict[str, Any]:
    dataset = Path(dataset_directory)
    pilot = Path(pilot_directory)
    validation = validate_dataset_v2(dataset, recompute_views=False)
    seal = _json(dataset / "sealed_test.json")
    if (
        seal.get("test_open_count") != 0
        or seal.get("test_labels_present_in_core") is not False
        or seal.get("test_result_json_lines_decoded") != 0
    ):
        raise InputValidationError(
            "proxy learning requires the strict opaque-label sealed-test package"
        )
    core = load_dataset_v2_core(dataset)
    inputs = core["inputs"]
    labels = core["component_labels"]
    weights = core["sampling_weights"]
    case_ids = inputs["case_ids"].astype(str)
    splits = weights["split"].astype(str)
    component_ids = labels["component_ids"].astype(str).tolist()
    expected_feature_names = list(
        LOITERING_IMPACT_FIRST_ARMOR_STATE_MODEL_INPUT_DEFINITION.feature_names
    )
    if inputs["feature_names"].astype(str).tolist() != expected_feature_names:
        raise InputValidationError(
            "dataset-v2 input contract is not the 16D first-armour state"
        )
    development_input = np.isin(splits, sorted(_DEVELOPMENT_SPLITS))
    if not np.all(inputs["input_mask"][development_input]):
        raise InputValidationError("development learner inputs must be fully observed")
    if component_ids != list(engine.compiled_target.component_order):
        raise InputValidationError("dataset-v2 component order differs from proxy engine")
    source_dataset = _pilot_dataset_directory(pilot)
    source_manifest = _json(source_dataset / "manifest.json")
    expected_target = set(
        source_manifest.get("source_contracts", {}).get("target_hashes") or []
    )
    expected_registry = str(
        source_manifest.get("engine", {}).get("configuration_hash", "")
    )
    if expected_target and engine.target_hash not in expected_target:
        raise InputValidationError("proxy engine target differs from sampler dataset")
    if expected_registry and engine.registry.hash != expected_registry:
        raise InputValidationError("proxy physics registry differs from sampler dataset")

    cases = _formal_case_map(pilot)
    proxy_model = FastSamplingProxy(
        engine,
        source_seed=proxy_source_seed,
        azimuths_per_ring=azimuths_per_ring,
        layer_quadrature_samples=layer_quadrature_samples,
    )
    proxy_hit = np.full(
        (case_ids.size, len(component_ids)), np.nan, dtype=np.float64
    )
    proxy_domain_status: Counter[str] = Counter()
    for index, (case_id, split) in enumerate(
        zip(case_ids.tolist(), splits.tolist(), strict=True)
    ):
        if split not in _DEVELOPMENT_SPLITS:
            continue
        case = cases.get(case_id)
        if case is None:
            raise InputValidationError("dataset-v2 case is absent from formal ledger")
        probability, status = _proxy_hit_vector(
            proxy_model, case, component_ids
        )
        proxy_hit[index] = probability
        proxy_domain_status[status] += 1

    base_case = _json(pilot / "plan" / "base_case.json")
    sampling_spec = SamplingSpec.from_file(pilot / "plan" / "sampling_spec.json")
    pretrain_x, pretrain_proxy, pool_report = build_outcome_blind_proxy_pool(
        engine=engine,
        sampling_spec=sampling_spec,
        base_case=base_case,
        component_ids=component_ids,
        count=proxy_pool_size,
        design_seed=proxy_pool_design_seed,
        proxy_source_seed=proxy_source_seed,
        azimuths_per_ring=azimuths_per_ring,
        layer_quadrature_samples=layer_quadrature_samples,
    )
    report = run_proxy_learning_ablation_arrays(
        x=inputs["x"],
        proxy_hit_probability=proxy_hit,
        teacher_hit_probability=labels["p_hit"],
        label_mask=labels["label_mask"],
        case_weight=weights["design_weight"],
        splits=splits,
        proxy_pretrain_x=pretrain_x,
        proxy_pretrain_hit_probability=pretrain_proxy,
        fractions=fractions,
        seeds=seeds,
        epochs=epochs,
        pretrain_epochs=pretrain_epochs,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        device=device,
    )
    report.update(
        {
            "dataset": {
                "dataset_id": validation["dataset_id"],
                "manifest_hash": validation["manifest_hash"],
                "manifest_sha256": file_hash(dataset / "manifest.json"),
                "sealed_test_sha256": file_hash(dataset / "sealed_test.json"),
            },
            "proxy_contract": {
                "class": "FastSamplingProxy",
                "source_seed": int(proxy_source_seed),
                "azimuths_per_ring": int(azimuths_per_ring),
                "layer_quadrature_samples": int(layer_quadrature_samples),
                "target_hash": engine.target_hash,
                "physics_registry_hash": engine.registry.hash,
                "formal_development_domain_status_counts": dict(
                    sorted(proxy_domain_status.items())
                ),
                "p_hit_transform": "1-exp(-expected_hits)",
            },
            "proxy_pretraining_pool": pool_report,
            "x1_gate_status": str(x1_gate_status),
            "research_use": (
                "candidate_for_promotion"
                if x1_gate_status == "passed"
                else "diagnostic_only_not_promoted"
            ),
            "evidence_boundary": (
                "Validation-only software-teacher learning comparison. The proxy "
                "is an input/initialization aid and never changes formal labels."
            ),
        }
    )
    report["report_hash"] = content_hash(report)
    return report


def _build_inadmissible_proxy_learning_report(
    *,
    dataset: Path,
    target: Path,
    physics: Path,
    proxy_pool_size: int,
    fractions: Sequence[float],
    seeds: Sequence[int],
    epochs: int,
    pretrain_epochs: int,
    x1_gate_status: str,
) -> dict[str, Any] | None:
    """Return an auditable no-selection result for an empty validation gate."""

    validation = validate_dataset_v2(dataset, recompute_views=False)
    seal = _json(dataset / "sealed_test.json")
    if (
        seal.get("test_open_count") != 0
        or seal.get("test_labels_present_in_core") is not False
        or seal.get("test_result_json_lines_decoded") != 0
    ):
        raise InputValidationError(
            "proxy learning requires the strict opaque-label sealed-test package"
        )
    core = load_dataset_v2_core(dataset)
    labels = core["component_labels"]
    splits = core["sampling_weights"]["split"].astype(str)

    def observations(split: str) -> dict[str, int]:
        selected = splits == split
        return {
            "case_count": int(np.sum(selected)),
            "p_hit_observation_count": int(np.sum(labels["label_mask"][selected])),
            "conditional_energy_observation_count": int(
                np.sum(labels["conditional_log1p_energy_mask"][selected])
            ),
            "conditional_hit_count_observation_count": int(
                np.sum(labels["conditional_log1p_hit_count_mask"][selected])
            ),
        }

    observation_counts = {
        split: observations(split) for split in ("train", "validation", "test")
    }
    if observation_counts["validation"]["p_hit_observation_count"] > 0:
        return None

    reason = {
        "code": "no_observable_validation_p_hit",
        "message": "validation contains no observable p_hit labels",
        "selection_metric": "validation_p_hit_weighted_brier",
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "status": "all_arms_inadmissible",
        "reason": reason,
        "arms": list(ARMS),
        "selected_arm": None,
        "selection_scope": "reference_validation_only",
        "selection_metric": reason["selection_metric"],
        "requested_fractions": [float(value) for value in fractions],
        "requested_seeds": [int(value) for value in seeds],
        "requested_epochs": int(epochs),
        "requested_pretrain_epochs": int(pretrain_epochs),
        "arm_results": [
            {
                "arm": arm,
                "status": "inadmissible",
                "error": reason,
                "validation_brier": None,
                "validation_ece": None,
            }
            for arm in ARMS
        ],
        "runs": [],
        "observation_counts": observation_counts,
        "dataset": {
            "dataset_id": validation["dataset_id"],
            "manifest_hash": validation["manifest_hash"],
            "manifest_sha256": file_hash(dataset / "manifest.json"),
            "sealed_test_sha256": file_hash(dataset / "sealed_test.json"),
        },
        "requested_inputs": {
            "target_sha256": file_hash(target),
            "physics_sha256": file_hash(physics),
        },
        "proxy_pretraining_pool": {
            "requested_count": int(proxy_pool_size),
            "status": "skipped_primary_validation_gate",
            "teacher_labels_used": False,
        },
        "x1_gate_status": str(x1_gate_status),
        "research_use": "diagnostic_only_not_promoted",
        "test_access_policy": {
            "true_labels_present_in_development_package": False,
            "test_result_json_lines_decoded": int(
                seal["test_result_json_lines_decoded"]
            ),
            "open_count": int(seal["test_open_count"]),
            "metrics_computed": False,
            "selection_uses_test": False,
        },
        "evidence_boundary": (
            "The four-arm comparison is inadmissible because the frozen validation "
            "split has no observable p_hit labels. Conditional hurdle observations "
            "do not replace the primary validation gate."
        ),
    }
    report["report_hash"] = content_hash(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-pool-size", type=int, default=512)
    parser.add_argument("--proxy-pool-design-seed", type=int, default=2026112022)
    parser.add_argument("--proxy-source-seed", type=int, default=2026111401)
    parser.add_argument("--azimuths-per-ring", type=int, default=4)
    parser.add_argument("--layer-quadrature-samples", type=int, default=16)
    parser.add_argument("--fraction", type=float, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--x1-gate-status", choices=("failed", "passed"), default="failed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    fractions = args.fraction or (0.25, 0.50, 1.0)
    seeds = args.seed or (17, 29, 43)
    report = _build_inadmissible_proxy_learning_report(
        dataset=args.dataset,
        target=args.target,
        physics=args.physics,
        proxy_pool_size=args.proxy_pool_size,
        fractions=fractions,
        seeds=seeds,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        x1_gate_status=args.x1_gate_status,
    )
    if report is None:
        engine = DamageEngine(
            _json(args.target), PhysicsRegistry.from_file(args.physics)
        )
        report = run_proxy_learning_ablation(
            dataset_directory=args.dataset,
            pilot_directory=args.pilot,
            engine=engine,
            proxy_pool_size=args.proxy_pool_size,
            proxy_pool_design_seed=args.proxy_pool_design_seed,
            proxy_source_seed=args.proxy_source_seed,
            azimuths_per_ring=args.azimuths_per_ring,
            layer_quadrature_samples=args.layer_quadrature_samples,
            fractions=fractions,
            seeds=seeds,
            epochs=args.epochs,
            pretrain_epochs=args.pretrain_epochs,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            device=args.device,
            x1_gate_status=args.x1_gate_status,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_hash": report["report_hash"],
                "selected_arm": report["selected_arm"],
                "test_open_count": report["test_access_policy"]["open_count"],
                "status": report.get("status", "completed"),
                "research_use": report["research_use"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "REPORT_VERSION",
    "build_outcome_blind_proxy_pool",
    "run_proxy_learning_ablation",
    "run_proxy_learning_ablation_arrays",
]
