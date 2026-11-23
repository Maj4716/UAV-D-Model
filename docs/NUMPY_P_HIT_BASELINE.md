# Pure-NumPy dataset-v2 `p_hit` baseline

This diagnostic trains a multi-output, one-hidden-layer MLP without PyTorch.
It is intended for learning-curve and label-support checks before formal neural
training.

## Run

```powershell
python tools/run_dataset_v2_numpy_baseline.py `
  --dataset artifacts/datasets/<dataset>/dataset_v2_development `
  --output artifacts/validation/<run>/numpy_p_hit_baseline
```

The default design runs training fractions `0.25`, `0.50`, and `1.0` with
seeds `17`, `29`, and `43`. Repeat `--fraction` or `--seed` to override the
lists. Every run set must include the `1.0` fraction.

Useful optimization options are:

```text
--epochs 200
--learning-rate 0.01
--hidden-size 64
--interval-loss-weight 0.25
--l2-penalty 0.0001
```

## Statistical contract

- Inputs are standardized with the mean and population standard deviation of
  that run's training subset. Validation and test covariates do not contribute
  to the fitted statistics.
- Point-identified `p_hit_mask` entries use design-weighted soft Bernoulli
  cross-entropy, so fractional Monte Carlo probabilities remain valid targets.
- Masked entries with an informative identification interval use
  design-weighted squared interval-hinge loss. A vacuous `[0,1]` interval adds
  zero loss and zero gradient.
- Adam optimizes the full training subset. Epochs and the selected 100% run are
  chosen by validation weighted Brier score.
- Validation reports weighted Brier score, ECE, calibration bins, interval
  violations, per-component observed/positive/negative support, and estimable
  component counts. Learning-curve summaries include seed mean, population
  variance, and standard deviation.

## Sealed-test contract

The command requires `sealed_test.json` to declare:

- `test_open_count == 0`;
- `test_result_json_lines_decoded == 0`;
- `test_labels_present_in_core == false`.

All test `p_hit` masks must also be false. Test labels are excluded from loss,
metrics, epoch selection, seed selection, and normalization. The report records
these conditions under `test_access_policy`.

## Outputs

`results.json` contains:

- the dataset manifest identity and model/loss contract;
- one record per fraction and seed, including a compact training history;
- validation Brier/ECE and component support diagnostics;
- learning-curve seed aggregates;
- the validation-selected 100% run and its model hash;
- the sealed-test access declaration.

`selected_model.npz` contains `w1`, `b1`, `w2`, `b2`, feature names,
component IDs, and the selected run's training mean and scale.

## Verification

```powershell
python -m pytest tests/test_numpy_hurdle_baseline.py -q
```

The tests cover analytic gradients against finite differences, vacuous-interval
zero gradients, train-only normalization, validation-label gating, CLI output,
and an unchanged sealed-test file hash.

Runtime dependency: the project's base `numpy>=1.26`. Test execution uses the
optional `pytest>=8` dependency.
