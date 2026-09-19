# Replogle parallel analysis and steering controls

This protocol was prepared on 2026-09-19 while the first complete AdaCell
validation run was still in progress. It prepares experiments; it does not
claim they have run or that AdaCell improves upon the backbone.

## Work completed independently of model training

The four Mean variants, Linear, Scouter and CPA have full population diagnostics
on all 380 test conditions. The checks exclude copied controls and use fixed
random projections (64 directions, seed 1729). They report variance ratios,
projected effective rank, exact unique-vector fractions, and sliced W1 distance.
Diagnostic version 2 centers in float64, preventing an artificial nonzero rank
when float32 population means are repeated exactly.

Zero variance is expected for adapters that intentionally output one repeated
mean. Conversely, almost every vector being unique does not establish realistic
heterogeneity. Current Scouter predictions have a mean condition-wise
predicted/real variance ratio of 0.00777, and CPA 0.14001. These observations
motivate independent diversity checks, not a claim that steering solves them.

Formal completion still requires the entire frozen 14-metric evaluation. Use
the summary command below to exclude partial predictions or partial evaluation.
It validates ordered genes, all condition counts, identical controls, all 14
finite metrics, agreement of the per-condition and summary tables, and the
prediction/reference hashes of independent diagnostics.

```bash
python scripts/baselines/summarize_replogle_completed.py \
  --outdir results/replogle/parallel_analysis
```

The resulting Markdown report and provenance manifest are local artifacts.
Neither data nor full predictions are committed to Git.

## Fixed validation plan

Prepare either released backbone without starting a model process:

```bash
python scripts/baselines/prepare_replogle_steering_controls.py \
  --variant scratch --outdir results/replogle/plans/steering_scratch_v1
python scripts/baselines/prepare_replogle_steering_controls.py \
  --variant finetuned --outdir results/replogle/plans/steering_finetuned_v1
```

Each output contains `plan.json` and individual commands in `commands.txt`.
The planner never submits an automatic queue. Each command explicitly selects
the validation reference, conda environment, device and all exposed steering
settings, and uses `all` rather than a finite smoke-test group limit. A different
plan cannot silently overwrite an existing plan file.

| Case | Difference from the alpha=1 reference | Purpose |
|---|---|---|
| `adacell_alpha1` | None | Current population steering reference |
| `unselected16` | No reward-dependent selection or resampling | Unsteered control with 16 generated candidates |
| `best_of_16` | Select by terminal reward only, no intermediate resampling | Isolate intermediate population selection |
| `adacell_alpha05` | Temperature 0.5 | Stronger steering |
| `adacell_alpha2` | Temperature 2 | Weaker steering |
| `without_signature` | Signature weight 0 | Signature reward ablation |
| `without_direction` | Direction weight 0 | Geometric/directional reward ablation |
| `without_anchor` | Anchor weight 0 | Training-support reward ablation |
| `without_zscore` | No candidate-wise reward standardization | Normalization diagnostic |

All cases use 16 particles, at most 16 native 32-cell blocks per candidate
population, 100 DDIM steps, eta=0, CFG=1, and seed 42. Native backbone attention
sets are unchanged; only the joint reward and selection object spans blocks.
Temperature is in the denominator: smaller alpha means stronger steering.
No weight is increased when another reward is removed.

The temperature candidates are 0.5, 1 and 2. Select on mean validation PDS cos
only after all three complete predictions and full evaluations exist. Ties
within 1e-6 favor the larger temperature. Other independent diagnostic changes
must be reported, not concealed through a favorable primary metric. There is
no assumed diversity or KL guarantee. The reward ablations at alpha=1 are
validation diagnostics; final test controls and ablations must use the same
selected temperature as the final method.

Freeze the selected configuration before test generation. Test scores already
reported for baselines do not enter this selection rule. Multiple seeds,
biological case studies and final full test runs remain separate required work.

## Planned budget versus observed budget

The single-sample released PerturbDiff output is a quality reference, not the
equal-16-candidate compute control. In the current engine, `random` retains the
first exchangeable candidate, without score-based selection. All three modes
still compute rewards each step, including controls. Candidate selection and
SMC therefore have the same planned denoising work, but wall time can differ.

Audit the actual counters of an ongoing or completed run:

```bash
python scripts/baselines/audit_replogle_steering_budget.py \
  --shard-root results/replogle/validation/adacell_scratch_alpha1/shards \
  --out results/replogle/plans/steering_scratch_v1/sampling_audit_snapshot.json
```

After the control also runs, add `--compare-shard-root` with its shard directory.
The audit checks shared run settings, group IDs, perturbations, valid/padded
cell counts and recorded denoised cell-steps. Partial matching groups never
become a verified *full* comparison. It also reports final surviving ancestors
and resampling frequency. It does not certify equality of unrecorded control
tensors, equal wall time, final evaluation completion, or biological accuracy.

Padding counts toward compute but not toward cell coverage or rewards. With
eta=0, duplicate resampled trajectories do not rejuvenate, so ancestor retention
and independent population diversity must both be inspected before making any
diversity-preservation claim.
