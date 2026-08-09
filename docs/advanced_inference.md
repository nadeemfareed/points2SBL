# Advanced inference configuration

Normal users should use `--mode raw`, `--mode full`, or `--mode adaptive`
plus input/output paths. The packaged `point_transformer.yaml` supplies the
backend profile. CLI options always override the profile.

## Raw

Network-baseline / ablation output:

- 1 vote
- direct `pred_leaf_prob` threshold (`0.50`)
- no semantic smoothing/refinement
- existing class 2 ground is preserved/excluded

## Full

Preserves the mature historical inference behavior:

- 1.5 m tiles
- 8192 points per network block
- 6 `grid4` votes
- confidence weighting and spatial voting
- fixed three-zone probability decision
- denoising, smoothing, woody refinement, structural-core refinement,
  component cleanup, and uncertain-point reassignment

## Adaptive

Probability-distribution inference:

- 8 votes
- `hybrid8` = 4 deterministic half-tile layouts + 4 seeded random layouts
- empirical `pred_leaf_prob` histogram
- smoothed density
- wood peak, inter-peak valley, leaf peak
- peak-tail/valley transition boundaries
- high-confidence peak populations remain locked
- only transition-zone points are reconsidered
- transition resolution uses `linearity`, `planarity`, `scattering`,
  `curvature`, and local class support
- class 2 ground is excluded from the histogram and remains `-1` in
  `pred_leaf_prob`

### Adaptive parameters

`adaptive_hist_bins`
: Histogram resolution.

`adaptive_hist_smooth_sigma`
: Density smoothing before peak/valley detection.

`adaptive_shoulder_fraction`
: Density level used to locate where each peak tail enters the valley. Lower
  values preserve more of the complete peak shape and restrict post-processing
  to the low-density valley. Shipped default: `0.02`.

`adaptive_min_transition_width`
: Minimum allowed transition-zone probability width.

`adaptive_geom_ratio`
: Required relative geometric similarity for prototype-based assignment.

`adaptive_local_support_min`
: Minimum local support used after geometry comparison.

## Reproducibility contract

The checkpoint's embedded config remains authoritative for the model
architecture and training feature contract. The runtime YAML controls inference
profiles. This prevents accidental changes to the trained 7-channel
representation while retaining full expert control of inference.


## Backend profiles versus CLI overrides

Routine users should select only `raw`, `full`, or `adaptive`. The packaged
`point_transformer.yaml` supplies the normal defaults. An explicitly supplied
CLI argument always overrides the profile for that run.

## Vote layouts

- `grid4`: four deterministic half-tile layouts.
- `grid8`: eight deterministic stratified layouts.
- `hybrid8`: four deterministic half-tile layouts plus seeded random layouts.
- `random`: seeded random offsets.

The predictor reports unique-layout coverage so an expert can determine how
often points were actually seen across layouts.

## Adaptive probability interpretation

Ground values of `-1` are reserved output fill values and are not part of the
adaptive probability distribution. The distribution analysis operates on
non-ground network probabilities in `[0,1]`.

High-confidence peak populations are locked. Geometry/local evidence is applied
only to the detected transition population.
