# Bundle Adjustment

This page documents the bundle-adjustment routines implemented in `openptv_python.orientation` and the demo driver in `openptv_python.demo_bundle_adjustment`.

## Overview

The repository currently exposes two related calibration-refinement workflows:

1. `multi_camera_bundle_adjustment`: jointly refine camera parameters and optionally 3D points by minimizing reprojection error.
1. `guarded_two_step_bundle_adjustment`: run a pose-focused bundle-adjustment stage first, then a tightly constrained intrinsic stage, and reject the second stage if it makes the solution worse.

The demo module compares several configurations on a case folder such as `tests/testing_fodder/test_cavity` and can write one updated calibration folder per experiment.

## Case Layout

The demo expects a case folder with this structure:

```text
case_dir/
  cal/
    cam1.tif.ori
    cam1.tif.addpar
    ...
  parameters/
    ptv.par
    sequence.par
  res_orig/
    rt_is.*
  img_orig/
    cam1.%05d_targets
    ...
```

This is the same layout used by `tests/testing_fodder/test_cavity`.

## Objective Function

For each observed 2D image point $u\_{ij} = (x\_{ij}, y\_{ij})$ from camera $j$ and 3D point $X_i$, bundle adjustment minimizes reprojection residuals of the form

$$
r\_{ij} =
\\begin{bmatrix}
\\hat{x}_{ij}(X_i, \\theta_j) - x_{ij} \\
\\hat{y}_{ij}(X_i, \\theta_j) - y_{ij}
\\end{bmatrix},
$$

where $\\theta_j$ denotes the active calibration parameters for camera $j$.

The solver minimizes a robust or linear least-squares objective

$$
\\min\_{\\Theta, X}
\\sum\_{i,j} \\rho\\left(\\left|W r\_{ij}\\right|^2\\right)

- \\sum_k \\left(\\frac{p_k - p\_{k,0}}{\\sigma_k}\\right)^2
- \\sum\_{m \\in \\mathcal{M}} \\left|\\frac{X_m - X_m^\*}{\\sigma^{(X)}\_m}\\right|^2.
  $$

with:

1. $W$ converting pixel residuals into normalized units using `pix_x` and `pix_y`.
1. $\\rho(\\cdot)$ chosen by the `loss` argument, for example `linear` or `soft_l1`.
1. Optional Gaussian-style priors on selected parameters via `prior_sigmas`.
1. Optional soft 3D geometry anchors on selected points via `known_points` and `known_point_sigmas`.

The implementation evaluates the forward projection through OpenPTV's existing camera and multimedia model, so refraction and the Brown-affine distortion model remain part of the optimization.

## Implemented Algorithms

## `multi_camera_bundle_adjustment`

This is the general solver. It can optimize:

1. Exterior parameters: `x0`, `y0`, `z0`, `omega`, `phi`, `kappa`.
1. Optional intrinsic and distortion parameters controlled by `OrientPar` flags.
1. The 3D points themselves, if `optimize_points=True`.

Important implementation details:

1. It uses `scipy.optimize.least_squares`.
1. For `trf` and `dogbox`, the code supplies a Jacobian sparsity pattern so finite differencing scales with the true bundle-adjustment dependency graph instead of a dense matrix.
1. Residuals are assembled camera-by-camera to reduce Python overhead.
1. The function guards against scale ambiguity when points and camera poses are both free but too few cameras are fixed.
1. If `known_points` is provided, the solver appends three residuals per constrained point to keep selected 3D coordinates near supplied target positions.

In practice, the free variables are partitioned as:

$$
z = [\\theta\_{j_1}, \\theta\_{j_2}, \\dots, X_1, X_2, \\dots].
$$

Each observation residual depends only on:

1. One camera parameter block.
1. One 3D point block.

Each known-point anchor residual depends only on:

1. One 3D point block.

That sparse structure is the reason `jac_sparsity` matters so much for runtime.

## `guarded_two_step_bundle_adjustment`

This routine is designed for a more conservative refinement flow:

1. Start from baseline calibrations and points.
1. Run a pose-oriented bundle-adjustment stage.
1. Run a second intrinsic-focused stage with all cameras fixed in pose.
1. Accept the intrinsic stage only if it does not degrade reprojection RMS and, by default, does not worsen mean ray convergence.

When `known_points` is supplied, the same soft geometry anchors are applied consistently in both stages.

This gives three possible final outcomes:

1. `baseline`: both optimized stages are rejected.
1. `pose`: the pose stage is accepted but the intrinsic stage is rejected.
1. `intrinsics`: both stages are accepted.

This is useful when the intrinsic update is intentionally tiny and should only be kept if it is clearly beneficial.

## Metrics Reported

The current routines report and the demo prints:

1. `initial_reprojection_rms` and `final_reprojection_rms`.
1. Per-camera reprojection RMS for `multi_camera_bundle_adjustment`.
1. `baseline_mean_ray_convergence`, `pose_mean_ray_convergence`, and `final_mean_ray_convergence` for guarded runs.
1. `baseline_correspondence_rate`, `pose_correspondence_rate`, and `intrinsic_correspondence_rate` for guarded runs when original quadruplet identities are available.
1. Runtime in seconds for each experiment.
1. `known_point_indices` for constrained runs in `multi_camera_bundle_adjustment`.

Reprojection RMS answers "how well do the updated cameras explain the observed image measurements?"

Mean ray convergence answers "how tightly do back-projected camera rays meet in 3D?"

Lower is better for both metrics.

## Demo Script

The demo entry point is:

```bash
python -m openptv_python.demo_bundle_adjustment
```

By default it uses `tests/testing_fodder/test_cavity`, writes results into `tmp/bundle_adjustment_demo`, and evaluates several presets.

### Demo Options

The command-line options are:

1. `case_dir`: optional positional argument pointing at a compatible case folder.
1. `--max-frames N`: only load the first `N` frames from `sequence.par`.
1. `--max-points-per-frame N`: only keep the first `N` fully observed points per frame.
1. `--perturbation-scale S`: scale the deterministic starting pose perturbation.
1. `--output-dir PATH`: write one output case folder per experiment under this directory.
1. `--skip-write`: run the experiments without writing updated calibration folders.
1. `--known-points N`: select `N` evenly spaced input 3D points as soft geometry anchors. `0` disables constrained presets.
1. `--known-point-sigma S`: apply the same object-space sigma to each anchored 3D coordinate.
1. `--diagnose-fixed-pairs`: run one selected experiment across every two-camera fixed pair instead of running the normal preset table.
1. `--diagnostic-experiment NAME`: choose which preset to sweep when `--diagnose-fixed-pairs` is enabled.
1. `--diagnose-epipolar`: compare pairwise epipolar consistency before and after the selected diagnostic experiment.
1. `--diagnose-quadruplets`: compare leave-one-camera-out quadruplet stability before and after the selected diagnostic experiment.
1. `--epipolar-curve-points N`: sample `N` points along each epipolar curve when approximating point-to-curve distance in pixels.
1. `--geometry-guard-mode {auto,off,soft,hard}`: control whether guarded two-step BA rejects stages that move a known 3D calibration target too far from a trusted reference calibration. `auto` enables `hard` when `parameters/cal_ori.par` points to a known 3D target file such as `cal/target_on_a_side.txt`.
1. `--geometry-guard-threshold S`: pixel threshold used by `hard` geometry guards.
1. `--geometry-export-threshold S`: refuse to write output case folders whose final calibration-body drift exceeds `S` pixels. Use `0` to disable export blocking.
1. `--correspondence-guard-mode {auto,off,soft,hard}`: control whether guarded two-step BA rejects stages that replace too many original quadruplet target identities after reprojection and nearest-target reassignment. `auto` derives a threshold from the trusted reference calibration when tracking data is available.
1. `--correspondence-guard-threshold S`: replacement-rate threshold used by `hard` correspondence guards.
1. `--correspondence-export-threshold S`: refuse to write output case folders whose final correspondence replacement rate exceeds `S`. Use `0` to disable correspondence-based export blocking.

### Included Demo Presets

The current demo always compares five unconstrained presets:

1. `pose_trf_linear`: pose-only bundle adjustment with `method="trf"` and `loss="linear"`.
1. `pose_soft_l1`: pose-only bundle adjustment with robust `soft_l1` loss.
1. `pose_fixed_points`: pose-only bundle adjustment with `optimize_points=False`.
1. `intrinsics_only`: the demo resets to the reference camera poses, perturbs only `k1`, `p1`, and `p2`, and then refines just those intrinsic terms while all camera poses and 3D points remain fixed.
1. `guarded_two_step`: pose stage plus tightly constrained intrinsic stage.

When a case exposes known 3D calibration-target points through `cal_ori.par`, the demo now defaults to a geometry-preserving workflow:

1. Guarded presets use a `hard` acceptance check against the known target projections.
1. Output case folders are written only if the final calibration stays within the configured export drift threshold.
1. Cases without a known 3D target file still run normally; the geometry guard simply stays off unless you explicitly enable it another way.

When a case also exposes original tracked quadruplets and per-camera target files, the demo adds a correspondence-preserving workflow on top:

1. Guarded presets can compare original target identities with the identities implied by the refined calibration.
1. `auto` correspondence mode derives a threshold from the trusted reference calibration and rejects later stages that replace materially more quadruplets than that reference already does.
1. Output case folders are skipped if the final replacement rate exceeds the configured correspondence export threshold.

If `--known-points` is greater than zero, the demo also compares two constrained presets:

1. `pose_trf_known_points`: pose-only bundle adjustment with soft 3D anchors.
1. `guarded_two_step_known_points`: guarded two-step refinement with the same 3D anchors active in both stages.

### Example Commands

Small, fast smoke test:

```bash
python -m openptv_python.demo_bundle_adjustment \
  --max-frames 1 \
  --max-points-per-frame 16 \
  --output-dir .tmp/demo_bundle_adjustment_check
```

Larger comparison on `test_cavity`:

```bash
python -m openptv_python.demo_bundle_adjustment \
  tests/testing_fodder/test_cavity \
  --max-frames 2 \
  --max-points-per-frame 80 \
  --known-points 12 \
  --known-point-sigma 0.25 \
  --output-dir .tmp/demo_bundle_adjustment_runs
```

Run the same case but disable automatic geometry blocking if you intentionally want to inspect all candidate exports:

```bash
python -m openptv_python.demo_bundle_adjustment \
  tests/testing_fodder/test_cavity \
  --max-frames 2 \
  --max-points-per-frame 80 \
  --known-points 12 \
  --known-point-sigma 0.25 \
  --geometry-guard-mode off \
  --geometry-export-threshold 0 \
  --output-dir .tmp/demo_bundle_adjustment_runs_all
```

Keep export blocking but switch guarded acceptance to a softer monotonic geometry check:

```bash
python -m openptv_python.demo_bundle_adjustment \
  tests/testing_fodder/test_cavity \
  --max-frames 2 \
  --max-points-per-frame 80 \
  --geometry-guard-mode soft \
  --geometry-export-threshold 2.5 \
  --output-dir .tmp/demo_bundle_adjustment_runs_soft
```

Keep geometry blocking but also enforce a hard correspondence replacement limit derived from the trusted reference calibration:

```bash
python -m openptv_python.demo_bundle_adjustment \
  tests/testing_fodder/test_cavity \
  --max-frames 2 \
  --max-points-per-frame 80 \
  --correspondence-guard-mode auto \
  --output-dir .tmp/demo_bundle_adjustment_runs_correspondence
```

Run without writing calibration folders:

```bash
python -m openptv_python.demo_bundle_adjustment --skip-write
```

Disable constrained presets entirely:

```bash
python -m openptv_python.demo_bundle_adjustment --known-points 0 --skip-write
```

Sweep all two-camera anchor pairs for the guarded solver:

```bash
python -m openptv_python.demo_bundle_adjustment \
  tests/testing_fodder/test_cavity \
  --max-frames 2 \
  --max-points-per-frame 80 \
  --known-points 12 \
  --known-point-sigma 0.25 \
  --skip-write \
  --diagnose-fixed-pairs \
  --diagnostic-experiment guarded_two_step
```

Compare epipolar consistency and quadruplet sensitivity before and after one guarded run:

```bash
python -m openptv_python.demo_bundle_adjustment \
  tests/testing_fodder/test_cavity \
  --max-frames 2 \
  --max-points-per-frame 80 \
  --known-points 12 \
  --known-point-sigma 0.25 \
  --skip-write \
  --diagnostic-experiment guarded_two_step \
  --diagnose-epipolar \
  --diagnose-quadruplets
```

## Output Folders

When writing is enabled, each experiment produces a case copy like:

```text
tmp/bundle_adjustment_demo/
  pose_trf_linear/
    cal/
      cam1.tif.ori
      cam1.tif.addpar
      ...
    calibration_delta.txt
    geometry_check.txt
    correspondence_check.txt
```

`calibration_delta.txt` is generated with `openptv_python.calibration_compare` and shows camera-by-camera parameter differences relative to the source case.

`geometry_check.txt` reports per-camera drift of the known 3D calibration-body projections relative to the trusted reference calibration. If the final drift exceeds `--geometry-export-threshold`, the demo does not write that experiment's case folder at all.

`correspondence_check.txt` reports how many original quadruplet target identities are replaced by the refined calibration after nearest-target reassignment. If the final replacement rate exceeds `--correspondence-export-threshold`, the demo does not write that experiment's case folder at all.

## Choosing Options

These are the main tradeoffs when selecting settings.

### Loss Function

1. `linear`: use when correspondences are trusted and you want a pure least-squares solution.
1. `soft_l1`: use when some observations may behave like outliers.

### Optimizing Points

1. `optimize_points=True`: stronger joint refinement, but more variables and slower solves.
1. `optimize_points=False`: faster, more stable if initial 3D points are already good enough.

The `intrinsics_only` preset deliberately uses `optimize_points=False` so the run changes only the selected intrinsic parameters and leaves both poses and 3D points untouched.

Known-point constraints currently require `optimize_points=True` because the anchors act directly on the point coordinates.

### Known 3D Point Anchors

1. Use `known_points` when some reconstructed 3D points correspond to trusted reference-object coordinates or to positions you deliberately want to preserve.
1. Smaller `known_point_sigmas` keep the selected points closer to their supplied coordinates.
1. Larger `known_point_sigmas` let reprojection error dominate while still damping large geometric drift.
1. In the demo, anchored points are selected from the loaded input 3D points, so the constrained presets show how soft geometry anchoring changes the solution relative to the unconstrained runs rather than injecting an external ground-truth object.

### Fixed Cameras

Fixing multiple cameras helps remove similarity-gauge ambiguity. If both points and poses are free, the implementation requires either:

1. At least two fixed cameras.
1. Or translation priors for `x0`, `y0`, and `z0`.

If you suspect that one camera is drifting more than the others, the first thing to test is not a single "best" fixed camera. A single fixed camera is still gauge-ambiguous for free-point bundle adjustment. The more defensible diagnostic is to sweep all two-camera fixed pairs and compare:

1. Final reprojection RMS.
1. Final mean ray convergence.
1. Whether the fixed cameras stayed numerically unchanged.
1. How much the free cameras moved relative to the starting calibration.

The demo's `--diagnose-fixed-pairs` mode prints exactly this table so you can see whether the suspicious motion is tied to one anchor pair or appears across many pairs.

### Epipolar And Quadruplet Diagnostics

The new diagnostic mode is intended for the situation where reprojection error improves but calibration-body structure or correspondence quality still looks wrong.

`--diagnose-epipolar` reports pairwise point-to-epipolar-curve distances in pixels for every camera pair. This is not yet an optimization term in bundle adjustment; it is a diagnostic on the observed image correspondences under the current calibration.

`--diagnose-quadruplets` runs a leave-one-camera-out reconstruction check on fully observed points and reports how much the reconstructed 3D point moves when one camera is omitted. Large spreads indicate unstable quadruplets or geometry that is highly sensitive to one view.

This split is useful because the two diagnostics answer different questions:

1. Epipolar distance asks whether the image correspondences are pairwise consistent with the current camera geometry.
1. Quadruplet sensitivity asks whether a nominal four-camera match remains stable when one view is removed.
1. Reprojection RMS asks whether the final 3D points and cameras explain the full observation set.

If epipolar distance stays bad while reprojection improves, the next step is usually not to add an epipolar term blindly. First identify whether a subset of camera pairs or quadruplets is driving the inconsistency, then decide whether to reject or downweight those observations.

The correspondence guard is aimed at a related failure mode: a calibration can reduce RMS by drifting until many refined projections land closer to different detected targets than the original quadruplet used. In that case, the solution is numerically fitting the images better while no longer preserving the same tracked structure. The new replacement-rate check gives you an explicit acceptance and export criterion for that situation.

### Priors and Bounds

1. `prior_sigmas` softly regularize motion away from the starting calibration.
1. `parameter_bounds` clip the allowed movement relative to the starting calibration.

This is especially useful for guarded intrinsic updates where you only want very small corrections.

## Practical Notes

1. `lm` can be effective on small synthetic problems, but `trf` is generally the better default for larger real cases because it supports bounds and Jacobian sparsity.
1. Real-case runtime depends strongly on the number of frames and points included.
1. `read_targets` currently prints filenames as it loads target files, so demo output includes those lines.

## Related Code

The main implementation lives in:

1. `openptv_python.orientation`
1. `openptv_python.demo_bundle_adjustment`
1. `openptv_python.calibration_compare`
