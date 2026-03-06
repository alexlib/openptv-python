# Bundle Adjustment

This page documents the bundle-adjustment routines implemented in `openptv_python.orientation` and the demo driver in `openptv_python.demo_bundle_adjustment`.

## Overview

The repository currently exposes two related calibration-refinement workflows:

1. `multi_camera_bundle_adjustment`: jointly refine camera parameters and optionally 3D points by minimizing reprojection error.
2. `guarded_two_step_bundle_adjustment`: run a pose-focused bundle-adjustment stage first, then a tightly constrained intrinsic stage, and reject the second stage if it makes the solution worse.

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

For each observed 2D image point $u_{ij} = (x_{ij}, y_{ij})$ from camera $j$ and 3D point $X_i$, bundle adjustment minimizes reprojection residuals of the form

$$
r_{ij} =
\begin{bmatrix}
\hat{x}_{ij}(X_i, \theta_j) - x_{ij} \\
\hat{y}_{ij}(X_i, \theta_j) - y_{ij}
\end{bmatrix},
$$

where $\theta_j$ denotes the active calibration parameters for camera $j$.

The solver minimizes a robust or linear least-squares objective

$$
\min_{\Theta, X}
\sum_{i,j} \rho\left(\left\|W r_{ij}\right\|^2\right)
+ \sum_k \left(\frac{p_k - p_{k,0}}{\sigma_k}\right)^2,
$$

with:

1. $W$ converting pixel residuals into normalized units using `pix_x` and `pix_y`.
2. $\rho(\cdot)$ chosen by the `loss` argument, for example `linear` or `soft_l1`.
3. Optional Gaussian-style priors on selected parameters via `prior_sigmas`.

The implementation evaluates the forward projection through OpenPTV's existing camera and multimedia model, so refraction and the Brown-affine distortion model remain part of the optimization.

## Implemented Algorithms

## `multi_camera_bundle_adjustment`

This is the general solver. It can optimize:

1. Exterior parameters: `x0`, `y0`, `z0`, `omega`, `phi`, `kappa`.
2. Optional intrinsic and distortion parameters controlled by `OrientPar` flags.
3. The 3D points themselves, if `optimize_points=True`.

Important implementation details:

1. It uses `scipy.optimize.least_squares`.
2. For `trf` and `dogbox`, the code supplies a Jacobian sparsity pattern so finite differencing scales with the true bundle-adjustment dependency graph instead of a dense matrix.
3. Residuals are assembled camera-by-camera to reduce Python overhead.
4. The function guards against scale ambiguity when points and camera poses are both free but too few cameras are fixed.

In practice, the free variables are partitioned as:

$$
z = [\theta_{j_1}, \theta_{j_2}, \dots, X_1, X_2, \dots].
$$

Each observation residual depends only on:

1. One camera parameter block.
2. One 3D point block.

That sparse structure is the reason `jac_sparsity` matters so much for runtime.

## `guarded_two_step_bundle_adjustment`

This routine is designed for a more conservative refinement flow:

1. Start from baseline calibrations and points.
2. Run a pose-oriented bundle-adjustment stage.
3. Run a second intrinsic-focused stage with all cameras fixed in pose.
4. Accept the intrinsic stage only if it does not degrade reprojection RMS and, by default, does not worsen mean ray convergence.

This gives three possible final outcomes:

1. `baseline`: both optimized stages are rejected.
2. `pose`: the pose stage is accepted but the intrinsic stage is rejected.
3. `intrinsics`: both stages are accepted.

This is useful when the intrinsic update is intentionally tiny and should only be kept if it is clearly beneficial.

## Metrics Reported

The current routines report and the demo prints:

1. `initial_reprojection_rms` and `final_reprojection_rms`.
2. Per-camera reprojection RMS for `multi_camera_bundle_adjustment`.
3. `baseline_mean_ray_convergence`, `pose_mean_ray_convergence`, and `final_mean_ray_convergence` for guarded runs.
4. Runtime in seconds for each experiment.

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
2. `--max-frames N`: only load the first `N` frames from `sequence.par`.
3. `--max-points-per-frame N`: only keep the first `N` fully observed points per frame.
4. `--perturbation-scale S`: scale the deterministic starting pose perturbation.
5. `--output-dir PATH`: write one output case folder per experiment under this directory.
6. `--skip-write`: run the experiments without writing updated calibration folders.

### Included Demo Presets

The current demo compares four presets:

1. `pose_trf_linear`: pose-only bundle adjustment with `method="trf"` and `loss="linear"`.
2. `pose_soft_l1`: pose-only bundle adjustment with robust `soft_l1` loss.
3. `pose_fixed_points`: pose-only bundle adjustment with `optimize_points=False`.
4. `guarded_two_step`: pose stage plus tightly constrained intrinsic stage.

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
  --output-dir .tmp/demo_bundle_adjustment_runs
```

Run without writing calibration folders:

```bash
python -m openptv_python.demo_bundle_adjustment --skip-write
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
```

`calibration_delta.txt` is generated with `openptv_python.calibration_compare` and shows camera-by-camera parameter differences relative to the source case.

## Choosing Options

These are the main tradeoffs when selecting settings.

### Loss Function

1. `linear`: use when correspondences are trusted and you want a pure least-squares solution.
2. `soft_l1`: use when some observations may behave like outliers.

### Optimizing Points

1. `optimize_points=True`: stronger joint refinement, but more variables and slower solves.
2. `optimize_points=False`: faster, more stable if initial 3D points are already good enough.

### Fixed Cameras

Fixing multiple cameras helps remove similarity-gauge ambiguity. If both points and poses are free, the implementation requires either:

1. At least two fixed cameras.
2. Or translation priors for `x0`, `y0`, and `z0`.

### Priors and Bounds

1. `prior_sigmas` softly regularize motion away from the starting calibration.
2. `parameter_bounds` clip the allowed movement relative to the starting calibration.

This is especially useful for guarded intrinsic updates where you only want very small corrections.

## Practical Notes

1. `lm` can be effective on small synthetic problems, but `trf` is generally the better default for larger real cases because it supports bounds and Jacobian sparsity.
2. Real-case runtime depends strongly on the number of frames and points included.
3. `read_targets` currently prints filenames as it loads target files, so demo output includes those lines.

## Related Code

The main implementation lives in:

1. `openptv_python.orientation`
2. `openptv_python.demo_bundle_adjustment`
3. `openptv_python.calibration_compare`
