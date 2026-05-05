# 3D-PTV Full Pipeline Test — Refactoring Plan

## Pipeline Overview

The 3D-PTV pipeline has six steps. Currently only step 6 (tracking) is tested end-to-end on real data.

| Step | Module | Key Function(s) | Tested E2E? |
|------|--------|------------------|-------------|
| 1. Image preprocessing | `image_processing.py` | `preprocess_image()`, `prepare_image()` | No |
| 2. Target detection | `segmentation.py` | `target_recognition()`, `targ_rec()` | No |
| 3. Coordinate correction | `trafo.py` | `pixel_to_metric()`, `dist_to_flat()` | No |
| 4. Stereo correspondence | `correspondences.py` | `correspondences()` | No |
| 5. 3D triangulation | `orientation.py` | `point_position()` | No |
| 6. Tracking | `track.py` | `track_forward_start()`, `trackcorr_c_loop()`, `trackcorr_c_finish()` | Yes |

## Current State of Tests

### `test_burgers.py`

- Tests only step 6 (tracking).
- Copies pre-computed `res_orig/rt_is.*` and `img_orig/*_targets` into working directories.
- Calls `tr_new()` then runs the tracking loop.
- Asserts `npart=19, nlinks=17` (without add) and `npart=20, nlinks=20` (with add).
- Uses `unittest.TestCase` and `os.chdir()`.

### `test_sequence_and_tracking_burgers.py`

- Attempts steps 3-6 (coordinate correction, correspondence, triangulation, tracking).
- Has the stereo correspondence + triangulation loop inline.
- Compares generated `rt_is` files against reference.
- Missing step 1 (preprocessing) and step 2 (detection) because **burgers has no raw images**.
- Missing `dist_to_flat()` distortion correction — only calls `pixel_to_metric()`.
- Contains excessive debug prints.

### `test_sequence_and_tracking_cavity.py`

- Copies pre-computed `rt_is.*` from `res_orig/` and runs tracking only.
- Does not exercise steps 1-5 despite test_cavity having raw TIFF images.

### Unit tests (existing but isolated)

These modules have standalone unit tests on synthetic data but are never connected in a pipeline:

- `test_image_processing.py` — synthetic images
- `test_segmentation.py` — synthetic spots
- `test_corresp.py` — synthetic setup
- `test_trafo.py` — unit transforms

## Data Availability

| Dataset | Raw images | `_targets` files | `rt_is.*` files | Calibration |
|---------|-----------|------------------|-----------------|-------------|
| `burgers` | No | Yes (5 particles/frame, 5 frames, 4 cameras) | Yes | Yes (4 cameras, 1024x1024) |
| `test_cavity` | Yes (TIFF, 1280x1024) | Yes | Yes | Yes (4 cameras) |

The **test_cavity** dataset is the only one with raw images and is the natural candidate for a full pipeline test.

## Plan

### Task 1 — Create `tests/test_full_pipeline_cavity.py` (High Priority)

A single `@pytest.mark.integration` test that exercises all six steps on the test_cavity data in a temporary directory.

#### Step 1: Image Preprocessing

- Load raw TIFF images `cam{1..4}.{10001..10004}` as uint8 numpy arrays.
- Read `ControlPar` from `parameters/ptv.par` (imx, imy, pixel sizes).
- Read filter parameters from `parameters/unsharp_mask.par` (dim_lp value).
- Call `preprocess_image(img, filter_hp, cpar, dim_lp)` for each camera and frame.
- Assert output is uint8, same shape as input, mean intensity lower than original (highpass removes background).

#### Step 2: Target Detection (Segmentation)

- Read `TargetPar` from `parameters/targ_rec.par`.
- Call `target_recognition(preprocessed_img, tpar, cam, cpar)` for each camera and frame.
- Assert detected target count is within tolerance of the reference `_targets` file count.
- Write detected targets with `write_targets()` into `img/` directory.

#### Step 3: Coordinate Correction

- For each detected target, call `pixel_to_metric(t.x, t.y, cpar)` to convert from pixel to metric coordinates.
- Call `dist_to_flat(x_metric, y_metric, cal)` to remove lens distortion.
- Build sorted corrected coordinate arrays (sorted by x) for the correspondence step.

#### Step 4: Stereo Correspondence

- Build a `Frame` object with sorted targets per camera.
- Read `VolumePar` from `parameters/criteria.par`.
- Call `correspondences(frm, corrected, vpar, cpar, cal, match_counts)`.
- Assert `match_counts` shows a reasonable number of matches.
- Compare against reference `rt_is` files if strict reproducibility is expected.

#### Step 5: 3D Triangulation

- For each correspondence, extract image points from matched targets.
- Call `point_position(img_pts, num_cams, mm_par, cal)` to triangulate.
- Build `Corres_dtype` and `Pathinfo` arrays.
- Write `rt_is`, `ptv_is`, `added` files with `write_path_frame()`.
- Assert 3D positions fall within the calibrated volume bounds.

#### Step 6: Tracking

- Create `TrackingRun` from the generated correspondence files.
- Run `track_forward_start()` then `trackcorr_c_loop()` for each step, then `trackcorr_c_finish()`.
- Assert `npart > 0` and `nlinks > 0`.

#### Required imports

```python
from openptv_python.image_processing import preprocess_image
from openptv_python.segmentation import target_recognition
from openptv_python.parameters import (
    read_control_par, read_sequence_par, read_track_par, read_volume_par,
    read_target_par, convert_track_par_to_tuple, MultimediaPar,
)
from openptv_python.trafo import pixel_to_metric, dist_to_flat
from openptv_python.correspondences import correspondences
from openptv_python.orientation import point_position
from openptv_python.calibration import read_calibration
from openptv_python.tracking_run import TrackingRun
from openptv_python.track import track_forward_start, trackcorr_c_loop, trackcorr_c_finish
from openptv_python.tracking_frame_buf import (
    Frame, Target, Pathinfo, Corres_dtype,
    read_targets, write_targets, write_path_frame,
)
from openptv_python.constants import COORD_UNUSED
```

### Task 2 — Clean up `test_sequence_and_tracking_burgers.py` (Medium Priority)

- Remove excessive `[DEBUG]` print statements.
- Add `dist_to_flat()` after `pixel_to_metric()` in the coordinate correction chain — the current code skips lens distortion correction, which may cause correspondence mismatches.
- Add proper assertions for correspondence counts per frame.
- Add assertions for tracking results (`npart`, `nlinks`).

### Task 3 — Modernize `test_burgers.py` (Low Priority)

- Convert from `unittest.TestCase` to pytest functions.
- Replace `os.chdir()` with `tmp_path` fixture and absolute paths.
- Replace manual `remove_directory()` / `copy_directory()` with `shutil` + `tmp_path` cleanup.
- Keep the existing known-good assertions (`npart=19/20`, `nlinks=17/20`) as a tracking regression test.

## Key Risks and Open Questions

1. **Segmentation reproducibility**: `target_recognition()` results may not exactly match the pre-computed `_targets` files, especially if the original files were generated by the C implementation. The test should use tolerance-based comparison (target count within N%, centroid positions within a pixel).

2. **Correspondence sensitivity to coordinate correction**: The missing `dist_to_flat()` in `test_sequence_and_tracking_burgers.py` may be why that test needs to compare against reference files rather than computing from scratch. Need to verify whether adding distortion correction changes the results.

3. **MultimediaPar defaults**: The existing tests use `MultimediaPar()` default constructor. Need to verify this matches the actual experimental setup (air/glass/water layers). The burgers and cavity datasets may have different multimedia configurations.

4. **test_cavity parameter completeness**: Need to confirm `test_cavity/parameters/` contains all required files: `criteria.par` (VolumePar), `track.par`, `targ_rec.par`, `unsharp_mask.par`.
