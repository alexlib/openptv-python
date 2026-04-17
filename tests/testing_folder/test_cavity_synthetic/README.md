# Synthetic Cavity Case

This case is generated deterministically from the geometry of `test_cavity`, but all observations come from known ground truth.

Contents:

- `cal/`: working calibrations recovered from synthetic calibration-body targets using `full_calibration`.
- `ground_truth/cal/`: exact camera models used to project the synthetic data.
- `ground_truth/calibration_body_points.txt`: known 3D calibration-body points.
- `calibration_targets/`: synthetic target files for that calibration body.
- `img_orig/`: synthetic particle target files for two frames.
- `res_orig/`: synthetic `rt_is`, `ptv_is`, and `added` files for those frames.
- `ground_truth/particles/`: exact 3D particle coordinates per frame.
- `ground_truth/manifest.json`: generation seed and calibration-recovery errors.
