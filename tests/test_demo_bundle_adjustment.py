import unittest
from pathlib import Path

import numpy as np

from openptv_python.demo_bundle_adjustment import (
    ExperimentResult,
    all_fixed_camera_pairs,
    build_experiment_start_calibrations,
    build_known_point_constraints,
    calibration_body_projection_drift,
    default_experiments,
    format_quadruplet_sensitivity,
    load_calibrations,
    load_reference_geometry_points,
    perturb_calibrations,
    summarize_epipolar_consistency,
    summarize_fixed_camera_diagnostics,
    summarize_quadruplet_sensitivity,
)
from openptv_python.calibration import read_calibration
from openptv_python.parameters import ControlPar, read_volume_par
from openptv_python.tracking_frame_buf import read_path_frame, read_targets


class TestBundleAdjustmentDemo(unittest.TestCase):
    def test_build_known_point_constraints_returns_copied_subset(self):
        point_init = np.arange(30, dtype=float).reshape(10, 3)

        known_points = build_known_point_constraints(point_init, 4)

        self.assertEqual(sorted(known_points), [0, 3, 6, 9])
        np.testing.assert_allclose(known_points[6], point_init[6])

        point_init[6, 0] = -999.0
        self.assertNotEqual(known_points[6][0], point_init[6, 0])

    def test_default_experiments_adds_known_point_presets(self):
        known_points = {0: np.array([0.0, 0.0, 0.0]), 5: np.array([1.0, 2.0, 3.0])}

        experiments = default_experiments(
            known_points=known_points,
            known_point_sigmas=0.25,
        )

        names = [spec.name for spec in experiments]
        self.assertIn("intrinsics_only", names)
        self.assertIn("pose_trf_known_points", names)
        self.assertIn("guarded_two_step_known_points", names)

        intrinsics_only = next(
            spec for spec in experiments if spec.name == "intrinsics_only"
        )
        self.assertFalse(intrinsics_only.ba_kwargs["optimize_extrinsics"])
        self.assertFalse(intrinsics_only.ba_kwargs["optimize_points"])
        self.assertEqual(
            intrinsics_only.ba_kwargs["fixed_camera_indices"],
            [0, 1, 2, 3],
        )
        self.assertTrue(intrinsics_only.ba_kwargs["use_reference_cals"])
        self.assertTrue(intrinsics_only.ba_kwargs["perturb_intrinsics_only"])
        self.assertEqual(intrinsics_only.ba_kwargs["intrinsic_perturbation_scale"], 1.0)

        pose_spec = next(
            spec for spec in experiments if spec.name == "pose_trf_known_points"
        )
        guarded_spec = next(
            spec
            for spec in experiments
            if spec.name == "guarded_two_step_known_points"
        )
        self.assertIs(pose_spec.ba_kwargs["known_points"], known_points)
        self.assertEqual(pose_spec.ba_kwargs["known_point_sigmas"], 0.25)
        self.assertIs(guarded_spec.ba_kwargs["known_points"], known_points)
        self.assertEqual(guarded_spec.ba_kwargs["known_point_sigmas"], 0.25)
        self.assertEqual(guarded_spec.ba_kwargs["geometry_guard_mode"], "off")
        self.assertIsNone(guarded_spec.ba_kwargs["geometry_guard_threshold"])

    def test_default_experiments_accepts_geometry_guard_configuration(self):
        experiments = default_experiments(
            perturbation_scale=1.0,
            geometry_guard_mode="hard",
            geometry_guard_threshold=2.5,
        )

        guarded_spec = next(
            spec for spec in experiments if spec.name == "guarded_two_step"
        )

        self.assertEqual(guarded_spec.ba_kwargs["geometry_guard_mode"], "hard")
        self.assertEqual(guarded_spec.ba_kwargs["geometry_guard_threshold"], 2.5)

    def test_pose_demo_keeps_fixed_cameras_on_reference_geometry(self):
        cavity_dir = Path("tests/testing_fodder/test_cavity")
        control = ControlPar(4).from_file(cavity_dir / "parameters/ptv.par")
        true_cals = load_calibrations(cavity_dir, 4)
        start_cals = perturb_calibrations(true_cals, 1.0)
        spec = next(
            experiment
            for experiment in default_experiments(perturbation_scale=1.0)
            if experiment.name == "pose_trf_linear"
        )

        working_cals = build_experiment_start_calibrations(
            spec,
            start_cals=start_cals,
            reference_cals=true_cals,
        )
        geometry_points = load_reference_geometry_points(cavity_dir, 4)
        drift = calibration_body_projection_drift(
            true_cals,
            working_cals,
            control,
            geometry_points,
        )

        self.assertIsNotNone(drift)
        np.testing.assert_allclose(working_cals[0].get_pos(), true_cals[0].get_pos())
        np.testing.assert_allclose(working_cals[1].get_pos(), true_cals[1].get_pos())
        np.testing.assert_allclose(
            working_cals[0].get_angles(),
            true_cals[0].get_angles(),
        )
        np.testing.assert_allclose(
            working_cals[1].get_angles(),
            true_cals[1].get_angles(),
        )
        self.assertLess(drift[0].max_distance, 1e-9)
        self.assertLess(drift[1].max_distance, 1e-9)
        self.assertGreater(drift[2].max_distance, 0.1)

    def test_all_fixed_camera_pairs_returns_all_unique_pairs(self):
        self.assertEqual(
            all_fixed_camera_pairs(4),
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        )

    def test_summarize_fixed_camera_diagnostics_orders_by_fixed_drift_then_rms(self):
        results = [
            ExperimentResult(
                name="run_b",
                description="",
                duration_sec=0.2,
                success=True,
                initial_rms=4.0,
                final_rms=1.3,
                baseline_ray_convergence=2.0,
                final_ray_convergence=1.1,
                notes="b",
                cal_dir=None,
                fixed_camera_indices=(0, 2),
                camera_position_shifts=[0.0, 0.8, 0.0, 0.6],
                camera_angle_shifts=[0.0, 0.02, 0.0, 0.01],
                refined_cals=None,
                refined_points=None,
            ),
            ExperimentResult(
                name="run_a",
                description="",
                duration_sec=0.1,
                success=True,
                initial_rms=4.0,
                final_rms=1.2,
                baseline_ray_convergence=2.0,
                final_ray_convergence=1.0,
                notes="a",
                cal_dir=None,
                fixed_camera_indices=(0, 1),
                camera_position_shifts=[0.0, 0.0, 0.5, 0.4],
                camera_angle_shifts=[0.0, 0.0, 0.01, 0.02],
                refined_cals=None,
                refined_points=None,
            ),
        ]

        diagnostics = summarize_fixed_camera_diagnostics(results)

        self.assertEqual(diagnostics[0].fixed_camera_indices, (0, 1))
        self.assertEqual(diagnostics[1].fixed_camera_indices, (0, 2))
        self.assertEqual(diagnostics[0].fixed_position_shift, 0.0)
        self.assertAlmostEqual(diagnostics[0].mean_free_position_shift, 0.45)

    def test_epipolar_and_quadruplet_diagnostics_detect_perturbation(self):
        cavity_dir = Path("tests/testing_fodder/test_cavity")
        control = ControlPar(4).from_file(cavity_dir / "parameters/ptv.par")
        vpar = read_volume_par(cavity_dir / "parameters/criteria.par")
        cals = [
            read_calibration(
                cavity_dir / f"cal/cam{cam_num}.tif.ori",
                cavity_dir / f"cal/cam{cam_num}.tif.addpar",
            )
            for cam_num in range(1, 5)
        ]

        cor_buf, path_buf = read_path_frame(str(cavity_dir / "res_orig/rt_is"), "", "", 10001)
        targets = [
            read_targets(str(cavity_dir / f"img_orig/cam{cam_num}.%05d"), 10001)
            for cam_num in range(1, 5)
        ]
        subset = [
            pt_num for pt_num, corres in enumerate(cor_buf) if np.all(corres.p >= 0)
        ][:8]
        observed_pixels = np.full((len(subset), 4, 2), np.nan, dtype=float)
        for out_num, pt_num in enumerate(subset):
            for cam in range(4):
                target_index = cor_buf[pt_num].p[cam]
                observed_pixels[out_num, cam, 0] = targets[cam][target_index].x
                observed_pixels[out_num, cam, 1] = targets[cam][target_index].y

        baseline_epipolar = summarize_epipolar_consistency(
            observed_pixels,
            cals,
            control,
            vpar,
            num_curve_points=16,
        )
        baseline_quad = summarize_quadruplet_sensitivity(observed_pixels, cals, control)

        perturbed = observed_pixels.copy()
        perturbed[0, 1, 0] += 25.0
        perturbed[0, 1, 1] -= 15.0
        perturbed_epipolar = summarize_epipolar_consistency(
            perturbed,
            cals,
            control,
            vpar,
            num_curve_points=16,
        )
        perturbed_quad = summarize_quadruplet_sensitivity(perturbed, cals, control)

        self.assertGreater(
            max(item.max_distance for item in perturbed_epipolar),
            max(item.max_distance for item in baseline_epipolar),
        )
        self.assertGreater(perturbed_quad.max_spread, baseline_quad.max_spread)
        self.assertIn("baseline", format_quadruplet_sensitivity(baseline_quad, perturbed_quad))


if __name__ == "__main__":
    unittest.main()