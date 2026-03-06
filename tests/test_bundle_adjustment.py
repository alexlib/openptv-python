import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.imgcoord import image_coordinates
from openptv_python.orientation import (
    guarded_two_step_bundle_adjustment,
    mean_ray_convergence,
    multi_camera_bundle_adjustment,
    reprojection_rms,
)
from openptv_python.parameters import ControlPar, OrientPar, SequencePar
from openptv_python.tracking_frame_buf import read_path_frame, read_targets
from openptv_python.trafo import arr_metric_to_pixel


class TestBundleAdjustment(unittest.TestCase):
    def setUp(self):
        self.control = ControlPar(4).from_file(
            Path("tests/testing_folder/control_parameters/control.par")
        )
        self.add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
        self.true_cals = [
            read_calibration(
                Path(f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"),
                self.add_file,
            )
            for cam_num in range(1, 5)
        ]

    @staticmethod
    def clone_calibration(cal: Calibration) -> Calibration:
        return Calibration(
            ext_par=cal.ext_par.copy(),
            int_par=cal.int_par.copy(),
            glass_par=cal.glass_par.copy(),
            added_par=cal.added_par.copy(),
            mmlut=cal.mmlut,
            mmlut_data=cal.mmlut_data,
        )

    def perturb_calibrations(self, true_cals: list[Calibration]) -> list[Calibration]:
        perturbed = [self.clone_calibration(cal) for cal in true_cals]
        deltas = [
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            (np.array([1.2, -0.8, 0.7]), np.array([0.012, -0.010, 0.006])),
            (np.array([-0.9, 0.7, -0.6]), np.array([-0.009, 0.008, -0.005])),
            (np.array([0.8, 1.0, -0.5]), np.array([0.011, 0.007, -0.004])),
        ]
        for cal, (pos_delta, angle_delta) in zip(perturbed, deltas):
            cal.set_pos(cal.get_pos() + pos_delta)
            cal.set_angles(cal.get_angles() + angle_delta)
        return perturbed

    def lightly_perturb_calibrations(
        self, true_cals: list[Calibration]
    ) -> list[Calibration]:
        perturbed = [self.clone_calibration(cal) for cal in true_cals]
        deltas = [
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            (np.array([0.5, -0.3, 0.2]), np.array([0.004, -0.003, 0.002])),
            (np.array([-0.4, 0.3, -0.2]), np.array([-0.003, 0.003, -0.002])),
            (np.array([0.3, 0.4, -0.2]), np.array([0.003, 0.002, -0.002])),
        ]
        for cal, (pos_delta, angle_delta) in zip(perturbed, deltas):
            cal.set_pos(cal.get_pos() + pos_delta)
            cal.set_angles(cal.get_angles() + angle_delta)
        return perturbed

    @staticmethod
    def cavity_quadruplet_observations(cavity_dir: Path, control: ControlPar):
        seq = SequencePar.from_file(cavity_dir / "parameters/sequence.par", 4)
        observed_batches = []
        point_batches = []

        for frame in range(seq.first, seq.last + 1):
            cor_buf, path_buf = read_path_frame(
                str(cavity_dir / "res_orig/rt_is"),
                "",
                "",
                frame,
            )
            targets = [
                read_targets(str(cavity_dir / f"img_orig/cam{cam_num}.%05d"), frame)
                for cam_num in range(1, 5)
            ]
            subset = [
                pt_num for pt_num, corres in enumerate(cor_buf) if np.all(corres.p >= 0)
            ]
            observed_pixels = np.full((len(subset), 4, 2), np.nan, dtype=float)
            point_init = np.empty((len(subset), 3), dtype=float)
            for out_num, pt_num in enumerate(subset):
                point_init[out_num] = path_buf[pt_num].x
                for cam in range(4):
                    target_index = cor_buf[pt_num].p[cam]
                    observed_pixels[out_num, cam, 0] = targets[cam][target_index].x
                    observed_pixels[out_num, cam, 1] = targets[cam][target_index].y
            observed_batches.append(observed_pixels)
            point_batches.append(point_init)

        return np.concatenate(observed_batches, axis=0), np.concatenate(
            point_batches, axis=0
        )

    def test_multi_camera_bundle_adjustment_improves_synthetic_reprojection(self):
        xs = np.linspace(-20.0, 20.0, 3)
        ys = np.linspace(-15.0, 15.0, 2)
        zs = np.array([-4.0, 5.0])
        points = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)

        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        start_cals = self.perturb_calibrations(self.true_cals)
        refined_cals, refined_points, result = multi_camera_bundle_adjustment(
            observed_pixels,
            start_cals,
            self.control,
            OrientPar(),
            point_init=None,
            fixed_camera_indices=[0, 1],
            loss="linear",
            method="lm",
            prior_sigmas={
                "x0": 1.0,
                "y0": 1.0,
                "z0": 1.0,
                "omega": 0.01,
                "phi": 0.01,
                "kappa": 0.01,
            },
            max_nfev=50,
        )

        self.assertTrue(result.success, msg=result.message)
        before_rms = result["initial_reprojection_rms"]
        after_rms = result["final_reprojection_rms"]
        self.assertLess(after_rms, before_rms * 0.4)
        self.assertLess(after_rms, 3e-2)

    def test_cavity_reprojection_improves(self):
        cavity_dir = Path("tests/testing_fodder/test_cavity")
        control = ControlPar(4).from_file(cavity_dir / "parameters/ptv.par")
        true_cals = [
            read_calibration(
                cavity_dir / f"cal/cam{cam_num}.tif.ori",
                cavity_dir / f"cal/cam{cam_num}.tif.addpar",
            )
            for cam_num in range(1, 5)
        ]
        start_cals = self.lightly_perturb_calibrations(true_cals)

        cor_buf, path_buf = read_path_frame(
            str(cavity_dir / "res_orig/rt_is"),
            "",
            "",
            10001,
        )
        targets = [
            read_targets(str(cavity_dir / f"img_orig/cam{cam_num}.%05d"), 10001)
            for cam_num in range(1, 5)
        ]

        subset = [
            pt_num for pt_num, corres in enumerate(cor_buf) if np.all(corres.p >= 0)
        ][:12]
        observed_pixels = np.full((len(subset), 4, 2), np.nan, dtype=float)
        point_init = np.empty((len(subset), 3), dtype=float)
        for out_num, pt_num in enumerate(subset):
            path_info = path_buf[pt_num]
            point_init[out_num] = path_info.x
            for cam in range(4):
                target_index = cor_buf[pt_num].p[cam]
                if target_index < 0:
                    continue
                observed_pixels[out_num, cam, 0] = targets[cam][target_index].x
                observed_pixels[out_num, cam, 1] = targets[cam][target_index].y

        before_rms = reprojection_rms(observed_pixels, point_init, start_cals, control)
        refined_cals, refined_points, result = multi_camera_bundle_adjustment(
            observed_pixels,
            start_cals,
            control,
            OrientPar(),
            point_init=point_init,
            fixed_camera_indices=[0, 1],
            loss="linear",
            method="lm",
            max_nfev=80,
        )
        after_rms = reprojection_rms(
            observed_pixels, refined_points, refined_cals, control
        )

        print(
            f"test_cavity reprojection RMS improved from {before_rms:.6f} px to {after_rms:.6f} px"
        )

        self.assertTrue(result.success, msg=result.message)
        self.assertLess(after_rms, before_rms * 0.6)
        self.assertLess(after_rms, before_rms - 0.2)
        np.testing.assert_allclose(after_rms, result["final_reprojection_rms"])

    def test_bundle_adjustment_rejects_scale_ambiguous_configuration(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )

        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        with self.assertRaises(ValueError):
            multi_camera_bundle_adjustment(
                observed_pixels,
                self.perturb_calibrations(self.true_cals),
                self.control,
                OrientPar(),
                point_init=None,
                fix_first_camera=True,
                loss="linear",
                method="lm",
                max_nfev=10,
            )

    def test_multi_camera_bundle_adjustment_known_points_constrain_geometry(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
                [0.0, 0.0, 3.0],
                [6.0, -4.0, -2.0],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        point_init = points.copy()
        point_init[:2] += np.array([[3.0, -2.0, 1.5], [-2.5, 1.0, -1.0]])
        start_cals = self.perturb_calibrations(self.true_cals)
        ba_kwargs = {
            "point_init": point_init,
            "fixed_camera_indices": [0, 1],
            "loss": "linear",
            "method": "trf",
            "prior_sigmas": {
                "x0": 1.0,
                "y0": 1.0,
                "z0": 1.0,
                "omega": 0.01,
                "phi": 0.01,
                "kappa": 0.01,
            },
            "max_nfev": 8,
        }

        unconstrained_cals, unconstrained_points, unconstrained_result = (
            multi_camera_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                **ba_kwargs,
            )
        )
        constrained_cals, constrained_points, constrained_result = (
            multi_camera_bundle_adjustment(
                observed_pixels,
                self.perturb_calibrations(self.true_cals),
                self.control,
                OrientPar(),
                known_points={0: points[0], 1: points[1]},
                known_point_sigmas=1e-3,
                **ba_kwargs,
            )
        )

        unconstrained_error = float(
            np.mean(np.linalg.norm(unconstrained_points[:2] - points[:2], axis=1))
        )
        constrained_error = float(
            np.mean(np.linalg.norm(constrained_points[:2] - points[:2], axis=1))
        )

        self.assertLess(constrained_error, unconstrained_error)
        self.assertLess(constrained_error, 1e-2)
        self.assertIn(0, constrained_result["known_point_indices"])
        self.assertIn(1, constrained_result["known_point_indices"])
        self.assertLess(
            reprojection_rms(
                observed_pixels,
                constrained_points,
                constrained_cals,
                self.control,
            ),
            unconstrained_result["initial_reprojection_rms"],
        )

    def test_fixed_camera_indices_preserve_selected_camera_poses(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
                [0.0, 0.0, 3.0],
                [6.0, -4.0, -2.0],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        start_cals = self.perturb_calibrations(self.true_cals)
        fixed_indices = [0, 1]
        refined_cals, _, result = multi_camera_bundle_adjustment(
            observed_pixels,
            start_cals,
            self.control,
            OrientPar(),
            point_init=points,
            fixed_camera_indices=fixed_indices,
            loss="linear",
            method="lm",
            prior_sigmas={
                "x0": 1.0,
                "y0": 1.0,
                "z0": 1.0,
                "omega": 0.01,
                "phi": 0.01,
                "kappa": 0.01,
            },
            max_nfev=50,
        )

        self.assertTrue(result.success, msg=result.message)
        for cam_index in fixed_indices:
            np.testing.assert_allclose(
                refined_cals[cam_index].get_pos(),
                start_cals[cam_index].get_pos(),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                refined_cals[cam_index].get_angles(),
                start_cals[cam_index].get_angles(),
                atol=1e-12,
            )
        self.assertEqual(result["optimized_camera_indices"], [2, 3])
        self.assertLess(
            result["final_reprojection_rms"], result["initial_reprojection_rms"]
        )

    def test_known_points_require_point_optimization(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        with self.assertRaises(ValueError):
            multi_camera_bundle_adjustment(
                observed_pixels,
                self.perturb_calibrations(self.true_cals),
                self.control,
                OrientPar(),
                point_init=points,
                fixed_camera_indices=[0, 1],
                optimize_points=False,
                known_points={0: points[0]},
                known_point_sigmas=1e-3,
            )

    def test_guarded_two_step_bundle_adjustment_rejects_bad_intrinsics_stage(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
                [0.0, 0.0, 3.0],
                [6.0, -4.0, -2.0],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        start_cals = self.lightly_perturb_calibrations(self.true_cals)
        pose_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        bad_intrinsic_cals = [self.clone_calibration(cal) for cal in pose_cals]
        for cal in bad_intrinsic_cals[1:]:
            cal.added_par[0] += 1e-3
            cal.added_par[3] += 5e-4
            cal.added_par[4] -= 5e-4

        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1

        with patch(
            "openptv_python.orientation.multi_camera_bundle_adjustment",
            side_effect=[
                (pose_cals, points.copy(), {"success": True, "stage": "pose"}),
                (
                    bad_intrinsic_cals,
                    points.copy(),
                    {"success": True, "stage": "intrinsics"},
                ),
            ],
        ) as mocked_adjustment:
            final_cals, final_points, summary = guarded_two_step_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                fixed_camera_indices=[0, 1],
                pose_prior_sigmas={
                    "x0": 0.5,
                    "y0": 0.5,
                    "z0": 0.5,
                    "omega": 0.005,
                    "phi": 0.005,
                    "kappa": 0.005,
                },
                pose_parameter_bounds={
                    "x0": (-2.0, 2.0),
                    "y0": (-2.0, 2.0),
                    "z0": (-2.0, 2.0),
                    "omega": (-0.02, 0.02),
                    "phi": (-0.02, 0.02),
                    "kappa": (-0.02, 0.02),
                },
                pose_max_nfev=60,
                intrinsic_prior_sigmas={
                    "k1": 1e-12,
                    "k2": 1e-12,
                    "k3": 1e-12,
                    "p1": 1e-12,
                    "p2": 1e-12,
                    "scx": 1e-12,
                    "she": 1e-12,
                    "cc": 1e-12,
                    "xh": 1e-12,
                    "yh": 1e-12,
                },
                intrinsic_parameter_bounds={
                    "k1": (-1e-10, 1e-10),
                    "k2": (-1e-10, 1e-10),
                    "k3": (-1e-10, 1e-10),
                    "p1": (-1e-10, 1e-10),
                    "p2": (-1e-10, 1e-10),
                    "scx": (-1e-12, 1e-12),
                    "she": (-1e-12, 1e-12),
                    "cc": (-1e-12, 1e-12),
                    "xh": (-1e-12, 1e-12),
                    "yh": (-1e-12, 1e-12),
                },
                intrinsic_max_nfev=20,
            )

        self.assertEqual(mocked_adjustment.call_count, 2)
        for call in mocked_adjustment.call_args_list:
            self.assertIsNone(call.kwargs.get("known_points"))
            self.assertIsNone(call.kwargs.get("known_point_sigmas"))
        self.assertEqual(summary["accepted_stage"], "pose")
        self.assertLess(
            summary["pose_reprojection_rms"], summary["baseline_reprojection_rms"]
        )
        self.assertGreater(
            summary["intrinsic_reprojection_rms"], summary["pose_reprojection_rms"]
        )
        self.assertLessEqual(
            summary["final_reprojection_rms"], summary["baseline_reprojection_rms"]
        )
        self.assertLessEqual(
            summary["final_mean_ray_convergence"],
            summary["baseline_mean_ray_convergence"],
        )
        np.testing.assert_allclose(
            reprojection_rms(observed_pixels, final_points, final_cals, self.control),
            summary["final_reprojection_rms"],
        )

    def test_guarded_two_step_bundle_adjustment_preserves_pose_when_intrinsics_are_tight(
        self,
    ):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
                [0.0, 0.0, 3.0],
                [6.0, -4.0, -2.0],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        start_cals = self.perturb_calibrations(self.true_cals)
        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1
        final_cals, final_points, summary = guarded_two_step_bundle_adjustment(
            observed_pixels,
            start_cals,
            self.control,
            OrientPar(),
            intrinsics,
            point_init=points,
            fixed_camera_indices=[0, 1],
            pose_prior_sigmas={
                "x0": 1.0,
                "y0": 1.0,
                "z0": 1.0,
                "omega": 0.01,
                "phi": 0.01,
                "kappa": 0.01,
            },
            pose_max_nfev=50,
            intrinsic_prior_sigmas={
                "k1": 1e-12,
                "k2": 1e-12,
                "k3": 1e-12,
                "p1": 1e-12,
                "p2": 1e-12,
                "scx": 1e-12,
                "she": 1e-12,
                "cc": 1e-12,
                "xh": 1e-12,
                "yh": 1e-12,
            },
            intrinsic_parameter_bounds={
                "k1": (-1e-10, 1e-10),
                "k2": (-1e-10, 1e-10),
                "k3": (-1e-10, 1e-10),
                "p1": (-1e-10, 1e-10),
                "p2": (-1e-10, 1e-10),
                "scx": (-1e-12, 1e-12),
                "she": (-1e-12, 1e-12),
                "cc": (-1e-12, 1e-12),
                "xh": (-1e-12, 1e-12),
                "yh": (-1e-12, 1e-12),
            },
            intrinsic_max_nfev=20,
        )

        self.assertIn(summary["accepted_stage"], {"pose", "intrinsics"})
        self.assertLess(
            summary["final_reprojection_rms"], summary["baseline_reprojection_rms"]
        )
        self.assertLessEqual(
            mean_ray_convergence(observed_pixels, final_cals, self.control),
            summary["baseline_mean_ray_convergence"],
        )
        self.assertEqual(final_points.shape, points.shape)

    def test_guarded_two_step_bundle_adjustment_passes_known_point_constraints(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1
        known_points = {0: points[0], 1: points[1]}

        with patch(
            "openptv_python.orientation.multi_camera_bundle_adjustment",
            side_effect=[
                (self.true_cals, points.copy(), {"success": True}),
                (self.true_cals, points.copy(), {"success": True}),
            ],
        ) as mocked_adjustment:
            guarded_two_step_bundle_adjustment(
                observed_pixels,
                self.lightly_perturb_calibrations(self.true_cals),
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                fixed_camera_indices=[0, 1],
                known_points=known_points,
                known_point_sigmas=1e-3,
            )

        self.assertEqual(mocked_adjustment.call_count, 2)
        for call in mocked_adjustment.call_args_list:
            self.assertEqual(call.kwargs["known_points"], known_points)
            self.assertEqual(call.kwargs["known_point_sigmas"], 1e-3)

    def test_guarded_two_step_bundle_adjustment_supports_staged_camera_release(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        start_cals = self.lightly_perturb_calibrations(self.true_cals)
        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1

        staged_returns = [
            (self.true_cals, points.copy(), {"success": True, "stage": f"pose_{idx}"})
            for idx in range(4)
        ] + [(self.true_cals, points.copy(), {"success": True, "stage": "intrinsics"})]

        with (
            patch(
                "openptv_python.orientation.multi_camera_bundle_adjustment",
                side_effect=staged_returns,
            ) as mocked_adjustment,
            patch(
                "openptv_python.orientation.reprojection_rms",
                side_effect=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            ),
            patch(
                "openptv_python.orientation.mean_ray_convergence",
                side_effect=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ),
        ):
            _, final_points, summary = guarded_two_step_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                pose_release_camera_order=[0, 1, 2, 3],
                pose_prior_sigmas={
                    "x0": 0.5,
                    "y0": 0.5,
                    "z0": 0.5,
                    "omega": 0.005,
                    "phi": 0.005,
                    "kappa": 0.005,
                },
                pose_parameter_bounds={
                    "x0": (-2.0, 2.0),
                    "y0": (-2.0, 2.0),
                    "z0": (-2.0, 2.0),
                    "omega": (-0.02, 0.02),
                    "phi": (-0.02, 0.02),
                    "kappa": (-0.02, 0.02),
                },
                pose_max_nfev=20,
                intrinsic_max_nfev=10,
            )

        self.assertEqual(mocked_adjustment.call_count, 5)
        fixed_sequences = [
            call.kwargs.get("fixed_camera_indices")
            for call in mocked_adjustment.call_args_list
        ]
        self.assertEqual(
            fixed_sequences,
            [[1, 2, 3], [2, 3], [3], [], [0, 1, 2, 3]],
        )
        self.assertEqual(summary["pose_release_camera_order"], [0, 1, 2, 3])
        self.assertEqual(summary["accepted_pose_stage_count"], 4)
        self.assertEqual(summary["accepted_stage"], "intrinsics")
        self.assertEqual(len(summary["pose_stage_summaries"]), 4)
        self.assertEqual(
            [
                stage["released_camera_index"]
                for stage in summary["pose_stage_summaries"]
            ],
            [0, 1, 2, 3],
        )
        self.assertEqual(final_points.shape, points.shape)

    def test_guarded_two_step_bundle_adjustment_stagewise_ray_slack_allows_near_miss(
        self,
    ):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.empty((len(points), 4, 2), dtype=float)
        for cam_num, cal in enumerate(self.true_cals):
            observed_pixels[:, cam_num, :] = arr_metric_to_pixel(
                image_coordinates(points, cal, self.control.mm), self.control
            )

        start_cals = self.lightly_perturb_calibrations(self.true_cals)
        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1

        staged_returns = [
            (self.true_cals, points.copy(), {"success": True, "stage": f"pose_{idx}"})
            for idx in range(4)
        ] + [(self.true_cals, points.copy(), {"success": True, "stage": "intrinsics"})]

        with (
            patch(
                "openptv_python.orientation.multi_camera_bundle_adjustment",
                side_effect=staged_returns,
            ),
            patch(
                "openptv_python.orientation.reprojection_rms",
                side_effect=[10.0, 9.0, 8.0, 7.0, 6.0, 5.5],
            ),
            patch(
                "openptv_python.orientation.mean_ray_convergence",
                side_effect=[6.0, 5.0, 4.0, 4.0005, 3.5, 3.4],
            ),
        ):
            _, _, summary = guarded_two_step_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                pose_release_camera_order=[0, 1, 2, 3],
                pose_stage_ray_slack=1e-3,
                pose_prior_sigmas={
                    "x0": 0.5,
                    "y0": 0.5,
                    "z0": 0.5,
                    "omega": 0.005,
                    "phi": 0.005,
                    "kappa": 0.005,
                },
                pose_parameter_bounds={
                    "x0": (-2.0, 2.0),
                    "y0": (-2.0, 2.0),
                    "z0": (-2.0, 2.0),
                    "omega": (-0.02, 0.02),
                    "phi": (-0.02, 0.02),
                    "kappa": (-0.02, 0.02),
                },
            )

        self.assertEqual(summary["accepted_pose_stage_count"], 4)
        self.assertEqual(summary["pose_stage_ray_slack"], 1e-3)
        self.assertTrue(summary["pose_stage_summaries"][2]["accepted"])

    def test_guarded_two_step_bundle_adjustment_rejects_on_hard_geometry_guard(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.zeros((len(points), 4, 2), dtype=float)
        start_cals = self.lightly_perturb_calibrations(self.true_cals)
        pose_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        bad_intrinsic_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        bad_intrinsic_cals[2].set_pos(
            bad_intrinsic_cals[2].get_pos() + np.array([20.0, 0.0, 0.0])
        )
        bad_intrinsic_cals[2].set_angles(
            bad_intrinsic_cals[2].get_angles() + np.array([0.0, 0.05, 0.0])
        )

        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1

        with (
            patch(
                "openptv_python.orientation.multi_camera_bundle_adjustment",
                side_effect=[
                    (pose_cals, points.copy(), {"success": True}),
                    (bad_intrinsic_cals, points.copy(), {"success": True}),
                ],
            ),
            patch(
                "openptv_python.orientation.reprojection_rms",
                side_effect=[10.0, 5.0, 4.0],
            ),
            patch(
                "openptv_python.orientation.mean_ray_convergence",
                side_effect=[3.0, 2.0, 1.0],
            ),
            patch(
                "openptv_python.orientation.img_coord",
                side_effect=lambda point, cal, _mm: (
                    float(point[0] + cal.get_pos()[0]),
                    float(point[1] + cal.get_pos()[1]),
                ),
            ),
            patch(
                "openptv_python.orientation.metric_to_pixel",
                side_effect=lambda x, y, _cpar: np.array([x, y], dtype=float),
            ),
        ):
            final_cals, _, summary = guarded_two_step_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                fixed_camera_indices=[0, 1],
                geometry_reference_points=points,
                geometry_reference_cals=self.true_cals,
                geometry_guard_mode="hard",
                geometry_guard_threshold=1.0,
            )

        self.assertEqual(summary["accepted_stage"], "pose")
        self.assertTrue(summary["pose_geometry_ok"])
        self.assertFalse(summary["intrinsic_geometry_ok"])
        self.assertGreater(summary["intrinsic_geometry_max"], 1.0)
        np.testing.assert_allclose(final_cals[2].get_pos(), pose_cals[2].get_pos())

    def test_guarded_two_step_bundle_adjustment_rejects_on_soft_geometry_guard(self):
        points = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 1.0],
                [-10.0, 10.0, -1.0],
                [10.0, 10.0, 0.5],
            ],
            dtype=float,
        )
        observed_pixels = np.zeros((len(points), 4, 2), dtype=float)
        start_cals = self.lightly_perturb_calibrations(self.true_cals)
        pose_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        bad_intrinsic_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        bad_intrinsic_cals[3].set_pos(
            bad_intrinsic_cals[3].get_pos() + np.array([2.0, 0.0, 0.0])
        )

        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1

        with (
            patch(
                "openptv_python.orientation.multi_camera_bundle_adjustment",
                side_effect=[
                    (pose_cals, points.copy(), {"success": True}),
                    (bad_intrinsic_cals, points.copy(), {"success": True}),
                ],
            ),
            patch(
                "openptv_python.orientation.reprojection_rms",
                side_effect=[10.0, 5.0, 4.0],
            ),
            patch(
                "openptv_python.orientation.mean_ray_convergence",
                side_effect=[3.0, 2.0, 1.0],
            ),
            patch(
                "openptv_python.orientation.img_coord",
                side_effect=lambda point, cal, _mm: (
                    float(point[0] + cal.get_pos()[0]),
                    float(point[1] + cal.get_pos()[1]),
                ),
            ),
            patch(
                "openptv_python.orientation.metric_to_pixel",
                side_effect=lambda x, y, _cpar: np.array([x, y], dtype=float),
            ),
        ):
            final_cals, _, summary = guarded_two_step_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                fixed_camera_indices=[0, 1],
                geometry_reference_points=points,
                geometry_reference_cals=self.true_cals,
                geometry_guard_mode="soft",
            )

        self.assertEqual(summary["accepted_stage"], "pose")
        self.assertTrue(summary["pose_geometry_ok"])
        self.assertFalse(summary["intrinsic_geometry_ok"])
        self.assertGreater(
            summary["intrinsic_geometry_max"],
            summary["pose_geometry_max"],
        )
        np.testing.assert_allclose(final_cals[3].get_pos(), pose_cals[3].get_pos())

    def test_guarded_two_step_bundle_adjustment_rejects_on_hard_correspondence_guard(
        self,
    ):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        observed_pixels = np.zeros((len(points), 4, 2), dtype=float)
        pose_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        for cal in pose_cals:
            cal.set_pos(np.zeros(3, dtype=float))
            cal.set_angles(np.zeros(3, dtype=float))
        start_cals = [self.clone_calibration(cal) for cal in pose_cals]
        bad_intrinsic_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        for cal in bad_intrinsic_cals:
            cal.set_pos(np.zeros(3, dtype=float))
            cal.set_angles(np.zeros(3, dtype=float))
        bad_intrinsic_cals[2].set_pos(
            bad_intrinsic_cals[2].get_pos() + np.array([25.0, 0.0, 0.0])
        )

        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1
        target_sets = [
            np.asarray([[point[0], point[1]] for point in points], dtype=float)
            for _ in range(4)
        ]

        with (
            patch(
                "openptv_python.orientation.multi_camera_bundle_adjustment",
                side_effect=[
                    (pose_cals, points.copy(), {"success": True}),
                    (bad_intrinsic_cals, points.copy(), {"success": True}),
                ],
            ),
            patch(
                "openptv_python.orientation.reprojection_rms",
                side_effect=[10.0, 5.0, 4.0],
            ),
            patch(
                "openptv_python.orientation.mean_ray_convergence",
                side_effect=[3.0, 2.0, 1.0],
            ),
            patch(
                "openptv_python.orientation.image_coordinates",
                side_effect=lambda pts, cal, _mm: pts[:, :2] + cal.get_pos()[:2],
            ),
            patch(
                "openptv_python.orientation.arr_metric_to_pixel",
                side_effect=lambda coords, _cpar: coords,
            ),
        ):
            final_cals, _, summary = guarded_two_step_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                fixed_camera_indices=[0, 1],
                correspondence_original_ids=np.tile(
                    np.arange(len(points))[:, None], (1, 4)
                ),
                correspondence_point_frame_indices=np.zeros(len(points), dtype=int),
                correspondence_frame_target_pixels=[target_sets],
                correspondence_guard_mode="hard",
                correspondence_guard_threshold=0.2,
            )

        self.assertEqual(summary["accepted_stage"], "pose")
        self.assertTrue(summary["pose_correspondence_ok"])
        self.assertFalse(summary["intrinsic_correspondence_ok"])
        self.assertGreater(summary["intrinsic_correspondence_rate"], 0.2)
        np.testing.assert_allclose(final_cals[2].get_pos(), pose_cals[2].get_pos())

    def test_guarded_two_step_bundle_adjustment_rejects_on_soft_correspondence_guard(
        self,
    ):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        observed_pixels = np.zeros((len(points), 4, 2), dtype=float)
        pose_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        for cal in pose_cals:
            cal.set_pos(np.zeros(3, dtype=float))
            cal.set_angles(np.zeros(3, dtype=float))
        start_cals = [self.clone_calibration(cal) for cal in pose_cals]
        bad_intrinsic_cals = [self.clone_calibration(cal) for cal in self.true_cals]
        for cal in bad_intrinsic_cals:
            cal.set_pos(np.zeros(3, dtype=float))
            cal.set_angles(np.zeros(3, dtype=float))
        bad_intrinsic_cals[3].set_pos(
            bad_intrinsic_cals[3].get_pos() + np.array([25.0, 0.0, 0.0])
        )

        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1
        target_sets = [
            np.asarray([[point[0], point[1]] for point in points], dtype=float)
            for _ in range(4)
        ]

        with (
            patch(
                "openptv_python.orientation.multi_camera_bundle_adjustment",
                side_effect=[
                    (pose_cals, points.copy(), {"success": True}),
                    (bad_intrinsic_cals, points.copy(), {"success": True}),
                ],
            ),
            patch(
                "openptv_python.orientation.reprojection_rms",
                side_effect=[10.0, 5.0, 4.0],
            ),
            patch(
                "openptv_python.orientation.mean_ray_convergence",
                side_effect=[3.0, 2.0, 1.0],
            ),
            patch(
                "openptv_python.orientation.image_coordinates",
                side_effect=lambda pts, cal, _mm: pts[:, :2] + cal.get_pos()[:2],
            ),
            patch(
                "openptv_python.orientation.arr_metric_to_pixel",
                side_effect=lambda coords, _cpar: coords,
            ),
        ):
            final_cals, _, summary = guarded_two_step_bundle_adjustment(
                observed_pixels,
                start_cals,
                self.control,
                OrientPar(),
                intrinsics,
                point_init=points,
                fixed_camera_indices=[0, 1],
                correspondence_original_ids=np.tile(
                    np.arange(len(points))[:, None], (1, 4)
                ),
                correspondence_point_frame_indices=np.zeros(len(points), dtype=int),
                correspondence_frame_target_pixels=[target_sets],
                correspondence_guard_mode="soft",
                correspondence_guard_reference_rate=0.0,
            )

        self.assertEqual(summary["accepted_stage"], "pose")
        self.assertTrue(summary["pose_correspondence_ok"])
        self.assertFalse(summary["intrinsic_correspondence_ok"])
        self.assertGreater(summary["intrinsic_correspondence_rate"], 0.0)
        np.testing.assert_allclose(final_cals[3].get_pos(), pose_cals[3].get_pos())

    def test_cavity_intrinsics_only_improves_from_intrinsic_perturbation(self):
        cavity_dir = Path("tests/testing_fodder/test_cavity")
        control = ControlPar(4).from_file(cavity_dir / "parameters/ptv.par")
        true_cals = [
            read_calibration(
                cavity_dir / f"cal/cam{cam_num}.tif.ori",
                cavity_dir / f"cal/cam{cam_num}.tif.addpar",
            )
            for cam_num in range(1, 5)
        ]

        cor_buf, path_buf = read_path_frame(
            str(cavity_dir / "res_orig/rt_is"),
            "",
            "",
            10001,
        )
        targets = [
            read_targets(str(cavity_dir / f"img_orig/cam{cam_num}.%05d"), 10001)
            for cam_num in range(1, 5)
        ]
        subset = [
            pt_num for pt_num, corres in enumerate(cor_buf) if np.all(corres.p >= 0)
        ][:24]
        observed_pixels = np.full((len(subset), 4, 2), np.nan, dtype=float)
        point_init = np.empty((len(subset), 3), dtype=float)
        for out_num, pt_num in enumerate(subset):
            point_init[out_num] = path_buf[pt_num].x
            for cam in range(4):
                target_index = cor_buf[pt_num].p[cam]
                observed_pixels[out_num, cam, 0] = targets[cam][target_index].x
                observed_pixels[out_num, cam, 1] = targets[cam][target_index].y

        start_cals = [self.clone_calibration(cal) for cal in true_cals]
        for cal in start_cals:
            cal.added_par[0] += 2e-5
            cal.added_par[3] += 8e-5
            cal.added_par[4] -= 6e-5

        before_rms = reprojection_rms(observed_pixels, point_init, start_cals, control)
        intrinsics = OrientPar()
        intrinsics.k1flag = 1
        intrinsics.p1flag = 1
        intrinsics.p2flag = 1

        refined_cals, refined_points, result = multi_camera_bundle_adjustment(
            observed_pixels,
            start_cals,
            control,
            intrinsics,
            point_init=point_init.copy(),
            fixed_camera_indices=[0, 1, 2, 3],
            optimize_extrinsics=False,
            optimize_points=False,
            loss="linear",
            method="trf",
            prior_sigmas={
                "k1": 5e-5,
                "p1": 1e-4,
                "p2": 1e-4,
            },
            parameter_bounds={
                "k1": (-5e-5, 5e-5),
                "p1": (-2e-4, 2e-4),
                "p2": (-2e-4, 2e-4),
            },
            max_nfev=40,
        )
        after_rms = reprojection_rms(
            observed_pixels,
            refined_points,
            refined_cals,
            control,
        )

        self.assertTrue(result.success, msg=result.message)
        self.assertLess(after_rms, before_rms - 0.1)
        for refined, start in zip(refined_cals, start_cals):
            np.testing.assert_allclose(refined.get_pos(), start.get_pos(), atol=1e-12)
            np.testing.assert_allclose(
                refined.get_angles(), start.get_angles(), atol=1e-12
            )
        np.testing.assert_allclose(refined_points, point_init)


if __name__ == "__main__":
    unittest.main()
