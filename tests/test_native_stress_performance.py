"""Stress benchmarks comparing native and non-native backends."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np

import openptv_python.image_processing as image_processing
import openptv_python.orientation as orientation
import openptv_python.segmentation as segmentation
from openptv_python._native_compat import (
    HAS_OPTV,
    HAS_NATIVE_PREPROCESS,
    HAS_NATIVE_SEGMENTATION,
)
from openptv_python._native_convert import to_native_calibration, to_native_control_par
from openptv_python.calibration import Calibration
from openptv_python.imgcoord import image_coordinates
from openptv_python.parameters import ControlPar, TargetPar, VolumePar

try:
    from optv.orientation import point_positions as native_point_positions
    from optv.parameters import VolumeParams as NativeVolumeParams

    HAS_NATIVE_RECONSTRUCTION = True
except ImportError:
    native_point_positions = None
    NativeVolumeParams = None
    HAS_NATIVE_RECONSTRUCTION = False


def _env_flag_enabled(name: str) -> bool:
    """Interpret common truthy environment variable values."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


RUN_STRESS_BENCHMARKS = _env_flag_enabled("OPENPTV_RUN_STRESS_BENCHMARKS")


def _benchmark(function, *, warmups: int = 1, runs: int = 3):
    """Run a callable repeatedly and return the final result and timings."""
    result = None
    for _ in range(warmups):
        result = function()

    timings = []
    for _ in range(runs):
        start = perf_counter()
        result = function()
        timings.append(perf_counter() - start)

    return result, timings


def _timing_summary(label: str, timings: list[float]) -> str:
    """Return a compact timing summary for benchmark output."""
    return (
        f"{label}: median={median(timings):.6f}s "
        f"min={min(timings):.6f}s max={max(timings):.6f}s"
    )


def _serialize_targets(targets) -> list[tuple[int, float, float, int, int, int, int, int]]:
    """Convert target objects into stable tuples for comparisons."""
    return [
        (
            int(target.pnr),
            round(float(target.x), 9),
            round(float(target.y), 9),
            int(target.n),
            int(target.nx),
            int(target.ny),
            int(target.sumg),
            int(target.tnr),
        )
        for target in targets
    ]


def _build_segmentation_stress_image(
    width: int = 512,
    height: int = 512,
    spacing: int = 24,
) -> np.ndarray:
    """Create a deterministic image with many isolated synthetic targets."""
    image = np.zeros((height, width), dtype=np.uint8)
    patch = np.array(
        [
            [0, 220, 230, 220, 0],
            [220, 240, 248, 240, 220],
            [230, 248, 255, 248, 230],
            [220, 240, 248, 240, 220],
            [0, 220, 230, 220, 0],
        ],
        dtype=np.uint8,
    )

    for center_y in range(12, height - 12, spacing):
        for center_x in range(12, width - 12, spacing):
            image[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3] = patch

    return image


def _build_reconstruction_stress_case(
    num_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray, ControlPar, list[Calibration]]:
    """Build a deterministic multi-camera reconstruction workload."""
    rng = np.random.default_rng(20260307)
    points = np.empty((num_points, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(-25.0, 25.0, size=num_points)
    points[:, 1] = rng.uniform(20.0, 65.0, size=num_points)
    points[:, 2] = rng.uniform(-15.0, 15.0, size=num_points)

    cpar = ControlPar(4).from_file(
        Path("tests/testing_folder/control_parameters/control.par")
    )
    cpar.mm.set_n1(1.0)
    cpar.mm.set_layers([1.0], [1.0])
    cpar.mm.set_n3(1.0)

    add_file = Path("tests/testing_folder/calibration/cam1.tif.addpar")
    calibs = [
        Calibration().from_file(
            ori_file=Path(
                f"tests/testing_folder/calibration/sym_cam{cam_num}.tif.ori"
            ),
            add_file=add_file,
        )
        for cam_num in range(1, 5)
    ]

    projections = [image_coordinates(points, cal, cpar.mm) for cal in calibs]
    targets = np.asarray(projections, dtype=np.float64).transpose(1, 0, 2)
    return points, targets, cpar, calibs


@unittest.skipUnless(
    RUN_STRESS_BENCHMARKS,
    "set OPENPTV_RUN_STRESS_BENCHMARKS=1 to run stress benchmarks",
)
class TestNativeStressPerformance(unittest.TestCase):
    """Stress tests comparing native and non-native runtime paths."""

    @unittest.skipUnless(
        HAS_NATIVE_PREPROCESS,
        "optv native preprocess_image is not available",
    )
    def test_preprocess_image_stress_timing(self):
        """Compare native preprocessing timing against the Python path."""
        rng = np.random.default_rng(20260307)
        image = rng.integers(0, 256, size=(1024, 1024), dtype=np.uint8)

        cpar = ControlPar(1)
        cpar.set_image_size((image.shape[1], image.shape[0]))

        def python_path():
            return image_processing.prepare_image(
                image,
                dim_lp=1,
                filter_hp=0,
                filter_file="",
            )

        def native_path():
            return image_processing.preprocess_image(
                image,
                filter_hp=0,
                cpar=cpar,
                dim_lp=1,
            )

        python_result, python_timings = _benchmark(python_path)
        with patch.object(
            image_processing,
            "native_preprocess_image",
            wraps=image_processing.native_preprocess_image,
        ) as native_call:
            native_result, native_timings = _benchmark(native_path)

        np.testing.assert_array_equal(native_result, python_result)
        self.assertGreaterEqual(native_call.call_count, 3)

        speedup = median(python_timings) / median(native_timings)
        print(
            "preprocess stress benchmark: "
            f"{_timing_summary('python', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )

    @unittest.skipUnless(
        HAS_NATIVE_SEGMENTATION,
        "optv native target recognition is not available",
    )
    def test_target_recognition_stress_timing(self):
        """Compare native target recognition timing against Python+Numba."""
        image = _build_segmentation_stress_image()

        cpar = ControlPar(1)
        cpar.set_image_size((image.shape[1], image.shape[0]))
        tpar = TargetPar(
            gvthresh=[200],
            discont=20,
            nnmin=1,
            nnmax=100,
            sumg_min=1,
            nxmin=1,
            nxmax=10,
            nymin=1,
            nymax=10,
        )

        def python_numba_path():
            with patch.object(segmentation, "HAS_NATIVE_SEGMENTATION", False):
                return segmentation.target_recognition(image, tpar, 0, cpar)

        def native_path():
            return segmentation.target_recognition(image, tpar, 0, cpar)

        python_targets, python_timings = _benchmark(python_numba_path)
        with patch.object(
            segmentation,
            "native_target_recognition",
            wraps=segmentation.native_target_recognition,
        ) as native_call:
            native_targets, native_timings = _benchmark(native_path)

        self.assertEqual(
            _serialize_targets(native_targets),
            _serialize_targets(python_targets),
        )
        self.assertGreaterEqual(native_call.call_count, 3)

        speedup = median(python_timings) / median(native_timings)
        print(
            "segmentation stress benchmark: "
            f"{_timing_summary('python+numba', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )

    @unittest.skipUnless(
        HAS_OPTV and HAS_NATIVE_RECONSTRUCTION,
        "optv native point_positions is not available",
    )
    def test_point_reconstruction_stress_timing(self):
        """Compare native multi-camera reconstruction timing against Python."""
        expected_points, targets, cpar, calibs = _build_reconstruction_stress_case()
        native_cpar = to_native_control_par(cpar)
        native_cals = [to_native_calibration(cal) for cal in calibs]
        native_vpar = NativeVolumeParams()
        python_vpar = VolumePar()

        def python_path():
            return orientation.point_positions(targets, cpar.mm, calibs, python_vpar)

        def native_path():
            assert native_point_positions is not None
            return native_point_positions(targets, native_cpar, native_cals, native_vpar)

        python_result, python_timings = _benchmark(python_path)
        native_result, native_timings = _benchmark(native_path)
        python_points, python_rcm = python_result
        native_points, native_rcm = native_result

        np.testing.assert_allclose(python_points, expected_points, atol=1e-6)
        np.testing.assert_allclose(native_points, expected_points, atol=1e-6)
        np.testing.assert_allclose(native_points, python_points, atol=1e-9)
        np.testing.assert_allclose(native_rcm, python_rcm, atol=1e-9)

        speedup = median(python_timings) / median(native_timings)
        print(
            "reconstruction stress benchmark: "
            f"{_timing_summary('python', python_timings)}; "
            f"{_timing_summary('native', native_timings)}; "
            f"speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    unittest.main()