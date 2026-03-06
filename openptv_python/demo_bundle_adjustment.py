"""Command-line demo for bundle-adjustment experiments on OpenPTV cases."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, cast

import numpy as np

from .calibration import Calibration, read_calibration, write_calibration
from .calibration_compare import compare_calibration_folders, format_calibration_comparison
from .orientation import (
    guarded_two_step_bundle_adjustment,
    mean_ray_convergence,
    multi_camera_bundle_adjustment,
    reprojection_rms,
)
from .parameters import ControlPar, OrientPar, SequencePar
from .tracking_frame_buf import read_path_frame, read_targets


@dataclass(frozen=True)
class ExperimentSpec:
    """Configuration for one bundle-adjustment demo run."""

    name: str
    description: str
    mode: str
    ba_kwargs: Dict[str, object]


@dataclass
class ExperimentResult:
    """Collected metrics for one bundle-adjustment demo run."""

    name: str
    description: str
    duration_sec: float
    success: bool
    initial_rms: float
    final_rms: float
    baseline_ray_convergence: float
    final_ray_convergence: float
    notes: str
    cal_dir: Path | None


def clone_calibration(cal: Calibration) -> Calibration:
    """Return a detached copy of a calibration object."""
    return Calibration(
        ext_par=cal.ext_par.copy(),
        int_par=cal.int_par.copy(),
        glass_par=cal.glass_par.copy(),
        added_par=cal.added_par.copy(),
        mmlut=cal.mmlut,
        mmlut_data=cal.mmlut_data,
    )


def perturb_calibrations(cals: List[Calibration], scale: float) -> List[Calibration]:
    """Apply deterministic pose perturbations so BA has something to recover."""
    perturbed = [clone_calibration(cal) for cal in cals]
    deltas = [
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.5, -0.3, 0.2]), np.array([0.004, -0.003, 0.002])),
        (np.array([-0.4, 0.3, -0.2]), np.array([-0.003, 0.003, -0.002])),
        (np.array([0.3, 0.4, -0.2]), np.array([0.003, 0.002, -0.002])),
    ]
    for cal, (pos_delta, angle_delta) in zip(perturbed, deltas):
        cal.set_pos(cal.get_pos() + pos_delta * scale)
        cal.set_angles(cal.get_angles() + angle_delta * scale)
    return perturbed


def discover_num_cams(cal_dir: Path) -> int:
    """Infer the number of cameras from calibration .ori files."""
    return len(sorted(cal_dir.glob("*.ori")))


def load_calibrations(case_dir: Path, num_cams: int) -> List[Calibration]:
    """Load all camera calibrations from a case folder."""
    cal_dir = case_dir / "cal"
    return [
        read_calibration(
            cal_dir / f"cam{cam_num}.tif.ori",
            cal_dir / f"cam{cam_num}.tif.addpar",
        )
        for cam_num in range(1, num_cams + 1)
    ]


def load_case_observations(
    case_dir: Path,
    num_cams: int,
    *,
    max_frames: int | None,
    max_points_per_frame: int | None,
) -> tuple[ControlPar, np.ndarray, np.ndarray]:
    """Load quadruplet observations and initial 3D points from a case folder."""
    control = ControlPar(num_cams).from_file(case_dir / "parameters/ptv.par")
    seq = SequencePar.from_file(case_dir / "parameters/sequence.par", num_cams)

    observed_batches = []
    point_batches = []
    frames = list(range(seq.first, seq.last + 1))
    if max_frames is not None:
        frames = frames[:max_frames]

    for frame in frames:
        cor_buf, path_buf = read_path_frame(
            str(case_dir / "res_orig/rt_is"),
            "",
            "",
            frame,
        )
        targets = [
            read_targets(str(case_dir / f"img_orig/cam{cam_num}.%05d"), frame)
            for cam_num in range(1, num_cams + 1)
        ]
        subset = [
            pt_num
            for pt_num, corres in enumerate(cor_buf)
            if np.all(corres.p[:num_cams] >= 0)
        ]
        if max_points_per_frame is not None:
            subset = subset[:max_points_per_frame]

        observed_pixels = np.full((len(subset), num_cams, 2), np.nan, dtype=float)
        point_init = np.empty((len(subset), 3), dtype=float)
        for out_num, pt_num in enumerate(subset):
            point_init[out_num] = path_buf[pt_num].x
            for cam in range(num_cams):
                target_index = cor_buf[pt_num].p[cam]
                observed_pixels[out_num, cam, 0] = targets[cam][target_index].x
                observed_pixels[out_num, cam, 1] = targets[cam][target_index].y

        observed_batches.append(observed_pixels)
        point_batches.append(point_init)

    if not observed_batches:
        raise ValueError(f"No observations loaded from {case_dir}")

    return (
        control,
        np.concatenate(observed_batches, axis=0),
        np.concatenate(point_batches, axis=0),
    )


def default_experiments() -> List[ExperimentSpec]:
    """Return a set of representative BA configurations."""
    pose_priors = {
        "x0": 0.5,
        "y0": 0.5,
        "z0": 0.5,
        "omega": 0.005,
        "phi": 0.005,
        "kappa": 0.005,
    }
    pose_bounds = {
        "x0": (-2.0, 2.0),
        "y0": (-2.0, 2.0),
        "z0": (-2.0, 2.0),
        "omega": (-0.02, 0.02),
        "phi": (-0.02, 0.02),
        "kappa": (-0.02, 0.02),
    }
    tight_intrinsic_priors = {
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
    }
    tight_intrinsic_bounds = {
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
    }

    intrinsics = OrientPar()
    intrinsics.k1flag = 1
    intrinsics.p1flag = 1
    intrinsics.p2flag = 1

    return [
        ExperimentSpec(
            name="pose_trf_linear",
            description="Pose-only BA with linear loss and TRF",
            mode="multi",
            ba_kwargs={
                "orient_par": OrientPar(),
                "fixed_camera_indices": [0, 1],
                "loss": "linear",
                "method": "trf",
                "prior_sigmas": pose_priors,
                "parameter_bounds": pose_bounds,
                "max_nfev": 12,
            },
        ),
        ExperimentSpec(
            name="pose_soft_l1",
            description="Pose-only BA with robust soft_l1 loss",
            mode="multi",
            ba_kwargs={
                "orient_par": OrientPar(),
                "fixed_camera_indices": [0, 1],
                "loss": "soft_l1",
                "f_scale": 1.0,
                "method": "trf",
                "prior_sigmas": pose_priors,
                "parameter_bounds": pose_bounds,
                "max_nfev": 12,
            },
        ),
        ExperimentSpec(
            name="pose_fixed_points",
            description="Pose-only BA with fixed 3D points",
            mode="multi",
            ba_kwargs={
                "orient_par": OrientPar(),
                "fixed_camera_indices": [0, 1],
                "loss": "linear",
                "method": "trf",
                "prior_sigmas": pose_priors,
                "parameter_bounds": pose_bounds,
                "max_nfev": 12,
                "optimize_points": False,
            },
        ),
        ExperimentSpec(
            name="guarded_two_step",
            description="Pose stage followed by tightly constrained intrinsic stage",
            mode="guarded",
            ba_kwargs={
                "pose_orient_par": OrientPar(),
                "intrinsic_orient_par": intrinsics,
                "fixed_camera_indices": [0, 1],
                "pose_prior_sigmas": pose_priors,
                "pose_parameter_bounds": pose_bounds,
                "pose_max_nfev": 8,
                "intrinsic_prior_sigmas": tight_intrinsic_priors,
                "intrinsic_parameter_bounds": tight_intrinsic_bounds,
                "intrinsic_max_nfev": 4,
            },
        ),
    ]


def ensure_output_case_layout(source_case_dir: Path, output_case_dir: Path) -> Path:
    """Copy the source case and return the writable calibration directory."""
    shutil.copytree(source_case_dir, output_case_dir, dirs_exist_ok=True)
    cal_dir = output_case_dir / "cal"
    cal_dir.mkdir(parents=True, exist_ok=True)
    return cal_dir


def write_calibration_folder(cals: List[Calibration], cal_dir: Path) -> None:
    """Write one calibration folder using OpenPTV naming conventions."""
    cal_dir.mkdir(parents=True, exist_ok=True)
    for cam_num, cal in enumerate(cals, start=1):
        stem = f"cam{cam_num}.tif"
        write_calibration(cal, cal_dir / f"{stem}.ori", cal_dir / f"{stem}.addpar")


def run_experiment(
    spec: ExperimentSpec,
    *,
    observed_pixels: np.ndarray,
    point_init: np.ndarray,
    control: ControlPar,
    start_cals: List[Calibration],
    source_case_dir: Path,
    output_dir: Path | None,
) -> ExperimentResult:
    """Execute one experiment and collect metrics and optional outputs."""
    working_cals = [clone_calibration(cal) for cal in start_cals]
    baseline_rms = reprojection_rms(observed_pixels, point_init, working_cals, control)
    baseline_ray = mean_ray_convergence(observed_pixels, working_cals, control)

    start = perf_counter()
    notes = ""
    if spec.mode == "multi":
        orient_par = cast(OrientPar, spec.ba_kwargs["orient_par"])
        refined_cals, refined_points, result = multi_camera_bundle_adjustment(
            observed_pixels,
            working_cals,
            control,
            orient_par,
            point_init=point_init,
            fix_first_camera=cast(bool, spec.ba_kwargs.get("fix_first_camera", True)),
            fixed_camera_indices=cast(
                List[int] | None,
                spec.ba_kwargs.get("fixed_camera_indices"),
            ),
            loss=cast(str, spec.ba_kwargs.get("loss", "soft_l1")),
            f_scale=cast(float, spec.ba_kwargs.get("f_scale", 1.0)),
            method=cast(str, spec.ba_kwargs.get("method", "trf")),
            prior_sigmas=cast(
                Dict[str, float] | None,
                spec.ba_kwargs.get("prior_sigmas"),
            ),
            parameter_bounds=cast(
                Dict[str, tuple[float, float]] | None,
                spec.ba_kwargs.get("parameter_bounds"),
            ),
            max_nfev=cast(int | None, spec.ba_kwargs.get("max_nfev")),
            optimize_extrinsics=cast(
                bool,
                spec.ba_kwargs.get("optimize_extrinsics", True),
            ),
            optimize_points=cast(bool, spec.ba_kwargs.get("optimize_points", True)),
            ftol=cast(float | None, spec.ba_kwargs.get("ftol")),
            xtol=cast(float | None, spec.ba_kwargs.get("xtol")),
            gtol=cast(float | None, spec.ba_kwargs.get("gtol")),
        )
        success = bool(result.success)
        final_rms = float(result["final_reprojection_rms"])
        final_ray = mean_ray_convergence(observed_pixels, refined_cals, control)
        notes = str(result.message)
    elif spec.mode == "guarded":
        pose_orient_par = cast(OrientPar, spec.ba_kwargs["pose_orient_par"])
        intrinsic_orient_par = cast(OrientPar, spec.ba_kwargs["intrinsic_orient_par"])
        refined_cals, refined_points, summary = guarded_two_step_bundle_adjustment(
            observed_pixels,
            working_cals,
            control,
            pose_orient_par,
            intrinsic_orient_par,
            point_init=point_init,
            fixed_camera_indices=cast(
                List[int] | None,
                spec.ba_kwargs.get("fixed_camera_indices"),
            ),
            pose_prior_sigmas=cast(
                Dict[str, float] | None,
                spec.ba_kwargs.get("pose_prior_sigmas"),
            ),
            pose_parameter_bounds=cast(
                Dict[str, tuple[float, float]] | None,
                spec.ba_kwargs.get("pose_parameter_bounds"),
            ),
            pose_loss=cast(str, spec.ba_kwargs.get("pose_loss", "linear")),
            pose_method=cast(str, spec.ba_kwargs.get("pose_method", "trf")),
            pose_max_nfev=cast(int | None, spec.ba_kwargs.get("pose_max_nfev")),
            intrinsic_prior_sigmas=cast(
                Dict[str, float] | None,
                spec.ba_kwargs.get("intrinsic_prior_sigmas"),
            ),
            intrinsic_parameter_bounds=cast(
                Dict[str, tuple[float, float]] | None,
                spec.ba_kwargs.get("intrinsic_parameter_bounds"),
            ),
            intrinsic_loss=cast(
                str,
                spec.ba_kwargs.get("intrinsic_loss", "linear"),
            ),
            intrinsic_method=cast(
                str,
                spec.ba_kwargs.get("intrinsic_method", "trf"),
            ),
            intrinsic_max_nfev=cast(
                int | None,
                spec.ba_kwargs.get("intrinsic_max_nfev"),
            ),
            intrinsic_ftol=cast(float | None, spec.ba_kwargs.get("intrinsic_ftol", 1e-12)),
            intrinsic_xtol=cast(float | None, spec.ba_kwargs.get("intrinsic_xtol", 1e-12)),
            intrinsic_gtol=cast(float | None, spec.ba_kwargs.get("intrinsic_gtol", 1e-12)),
            pose_optimize_points=cast(
                bool,
                spec.ba_kwargs.get("pose_optimize_points", True),
            ),
            intrinsic_optimize_points=cast(
                bool,
                spec.ba_kwargs.get("intrinsic_optimize_points", True),
            ),
            reject_worse_solution=cast(
                bool,
                spec.ba_kwargs.get("reject_worse_solution", True),
            ),
            reject_on_ray_convergence=cast(
                bool,
                spec.ba_kwargs.get("reject_on_ray_convergence", True),
            ),
        )
        success = True
        final_rms = cast(float, summary["final_reprojection_rms"])
        final_ray = cast(float, summary["final_mean_ray_convergence"])
        notes = f"accepted_stage={summary['accepted_stage']}"
    else:
        raise ValueError(f"Unknown experiment mode: {spec.mode}")

    duration_sec = perf_counter() - start

    cal_dir = None
    if output_dir is not None:
        case_out_dir = output_dir / spec.name
        cal_dir = ensure_output_case_layout(source_case_dir, case_out_dir)
        write_calibration_folder(refined_cals, cal_dir)
        comparison = compare_calibration_folders(source_case_dir / "cal", cal_dir)
        (case_out_dir / "calibration_delta.txt").write_text(
            format_calibration_comparison(
                comparison,
                reference_dir=source_case_dir / "cal",
                candidate_dir=cal_dir,
            )
            + "\n",
            encoding="utf-8",
        )

    return ExperimentResult(
        name=spec.name,
        description=spec.description,
        duration_sec=duration_sec,
        success=success,
        initial_rms=baseline_rms,
        final_rms=final_rms,
        baseline_ray_convergence=baseline_ray,
        final_ray_convergence=final_ray,
        notes=notes,
        cal_dir=cal_dir,
    )


def format_results(results: Iterable[ExperimentResult]) -> str:
    """Render a compact plain-text summary table."""
    rows = list(results)
    headers = (
        "name",
        "success",
        "seconds",
        "rms_before",
        "rms_after",
        "ray_before",
        "ray_after",
        "notes",
    )
    data = [
        [
            row.name,
            "yes" if row.success else "no",
            f"{row.duration_sec:.2f}",
            f"{row.initial_rms:.6f}",
            f"{row.final_rms:.6f}",
            f"{row.baseline_ray_convergence:.6f}",
            f"{row.final_ray_convergence:.6f}",
            row.notes,
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in data:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: List[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(list(headers)), separator]
    lines.extend(render_row(row) for row in data)
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the BA demo."""
    parser = argparse.ArgumentParser(
        description="Demonstrate bundle-adjustment options on test_cavity or another compatible case.",
    )
    parser.add_argument(
        "case_dir",
        type=Path,
        nargs="?",
        default=Path("tests/testing_fodder/test_cavity"),
        help="Case folder containing cal/, parameters/, res_orig/, and img_orig/.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only use the first N frames from sequence.par.",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=80,
        help="Only use the first N fully observed points from each frame.",
    )
    parser.add_argument(
        "--perturbation-scale",
        type=float,
        default=1.0,
        help="Scale factor for the deterministic starting-calibration perturbation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/bundle_adjustment_demo"),
        help="Where to write one output case folder per experiment.",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Run comparisons but do not write updated calibration folders.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the bundle-adjustment demo."""
    args = parse_args(argv)
    case_dir = args.case_dir.resolve()
    num_cams = discover_num_cams(case_dir / "cal")

    control, observed_pixels, point_init = load_case_observations(
        case_dir,
        num_cams,
        max_frames=args.max_frames,
        max_points_per_frame=args.max_points_per_frame,
    )
    true_cals = load_calibrations(case_dir, num_cams)
    start_cals = perturb_calibrations(true_cals, args.perturbation_scale)

    output_dir = None if args.skip_write else args.output_dir.resolve()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Case: {case_dir}")
    print(f"Cameras: {num_cams}")
    print(f"Observations: {observed_pixels.shape[0]} points across {observed_pixels.shape[1]} cameras")
    print(f"Output folders: {output_dir if output_dir is not None else 'disabled'}")
    print()

    results = []
    for spec in default_experiments():
        print(f"Running {spec.name}: {spec.description}")
        result = run_experiment(
            spec,
            observed_pixels=observed_pixels,
            point_init=point_init,
            control=control,
            start_cals=start_cals,
            source_case_dir=case_dir,
            output_dir=output_dir,
        )
        results.append(result)
        if result.cal_dir is not None:
            print(f"  wrote calibration folder: {result.cal_dir}")
        print(f"  final RMS: {result.final_rms:.6f} px in {result.duration_sec:.2f} s")
        print()

    print(format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())