"""Command-line demo for bundle-adjustment experiments on OpenPTV cases."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, cast

import numpy as np

from .calibration import Calibration, read_calibration, write_calibration
from .calibration_compare import (
    compare_calibration_folders,
    format_calibration_comparison,
)
from .epi import epipolar_curve
from .imgcoord import img_coord
from .orientation import (
    guarded_two_step_bundle_adjustment,
    initialize_bundle_adjustment_points,
    mean_ray_convergence,
    metric_observations_from_pixels,
    multi_camera_bundle_adjustment,
    reprojection_rms,
)
from .parameters import (
    ControlPar,
    OrientPar,
    SequencePar,
    VolumePar,
    read_cal_ori_parameters,
    read_volume_par,
)
from .sortgrid import read_calblock as read_sortgrid_calblock
from .tracking_frame_buf import read_path_frame, read_targets
from .trafo import metric_to_pixel


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
    fixed_camera_indices: tuple[int, ...]
    camera_position_shifts: List[float]
    camera_angle_shifts: List[float]
    geometry_projection_drift: List[ProjectionDriftSummary] | None = None
    refined_cals: List[Calibration] | None = None
    refined_points: np.ndarray | None = None


@dataclass
class FixedCameraDiagnostic:
    """Summary of one anchor-pair diagnostic run."""

    fixed_camera_indices: tuple[int, int]
    final_rms: float
    final_ray_convergence: float
    fixed_position_shift: float
    fixed_angle_shift: float
    mean_free_position_shift: float
    max_free_position_shift: float
    mean_free_angle_shift: float
    notes: str


@dataclass
class CameraPairEpipolarSummary:
    """Summary of pairwise epipolar consistency for one camera pair."""

    camera_pair: tuple[int, int]
    mean_distance: float
    median_distance: float
    p95_distance: float
    max_distance: float
    acceptance_rates: Dict[float, float]


@dataclass
class QuadrupletSensitivitySummary:
    """Leave-one-camera-out stability summary for fully observed points."""

    num_points: int
    mean_spread: float
    median_spread: float
    p95_spread: float
    max_spread: float
    mean_full_ray_convergence: float
    mean_leave_one_out_ray_convergence: float
    worst_point_indices: List[int]


@dataclass(frozen=True)
class ProjectionDriftSummary:
    """Reference calibration-body projection drift for one camera."""

    camera_index: int
    mean_distance: float
    p95_distance: float
    max_distance: float


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


def perturb_intrinsic_parameters(cals: List[Calibration], scale: float) -> List[Calibration]:
    """Apply deterministic distortion perturbations while keeping poses fixed."""
    perturbed = [clone_calibration(cal) for cal in cals]
    deltas = [
        (2.5e-5, -8.0e-5, 6.0e-5),
        (-2.0e-5, 7.0e-5, -5.0e-5),
        (1.8e-5, 5.0e-5, 4.0e-5),
        (-1.6e-5, -6.0e-5, 5.0e-5),
    ]
    for cal, (k1_delta, p1_delta, p2_delta) in zip(perturbed, deltas):
        radial = cal.get_radial_distortion().copy()
        radial[0] += k1_delta * scale
        cal.set_radial_distortion(radial)

        decentering = cal.get_decentering().copy()
        decentering[0] += p1_delta * scale
        decentering[1] += p2_delta * scale
        cal.set_decentering(decentering)

    return perturbed


def build_experiment_start_calibrations(
    spec: ExperimentSpec,
    *,
    start_cals: List[Calibration],
    reference_cals: List[Calibration],
) -> List[Calibration]:
    """Construct the initial calibration set for one experiment."""
    if bool(spec.ba_kwargs.get("use_reference_cals", False)):
        base_cals = reference_cals
    else:
        base_cals = start_cals

    working_cals = [clone_calibration(cal) for cal in base_cals]
    if bool(spec.ba_kwargs.get("perturb_intrinsics_only", False)):
        scale = cast(float, spec.ba_kwargs.get("intrinsic_perturbation_scale", 1.0))
        working_cals = perturb_intrinsic_parameters(working_cals, scale)

    fixed_camera_indices = tuple(
        cast(List[int] | None, spec.ba_kwargs.get("fixed_camera_indices")) or []
    )
    for camera_index in fixed_camera_indices:
        if camera_index < 0 or camera_index >= len(working_cals):
            continue
        working_cals[camera_index].set_pos(reference_cals[camera_index].get_pos().copy())
        working_cals[camera_index].set_angles(
            reference_cals[camera_index].get_angles().copy()
        )

    return working_cals


def load_reference_geometry_points(
    case_dir: Path,
    num_cams: int,
) -> np.ndarray | None:
    """Load known calibration-body points referenced by cal_ori.par, if present."""
    cal_ori_path = case_dir / "parameters/cal_ori.par"
    if not cal_ori_path.exists():
        return None

    calibration_par = read_cal_ori_parameters(cal_ori_path, num_cams)
    if not calibration_par.fixp_name:
        return None

    calblock_path = case_dir / calibration_par.fixp_name
    if not calblock_path.exists():
        return None

    calblock = read_sortgrid_calblock(calblock_path)
    if len(calblock) == 0:
        return None

    return np.column_stack([calblock.x, calblock.y, calblock.z]).astype(float)


def calibration_body_projection_drift(
    reference_cals: Sequence[Calibration],
    candidate_cals: Sequence[Calibration],
    control: ControlPar,
    reference_points: np.ndarray | None,
) -> List[ProjectionDriftSummary] | None:
    """Compare candidate calibration-body projections to a trusted reference set."""
    if reference_points is None:
        return None

    summaries: List[ProjectionDriftSummary] = []
    for camera_index, (reference_cal, candidate_cal) in enumerate(
        zip(reference_cals, candidate_cals),
        start=1,
    ):
        reference_pixels = []
        candidate_pixels = []
        for point in reference_points:
            ref_x, ref_y = img_coord(point, reference_cal, control.mm)
            cand_x, cand_y = img_coord(point, candidate_cal, control.mm)
            reference_pixels.append(metric_to_pixel(ref_x, ref_y, control))
            candidate_pixels.append(metric_to_pixel(cand_x, cand_y, control))

        displacement = np.linalg.norm(
            np.asarray(candidate_pixels) - np.asarray(reference_pixels),
            axis=1,
        )
        summaries.append(
            ProjectionDriftSummary(
                camera_index=camera_index,
                mean_distance=float(displacement.mean()),
                p95_distance=float(np.percentile(displacement, 95.0)),
                max_distance=float(displacement.max()),
            )
        )

    return summaries


def format_projection_drift(summaries: Sequence[ProjectionDriftSummary] | None) -> str:
    """Render calibration-body projection drift summaries."""
    if not summaries:
        return "No reference calibration-body geometry available."

    lines = ["camera  mean_px  p95_px  max_px", "------  -------  ------  ------"]
    for item in summaries:
        lines.append(
            f"{item.camera_index:>6}  {item.mean_distance:>7.3f}  {item.p95_distance:>6.3f}  {item.max_distance:>6.3f}"
        )
    return "\n".join(lines)


def max_projection_drift(
    summaries: Sequence[ProjectionDriftSummary] | None,
) -> float | None:
    """Return the worst per-camera maximum projection drift in pixels."""
    if not summaries:
        return None
    return max(item.max_distance for item in summaries)


def should_block_export_on_geometry(
    summaries: Sequence[ProjectionDriftSummary] | None,
    threshold_px: float | None,
) -> tuple[bool, str]:
    """Return whether an export should be blocked by geometry drift."""
    if summaries is None or threshold_px is None or threshold_px <= 0:
        return False, ""

    drift_max = max_projection_drift(summaries)
    if drift_max is None or drift_max <= threshold_px:
        return False, ""

    return True, f"geometry_export_blocked=max_drift={drift_max:.3f}px>{threshold_px:.3f}px"


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


def build_known_point_constraints(
    point_init: np.ndarray,
    count: int,
) -> Dict[int, np.ndarray]:
    """Select a deterministic subset of 3D points to use as soft anchors."""
    if count <= 0:
        return {}

    num_points = int(point_init.shape[0])
    if num_points == 0:
        raise ValueError("Cannot build known-point constraints from an empty point set")

    target_count = min(count, num_points)
    selected_indices = np.linspace(0, num_points - 1, num=target_count, dtype=int)
    return {
        int(point_index): point_init[int(point_index)].copy()
        for point_index in selected_indices.tolist()
    }


def all_fixed_camera_pairs(num_cams: int) -> List[tuple[int, int]]:
    """Return all unique two-camera anchor pairs."""
    return [
        (first, second)
        for first in range(num_cams - 1)
        for second in range(first + 1, num_cams)
    ]


def calibration_pose_shifts(
    reference_cals: Sequence[Calibration],
    candidate_cals: Sequence[Calibration],
) -> tuple[List[float], List[float]]:
    """Measure per-camera pose changes relative to a reference calibration set."""
    position_shifts = [
        float(np.linalg.norm(candidate.get_pos() - reference.get_pos()))
        for reference, candidate in zip(reference_cals, candidate_cals)
    ]
    angle_shifts = [
        float(np.linalg.norm(candidate.get_angles() - reference.get_angles()))
        for reference, candidate in zip(reference_cals, candidate_cals)
    ]
    return position_shifts, angle_shifts


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """Return the minimum Euclidean distance from a point to a polyline in pixels."""
    if polyline.shape[0] < 2:
        raise ValueError("Polyline must have at least two vertices")

    best = np.inf
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq == 0.0:
            distance = float(np.linalg.norm(point - start))
        else:
            t = float(np.dot(point - start, segment) / segment_length_sq)
            t = min(1.0, max(0.0, t))
            projection = start + t * segment
            distance = float(np.linalg.norm(point - projection))
        best = min(best, distance)
    return float(best)


def symmetric_epipolar_distance(
    origin_obs: np.ndarray,
    target_obs: np.ndarray,
    origin_cal: Calibration,
    target_cal: Calibration,
    cpar: ControlPar,
    vpar: VolumePar,
    num_curve_points: int,
) -> float:
    """Return a symmetric epipolar inconsistency score in pixels."""
    forward_curve = epipolar_curve(
        origin_obs,
        origin_cal,
        target_cal,
        num_curve_points,
        cpar,
        vpar,
    )
    backward_curve = epipolar_curve(
        target_obs,
        target_cal,
        origin_cal,
        num_curve_points,
        cpar,
        vpar,
    )
    forward_distance = _point_to_polyline_distance(target_obs, forward_curve)
    backward_distance = _point_to_polyline_distance(origin_obs, backward_curve)
    return 0.5 * (forward_distance + backward_distance)


def summarize_epipolar_consistency(
    observed_pixels: np.ndarray,
    cals: Sequence[Calibration],
    cpar: ControlPar,
    vpar: VolumePar,
    *,
    num_curve_points: int = 64,
    threshold_scales: Sequence[float] = (0.5, 1.0, 2.0),
) -> List[CameraPairEpipolarSummary]:
    """Summarize pairwise epipolar distance statistics for observed correspondences."""
    thresholds = [float(vpar.eps0 * scale) for scale in threshold_scales]
    summaries = []
    for cam_a, cam_b in all_fixed_camera_pairs(len(cals)):
        distances = []
        for point_index in range(observed_pixels.shape[0]):
            obs_a = observed_pixels[point_index, cam_a]
            obs_b = observed_pixels[point_index, cam_b]
            if not (np.all(np.isfinite(obs_a)) and np.all(np.isfinite(obs_b))):
                continue
            distances.append(
                symmetric_epipolar_distance(
                    obs_a,
                    obs_b,
                    cals[cam_a],
                    cals[cam_b],
                    cpar,
                    vpar,
                    num_curve_points,
                )
            )

        if not distances:
            continue
        distance_array = np.asarray(distances, dtype=np.float64)
        summaries.append(
            CameraPairEpipolarSummary(
                camera_pair=(cam_a, cam_b),
                mean_distance=float(np.mean(distance_array)),
                median_distance=float(np.median(distance_array)),
                p95_distance=float(np.percentile(distance_array, 95.0)),
                max_distance=float(np.max(distance_array)),
                acceptance_rates={
                    threshold: float(np.mean(distance_array <= threshold))
                    for threshold in thresholds
                },
            )
        )

    summaries.sort(key=lambda item: item.mean_distance)
    return summaries


def format_epipolar_diagnostics(
    summaries: Sequence[CameraPairEpipolarSummary],
) -> str:
    """Render pairwise epipolar statistics as a compact table."""
    thresholds = sorted(
        {threshold for item in summaries for threshold in item.acceptance_rates}
    )
    headers = [
        "pair",
        "mean_epi",
        "median_epi",
        "p95_epi",
        "max_epi",
    ]
    headers.extend([f"<= {threshold:.3f}" for threshold in thresholds])
    data = []
    for item in summaries:
        row = [
            f"{item.camera_pair[0] + 1},{item.camera_pair[1] + 1}",
            f"{item.mean_distance:.6f}",
            f"{item.median_distance:.6f}",
            f"{item.p95_distance:.6f}",
            f"{item.max_distance:.6f}",
        ]
        row.extend([f"{100.0 * item.acceptance_rates[threshold]:.1f}%" for threshold in thresholds])
        data.append(row)

    widths = [len(header) for header in headers]
    for row in data:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: List[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in data)
    return "\n".join(lines)


def summarize_quadruplet_sensitivity(
    observed_pixels: np.ndarray,
    cals: Sequence[Calibration],
    cpar: ControlPar,
) -> QuadrupletSensitivitySummary:
    """Measure leave-one-camera-out instability for fully observed points."""
    if observed_pixels.shape[1] < 4:
        raise ValueError("Quadruplet sensitivity requires at least four cameras")

    full_mask = np.all(np.isfinite(observed_pixels), axis=(1, 2))
    full_indices = np.flatnonzero(full_mask)
    if full_indices.size == 0:
        raise ValueError("No fully observed quadruplets available for sensitivity analysis")

    full_points, full_ray_convergence = initialize_bundle_adjustment_points(
        observed_pixels[full_indices],
        list(cals),
        cpar,
    )

    spreads = np.empty(full_indices.size, dtype=np.float64)
    leave_one_out_rays = np.empty(full_indices.size, dtype=np.float64)
    for local_index, point_index in enumerate(full_indices):
        subset_points = []
        subset_rays = []
        for omitted_camera in range(observed_pixels.shape[1]):
            keep = [cam for cam in range(observed_pixels.shape[1]) if cam != omitted_camera]
            subset_observed = observed_pixels[point_index : point_index + 1, keep, :]
            subset_cals = [cals[cam] for cam in keep]
            subset_point, subset_ray = initialize_bundle_adjustment_points(
                subset_observed,
                subset_cals,
                cpar,
            )
            subset_points.append(subset_point[0])
            subset_rays.append(float(subset_ray[0]))

        subset_stack = np.asarray(subset_points, dtype=np.float64)
        spreads[local_index] = float(
            np.max(np.linalg.norm(subset_stack - full_points[local_index], axis=1))
        )
        leave_one_out_rays[local_index] = float(np.mean(subset_rays))

    worst_order = np.argsort(spreads)[::-1][:5]
    return QuadrupletSensitivitySummary(
        num_points=int(full_indices.size),
        mean_spread=float(np.mean(spreads)),
        median_spread=float(np.median(spreads)),
        p95_spread=float(np.percentile(spreads, 95.0)),
        max_spread=float(np.max(spreads)),
        mean_full_ray_convergence=float(np.mean(full_ray_convergence)),
        mean_leave_one_out_ray_convergence=float(np.mean(leave_one_out_rays)),
        worst_point_indices=[int(full_indices[index]) for index in worst_order.tolist()],
    )


def format_quadruplet_sensitivity(
    baseline: QuadrupletSensitivitySummary,
    final: QuadrupletSensitivitySummary,
) -> str:
    """Render before/after quadruplet sensitivity diagnostics."""
    headers = (
        "phase",
        "points",
        "mean_spread",
        "median_spread",
        "p95_spread",
        "max_spread",
        "mean_full_ray",
        "mean_loo_ray",
        "worst_points",
    )
    data = []
    for phase, summary in (("baseline", baseline), ("final", final)):
        data.append(
            [
                phase,
                str(summary.num_points),
                f"{summary.mean_spread:.6f}",
                f"{summary.median_spread:.6f}",
                f"{summary.p95_spread:.6f}",
                f"{summary.max_spread:.6f}",
                f"{summary.mean_full_ray_convergence:.6f}",
                f"{summary.mean_leave_one_out_ray_convergence:.6f}",
                ",".join(str(index) for index in summary.worst_point_indices),
            ]
        )

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


def summarize_fixed_camera_diagnostics(
    results: Sequence[ExperimentResult],
) -> List[FixedCameraDiagnostic]:
    """Collapse experiment results into anchor-pair diagnostic metrics."""
    diagnostics = []
    for result in results:
        if len(result.fixed_camera_indices) != 2:
            raise ValueError("Fixed-camera diagnostics require exactly two fixed cameras")

        fixed_indices = set(result.fixed_camera_indices)
        fixed_position_shifts = [
            result.camera_position_shifts[index] for index in result.fixed_camera_indices
        ]
        fixed_angle_shifts = [
            result.camera_angle_shifts[index] for index in result.fixed_camera_indices
        ]
        free_position_shifts = [
            shift
            for index, shift in enumerate(result.camera_position_shifts)
            if index not in fixed_indices
        ]
        free_angle_shifts = [
            shift
            for index, shift in enumerate(result.camera_angle_shifts)
            if index not in fixed_indices
        ]

        diagnostics.append(
            FixedCameraDiagnostic(
                fixed_camera_indices=cast(tuple[int, int], result.fixed_camera_indices),
                final_rms=result.final_rms,
                final_ray_convergence=result.final_ray_convergence,
                fixed_position_shift=max(fixed_position_shifts, default=0.0),
                fixed_angle_shift=max(fixed_angle_shifts, default=0.0),
                mean_free_position_shift=float(np.mean(free_position_shifts)),
                max_free_position_shift=max(free_position_shifts, default=0.0),
                mean_free_angle_shift=float(np.mean(free_angle_shifts)),
                notes=result.notes,
            )
        )

    diagnostics.sort(
        key=lambda item: (
            item.fixed_position_shift,
            item.fixed_angle_shift,
            item.final_rms,
            item.final_ray_convergence,
        )
    )
    return diagnostics


def default_experiments(
    *,
    known_points: Dict[int, np.ndarray] | None = None,
    known_point_sigmas: float | np.ndarray | None = None,
    perturbation_scale: float = 1.0,
    geometry_guard_mode: str = "off",
    geometry_guard_threshold: float | None = None,
) -> List[ExperimentSpec]:
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

    intrinsics_only = OrientPar()
    intrinsics_only.k1flag = 1
    intrinsics_only.p1flag = 1
    intrinsics_only.p2flag = 1

    experiments = [
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
            name="intrinsics_only",
            description="Intrinsics-only BA from fixed reference poses with tightly bounded distortion updates",
            mode="multi",
            ba_kwargs={
                "orient_par": intrinsics_only,
                "fixed_camera_indices": [0, 1, 2, 3],
                "loss": "linear",
                "method": "trf",
                "prior_sigmas": {
                    "k1": 5e-5,
                    "p1": 1e-4,
                    "p2": 1e-4,
                },
                "parameter_bounds": {
                    "k1": (-5e-5, 5e-5),
                    "p1": (-2e-4, 2e-4),
                    "p2": (-2e-4, 2e-4),
                },
                "max_nfev": 40,
                "optimize_extrinsics": False,
                "optimize_points": False,
                "use_reference_cals": True,
                "perturb_intrinsics_only": True,
                "intrinsic_perturbation_scale": perturbation_scale,
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
                "geometry_guard_mode": geometry_guard_mode,
                "geometry_guard_threshold": geometry_guard_threshold,
            },
        ),
    ]

    if known_points:
        experiments.extend(
            [
                ExperimentSpec(
                    name="pose_trf_known_points",
                    description="Pose-only BA with soft known-point geometry anchors",
                    mode="multi",
                    ba_kwargs={
                        "orient_par": OrientPar(),
                        "fixed_camera_indices": [0, 1],
                        "loss": "linear",
                        "method": "trf",
                        "prior_sigmas": pose_priors,
                        "parameter_bounds": pose_bounds,
                        "max_nfev": 12,
                        "known_points": known_points,
                        "known_point_sigmas": known_point_sigmas,
                    },
                ),
                ExperimentSpec(
                    name="guarded_two_step_known_points",
                    description="Guarded two-step BA with soft known-point geometry anchors",
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
                        "known_points": known_points,
                        "known_point_sigmas": known_point_sigmas,
                        "geometry_guard_mode": geometry_guard_mode,
                        "geometry_guard_threshold": geometry_guard_threshold,
                    },
                ),
            ]
        )

    return experiments


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
    reference_cals: List[Calibration],
    reference_geometry_points: np.ndarray | None,
    geometry_export_threshold: float | None,
    source_case_dir: Path,
    output_dir: Path | None,
) -> ExperimentResult:
    """Execute one experiment and collect metrics and optional outputs."""
    working_cals = build_experiment_start_calibrations(
        spec,
        start_cals=start_cals,
        reference_cals=reference_cals,
    )
    fixed_camera_indices = tuple(
        cast(List[int] | None, spec.ba_kwargs.get("fixed_camera_indices")) or []
    )
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
            known_points=cast(
                Dict[int, np.ndarray] | None,
                spec.ba_kwargs.get("known_points"),
            ),
            known_point_sigmas=cast(
                float | np.ndarray | None,
                spec.ba_kwargs.get("known_point_sigmas"),
            ),
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
            intrinsic_ftol=cast(
                float | None, spec.ba_kwargs.get("intrinsic_ftol", 1e-12)
            ),
            intrinsic_xtol=cast(
                float | None, spec.ba_kwargs.get("intrinsic_xtol", 1e-12)
            ),
            intrinsic_gtol=cast(
                float | None, spec.ba_kwargs.get("intrinsic_gtol", 1e-12)
            ),
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
            known_points=cast(
                Dict[int, np.ndarray] | None,
                spec.ba_kwargs.get("known_points"),
            ),
            known_point_sigmas=cast(
                float | np.ndarray | None,
                spec.ba_kwargs.get("known_point_sigmas"),
            ),
            geometry_reference_points=reference_geometry_points,
            geometry_reference_cals=reference_cals,
            geometry_guard_mode=cast(
                str,
                spec.ba_kwargs.get("geometry_guard_mode", "off"),
            ),
            geometry_guard_threshold=cast(
                float | None,
                spec.ba_kwargs.get("geometry_guard_threshold"),
            ),
        )
        success = True
        final_rms = cast(float, summary["final_reprojection_rms"])
        final_ray = cast(float, summary["final_mean_ray_convergence"])
        notes = f"accepted_stage={summary['accepted_stage']}"
    else:
        raise ValueError(f"Unknown experiment mode: {spec.mode}")

    duration_sec = perf_counter() - start
    camera_position_shifts, camera_angle_shifts = calibration_pose_shifts(
        working_cals,
        refined_cals,
    )
    geometry_projection_drift = calibration_body_projection_drift(
        reference_cals,
        refined_cals,
        control,
        reference_geometry_points,
    )
    export_blocked, export_note = should_block_export_on_geometry(
        geometry_projection_drift,
        geometry_export_threshold,
    )
    if export_note:
        notes = f"{notes}; {export_note}" if notes else export_note
    if export_blocked:
        success = False

    cal_dir = None
    if output_dir is not None and not export_blocked:
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
        (case_out_dir / "geometry_check.txt").write_text(
            format_projection_drift(geometry_projection_drift) + "\n",
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
        fixed_camera_indices=fixed_camera_indices,
        camera_position_shifts=camera_position_shifts,
        camera_angle_shifts=camera_angle_shifts,
        geometry_projection_drift=geometry_projection_drift,
        refined_cals=refined_cals,
        refined_points=refined_points,
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
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(list(headers)), separator]
    lines.extend(render_row(row) for row in data)
    return "\n".join(lines)


def format_fixed_camera_diagnostics(
    diagnostics: Sequence[FixedCameraDiagnostic],
) -> str:
    """Render a compact plain-text summary of anchor-pair diagnostics."""
    headers = (
        "fixed_pair",
        "rms_after",
        "ray_after",
        "fixed_pos_shift",
        "fixed_ang_shift",
        "mean_free_pos",
        "max_free_pos",
        "mean_free_ang",
        "notes",
    )
    data = [
        [
            f"{item.fixed_camera_indices[0] + 1},{item.fixed_camera_indices[1] + 1}",
            f"{item.final_rms:.6f}",
            f"{item.final_ray_convergence:.6f}",
            f"{item.fixed_position_shift:.6e}",
            f"{item.fixed_angle_shift:.6e}",
            f"{item.mean_free_position_shift:.6f}",
            f"{item.max_free_position_shift:.6f}",
            f"{item.mean_free_angle_shift:.6f}",
            item.notes,
        ]
        for item in diagnostics
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


def summarize_anchor_participation(
    diagnostics: Sequence[FixedCameraDiagnostic],
    num_cams: int,
) -> str:
    """Aggregate anchor-pair diagnostics by camera participation."""
    headers = (
        "camera",
        "pair_count",
        "avg_rms",
        "avg_ray",
        "avg_free_pos",
        "avg_free_ang",
    )
    rows = []
    for camera_index in range(num_cams):
        participating = [
            item for item in diagnostics if camera_index in item.fixed_camera_indices
        ]
        if not participating:
            continue
        rows.append(
            [
                str(camera_index + 1),
                str(len(participating)),
                f"{np.mean([item.final_rms for item in participating]):.6f}",
                f"{np.mean([item.final_ray_convergence for item in participating]):.6f}",
                f"{np.mean([item.mean_free_position_shift for item in participating]):.6f}",
                f"{np.mean([item.mean_free_angle_shift for item in participating]):.6f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: List[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(list(headers)), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def find_experiment_spec(
    experiments: Sequence[ExperimentSpec],
    name: str,
) -> ExperimentSpec:
    """Return one experiment by name."""
    for spec in experiments:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown experiment: {name}")


def run_fixed_camera_pair_diagnostics(
    spec: ExperimentSpec,
    *,
    observed_pixels: np.ndarray,
    point_init: np.ndarray,
    control: ControlPar,
    start_cals: List[Calibration],
    reference_cals: List[Calibration],
    reference_geometry_points: np.ndarray | None,
    geometry_export_threshold: float | None,
    source_case_dir: Path,
) -> List[FixedCameraDiagnostic]:
    """Run one experiment over every two-camera anchor pair and summarize results."""
    diagnostic_results = []
    for fixed_pair in all_fixed_camera_pairs(len(start_cals)):
        pair_kwargs = dict(spec.ba_kwargs)
        pair_kwargs["fixed_camera_indices"] = list(fixed_pair)
        pair_spec = ExperimentSpec(
            name=f"{spec.name}_fixed_{fixed_pair[0] + 1}_{fixed_pair[1] + 1}",
            description=f"{spec.description} with cameras {fixed_pair[0] + 1} and {fixed_pair[1] + 1} fixed",
            mode=spec.mode,
            ba_kwargs=pair_kwargs,
        )
        diagnostic_results.append(
            run_experiment(
                pair_spec,
                observed_pixels=observed_pixels,
                point_init=point_init,
                control=control,
                start_cals=start_cals,
                reference_cals=reference_cals,
                reference_geometry_points=reference_geometry_points,
                geometry_export_threshold=geometry_export_threshold,
                source_case_dir=source_case_dir,
                output_dir=None,
            )
        )

    return summarize_fixed_camera_diagnostics(diagnostic_results)


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
    parser.add_argument(
        "--known-points",
        type=int,
        default=12,
        help="Use N evenly spaced input 3D points as soft geometry anchors; 0 disables constrained presets.",
    )
    parser.add_argument(
        "--known-point-sigma",
        type=float,
        default=0.25,
        help="Standard deviation for each anchored 3D coordinate in object-space units.",
    )
    parser.add_argument(
        "--diagnose-fixed-pairs",
        action="store_true",
        help="Run one experiment across every two-camera fixed pair and print anchor-pair diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-experiment",
        type=str,
        default="guarded_two_step",
        help="Experiment name to use with --diagnose-fixed-pairs.",
    )
    parser.add_argument(
        "--diagnose-epipolar",
        action="store_true",
        help="Compare pairwise epipolar consistency before and after the selected diagnostic experiment.",
    )
    parser.add_argument(
        "--diagnose-quadruplets",
        action="store_true",
        help="Compare leave-one-camera-out quadruplet stability before and after the selected diagnostic experiment.",
    )
    parser.add_argument(
        "--epipolar-curve-points",
        type=int,
        default=64,
        help="Number of points sampled along each epipolar curve for the diagnostic distance calculation.",
    )
    parser.add_argument(
        "--geometry-guard-mode",
        type=str,
        choices=("auto", "off", "soft", "hard"),
        default="auto",
        help="Guarded-BA geometry acceptance mode. 'auto' uses 'hard' when cal_ori.par exposes known 3D target points.",
    )
    parser.add_argument(
        "--geometry-guard-threshold",
        type=float,
        default=2.5,
        help="Maximum allowed calibration-body projection drift in pixels for hard geometry guards.",
    )
    parser.add_argument(
        "--geometry-export-threshold",
        type=float,
        default=2.5,
        help="Do not write output case folders whose final calibration-body drift exceeds this many pixels; 0 disables export blocking.",
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
    reference_geometry_points = load_reference_geometry_points(case_dir, num_cams)
    if args.geometry_guard_mode == "auto":
        geometry_guard_mode = "hard" if reference_geometry_points is not None else "off"
    else:
        geometry_guard_mode = args.geometry_guard_mode
    geometry_export_threshold = (
        args.geometry_export_threshold if reference_geometry_points is not None else None
    )
    known_points = build_known_point_constraints(point_init, args.known_points)
    known_point_sigmas = args.known_point_sigma if known_points else None

    output_dir = None if args.skip_write else args.output_dir.resolve()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Case: {case_dir}")
    print(f"Cameras: {num_cams}")
    print(
        f"Observations: {observed_pixels.shape[0]} points across {observed_pixels.shape[1]} cameras"
    )
    if known_points:
        print(
            f"Known-point anchors: {len(known_points)} (sigma={args.known_point_sigma:.3f})"
        )
    else:
        print("Known-point anchors: disabled")
    if reference_geometry_points is not None:
        print(
            f"Geometry guard: mode={geometry_guard_mode}, guard_threshold={args.geometry_guard_threshold:.3f}px, export_threshold={args.geometry_export_threshold:.3f}px"
        )
    else:
        print("Geometry guard: unavailable for this case (no known 3D target file found)")
    print(f"Output folders: {output_dir if output_dir is not None else 'disabled'}")
    print()

    experiments = default_experiments(
        known_points=known_points or None,
        known_point_sigmas=known_point_sigmas,
        perturbation_scale=args.perturbation_scale,
        geometry_guard_mode=geometry_guard_mode,
        geometry_guard_threshold=args.geometry_guard_threshold,
    )

    if args.diagnose_fixed_pairs:
        spec = find_experiment_spec(experiments, args.diagnostic_experiment)
        diagnostics = run_fixed_camera_pair_diagnostics(
            spec,
            observed_pixels=observed_pixels,
            point_init=point_init,
            control=control,
            start_cals=start_cals,
            reference_cals=true_cals,
            reference_geometry_points=reference_geometry_points,
            geometry_export_threshold=geometry_export_threshold,
            source_case_dir=case_dir,
        )
        print(
            f"Fixed-pair diagnostics for {spec.name}: {spec.description}"
        )
        print(format_fixed_camera_diagnostics(diagnostics))
        print()
        print("Anchor participation summary")
        print(summarize_anchor_participation(diagnostics, num_cams))
        return 0

    if args.diagnose_epipolar or args.diagnose_quadruplets:
        spec = find_experiment_spec(experiments, args.diagnostic_experiment)
        diagnostic_start_cals = build_experiment_start_calibrations(
            spec,
            start_cals=start_cals,
            reference_cals=true_cals,
        )
        diagnostic_result = run_experiment(
            spec,
            observed_pixels=observed_pixels,
            point_init=point_init,
            control=control,
            start_cals=start_cals,
            reference_cals=true_cals,
            reference_geometry_points=reference_geometry_points,
            geometry_export_threshold=geometry_export_threshold,
            source_case_dir=case_dir,
            output_dir=None,
        )
        if diagnostic_result.refined_cals is None:
            raise ValueError("Diagnostic experiment did not return refined calibrations")

        print(
            f"Diagnostics for {spec.name}: {spec.description}"
        )
        print(
            f"Result RMS: {diagnostic_result.final_rms:.6f} px, ray: {diagnostic_result.final_ray_convergence:.6f}, notes: {diagnostic_result.notes}"
        )
        print()

        if args.diagnose_epipolar:
            vpar = read_volume_par(case_dir / "parameters/criteria.par")
            baseline_epipolar = summarize_epipolar_consistency(
                observed_pixels,
                diagnostic_start_cals,
                control,
                vpar,
                num_curve_points=args.epipolar_curve_points,
            )
            final_epipolar = summarize_epipolar_consistency(
                observed_pixels,
                diagnostic_result.refined_cals,
                control,
                vpar,
                num_curve_points=args.epipolar_curve_points,
            )
            print(f"Epipolar diagnostics against criteria.par eps0={vpar.eps0:.6f}")
            print("Baseline")
            print(format_epipolar_diagnostics(baseline_epipolar))
            print()
            print("Final")
            print(format_epipolar_diagnostics(final_epipolar))
            print()

        if args.diagnose_quadruplets:
            baseline_quadruplets = summarize_quadruplet_sensitivity(
                observed_pixels,
                diagnostic_start_cals,
                control,
            )
            final_quadruplets = summarize_quadruplet_sensitivity(
                observed_pixels,
                diagnostic_result.refined_cals,
                control,
            )
            print("Quadruplet leave-one-camera-out sensitivity")
            print(format_quadruplet_sensitivity(baseline_quadruplets, final_quadruplets))

        return 0

    results = []
    for spec in experiments:
        print(f"Running {spec.name}: {spec.description}")
        result = run_experiment(
            spec,
            observed_pixels=observed_pixels,
            point_init=point_init,
            control=control,
            start_cals=start_cals,
            reference_cals=true_cals,
            reference_geometry_points=reference_geometry_points,
            geometry_export_threshold=geometry_export_threshold,
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
