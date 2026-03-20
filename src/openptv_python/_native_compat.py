"""Compatibility layer for selecting between optv and Python/Numba engines."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Literal
import warnings

EngineName = Literal["optv", "python"]

DEFAULT_ENGINE: EngineName = "optv"
ENGINE_OPTV: EngineName = "optv"
ENGINE_PYTHON: EngineName = "python"

_ENGINE_PREFERENCE: EngineName = DEFAULT_ENGINE
_ENGINE_REASON: str = ""
_ENGINE_WARNING_EMITTED = False


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except Exception:
        return None


optv_calibration = _optional_import("optv.calibration")
optv_image_processing = _optional_import("optv.image_processing")
optv_parameters = _optional_import("optv.parameters")
optv_segmentation = _optional_import("optv.segmentation")
optv_tracking_framebuf = _optional_import("optv.tracking_framebuf")

HAS_OPTV = any(
    module is not None
    for module in (
        optv_calibration,
        optv_image_processing,
        optv_parameters,
        optv_segmentation,
        optv_tracking_framebuf,
    )
)

HAS_NATIVE_PREPROCESS = (
    optv_image_processing is not None
    and hasattr(optv_image_processing, "preprocess_image")
    and optv_parameters is not None
    and hasattr(optv_parameters, "ControlParams")
)

HAS_NATIVE_SEGMENTATION = (
    optv_segmentation is not None
    and hasattr(optv_segmentation, "target_recognition")
    and optv_parameters is not None
    and hasattr(optv_parameters, "TargetParams")
    and hasattr(optv_parameters, "ControlParams")
    and optv_tracking_framebuf is not None
    and hasattr(optv_tracking_framebuf, "Target")
)

HAS_NATIVE_CALIBRATION = optv_calibration is not None and hasattr(
    optv_calibration, "Calibration"
)

HAS_NATIVE_TARGETS = optv_tracking_framebuf is not None and hasattr(
    optv_tracking_framebuf, "Target"
)


def _emit_engine_warning(reason: str) -> None:
    global _ENGINE_WARNING_EMITTED
    if _ENGINE_WARNING_EMITTED:
        return

    warnings.warn(reason, RuntimeWarning, stacklevel=3)
    _ENGINE_WARNING_EMITTED = True


def _set_engine_state(engine: EngineName, reason: str) -> None:
    global _ENGINE_PREFERENCE, _ENGINE_REASON, _ENGINE_WARNING_EMITTED
    _ENGINE_PREFERENCE = engine
    _ENGINE_REASON = reason
    _ENGINE_WARNING_EMITTED = False


def _resolve_engine_request(engine: EngineName | str | None) -> EngineName:
    if engine is None:
        return DEFAULT_ENGINE

    normalized = str(engine).strip().lower()
    if normalized in {"optv", "native", "c", "fast"}:
        return ENGINE_OPTV
    if normalized in {"python", "numba", "pure-python", "fallback"}:
        return ENGINE_PYTHON

    raise ValueError(f"Unknown engine '{engine}'. Use 'optv' or 'python'.")


def _resolve_active_engine() -> tuple[EngineName, str]:
    if _ENGINE_PREFERENCE == ENGINE_PYTHON:
        return ENGINE_PYTHON, "Forced Python engine"

    if HAS_OPTV:
        return ENGINE_OPTV, "Using optv engine"

    return ENGINE_PYTHON, "optv is unavailable; using Python/Numba fallback"


def set_engine(engine: EngineName | str | None = None, *, warn_once: bool = True) -> EngineName:
    """Set the preferred engine for native-backed calls.

    Parameters
    ----------
    engine:
        Preferred engine. ``optv`` is the default and ``python`` forces the
        Python/Numba code path.
    warn_once:
        Emit a one-time warning when the selected engine cannot be used.
    """

    requested = _resolve_engine_request(engine)

    if requested == ENGINE_PYTHON:
        _set_engine_state(requested, "Forced Python engine")
        return ENGINE_PYTHON

    if HAS_OPTV:
        _set_engine_state(requested, "Using optv engine")
        return ENGINE_OPTV

    reason = "optv is unavailable; using Python/Numba fallback"
    _set_engine_state(requested, reason)
    if warn_once:
        _emit_engine_warning(reason)
    return ENGINE_PYTHON


def get_engine_preference() -> EngineName:
    """Return the user-requested engine preference."""

    return _ENGINE_PREFERENCE


def get_active_engine() -> EngineName:
    """Return the engine currently used for native-backed calls."""

    active, _ = _resolve_active_engine()
    return active


def get_engine_reason() -> str:
    """Return a human-readable explanation of the active engine choice."""

    active, reason = _resolve_active_engine()
    if active == ENGINE_PYTHON and _ENGINE_PREFERENCE == ENGINE_PYTHON:
        return reason
    if active == ENGINE_PYTHON and _ENGINE_PREFERENCE == ENGINE_OPTV:
        return reason
    return reason


def get_engine_status() -> str:
    """Return a short status string for GUI and batch reporting."""

    return f"engine={get_active_engine()} ({get_engine_reason()})"


def should_use_native(feature_name: str | None = None) -> bool:
    """Return True when the native optv implementation should be used.

    The selector prefers optv by default and falls back to Python/Numba when
    optv is unavailable or the user forced the Python engine.
    """

    if get_engine_preference() == ENGINE_PYTHON:
        return False

    if feature_name in {None, "", "preprocess_image"}:
        return HAS_NATIVE_PREPROCESS

    if feature_name == "target_recognition":
        return HAS_NATIVE_SEGMENTATION

    if feature_name == "calibration":
        return HAS_NATIVE_CALIBRATION

    if feature_name == "targets":
        return HAS_NATIVE_TARGETS

    return HAS_OPTV


def get_num_cams(control_params: Any) -> int:
    """Return the camera count from either backend's control parameter object."""

    num_cams = getattr(control_params, "num_cams", None)
    if num_cams is not None:
        return int(num_cams)

    getter = getattr(control_params, "get_num_cams", None)
    if callable(getter):
        return int(getter())

    raise AttributeError("ControlParams object does not expose a camera count")
    
def get_multimedia_par(control_params: Any) -> Any:
    """Return the multimedia-parameter object from either backend."""

    multimedia = getattr(control_params, "mm", None)
    if multimedia is not None:
        return multimedia

    getter = getattr(control_params, "get_multimedia_par", None)
    if callable(getter):
        multimedia = getter()
    else:
        getter = getattr(control_params, "get_multimedia_params", None)
        if callable(getter):
            multimedia = getter()
        else:
            raise AttributeError(
                "ControlParams object does not expose multimedia parameters"
            )

    try:
        from openptv_python.parameters import MultimediaPar
    except Exception:
        return multimedia

    if isinstance(multimedia, MultimediaPar):
        return multimedia

    converted = MultimediaPar(
        nlay=int(multimedia.get_nlay()) if hasattr(multimedia, "get_nlay") else 1,
        n1=float(multimedia.get_n1()) if hasattr(multimedia, "get_n1") else float(getattr(multimedia, "n1", 1.0)),
        n2=list(multimedia.get_n2()) if hasattr(multimedia, "get_n2") else list(getattr(multimedia, "n2", [1.0])),
        d=list(multimedia.get_d()) if hasattr(multimedia, "get_d") else list(getattr(multimedia, "d", [0.0])),
        n3=float(multimedia.get_n3()) if hasattr(multimedia, "get_n3") else float(getattr(multimedia, "n3", 1.0)),
    )
    return converted


def get_image_size(control_params: Any) -> tuple[int, int]:
    """Return the image size from either backend's control parameter object."""

    getter = getattr(control_params, "get_image_size", None)
    if callable(getter):
        imx, imy = getter()
        return int(imx), int(imy)

    return int(getattr(control_params, "imx")), int(getattr(control_params, "imy"))


def get_pixel_size(control_params: Any) -> tuple[float, float]:
    """Return the pixel size from either backend's control parameter object."""

    getter = getattr(control_params, "get_pixel_size", None)
    if callable(getter):
        pix_x, pix_y = getter()
        return float(pix_x), float(pix_y)

    return float(getattr(control_params, "pix_x")), float(getattr(control_params, "pix_y"))


# Initialize once so the default preference is explicit and optv availability
# is reported immediately in sessions where it is missing.
set_engine(DEFAULT_ENGINE, warn_once=True)


def native_preprocess_image(*args: Any, **kwargs: Any) -> Any:
    if not HAS_NATIVE_PREPROCESS or optv_image_processing is None:
        raise RuntimeError("optv native preprocess_image is not available")
    return optv_image_processing.preprocess_image(*args, **kwargs)


def native_target_recognition(*args: Any, **kwargs: Any) -> Any:
    if not HAS_NATIVE_SEGMENTATION or optv_segmentation is None:
        raise RuntimeError("optv native target_recognition is not available")
    return optv_segmentation.target_recognition(*args, **kwargs)
