"""Regenerate the native stress benchmark demo assets for the docs.

This script reruns tests/test_native_stress_performance.py, parses the timing
summary lines printed by the test module, and writes machine-specific demo
artifacts under docs/_static/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Structured benchmark summary for one workload."""

    key: str
    label: str
    python_label: str
    python_seconds: float
    native_seconds: float
    speedup: float
    legacy_seconds: float | None = None
    compiled_vs_legacy: float | None = None


LINE_PATTERNS = {
    "sequence": re.compile(
        r"^sequence stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "preprocess": re.compile(
        r"^preprocess stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "segmentation": re.compile(
        r"^segmentation stress benchmark: python\+numba: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "stereomatching": re.compile(
        r"^stereomatching stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
    "reconstruction": re.compile(
        r"^reconstruction stress benchmark: compiled-python: median=(?P<python>[0-9.]+)s .*?"
        r"legacy-python: median=(?P<legacy>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?"
        r"compiled-vs-legacy=(?P<compiled_vs_legacy>[0-9.]+)x; "
        r"compiled-vs-native=(?P<speedup>[0-9.]+)x"
    ),
    "multilayer_reconstruction": re.compile(
        r"^multilayer reconstruction stress benchmark: compiled-python: median=(?P<python>[0-9.]+)s .*?"
        r"legacy-python: median=(?P<legacy>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?"
        r"compiled-vs-legacy=(?P<compiled_vs_legacy>[0-9.]+)x; "
        r"compiled-vs-native=(?P<speedup>[0-9.]+)x"
    ),
    "tracking": re.compile(
        r"^tracking stress benchmark: python: median=(?P<python>[0-9.]+)s .*?"
        r"native: median=(?P<native>[0-9.]+)s .*?speedup=(?P<speedup>[0-9.]+)x"
    ),
}


WORKLOAD_METADATA = {
    "sequence": ("Sequence params", "python"),
    "preprocess": ("Preprocess image", "python"),
    "segmentation": ("Segmentation", "python+numba"),
    "stereomatching": ("Stereo matching", "python"),
    "reconstruction": ("Reconstruction", "compiled-python"),
    "multilayer_reconstruction": ("Multilayer reconstruction", "compiled-python"),
    "tracking": ("Tracking", "python"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing tests/test_native_stress_performance.py",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=Path(__file__).resolve().parent / "_static" / "native-stress-demo.svg",
        help="Output path for the generated SVG chart",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(__file__).resolve().parent / "_static" / "native-stress-demo.json",
        help="Output path for the parsed benchmark JSON",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=Path(__file__).resolve().parent / "_static" / "native-stress-demo.log",
        help="Output path for the raw benchmark log",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run pytest",
    )
    return parser.parse_args()


def run_benchmarks(repo_root: Path, python_executable: str) -> str:
    command = [
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "tests/test_native_stress_performance.py",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            "stress benchmark run failed with exit code "
            f"{completed.returncode}\n\n{output}"
        )
    return output


def parse_benchmarks(output: str) -> list[BenchmarkResult]:
    result_map: dict[str, BenchmarkResult] = {}
    for raw_line in output.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        line = line.lstrip(".").strip()
        for key, pattern in LINE_PATTERNS.items():
            match = pattern.match(line)
            if match is None:
                continue
            label, python_label = WORKLOAD_METADATA[key]
            groups = match.groupdict()
            result_map[key] = BenchmarkResult(
                key=key,
                label=label,
                python_label=python_label,
                python_seconds=float(groups["python"]),
                native_seconds=float(groups["native"]),
                speedup=float(groups["speedup"]),
                legacy_seconds=(
                    float(groups["legacy"]) if groups.get("legacy") else None
                ),
                compiled_vs_legacy=(
                    float(groups["compiled_vs_legacy"])
                    if groups.get("compiled_vs_legacy")
                    else None
                ),
            )
            break

    expected_order = list(WORKLOAD_METADATA)
    missing = [key for key in expected_order if key not in result_map]
    if missing:
        raise RuntimeError(
            "failed to parse benchmark summaries for: " + ", ".join(missing)
        )
    return [result_map[key] for key in expected_order]


def _speedup_to_width(speedup: float, chart_width: float, max_decade: int) -> float:
    if speedup <= 1.0:
        return 18.0
    return max(18.0, chart_width * (math.log10(speedup) / max_decade))


def render_svg(results: list[BenchmarkResult], generated_at: datetime) -> str:
    width = 1180
    height = 760
    chart_left = 250
    chart_top = 180
    chart_width = 840
    row_height = 60
    max_speedup = max(result.speedup for result in results)
    max_decade = max(1, math.ceil(math.log10(max_speedup)))
    palette = {
        "python": "#5f8f77",
        "python+numba": "#5f8f77",
        "compiled-python": "#4473a8",
    }

    grid_lines = []
    grid_labels = []
    for decade in range(max_decade + 1):
        x = chart_left + chart_width * (decade / max_decade)
        grid_lines.append(
            f'<line x1="{x:.1f}" y1="{chart_top}" x2="{x:.1f}" y2="{chart_top + row_height * len(results)}" stroke="#d6d3d1" stroke-width="1"/>'
        )
        label = f"{10 ** decade:,}x"
        grid_labels.append(
            f'<text x="{x:.1f}" y="{chart_top + row_height * len(results) + 28}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#6b7280">{label}</text>'
        )

    bars = []
    for index, result in enumerate(results):
        y = chart_top + index * row_height
        bar_y = y - 20
        bar_width = _speedup_to_width(result.speedup, chart_width, max_decade)
        bar_color = palette[result.python_label]
        bars.append(
            f'<text x="64" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="17" font-weight="700" fill="#374151">{result.label}</text>'
        )
        bars.append(
            f'<rect x="{chart_left}" y="{bar_y}" width="{bar_width:.1f}" height="28" rx="8" fill="{bar_color}"/>'
        )
        bars.append(
            f'<text x="{chart_left + bar_width + 10:.1f}" y="{y - 1}" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#374151">{result.speedup:.2f}x</text>'
        )
        bars.append(
            f'<text x="{chart_left}" y="{y + 18}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#6b7280">{result.python_label}: {result.python_seconds:.6f}s, native: {result.native_seconds:.6f}s</text>'
        )

    generated_label = generated_at.strftime("%Y-%m-%d %H:%M UTC")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Native stress benchmark demo from this machine</title>
  <desc id="desc">Horizontal bar chart showing native speedup over the Python path for several openptv-python workloads.</desc>
  <rect width="{width}" height="{height}" fill="#f6f4ed"/>
  <rect x="36" y="30" width="1108" height="690" rx="24" fill="#fffdf8" stroke="#d7d1c3" stroke-width="2"/>

  <text x="64" y="74" font-family="Helvetica, Arial, sans-serif" font-size="30" font-weight="700" fill="#1f2933">Local demo from tests/test_native_stress_performance.py</text>
  <text x="64" y="104" font-family="Helvetica, Arial, sans-serif" font-size="17" fill="#4b5563">Regenerated from a fresh stress-suite run. Bars show native speedup over the routed Python path used by each test.</text>
  <text x="64" y="128" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#6b7280">Generated: {generated_label}</text>

  <line x1="{chart_left}" y1="{chart_top + row_height * len(results)}" x2="{chart_left + chart_width}" y2="{chart_top + row_height * len(results)}" stroke="#6b7280" stroke-width="2"/>
  <line x1="{chart_left}" y1="{chart_top - 25}" x2="{chart_left}" y2="{chart_top + row_height * len(results)}" stroke="#6b7280" stroke-width="2"/>
  {''.join(grid_lines)}
  {''.join(grid_labels)}
  {''.join(bars)}

  <rect x="760" y="146" width="320" height="92" rx="14" fill="#f7efe2" stroke="#d1b68b" stroke-width="1.5"/>
  <text x="784" y="178" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="700" fill="#4b3a20">Interpretation</text>
  <text x="784" y="202" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#5d4d34">Green: general Python or Python+Numba path versus native</text>
  <text x="784" y="224" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#5d4d34">Blue: compiled Python path versus native</text>

  <text x="64" y="690" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#4b5563">Machine-specific results are also saved as JSON and raw log files next to this SVG.</text>
</svg>
'''


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = run_benchmarks(repo_root, args.python)
    results = parse_benchmarks(output)
    generated_at = datetime.now(timezone.utc)

    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.svg_output.write_text(render_svg(results, generated_at), encoding="utf-8")
    args.json_output.write_text(
        json.dumps(
            {
                "generated_at": generated_at.isoformat(),
                "python": args.python,
                "command": [
                    args.python,
                    "-m",
                    "pytest",
                    "-q",
                    "-s",
                    "tests/test_native_stress_performance.py",
                ],
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.log_output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())