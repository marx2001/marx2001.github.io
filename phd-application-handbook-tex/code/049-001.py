#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 08 v5: direct Step 07 directory continuation, analytic boundaries, and WannierTools spectra
======================================================================

This script continues the Lieb8 workflow after Step 07. It does not replace the
strict full-BZ audit or the reliable Chern labels from Step 06/07.

Main tasks
----------
1. Add non-redundant mechanism features:
       M_plus  =  m_e + d1 + d2
       M_minus = -m_e + d1 + d2
       Delta_scale = sqrt(Delta1*Delta2)
       Delta_anisotropy = |log(Delta1/Delta2)|
   together with hopping-product/chirality descriptors.

2. Compare legacy and analytic compact logistic models using parameter-cluster
   group holdout, rather than random point splitting.

3. Refine the closest topology-trivial interpolation paths. Search for roots of
       M_plus = 0, M_minus = 0, t1 = 0, t2 = 0
   and verify each candidate by a dense direct-gap search and Chern calculations
   on both sides of the transition.

4. Calculate semi-infinite left/right edge spectral functions with the
   Sancho--Rubio iterative surface Green-function method used by WannierTools
   (the 1985 recursion implemented in surfgreen.f90/surfstat.f90).

5. Resolve spin-up/spin-down surface spectral weight and export data in a
   WannierTools-like dos.dat_l / dos.dat_r layout.

6. Independently diagonalize finite ribbons and check Ny = 30, 40, 60 width
   convergence. This is a validation channel, not the primary surface method.

Run
---
    python 08_Lieb8_Analytic_Mechanism_and_WannierTools_Edge_v5.py \
        --step07-dir outputs_step07_mechanism_boundary_consensus_...

Quick test
----------
    python 08_Lieb8_Analytic_Mechanism_and_WannierTools_Edge_v5.py \
        --step07-dir outputs_step07_mechanism_boundary_consensus_... --quick

Direct Step 07 handoff
----------------------
The script reads an already existing Step 07 output directory directly. It never
opens or extracts a ZIP archive. With no --step07-dir argument, it searches the
current project directory and the script directory for an extracted directory
named outputs_step07_mechanism_boundary_consensus_*. It reads the Step 07 output
registry, inherits the run tag, random seed, interpolation-pair count and ribbon
width, and writes a Step 08 output directory beside the Step 07 result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from scipy.linalg import eigh
from scipy.optimize import brentq, minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration
# =============================================================================

PHYS7 = ["m_e", "t1", "t2", "s1", "d1", "s2", "d2"]
RAW8 = ["e1", "e2", "t1", "t2", "r1", "r2", "r3", "r4"]
SPIN_INDICES = {"up": [0, 2], "down": [1, 3]}
SITE_1 = np.array([0.5, 0.0])
SITE_2 = np.array([0.0, 0.5])
EPS = 1.0e-12

LEGACY_COMPACT_FEATURES = [
    "delta_min",
    "delta_product",
    "delta_imbalance",
    "mass_product_XM",
    "mass_sign_change_XM",
    "crossing_fraction",
    "gap_to_delta_min",
]

# Non-redundant physical coordinates. delta_scale and delta_log_anisotropy are
# independent replacements for the redundant trio delta_min/product/imbalance.
ANALYTIC_COMPACT_FEATURES = [
    "delta_scale",
    "delta_log_anisotropy",
    "mass_boundary_min_abs",
    "mass_boundary_product",
    "abs_t_product",
    "crossing_fraction",
    "gap_to_delta_scale",
]

ANALYTIC_SIGNED_FEATURES = ANALYTIC_COMPACT_FEATURES + [
    "M_plus",
    "M_minus",
    "t_product",
]

DEFAULT_RANDOM_SEED = 20260711
DEFAULT_N_SPLITS = 20
DEFAULT_TEST_SIZE = 0.25
DEFAULT_INTERPOLATION_PAIRS = 4
DEFAULT_ROOT_GRID = 2001
DEFAULT_BZ_NK = 240
DEFAULT_CHERN_GRIDS = (21, 31)
DEFAULT_CHERN_SHIFTS = ((0.0, 0.0), (0.5, 0.5))
DEFAULT_RIBBON_WIDTHS = (30, 40, 60)
DEFAULT_RIBBON_NK = 241
DEFAULT_EDGE_CELLS = 3
DEFAULT_EDGE_THRESHOLD = 0.35

# Semi-infinite surface Green-function settings. The eta value is expressed in
# the normalized energy units of the Lieb8 Hamiltonian (E_* = 1).
DEFAULT_SURFACE_NK = 241
DEFAULT_SURFACE_ENERGY_POINTS = 601
DEFAULT_SURFACE_ETA = 0.003
DEFAULT_SURFACE_MAX_ITER = 100
DEFAULT_SURFACE_TOL = 1.0e-14
DEFAULT_SURFACE_LOG_CONTRAST = 80.0


@dataclass(frozen=True)
class Settings:
    random_seed: int
    n_splits: int
    test_size: float
    interpolation_pairs: int
    root_grid: int
    bz_nk: int
    chern_grids: tuple[int, ...]
    chern_shifts: tuple[tuple[float, float], ...]
    edge_method: str
    surface_nk: int
    surface_energy_points: int
    surface_eta: float
    surface_max_iter: int
    surface_tol: float
    surface_log_contrast: float
    surface_energy_window: float | None
    write_surface_text: bool
    ribbon_widths: tuple[int, ...]
    ribbon_nk: int
    edge_cells: int
    edge_threshold: float
    quick: bool


@dataclass(frozen=True)
class Step07Context:
    """Resolved, already-existing Step 07 output directory used by Step 08."""

    directory: Path
    source_parent: Path
    run_tag: str
    registry_file: Path
    files: Mapping[str, Path]
    configuration: Mapping[str, object]
    summary: Mapping[str, object]

    def require(self, key: str, fallback_pattern: str | None = None) -> Path:
        candidate = self.files.get(key)
        if candidate is not None and candidate.is_file():
            return candidate
        if fallback_pattern is not None:
            return find_unique(self.directory, fallback_pattern)
        raise FileNotFoundError(
            f"Step 07 registry does not contain required key {key!r}. "
            f"Registry: {self.registry_file}"
        )


# =============================================================================
# File discovery and utilities
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lieb8 Step 08 v5: direct continuation from an existing Step 07 "
            "output directory, with analytic boundaries and WannierTools-style "
            "semi-infinite edge spectra"
        )
    )
    parser.add_argument(
        "--step07-dir",
        type=Path,
        default=None,
        help=(
            "Existing extracted Step 07 output directory. When omitted, the "
            "newest compatible outputs_step07_mechanism_boundary_consensus_* "
            "directory is discovered automatically. ZIP files are not read."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default a run-tagged Step 08 directory is "
            "created beside the Step 07 output directory."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller grids for code/environment validation.",
    )
    parser.add_argument(
        "--skip-chern",
        action="store_true",
        help="Skip Chern calculations around refined roots.",
    )
    parser.add_argument(
        "--skip-edge",
        action="store_true",
        help="Skip all surface Green-function and finite-ribbon calculations.",
    )
    parser.add_argument(
        "--edge-method",
        choices=("green", "ribbon", "both"),
        default="both",
        help=(
            "green: semi-infinite iterative surface Green function; "
            "ribbon: finite-strip diagonalization; both: run both (default)."
        ),
    )
    parser.add_argument(
        "--skip-ribbon",
        action="store_true",
        help=(
            "When --edge-method=both, run only the surface Green-function part."
        ),
    )
    parser.add_argument(
        "--surface-eta",
        type=float,
        default=None,
        help="Imaginary broadening eta for the retarded surface Green function.",
    )
    parser.add_argument(
        "--surface-nk",
        type=int,
        default=None,
        help="Number of edge-momentum points for the surface spectral map.",
    )
    parser.add_argument(
        "--surface-energy-points",
        type=int,
        default=None,
        help="Number of energy points for the surface spectral map.",
    )
    parser.add_argument(
        "--surface-energy-window",
        type=float,
        default=None,
        help=(
            "Symmetric plotting window around E_F. By default it is selected "
            "from the audited indirect gap."
        ),
    )
    parser.add_argument(
        "--write-surface-text",
        action="store_true",
        help=(
            "Also export large WannierTools-like text files (dos.dat_l/r and "
            "spindos.dat_l/r). Compressed NPZ output is always written."
        ),
    )
    parser.add_argument(
        "--targeted-samples",
        type=int,
        default=1024,
        help=(
            "Generate this many new unit-sphere samples concentrated near "
            "M_plus=0, M_minus=0, t1=0, and t2=0. The production default is 1024; use 0 to skip."
        ),
    )
    parser.add_argument(
        "--targeted-width",
        type=float,
        default=0.12,
        help="Half-width of the targeted signed control parameter around each boundary.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the random seed inherited from the Step 07 run tag.",
    )
    parser.add_argument(
        "--interpolation-pairs",
        type=int,
        default=None,
        help="Override n_interpolation_pairs inherited from Step 07.",
    )
    parser.add_argument(
        "--ribbon-widths",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override finite-ribbon widths. By default Step 08 uses widths "
            "around Step 07 ribbon_ny, e.g. 30 40 60 for ribbon_ny=40."
        ),
    )
    return parser.parse_args()

def _looks_like_step07_output(directory: Path) -> bool:
    """Return True only for an extracted Step 07 output directory."""
    if not directory.is_dir():
        return False
    return bool(
        list(directory.glob("step07_00_output_file_registry__*.csv"))
        and list(directory.glob("step07_01_consensus_and_engineered_feature_master__*.csv"))
        and list(directory.glob("step07_04_nearest_topology_trivial_parameter_pairs__*.csv"))
    )


def _candidate_search_roots() -> list[Path]:
    roots = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    unique: list[Path] = []
    for root in roots:
        if root.exists() and root not in unique:
            unique.append(root)
    return unique


def _locate_step07_output_directory(path: Path) -> Path:
    """
    Resolve a directory-only Step 07 input.

    The supplied path may be the Step 07 output directory itself or a project
    directory containing exactly one compatible Step 07 output directory.
    Archive files are deliberately rejected.
    """
    selected = path.expanduser().resolve()
    if not selected.exists():
        raise FileNotFoundError(f"Step 07 path does not exist: {selected}")
    if selected.is_file():
        raise ValueError(
            f"Step 08 requires an extracted Step 07 output directory, not a file: {selected}. "
            "Do not pass the upload ZIP; place the extracted Step 07 directory in the project."
        )
    if _looks_like_step07_output(selected):
        return selected

    candidates: list[Path] = []
    for child in selected.glob("outputs_step07_mechanism_boundary_consensus_*"):
        if _looks_like_step07_output(child):
            candidates.append(child.resolve())
    for registry in selected.glob("*/step07_00_output_file_registry__*.csv"):
        if _looks_like_step07_output(registry.parent):
            candidates.append(registry.parent.resolve())
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No compatible extracted Step 07 output directory was found in {selected}. "
            "Expected outputs_step07_mechanism_boundary_consensus_* containing the "
            "Step 07 registry, analysis master, and nearest-pair table."
        )
    raise RuntimeError(
        "Multiple compatible Step 07 output directories were found. Pass the intended "
        "directory explicitly with --step07-dir:\n"
        + "\n".join(str(p) for p in candidates)
    )


def auto_find_step07_directory() -> Path:
    """Find the newest compatible extracted Step 07 directory; ignore all ZIPs."""
    candidates: list[Path] = []
    for root in _candidate_search_roots():
        if _looks_like_step07_output(root):
            candidates.append(root)
        for child in root.glob("outputs_step07_mechanism_boundary_consensus_*"):
            if _looks_like_step07_output(child):
                candidates.append(child.resolve())
        for registry in root.glob("*/step07_00_output_file_registry__*.csv"):
            if _looks_like_step07_output(registry.parent):
                candidates.append(registry.parent.resolve())
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise FileNotFoundError(
            "Cannot auto-detect an extracted Step 07 output directory. Place this Step 08 "
            "script in the project directory beside outputs_step07_mechanism_boundary_consensus_*, "
            "or pass that directory with --step07-dir. ZIP files are intentionally ignored."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def _extract_run_tag(directory: Path, master_file: Path) -> str:
    match = re.search(r"__([^/\\]+)\.csv$", master_file.name)
    if match:
        return match.group(1)
    prefix = "outputs_step07_mechanism_boundary_consensus_"
    if directory.name.startswith(prefix):
        return directory.name[len(prefix):]
    return directory.name


def _load_json_optional(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def resolve_step07_context(directory_path: Path | None) -> Step07Context:
    """Read Step 07 directly from its existing output directory."""
    directory = _locate_step07_output_directory(
        directory_path if directory_path is not None else auto_find_step07_directory()
    )

    registry_file = find_unique(directory, "step07_00_output_file_registry__*.csv")
    registry = pd.read_csv(registry_file)
    ensure_columns(registry, ["key", "filename"], "Step 07 output registry")
    duplicate_keys = registry[registry["key"].duplicated(keep=False)]["key"].tolist()
    if duplicate_keys:
        raise ValueError(f"Duplicate keys in Step 07 registry: {sorted(set(duplicate_keys))}")

    files: dict[str, Path] = {}
    for row in registry.itertuples(index=False):
        key = str(row.key)
        filename = Path(str(row.filename).replace("\\", "/")).name
        direct = directory / filename
        if direct.is_file():
            files[key] = direct
            continue
        matches = list(directory.rglob(filename))
        if len(matches) == 1:
            files[key] = matches[0]

    master_file = files.get("analysis_master") or find_unique(
        directory, "step07_01_consensus_and_engineered_feature_master__*.csv"
    )
    files["analysis_master"] = master_file

    config_file = files.get("config")
    if config_file is None:
        matches = list(directory.glob("step07_00_run_configuration__*.json"))
        config_file = matches[0] if len(matches) == 1 else None
    summary_file = files.get("summary")
    if summary_file is None:
        matches = list(directory.glob("step07_08_run_summary__*.json"))
        summary_file = matches[0] if len(matches) == 1 else None

    return Step07Context(
        directory=directory,
        source_parent=directory.parent,
        run_tag=_extract_run_tag(directory, master_file),
        registry_file=registry_file,
        files=files,
        configuration=_load_json_optional(config_file),
        summary=_load_json_optional(summary_file),
    )

def find_unique(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {directory}; "
            f"found {len(matches)}:\n" + "\n".join(str(x) for x in matches)
        )
    return matches[0]


def infer_seed(context: Step07Context) -> int:
    for source in (context.run_tag, context.directory.name):
        match = re.search(r"seed(\d+)", source)
        if match:
            return int(match.group(1))
    for payload in (context.configuration, context.summary):
        for key in ("random_seed", "seed"):
            value = payload.get(key)
            if value is not None:
                return int(value)
    return DEFAULT_RANDOM_SEED


def infer_interpolation_pairs(context: Step07Context) -> int:
    value = context.configuration.get("n_interpolation_pairs")
    if value is not None:
        return max(1, int(value))
    match = re.search(r"pairs(\d+)", context.run_tag)
    return int(match.group(1)) if match else DEFAULT_INTERPOLATION_PAIRS


def infer_ribbon_widths(context: Step07Context) -> tuple[int, ...]:
    value = context.configuration.get("ribbon_ny")
    if value is None:
        return DEFAULT_RIBBON_WIDTHS
    center = max(8, int(value))
    widths = (max(8, center - 10), center, center + 20)
    return tuple(sorted(set(widths)))


def write_input_compatibility_report(
    context: Step07Context,
    output_dir: Path,
    required_files: Mapping[str, Path],
) -> None:
    report = {
        "workflow_step": "step08",
        "input_mode": "existing_step07_directory_only",
        "step07_output_root": str(context.directory),
        "step07_run_tag": context.run_tag,
        "zip_reading_enabled": False,
        "step07_registry": str(context.registry_file),
        "required_files": {key: str(path) for key, path in required_files.items()},
        "required_files_all_exist": all(path.is_file() for path in required_files.values()),
        "inherited_configuration": dict(context.configuration),
    }
    write_json(output_dir / "step08_00_step07_compatibility_report.json", report)


def write_step08_registry(output_dir: Path, run_tag: str) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("step08_00_output_file_registry"):
            continue
        rows.append(
            {
                "filename": path.name,
                "relative_path": str(path.relative_to(output_dir)),
                "suffix": path.suffix.lower(),
                "size_bytes": int(path.stat().st_size),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / f"step08_00_output_file_registry__{run_tag}.csv", index=False
    )


def resolve_edge_selection_file(
    context: Step07Context,
    master: pd.DataFrame,
    pair_table: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, str]:
    """Use Step 07's ribbon selection or generate an equivalent fallback."""
    existing = context.files.get("ribbon_selection")
    if existing is not None and existing.is_file():
        selection = pd.read_csv(existing)
        ensure_columns(selection, ["role", "sample_id"], "Step 07 ribbon selection")
        known = set(master["sample_id"].astype(str))
        missing = [sid for sid in selection["sample_id"].astype(str) if sid not in known]
        if missing:
            raise KeyError(
                f"Step 07 ribbon selection contains sample IDs absent from the master: {missing}"
            )
        return existing, "step07_registry"

    ensure_columns(
        pair_table,
        ["topology_sample_id", "trivial_sample_id", "standardized_distance"],
        "Step 07 nearest-pair table",
    )
    candidates = pair_table.copy()
    known = set(master["sample_id"].astype(str))
    candidates = candidates[
        candidates["topology_sample_id"].astype(str).isin(known)
        & candidates["trivial_sample_id"].astype(str).isin(known)
    ].copy()
    if candidates.empty:
        raise ValueError(
            "No valid topology-trivial pair is available for edge-state analysis."
        )
    if {"topology_consensus_group", "trivial_consensus_group"}.issubset(candidates.columns):
        preferred = candidates[
            (candidates["topology_consensus_group"] == "both_topo")
            & (candidates["trivial_consensus_group"] == "both_trivial")
        ]
        if not preferred.empty:
            candidates = preferred
    if "mutual_nearest_neighbor" in candidates.columns:
        mutual = candidates[candidates["mutual_nearest_neighbor"].astype(bool)]
        if not mutual.empty:
            candidates = mutual
    selected = candidates.sort_values("standardized_distance").iloc[0]
    generated = pd.DataFrame(
        [
            {
                "role": "typeII_QSH",
                "sample_id": str(selected["topology_sample_id"]),
                "pair_distance": float(selected["standardized_distance"]),
            },
            {
                "role": "trivial_insulator",
                "sample_id": str(selected["trivial_sample_id"]),
                "pair_distance": float(selected["standardized_distance"]),
            },
        ]
    )
    path = output_dir / "step08_00_generated_edge_pair_selection.csv"
    generated.to_csv(path, index=False)
    return path, "generated_from_step07_nearest_pairs"

def ensure_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(np.asarray(denominator, dtype=float), EPS)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("png", 320), ("pdf", None), ("svg", None)):
        kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path_without_suffix.with_suffix(f".{extension}"), **kwargs)


# =============================================================================
# Hamiltonian, bands, and Chern number
# =============================================================================


def phys7_to_raw8(vector: Sequence[float]) -> dict[str, float]:
    values = {name: float(value) for name, value in zip(PHYS7, vector)}
    return {
        **values,
        "e1": values["m_e"],
        "e2": -values["m_e"],
        "r1": 0.5 * (values["s1"] + values["d1"]),
        "r3": 0.5 * (values["s1"] - values["d1"]),
        "r2": 0.5 * (values["s2"] + values["d2"]),
        "r4": 0.5 * (values["s2"] - values["d2"]),
    }


def slerp(vector_a: Sequence[float], vector_b: Sequence[float], t: float) -> np.ndarray:
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle < 1.0e-12:
        vector = (1.0 - t) * a + t * b
        return vector / np.linalg.norm(vector)
    sine = np.sin(angle)
    return np.sin((1.0 - t) * angle) / sine * a + np.sin(t * angle) / sine * b


def h_lieb_atomic(kx: float, ky: float, params: pd.Series | dict) -> np.ndarray:
    e1, e2, t1, t2, r1, r2, r3, r4 = [float(params[key]) for key in RAW8]
    a = e1 + 2.0 * r1 * np.cos(ky) + 2.0 * r3 * np.cos(kx)
    b = e2 + 2.0 * r2 * np.cos(ky) + 2.0 * r4 * np.cos(kx)
    c = e2 + 2.0 * r2 * np.cos(kx) + 2.0 * r4 * np.cos(ky)
    d = e1 + 2.0 * r1 * np.cos(kx) + 2.0 * r3 * np.cos(ky)
    cp = np.cos(0.5 * (kx + ky))
    cm = np.cos(0.5 * (kx - ky))
    u = 2.0 * (-1j * t1 + t2) * cp + 2.0 * (1j * t1 + t2) * cm
    v = 2.0 * (-1j * t1 + t2) * cm + 2.0 * (1j * t1 + t2) * cp
    return np.array(
        [
            [a, 0.0, u, 0.0],
            [0.0, b, 0.0, v],
            [np.conj(u), 0.0, c, 0.0],
            [0.0, np.conj(v), 0.0, d],
        ],
        dtype=np.complex128,
    )


def h_periodic(kx: float, ky: float, params: pd.Series | dict) -> np.ndarray:
    phase_1 = np.exp(1j * (kx * SITE_1[0] + ky * SITE_1[1]))
    phase_2 = np.exp(1j * (kx * SITE_2[0] + ky * SITE_2[1]))
    gauge = np.diag([phase_1, phase_1, phase_2, phase_2]).astype(np.complex128)
    return gauge.conj().T @ h_lieb_atomic(kx, ky, params) @ gauge


def spin_block_periodic(
    kx: float, ky: float, params: pd.Series | dict, spin: str
) -> np.ndarray:
    indices = SPIN_INDICES[spin]
    full = h_periodic(kx, ky, params)
    return full[np.ix_(indices, indices)]


def band_arrays(kx: np.ndarray, ky: np.ndarray, params: pd.Series | dict) -> np.ndarray:
    e1, e2, t1, t2, r1, r2, r3, r4 = [float(params[key]) for key in RAW8]
    a = e1 + 2.0 * r1 * np.cos(ky) + 2.0 * r3 * np.cos(kx)
    b = e2 + 2.0 * r2 * np.cos(ky) + 2.0 * r4 * np.cos(kx)
    c = e2 + 2.0 * r2 * np.cos(kx) + 2.0 * r4 * np.cos(ky)
    d = e1 + 2.0 * r1 * np.cos(kx) + 2.0 * r3 * np.cos(ky)
    cp = np.cos(0.5 * (kx + ky))
    cm = np.cos(0.5 * (kx - ky))
    u_re = 2.0 * t2 * (cp + cm)
    u_im = 2.0 * t1 * (cm - cp)
    v_re = 2.0 * t2 * (cp + cm)
    v_im = 2.0 * t1 * (cp - cm)
    up_center = 0.5 * (a + c)
    down_center = 0.5 * (b + d)
    up_radius = np.sqrt((0.5 * (a - c)) ** 2 + u_re**2 + u_im**2)
    down_radius = np.sqrt((0.5 * (b - d)) ** 2 + v_re**2 + v_im**2)
    energies = np.stack(
        [
            up_center - up_radius,
            up_center + up_radius,
            down_center - down_radius,
            down_center + down_radius,
        ],
        axis=-1,
    )
    energies.sort(axis=-1)
    return energies


def periodic_bz_grid(nk: int) -> tuple[np.ndarray, int]:
    """Return a periodic square-BZ grid that explicitly contains k=0.

    ``np.linspace(-pi, pi, odd_nk, endpoint=False)`` misses zero.  That was the
    source of the false finite gap at X/Y in the earlier Step 08 output.  The
    effective mesh is therefore always even, so that the periodic grid contains
    -pi and 0 exactly.  +pi is symmetry-equivalent to -pi.
    """
    if nk < 4:
        raise ValueError(f"nk must be >= 4, got {nk}")
    effective_nk = int(nk if nk % 2 == 0 else nk + 1)
    values = -np.pi + 2.0 * np.pi * np.arange(effective_nk) / effective_nk
    if not np.any(np.isclose(values, 0.0, atol=1.0e-14)):
        raise RuntimeError("Internal error: periodic BZ grid does not contain k=0")
    return values.astype(float), effective_nk


def _middle_bands(kx: float, ky: float, params: pd.Series | dict) -> tuple[float, float]:
    energies = np.linalg.eigvalsh(h_lieb_atomic(float(kx), float(ky), params))
    return float(energies[1]), float(energies[2])


def _unique_k_seeds(seeds: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for kx, ky in seeds:
        point = (float(np.clip(kx, -np.pi, np.pi)), float(np.clip(ky, -np.pi, np.pi)))
        if not any(np.hypot(point[0] - old[0], point[1] - old[1]) < 1.0e-10 for old in unique):
            unique.append(point)
    return unique


def _refine_band_extremum(
    params: pd.Series | dict,
    seeds: Sequence[tuple[float, float]],
    mode: str,
) -> dict[str, float]:
    """Refine a direct-gap minimum, VBM, or CBM from several BZ seeds."""
    if mode not in {"direct", "vbm", "cbm"}:
        raise ValueError(mode)

    def objective(point: np.ndarray) -> float:
        e2, e3 = _middle_bands(float(point[0]), float(point[1]), params)
        if mode == "direct":
            return e3 - e2
        if mode == "vbm":
            return -e2
        return e3

    best_value = float("inf")
    best_point = (float("nan"), float("nan"))
    for seed in _unique_k_seeds(seeds):
        result = minimize(
            objective,
            np.asarray(seed, dtype=float),
            method="Powell",
            bounds=((-np.pi, np.pi), (-np.pi, np.pi)),
            options={"xtol": 1.0e-10, "ftol": 1.0e-12, "maxiter": 180},
        )
        point = np.asarray(result.x, dtype=float)
        value = float(objective(point))
        if np.isfinite(value) and value < best_value:
            best_value = value
            best_point = (float(point[0]), float(point[1]))

    # Always compare against the exact high-symmetry points.  This protects the
    # X/Y/M/Γ roots even if a local optimizer terminates on a neighboring point.
    high_symmetry = [
        (0.0, 0.0),
        (-np.pi, 0.0),
        (0.0, -np.pi),
        (-np.pi, -np.pi),
        (np.pi, 0.0),
        (0.0, np.pi),
        (np.pi, np.pi),
    ]
    for point in high_symmetry:
        value = float(objective(np.asarray(point, dtype=float)))
        if value < best_value:
            best_value = value
            best_point = point

    if mode == "vbm":
        physical_value = -best_value
    else:
        physical_value = best_value
    return {
        "value": float(physical_value),
        "kx": float(best_point[0]),
        "ky": float(best_point[1]),
    }


def dense_gap_scan(
    params: pd.Series | dict,
    nk: int,
    *,
    refine_local: bool = True,
    n_local_seeds: int = 8,
) -> dict[str, float]:
    """Full-BZ direct/indirect gap scan with exact high-symmetry coverage.

    The periodic mesh is forced to an even size, so Γ, X, Y and M are sampled
    exactly.  When ``refine_local`` is true, the best grid candidates are refined
    continuously in two dimensions.  The returned direct gap is therefore not
    limited by the discrete BZ mesh and can resolve an off-grid axis closing.
    """
    k_values, effective_nk = periodic_bz_grid(nk)
    kx, ky = np.meshgrid(k_values, k_values, indexing="ij")
    energies = band_arrays(kx, ky, params)
    direct = energies[..., 2] - energies[..., 1]
    direct_index = np.unravel_index(int(np.argmin(direct)), direct.shape)
    vbm_index = np.unravel_index(int(np.argmax(energies[..., 1])), direct.shape)
    cbm_index = np.unravel_index(int(np.argmin(energies[..., 2])), direct.shape)

    result: dict[str, float] = {
        "direct_gap": float(direct[direct_index]),
        "direct_kx": float(k_values[direct_index[0]]),
        "direct_ky": float(k_values[direct_index[1]]),
        "indirect_gap": float(energies[..., 2][cbm_index] - energies[..., 1][vbm_index]),
        "vbm": float(energies[..., 1][vbm_index]),
        "vbm_kx": float(k_values[vbm_index[0]]),
        "vbm_ky": float(k_values[vbm_index[1]]),
        "cbm": float(energies[..., 2][cbm_index]),
        "cbm_kx": float(k_values[cbm_index[0]]),
        "cbm_ky": float(k_values[cbm_index[1]]),
        "requested_nk": int(nk),
        "effective_nk": int(effective_nk),
        "contains_zero": 1,
        "local_refined": int(refine_local),
    }
    if not refine_local:
        return result

    high_symmetry = [
        (0.0, 0.0),
        (-np.pi, 0.0),
        (0.0, -np.pi),
        (-np.pi, -np.pi),
        (np.pi, 0.0),
        (0.0, np.pi),
        (np.pi, np.pi),
    ]

    def candidate_seeds(array: np.ndarray, smallest: bool) -> list[tuple[float, float]]:
        flat = array.ravel()
        count = min(max(1, n_local_seeds), flat.size)
        if smallest:
            indices = np.argpartition(flat, count - 1)[:count]
        else:
            indices = np.argpartition(-flat, count - 1)[:count]
        points = []
        for flat_index in indices:
            index = np.unravel_index(int(flat_index), array.shape)
            points.append((float(k_values[index[0]]), float(k_values[index[1]])))
        return points + high_symmetry

    direct_refined = _refine_band_extremum(
        params, candidate_seeds(direct, True), "direct"
    )
    vbm_refined = _refine_band_extremum(
        params, candidate_seeds(energies[..., 1], False), "vbm"
    )
    cbm_refined = _refine_band_extremum(
        params, candidate_seeds(energies[..., 2], True), "cbm"
    )

    result.update(
        {
            "direct_gap": float(max(0.0, direct_refined["value"])),
            "direct_kx": direct_refined["kx"],
            "direct_ky": direct_refined["ky"],
            "indirect_gap": float(cbm_refined["value"] - vbm_refined["value"]),
            "vbm": vbm_refined["value"],
            "vbm_kx": vbm_refined["kx"],
            "vbm_ky": vbm_refined["ky"],
            "cbm": cbm_refined["value"],
            "cbm_kx": cbm_refined["kx"],
            "cbm_ky": cbm_refined["ky"],
        }
    )
    return result

def normalized_link(value: complex, tolerance: float = 1.0e-10) -> complex | None:
    amplitude = abs(value)
    if amplitude < tolerance:
        return None
    return value / amplitude


def fukui_single_band(
    h_function: Callable[[float, float], np.ndarray],
    nk: int,
    shift: tuple[float, float],
) -> float:
    shift_x, shift_y = shift
    kxs = 2.0 * np.pi * (np.arange(nk) + shift_x) / nk
    kys = 2.0 * np.pi * (np.arange(nk) + shift_y) / nk
    vectors = np.empty((nk, nk, 2), dtype=np.complex128)
    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            _, eigenvectors = np.linalg.eigh(h_function(float(kx), float(ky)))
            vectors[ix, iy] = eigenvectors[:, 0]
    total_phase = 0.0
    for ix in range(nk):
        for iy in range(nk):
            vector = vectors[ix, iy]
            vector_x = vectors[(ix + 1) % nk, iy]
            vector_y = vectors[ix, (iy + 1) % nk]
            vector_xy = vectors[(ix + 1) % nk, (iy + 1) % nk]
            ux = normalized_link(np.vdot(vector, vector_x))
            uy = normalized_link(np.vdot(vector, vector_y))
            ux_y = normalized_link(np.vdot(vector_y, vector_xy))
            uy_x = normalized_link(np.vdot(vector_x, vector_xy))
            if any(value is None for value in (ux, uy, ux_y, uy_x)):
                return float("nan")
            total_phase += np.angle(ux * uy_x / (ux_y * uy))
    return float(total_phase / (2.0 * np.pi))


def fukui_total(
    params: pd.Series | dict,
    nk: int,
    shift: tuple[float, float],
) -> float:
    shift_x, shift_y = shift
    kxs = 2.0 * np.pi * (np.arange(nk) + shift_x) / nk
    kys = 2.0 * np.pi * (np.arange(nk) + shift_y) / nk
    vectors = np.empty((nk, nk, 4, 2), dtype=np.complex128)
    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            _, eigenvectors = np.linalg.eigh(h_periodic(float(kx), float(ky), params))
            vectors[ix, iy] = eigenvectors[:, :2]
    total_phase = 0.0
    for ix in range(nk):
        for iy in range(nk):
            vector = vectors[ix, iy]
            vector_x = vectors[(ix + 1) % nk, iy]
            vector_y = vectors[ix, (iy + 1) % nk]
            vector_xy = vectors[(ix + 1) % nk, (iy + 1) % nk]
            ux = normalized_link(np.linalg.det(vector.conj().T @ vector_x))
            uy = normalized_link(np.linalg.det(vector.conj().T @ vector_y))
            ux_y = normalized_link(np.linalg.det(vector_y.conj().T @ vector_xy))
            uy_x = normalized_link(np.linalg.det(vector_x.conj().T @ vector_xy))
            if any(value is None for value in (ux, uy, ux_y, uy_x)):
                return float("nan")
            total_phase += np.angle(ux * uy_x / (ux_y * uy))
    return float(total_phase / (2.0 * np.pi))


def reliable_chern(
    params: pd.Series | dict,
    grids: Sequence[int],
    shifts: Sequence[tuple[float, float]],
) -> dict[str, object]:
    tuples: list[tuple[int, int, int]] = []
    raw_rows = []
    for nk in grids:
        for shift in shifts:
            cup = fukui_single_band(
                lambda kx, ky: spin_block_periodic(kx, ky, params, "up"), nk, shift
            )
            cdn = fukui_single_band(
                lambda kx, ky: spin_block_periodic(kx, ky, params, "down"), nk, shift
            )
            ctot = fukui_total(params, nk, shift)
            raw_rows.append((nk, shift, cup, cdn, ctot))
            if not all(np.isfinite(value) for value in (cup, cdn, ctot)):
                continue
            integers = (int(np.rint(cup)), int(np.rint(cdn)), int(np.rint(ctot)))
            if (
                abs(cup - integers[0]) <= 0.08
                and abs(cdn - integers[1]) <= 0.08
                and abs(ctot - integers[2]) <= 0.08
                and abs(ctot - cup - cdn) <= 0.10
            ):
                tuples.append(integers)
    if len(tuples) < 2:
        return {
            "chern_reliable": 0,
            "chern_consensus_count": 0,
            "chern_up_int": np.nan,
            "chern_down_int": np.nan,
            "chern_total_int": np.nan,
            "phase_label": "chern_unresolved",
            "raw": raw_rows,
        }
    consensus, count = Counter(tuples).most_common(1)[0]
    cup, cdn, ctot = consensus
    if cup == -cdn and cup != 0 and ctot == 0:
        phase = "typeII_QSH"
    elif cup == 0 and cdn == 0 and ctot == 0:
        phase = "trivial_insulator"
    else:
        phase = "other_gapped_phase"
    return {
        "chern_reliable": int(count >= 2),
        "chern_consensus_count": int(count),
        "chern_up_int": int(cup),
        "chern_down_int": int(cdn),
        "chern_total_int": int(ctot),
        "phase_label": phase,
        "raw": raw_rows,
    }


# =============================================================================
# Mechanism features and grouped model comparison
# =============================================================================


def add_analytic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delta1 = np.asarray(out["spin_splitting_1"], dtype=float)
    delta2 = np.asarray(out["spin_splitting_2"], dtype=float)
    out["M_plus"] = out["m_e"] + out["d1"] + out["d2"]
    out["M_minus"] = -out["m_e"] + out["d1"] + out["d2"]
    out["mass_boundary_product"] = out["M_plus"] * out["M_minus"]
    out["mass_boundary_sign_change"] = (out["mass_boundary_product"] < 0.0).astype(int)
    out["mass_boundary_min_abs"] = np.minimum(np.abs(out["M_plus"]), np.abs(out["M_minus"]))
    out["mass_boundary_max_abs"] = np.maximum(np.abs(out["M_plus"]), np.abs(out["M_minus"]))
    out["delta_scale"] = np.sqrt(np.maximum(delta1 * delta2, 0.0))
    out["delta_log_anisotropy"] = np.abs(np.log((delta1 + EPS) / (delta2 + EPS)))
    out["t_product"] = out["t1"] * out["t2"]
    out["abs_t_product"] = np.abs(out["t_product"])
    out["t_chirality"] = np.sign(out["t_product"]).astype(int)
    out["gap_to_delta_scale"] = safe_divide(
        np.asarray(out["final_direct_gap"], dtype=float),
        np.asarray(out["delta_scale"], dtype=float),
    )
    out["distance_to_any_analytic_boundary"] = np.minimum.reduce(
        [
            np.abs(np.asarray(out["M_plus"], dtype=float)),
            np.abs(np.asarray(out["M_minus"], dtype=float)),
            np.abs(np.asarray(out["t1"], dtype=float)),
            np.abs(np.asarray(out["t2"], dtype=float)),
        ]
    )
    return out


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    result = {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
    }
    try:
        result["roc_auc"] = roc_auc_score(y_true, probability)
    except ValueError:
        result["roc_auc"] = float("nan")
    return {key: float(value) for key, value in result.items()}


def audit_exact_mass_product_rule(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Audit the empirical rule M_plus*M_minus > 0 on audited insulators.

    A perfect score on the current dataset is evidence for an exact model rule,
    but it is not by itself a proof. Targeted out-of-sample validation and an
    analytic two-band derivation are still required.
    """
    prediction = (np.asarray(df["mass_boundary_product"], dtype=float) > 0.0).astype(int)
    truth = np.asarray(df["target_typeII"], dtype=int)
    metrics = binary_metrics(truth, prediction.astype(float))
    metrics.update(
        {
            "rule": "typeII iff M_plus*M_minus > 0",
            "n_samples": int(len(df)),
            "n_mismatches": int(np.sum(prediction != truth)),
        }
    )
    result = df[[
        "sample_id", "phase_label", "target_typeII", "M_plus", "M_minus",
        "mass_boundary_product", "final_direct_gap", "final_indirect_gap"
    ]].copy()
    result["analytic_rule_prediction"] = prediction
    result["analytic_rule_mismatch"] = (prediction != truth).astype(int)
    result.to_csv(output_dir / "step08_02_exact_mass_product_rule_audit.csv", index=False)
    write_json(output_dir / "step08_02_exact_mass_product_rule_metrics.json", metrics)
    return metrics, result


def grouped_logistic_comparison(
    df: pd.DataFrame,
    settings: Settings,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_columns(df, ["target_typeII", "parameter_cluster"], "Step 07 master")
    feature_sets = {
        "legacy_compact": LEGACY_COMPACT_FEATURES,
        "analytic_compact": ANALYTIC_COMPACT_FEATURES,
        "analytic_signed": ANALYTIC_SIGNED_FEATURES,
    }
    for feature_names in feature_sets.values():
        ensure_columns(df, feature_names, "analytic feature master")

    y = np.asarray(df["target_typeII"], dtype=int)
    groups = np.asarray(df["parameter_cluster"])
    splitter = GroupShuffleSplit(
        n_splits=settings.n_splits,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    split_rows = []
    coefficient_rows = []
    for split_id, (train_index, test_index) in enumerate(
        splitter.split(df, y, groups=groups), start=1
    ):
        for model_name, feature_names in feature_sets.items():
            x = np.asarray(df[feature_names], dtype=float)
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=1.0,
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=settings.random_seed,
                        ),
                    ),
                ]
            )
            model.fit(x[train_index], y[train_index])
            probability = model.predict_proba(x[test_index])[:, 1]
            metrics = binary_metrics(y[test_index], probability)
            split_rows.append(
                {
                    "split_id": split_id,
                    "model": model_name,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "n_train_clusters": len(np.unique(groups[train_index])),
                    "n_test_clusters": len(np.unique(groups[test_index])),
                    **metrics,
                }
            )
            coefficients = model.named_steps["logistic"].coef_[0]
            for feature, coefficient in zip(feature_names, coefficients):
                coefficient_rows.append(
                    {
                        "split_id": split_id,
                        "model": model_name,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                    }
                )

    split_df = pd.DataFrame(split_rows)
    coefficient_df = pd.DataFrame(coefficient_rows)
    summary_df = (
        split_df.groupby("model")
        .agg(
            n_splits=("split_id", "count"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .reset_index()
    )
    split_df.to_csv(output_dir / "step08_02_group_holdout_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "step08_02_model_comparison_summary.csv", index=False)
    coefficient_df.to_csv(output_dir / "step08_02_logistic_coefficients_by_split.csv", index=False)

    coefficient_summary = (
        coefficient_df.groupby(["model", "feature"])
        .agg(
            coefficient_mean=("standardized_coefficient", "mean"),
            coefficient_std=("standardized_coefficient", "std"),
            positive_fraction=("standardized_coefficient", lambda x: float(np.mean(x > 0.0))),
        )
        .reset_index()
    )
    coefficient_summary.to_csv(
        output_dir / "step08_02_logistic_coefficient_summary.csv", index=False
    )
    return split_df, summary_df, coefficient_summary


def mass_boundary_statistics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    sign_plus = np.where(df["M_plus"] >= 0.0, "+", "-")
    sign_minus = np.where(df["M_minus"] >= 0.0, "+", "-")
    work = df.copy()
    work["mass_quadrant"] = [f"M+{a}, M-{b}" for a, b in zip(sign_plus, sign_minus)]
    work["hopping_quadrant"] = [
        f"t1{'+' if a >= 0 else '-'}, t2{'+' if b >= 0 else '-'}"
        for a, b in zip(work["t1"], work["t2"])
    ]
    rows = []
    for group_columns in (["mass_quadrant"], ["hopping_quadrant"], ["mass_quadrant", "hopping_quadrant"]):
        grouped = work.groupby(group_columns, dropna=False)
        for key, part in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            record = {column: value for column, value in zip(group_columns, key)}
            rows.append(
                {
                    "grouping": "+".join(group_columns),
                    **record,
                    "n_samples": int(len(part)),
                    "n_typeII": int(part["target_typeII"].sum()),
                    "typeII_fraction": float(part["target_typeII"].mean()),
                    "mean_direct_gap": float(part["final_direct_gap"].mean()),
                    "mean_delta_scale": float(part["delta_scale"].mean()),
                    "mean_boundary_distance": float(part["mass_boundary_min_abs"].mean()),
                }
            )
    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "step08_03_mass_hopping_quadrant_statistics.csv", index=False)
    return stats


def plot_mechanism_overview(df: pd.DataFrame, figures_dir: Path) -> None:
    phase_numeric = np.asarray(df["target_typeII"], dtype=int)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    trivial = phase_numeric == 0
    topo = phase_numeric == 1
    ax.scatter(
        df.loc[trivial, "M_minus"],
        df.loc[trivial, "M_plus"],
        s=18,
        alpha=0.55,
        label="trivial insulator",
    )
    ax.scatter(
        df.loc[topo, "M_minus"],
        df.loc[topo, "M_plus"],
        s=20,
        alpha=0.70,
        label="type-II QSH",
    )
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"$M_-=-m_e+d_1+d_2$")
    ax.set_ylabel(r"$M_+=m_e+d_1+d_2$")
    ax.set_title("Two analytic X/Y mass branches")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures_dir / "step08_03_Mplus_Mminus_phase_scatter")
    plt.close(fig)

    # Non-redundant splitting coordinates.
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(
        df.loc[trivial, "delta_scale"],
        df.loc[trivial, "delta_log_anisotropy"],
        s=18,
        alpha=0.55,
        label="trivial insulator",
    )
    ax.scatter(
        df.loc[topo, "delta_scale"],
        df.loc[topo, "delta_log_anisotropy"],
        s=20,
        alpha=0.70,
        label="type-II QSH",
    )
    ax.set_xlabel(r"$\bar\Delta=\sqrt{\Delta_1\Delta_2}$")
    ax.set_ylabel(r"$A_\Delta=|\ln(\Delta_1/\Delta_2)|$")
    ax.set_title("Independent splitting strength and anisotropy")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures_dir / "step08_03_delta_scale_anisotropy_phase_scatter")
    plt.close(fig)

    # Phase fraction as a function of distance to the closest mass boundary.
    quantiles = np.unique(
        np.quantile(df["mass_boundary_min_abs"], np.linspace(0.0, 1.0, 9))
    )
    if len(quantiles) >= 3:
        binned = df.copy()
        binned["boundary_bin"] = pd.cut(
            binned["mass_boundary_min_abs"], bins=quantiles, include_lowest=True
        )
        profile = (
            binned.groupby("boundary_bin", observed=True)
            .agg(
                boundary_distance=("mass_boundary_min_abs", "mean"),
                topology_fraction=("target_typeII", "mean"),
                n_samples=("target_typeII", "size"),
            )
            .reset_index(drop=True)
        )
        profile.to_csv(
            figures_dir.parent / "step08_03_topology_fraction_vs_mass_boundary_distance.csv",
            index=False,
        )
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        ax.plot(
            profile["boundary_distance"],
            profile["topology_fraction"],
            marker="o",
            linewidth=1.5,
        )
        for row in profile.itertuples(index=False):
            ax.annotate(
                f"n={row.n_samples}",
                (row.boundary_distance, row.topology_fraction),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel(r"$\min(|M_+|,|M_-|)$")
        ax.set_ylabel("type-II QSH fraction")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("Topology probability versus analytic mass-boundary distance")
        fig.tight_layout()
        save_figure(fig, figures_dir / "step08_03_topology_fraction_vs_boundary_distance")
        plt.close(fig)


# =============================================================================
# Analytic root refinement along SLERP paths
# =============================================================================


def find_all_roots(function: Callable[[float], float], n_grid: int) -> list[float]:
    t_grid = np.linspace(0.0, 1.0, n_grid)
    values = np.asarray([function(float(t)) for t in t_grid], dtype=float)
    roots: list[float] = []
    for i in range(len(t_grid) - 1):
        a, b = float(t_grid[i]), float(t_grid[i + 1])
        fa, fb = float(values[i]), float(values[i + 1])
        if abs(fa) < 1.0e-12:
            roots.append(a)
        if fa * fb < 0.0:
            roots.append(float(brentq(function, a, b, xtol=1.0e-13, rtol=1.0e-13)))
    if abs(values[-1]) < 1.0e-12:
        roots.append(1.0)
    unique = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 1.0e-8:
            unique.append(root)
    return unique


def exact_four_band_gap(kx: float, ky: float, params: pd.Series | dict) -> float:
    energies = np.linalg.eigvalsh(h_lieb_atomic(kx, ky, params))
    return float(energies[2] - energies[1])


def spin_gap(kx: float, ky: float, params: pd.Series | dict, spin: str) -> float:
    energies = np.linalg.eigvalsh(spin_block_periodic(kx, ky, params, spin))
    return float(energies[1] - energies[0])


def axis_minimum(params: pd.Series | dict, spin: str, axis: str) -> dict[str, float]:
    if axis not in {"kx", "ky"}:
        raise ValueError(axis)

    def objective(q: float) -> float:
        kx, ky = (q, 0.0) if axis == "kx" else (0.0, q)
        return spin_gap(kx, ky, params, spin)

    q_grid = np.linspace(-np.pi, np.pi, 2001)
    values = np.asarray([objective(float(q)) for q in q_grid])
    candidate_indices = np.argsort(values)[:8]
    best = {"gap": float("inf"), "q": float("nan")}
    step = float(q_grid[1] - q_grid[0])
    for index in candidate_indices:
        center = float(q_grid[index])
        lower = max(-np.pi, center - 3.0 * step)
        upper = min(np.pi, center + 3.0 * step)
        result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
        if result.fun < best["gap"]:
            best = {"gap": float(result.fun), "q": float(result.x)}
    kx, ky = (best["q"], 0.0) if axis == "kx" else (0.0, best["q"])
    best.update({"kx": float(kx), "ky": float(ky), "spin": spin, "axis": axis})
    return best


def candidate_gap_location(boundary: str, params: pd.Series | dict) -> dict[str, object]:
    if boundary in {"M_plus", "M_minus"}:
        candidates = []
        for label, (kx, ky) in {
            "X": (np.pi, 0.0),
            "Y": (0.0, np.pi),
            "minus_X": (-np.pi, 0.0),
            "minus_Y": (0.0, -np.pi),
        }.items():
            candidates.append(
                {
                    "location_label": label,
                    "kx": float(kx),
                    "ky": float(ky),
                    "gap": exact_four_band_gap(kx, ky, params),
                    "spin": "up" if boundary == "M_plus" else "down",
                    "axis": "high_symmetry",
                }
            )
        return min(candidates, key=lambda row: row["gap"])

    # t1/t2 zeros can create axis closings. Test both spin blocks and axes.
    candidates = [
        axis_minimum(params, spin, axis)
        for spin in ("up", "down")
        for axis in ("kx", "ky")
    ]
    best = min(candidates, key=lambda row: row["gap"])
    best["location_label"] = f"{best['spin']}_{best['axis']}_axis"
    best["gap"] = exact_four_band_gap(best["kx"], best["ky"], params)
    return best


def refine_transition_paths(
    master: pd.DataFrame,
    pair_table: pd.DataFrame,
    settings: Settings,
    output_dir: Path,
    figures_dir: Path,
    skip_chern: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve analytic roots and produce gap profiles that pass through them.

    The previous implementation used an odd periodic mesh and plotted only a
    uniform interpolation grid.  Consequently X/Y and off-grid axis closings
    could be absent from the plotted curve.  This version:

    * uses the even, high-symmetry-complete BZ scan;
    * performs continuous two-dimensional local gap refinement;
    * inserts every analytic root and logarithmically spaced neighboring points
      into the path profile;
    * distinguishes a scalar-parameter zero from a verified bulk-gap closing;
    * searches for insulating points on both sides before accepting Chern data.
    """
    ensure_columns(pair_table, ["topology_sample_id", "trivial_sample_id"], "pair table")
    master_index = master.set_index("sample_id")
    selected = pair_table.nsmallest(settings.interpolation_pairs, "standardized_distance")
    root_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    closing_tolerance = 2.0e-6 if not settings.quick else 2.0e-5

    for pair_id, pair in enumerate(selected.itertuples(index=False), start=1):
        topo = master_index.loc[pair.topology_sample_id]
        trivial = master_index.loc[pair.trivial_sample_id]
        vector_a = topo[PHYS7].to_numpy(dtype=float)
        vector_b = trivial[PHYS7].to_numpy(dtype=float)

        def vector_at(t: float) -> np.ndarray:
            return slerp(vector_a, vector_b, float(t))

        scalar_functions = {
            "M_plus": lambda t: float(vector_at(t)[0] + vector_at(t)[4] + vector_at(t)[6]),
            "M_minus": lambda t: float(-vector_at(t)[0] + vector_at(t)[4] + vector_at(t)[6]),
            "t1": lambda t: float(vector_at(t)[1]),
            "t2": lambda t: float(vector_at(t)[2]),
        }

        root_specs: list[dict[str, object]] = []
        for boundary, function in scalar_functions.items():
            for root in find_all_roots(function, settings.root_grid):
                if root <= 1.0e-8 or root >= 1.0 - 1.0e-8:
                    continue
                vector = vector_at(root)
                params = phys7_to_raw8(vector)
                location = candidate_gap_location(boundary, params)
                global_gap = dense_gap_scan(params, settings.bz_nk, refine_local=True)
                verified_gap = min(float(location["gap"]), float(global_gap["direct_gap"]))
                true_closing = bool(verified_gap <= closing_tolerance)
                root_specs.append(
                    {
                        "boundary": boundary,
                        "function": function,
                        "t_critical": float(root),
                        "vector": vector,
                        "params": params,
                        "location": location,
                        "global_gap": global_gap,
                        "verified_gap": float(verified_gap),
                        "true_gap_closing": true_closing,
                    }
                )

        all_root_t = sorted(float(spec["t_critical"]) for spec in root_specs)
        for root_counter, spec in enumerate(root_specs, start=1):
            root = float(spec["t_critical"])
            vector = np.asarray(spec["vector"], dtype=float)
            location = spec["location"]
            global_gap = spec["global_gap"]
            record: dict[str, object] = {
                "pair_id": pair_id,
                "root_id_within_pair": root_counter,
                "topology_sample_id": pair.topology_sample_id,
                "trivial_sample_id": pair.trivial_sample_id,
                "boundary": spec["boundary"],
                "t_critical": root,
                "boundary_value": float(spec["function"](root)),
                **{name: float(value) for name, value in zip(PHYS7, vector)},
                "M_plus": float(vector[0] + vector[4] + vector[6]),
                "M_minus": float(-vector[0] + vector[4] + vector[6]),
                "candidate_location": location["location_label"],
                "candidate_kx": float(location["kx"]),
                "candidate_ky": float(location["ky"]),
                "candidate_direct_gap": float(location["gap"]),
                "global_direct_gap": float(global_gap["direct_gap"]),
                "global_direct_kx": float(global_gap["direct_kx"]),
                "global_direct_ky": float(global_gap["direct_ky"]),
                "global_indirect_gap": float(global_gap["indirect_gap"]),
                "verified_direct_gap": float(spec["verified_gap"]),
                "true_gap_closing": int(spec["true_gap_closing"]),
                "gap_closing_tolerance": closing_tolerance,
                "effective_bz_nk": int(global_gap["effective_nk"]),
                "high_symmetry_complete_grid": int(global_gap["contains_zero"]),
            }

            if not skip_chern and bool(spec["true_gap_closing"]):
                # Use the first offset that is both insulating and has a reliable
                # Chern consensus.  The offset is restricted so that it cannot
                # cross another analytic root on the same interpolation path.
                distances = [abs(root - other) for other in all_root_t if abs(root - other) > 1.0e-10]
                root_separation_cap = 0.35 * min(distances) if distances else 0.08
                for side_name, direction in (("before", -1.0), ("after", 1.0)):
                    accepted = False
                    minimum_chern_side_gap = 0.004 if settings.quick else 0.006
                    # Start far enough from the singular point for a finite Fukui
                    # mesh to resolve the concentrated Berry curvature.  Accepting
                    # the first 0.001-offset consensus can falsely return C=0 even
                    # though all coarse meshes agree.  The largest safe insulating
                    # offset is therefore tested first.
                    for trial_offset in (0.05, 0.025, 0.012, 0.006, 0.003, 0.001):
                        offset = min(trial_offset, root_separation_cap)
                        if offset <= 1.0e-6:
                            continue
                        side_t = float(np.clip(root + direction * offset, 0.0, 1.0))
                        if abs(side_t - root) < 1.0e-8:
                            continue
                        side_params = phys7_to_raw8(vector_at(side_t))
                        side_gap = dense_gap_scan(
                            side_params,
                            max(120, settings.bz_nk // 2),
                            refine_local=True,
                        )
                        if (
                            side_gap["direct_gap"] <= minimum_chern_side_gap
                            or side_gap["indirect_gap"] <= minimum_chern_side_gap
                        ):
                            continue
                        chern = reliable_chern(
                            side_params, settings.chern_grids, settings.chern_shifts
                        )
                        if not bool(chern["chern_reliable"]):
                            continue
                        record[f"{side_name}_t"] = side_t
                        record[f"{side_name}_offset"] = abs(side_t - root)
                        record[f"{side_name}_direct_gap"] = side_gap["direct_gap"]
                        record[f"{side_name}_indirect_gap"] = side_gap["indirect_gap"]
                        for key in (
                            "chern_reliable",
                            "chern_consensus_count",
                            "chern_up_int",
                            "chern_down_int",
                            "chern_total_int",
                            "phase_label",
                        ):
                            record[f"{side_name}_{key}"] = chern[key]
                        accepted = True
                        break
                    record[f"{side_name}_verified"] = int(accepted)
            root_rows.append(record)

        # Include the exact roots and nearby logarithmic offsets in the plotted
        # path.  This forces the profile to display the true closing rather than
        # interpolate over it between two coarse t points.
        base_count = 201 if not settings.quick else 81
        t_values = set(float(value) for value in np.linspace(0.0, 1.0, base_count))
        neighborhood = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2)
        for spec in root_specs:
            root = float(spec["t_critical"])
            t_values.add(root)
            for delta in neighborhood:
                t_values.add(float(np.clip(root - delta, 0.0, 1.0)))
                t_values.add(float(np.clip(root + delta, 0.0, 1.0)))

        root_by_t = {round(float(spec["t_critical"]), 13): spec for spec in root_specs}
        for t in sorted(t_values):
            vector = vector_at(t)
            params = phys7_to_raw8(vector)
            matched = root_by_t.get(round(float(t), 13))
            is_analytic_root = int(matched is not None)
            root_boundary = ""
            if matched is not None:
                # Reuse the already refined full-BZ result at the exact root.
                gap = dict(matched["global_gap"])
                gap["direct_gap"] = min(
                    float(gap["direct_gap"]),
                    float(matched["location"]["gap"]),
                )
                gap["direct_kx"] = float(matched["location"]["kx"])
                gap["direct_ky"] = float(matched["location"]["ky"])
                root_boundary = str(matched["boundary"])
            else:
                # A high-symmetry-complete even grid is sufficient away from a
                # transition.  Continuous refinement is restricted to a narrow
                # neighborhood of analytic roots to keep Step 08 tractable.
                near_root = any(abs(float(t) - root_t) <= 0.035 for root_t in all_root_t)
                gap = dense_gap_scan(
                    params,
                    settings.bz_nk,
                    refine_local=near_root,
                    n_local_seeds=4,
                )
            path_rows.append(
                {
                    "pair_id": pair_id,
                    "topology_sample_id": pair.topology_sample_id,
                    "trivial_sample_id": pair.trivial_sample_id,
                    "t": float(t),
                    **{name: float(value) for name, value in zip(PHYS7, vector)},
                    "M_plus": float(vector[0] + vector[4] + vector[6]),
                    "M_minus": float(-vector[0] + vector[4] + vector[6]),
                    "is_analytic_root": is_analytic_root,
                    "root_boundary": root_boundary,
                    **gap,
                }
            )

    root_df = pd.DataFrame(root_rows)
    path_df = pd.DataFrame(path_rows).sort_values(["pair_id", "t"]).reset_index(drop=True)
    root_df.to_csv(output_dir / "step08_04_refined_analytic_transition_roots.csv", index=False)
    path_df.to_csv(output_dir / "step08_04_dense_transition_path_profiles.csv", index=False)

    for pair_id, part in path_df.groupby("pair_id"):
        roots = root_df[root_df["pair_id"] == pair_id] if len(root_df) else pd.DataFrame()
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.4), sharex=True)
        axes[0].plot(part["t"], part["direct_gap"], label="direct gap", linewidth=1.4)
        axes[0].plot(part["t"], part["indirect_gap"], label="indirect gap", linewidth=1.2)
        axes[0].axhline(0.0, linewidth=0.8)
        axes[0].set_ylabel("Gap")
        axes[0].legend(frameon=False)
        axes[1].plot(part["t"], part["M_plus"], label=r"$M_+$")
        axes[1].plot(part["t"], part["M_minus"], label=r"$M_-$")
        axes[1].plot(part["t"], part["t1"], label=r"$t_1$", linestyle="--")
        axes[1].plot(part["t"], part["t2"], label=r"$t_2$", linestyle="--")
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_xlabel("SLERP interpolation coordinate t")
        axes[1].set_ylabel("Analytic control parameter")
        axes[1].legend(frameon=False, ncol=4, fontsize=8)
        if len(roots):
            for root in roots.itertuples(index=False):
                style = "-" if int(root.true_gap_closing) else ":"
                width = 1.0 if int(root.true_gap_closing) else 0.7
                for ax in axes:
                    ax.axvline(root.t_critical, linewidth=width, linestyle=style)
                axes[0].scatter(
                    [root.t_critical],
                    [root.verified_direct_gap],
                    marker="o" if int(root.true_gap_closing) else "x",
                    s=28,
                    zorder=5,
                )
        fig.suptitle(f"Pair {pair_id}: full-BZ gap closing and analytic roots")
        fig.tight_layout()
        save_figure(fig, figures_dir / f"step08_04_pair_{pair_id:02d}_refined_transition")
        plt.close(fig)

    return root_df, path_df


# =============================================================================
# Targeted out-of-sample validation near analytic boundaries
# =============================================================================


def sample_unit_sphere_with_linear_control(
    rng: np.random.Generator,
    normal: np.ndarray,
    target_value: float,
) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    n_hat = normal / norm
    if abs(target_value) >= norm:
        raise ValueError("target_value lies outside the unit-sphere range")
    tangent = rng.normal(size=7)
    tangent = tangent - np.dot(tangent, n_hat) * n_hat
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1.0e-12:
        return sample_unit_sphere_with_linear_control(rng, normal, target_value)
    tangent /= tangent_norm
    parallel_coefficient = target_value / norm
    vector = math.sqrt(max(0.0, 1.0 - parallel_coefficient**2)) * tangent + parallel_coefficient * n_hat
    return vector / np.linalg.norm(vector)


def targeted_boundary_validation(
    n_samples: int,
    control_width: float,
    settings: Settings,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if n_samples <= 0:
        return pd.DataFrame(), {}
    rng = np.random.default_rng(settings.random_seed + 808)
    families = {
        "M_plus": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        "M_minus": np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        "t1": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "t2": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    }
    coarse_nk = 50 if settings.quick else 100
    fine_nk = 80 if settings.quick else 180
    gap_accept = 0.012 if settings.quick else 0.020
    convergence_tolerance = 0.012 if settings.quick else 0.006
    rows = []
    for sample_index in range(n_samples):
        family_name = list(families)[sample_index % len(families)]
        target = float(rng.uniform(-control_width, control_width))
        vector = sample_unit_sphere_with_linear_control(rng, families[family_name], target)
        params = phys7_to_raw8(vector)
        coarse = dense_gap_scan(params, coarse_nk, refine_local=False)
        fine_grid = dense_gap_scan(params, fine_nk, refine_local=False)
        # Only candidates that look insulating on the even grids need the more
        # expensive continuous refinement.  The refined result is authoritative.
        if fine_grid["direct_gap"] > 0.5 * gap_accept and fine_grid["indirect_gap"] > 0.5 * gap_accept:
            fine = dense_gap_scan(params, fine_nk, refine_local=True)
        else:
            fine = fine_grid
        high_confidence = (
            fine["direct_gap"] > gap_accept
            and fine["indirect_gap"] > gap_accept
            and abs(fine["direct_gap"] - coarse["direct_gap"]) < convergence_tolerance
            and abs(fine["indirect_gap"] - coarse["indirect_gap"]) < convergence_tolerance
        )
        M_plus = float(vector[0] + vector[4] + vector[6])
        M_minus = float(-vector[0] + vector[4] + vector[6])
        product = M_plus * M_minus
        row = {
            "targeted_sample_id": f"targeted_{sample_index:06d}",
            "sampling_family": family_name,
            "requested_control_value": target,
            **{name: float(value) for name, value in zip(PHYS7, vector)},
            **{name: float(params[name]) for name in RAW8},
            "M_plus": M_plus,
            "M_minus": M_minus,
            "mass_boundary_product": product,
            "analytic_rule_prediction": int(product > 0.0),
            "coarse_direct_gap": coarse["direct_gap"],
            "coarse_indirect_gap": coarse["indirect_gap"],
            "fine_grid_direct_gap": fine_grid["direct_gap"],
            "fine_grid_indirect_gap": fine_grid["indirect_gap"],
            "fine_direct_gap": fine["direct_gap"],
            "fine_indirect_gap": fine["indirect_gap"],
            "fine_effective_nk": fine["effective_nk"],
            "fine_local_refined": fine["local_refined"],
            "gap_high_confidence": int(high_confidence),
        }
        if high_confidence:
            chern = reliable_chern(params, settings.chern_grids, settings.chern_shifts)
            for key in (
                "chern_reliable", "chern_consensus_count", "chern_up_int",
                "chern_down_int", "chern_total_int", "phase_label"
            ):
                row[key] = chern[key]
            if chern["chern_reliable"] and chern["phase_label"] in {"typeII_QSH", "trivial_insulator"}:
                row["chern_target_typeII"] = int(chern["phase_label"] == "typeII_QSH")
                row["analytic_rule_mismatch"] = int(
                    row["analytic_rule_prediction"] != row["chern_target_typeII"]
                )
            else:
                row["chern_target_typeII"] = np.nan
                row["analytic_rule_mismatch"] = np.nan
        rows.append(row)

    targeted = pd.DataFrame(rows)
    targeted.to_csv(output_dir / "step08_06_targeted_boundary_validation_samples.csv", index=False)
    accepted = targeted[
        (targeted["gap_high_confidence"] == 1)
        & (targeted.get("chern_reliable", 0) == 1)
        & targeted.get("chern_target_typeII", pd.Series(index=targeted.index, dtype=float)).notna()
    ]
    summary = {
        "n_generated": int(len(targeted)),
        "n_gap_high_confidence": int(targeted["gap_high_confidence"].sum()),
        "n_chern_labeled_typeII_or_trivial": int(len(accepted)),
        "n_rule_mismatches": int(accepted["analytic_rule_mismatch"].sum()) if len(accepted) else 0,
        "rule_accuracy": float(1.0 - accepted["analytic_rule_mismatch"].mean()) if len(accepted) else None,
        "sampling_families": list(families),
        "control_width": control_width,
        "gap_acceptance_threshold": gap_accept,
    }
    write_json(output_dir / "step08_06_targeted_boundary_validation_summary.json", summary)
    return targeted, summary


# =============================================================================
# Semi-infinite surface Green function and finite-ribbon validation
# =============================================================================


def ribbon_blocks(kx: float, params: pd.Series | dict) -> tuple[np.ndarray, np.ndarray]:
    """Return H00(kx) and H01(kx) for a y-normal principal-layer geometry.

    The periodic Hamiltonian is reconstructed as

        H(kx, ky) = H00 + H01 exp(+i ky) + H01^dagger exp(-i ky).

    This convention is passed directly to the iterative surface Green-function
    algorithm and to the finite-ribbon Hamiltonian.
    """
    e1, e2, t1, t2, r1, r2, r3, r4 = [float(params[key]) for key in RAW8]
    coupling_minus = t2 - 1j * t1
    coupling_plus = t2 + 1j * t1
    onsite = np.zeros((4, 4), dtype=np.complex128)
    hopping = np.zeros((4, 4), dtype=np.complex128)

    onsite[0, 0] = e1 + 2.0 * r3 * np.cos(kx)
    onsite[1, 1] = e2 + 2.0 * r4 * np.cos(kx)
    onsite[2, 2] = e2 + 2.0 * r2 * np.cos(kx)
    onsite[3, 3] = e1 + 2.0 * r1 * np.cos(kx)

    hopping[0, 0] = r1
    hopping[1, 1] = r2
    hopping[2, 2] = r4
    hopping[3, 3] = r3

    onsite[0, 2] = coupling_plus + coupling_minus * np.exp(-1j * kx)
    onsite[2, 0] = np.conj(onsite[0, 2])
    hopping[0, 2] = coupling_minus + coupling_plus * np.exp(-1j * kx)

    onsite[1, 3] = coupling_minus + coupling_plus * np.exp(-1j * kx)
    onsite[3, 1] = np.conj(onsite[1, 3])
    hopping[1, 3] = coupling_plus + coupling_minus * np.exp(-1j * kx)
    return onsite, hopping


def ribbon_reconstruction_error(params: pd.Series | dict, random_seed: int) -> float:
    """Verify that H00/H01 reproduce the analytic 2D Bloch Hamiltonian."""
    rng = np.random.default_rng(random_seed)
    maximum = 0.0
    for _ in range(40):
        kx = float(rng.uniform(-np.pi, np.pi))
        ky = float(rng.uniform(-np.pi, np.pi))
        onsite, hopping = ribbon_blocks(kx, params)
        reconstructed = (
            onsite
            + hopping * np.exp(1j * ky)
            + hopping.conj().T * np.exp(-1j * ky)
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(reconstructed - h_periodic(kx, ky, params)))),
        )
    return maximum


def surface_green_1985(
    energy: float,
    h00: np.ndarray,
    h01: np.ndarray,
    eta: float,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Iterative left/right surface Green functions (Sancho--Rubio recursion).

    This is a direct Python implementation of the recursion used in the
    WannierTools ``surfgreen_1985`` routine.  It returns the retarded Green
    functions of the left surface, right surface and renormalized bulk
    principal layer.
    """
    if eta <= 0.0:
        raise ValueError("surface eta must be positive")
    dimension = h00.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    z_identity = complex(float(energy), float(eta)) * identity

    epsilon_bulk = np.array(h00, dtype=np.complex128, copy=True)
    epsilon_left = np.array(h00, dtype=np.complex128, copy=True)
    epsilon_right = np.array(h00, dtype=np.complex128, copy=True)
    alpha = np.array(h01, dtype=np.complex128, copy=True)
    beta = alpha.conj().T
    residual = float("inf")

    for iteration in range(1, max_iter + 1):
        g0 = np.linalg.inv(z_identity - epsilon_bulk)
        alpha_g = alpha @ g0
        beta_g = beta @ g0

        alpha_g_beta = alpha_g @ beta
        beta_g_alpha = beta_g @ alpha
        epsilon_left = epsilon_left + alpha_g_beta
        epsilon_right = epsilon_right + beta_g_alpha
        epsilon_bulk = epsilon_bulk + alpha_g_beta + beta_g_alpha

        alpha_new = alpha_g @ alpha
        beta_new = beta_g @ beta
        residual = max(
            float(np.linalg.norm(alpha_new, ord="fro")),
            float(np.linalg.norm(beta_new, ord="fro")),
        )
        alpha, beta = alpha_new, beta_new
        if residual <= tolerance:
            break

    g_left = np.linalg.inv(z_identity - epsilon_left)
    g_right = np.linalg.inv(z_identity - epsilon_right)
    g_bulk = np.linalg.inv(z_identity - epsilon_bulk)
    return g_left, g_right, g_bulk, iteration, residual


def _spectral_trace(green: np.ndarray, indices: Sequence[int] | None = None) -> float:
    if indices is None:
        diagonal = np.diag(green)
    else:
        diagonal = np.diag(green)[list(indices)]
    value = -float(np.imag(np.sum(diagonal))) / np.pi
    return max(value, 0.0)


def select_surface_energy_window(sample: pd.Series, requested: float | None) -> float:
    if requested is not None:
        if requested <= 0.0:
            raise ValueError("--surface-energy-window must be positive")
        return float(requested)
    gap = max(float(sample["final_indirect_gap"]), 0.0)
    # Wide enough to expose both bulk continua and the in-gap edge branch, but
    # not so wide that the edge signal becomes visually compressed.
    return max(0.22, min(0.60, 2.2 * gap + 0.08))


def calculate_surface_spectral_function(
    sample: pd.Series,
    nk: int,
    energy_points: int,
    eta: float,
    tolerance: float,
    max_iter: int,
    energy_window: float | None,
) -> dict[str, np.ndarray | float | int | str]:
    """Calculate semi-infinite left/right edge spectral functions."""
    sample_id = str(sample.get("sample_id", sample.name))
    fermi = 0.5 * (float(sample["refined_vbm"]) + float(sample["refined_cbm"]))
    window = select_surface_energy_window(sample, energy_window)
    kx = np.linspace(-np.pi, np.pi, nk)
    kx_over_pi = kx / np.pi
    energy_relative = np.linspace(-window, window, energy_points)
    energy_absolute = energy_relative + fermi

    shape = (energy_points, nk)
    arrays = {
        name: np.zeros(shape, dtype=np.float64)
        for name in (
            "left_total",
            "right_total",
            "bulk_total",
            "left_up",
            "left_down",
            "right_up",
            "right_down",
            "bulk_up",
            "bulk_down",
        )
    }
    iteration_counts = np.zeros(shape, dtype=np.int16)
    residuals = np.zeros(shape, dtype=np.float64)

    progress_stride = max(1, nk // 10)
    for k_index, k_value in enumerate(kx):
        if k_index % progress_stride == 0 or k_index == nk - 1:
            print(
                f"  surface Green: {sample_id}  k={k_index + 1}/{nk}",
                flush=True,
            )
        h00, h01 = ribbon_blocks(float(k_value), sample)
        for energy_index, absolute_energy in enumerate(energy_absolute):
            g_left, g_right, g_bulk, iterations, residual = surface_green_1985(
                float(absolute_energy),
                h00,
                h01,
                eta=eta,
                tolerance=tolerance,
                max_iter=max_iter,
            )
            arrays["left_total"][energy_index, k_index] = _spectral_trace(g_left)
            arrays["right_total"][energy_index, k_index] = _spectral_trace(g_right)
            arrays["bulk_total"][energy_index, k_index] = _spectral_trace(g_bulk)
            arrays["left_up"][energy_index, k_index] = _spectral_trace(
                g_left, SPIN_INDICES["up"]
            )
            arrays["left_down"][energy_index, k_index] = _spectral_trace(
                g_left, SPIN_INDICES["down"]
            )
            arrays["right_up"][energy_index, k_index] = _spectral_trace(
                g_right, SPIN_INDICES["up"]
            )
            arrays["right_down"][energy_index, k_index] = _spectral_trace(
                g_right, SPIN_INDICES["down"]
            )
            arrays["bulk_up"][energy_index, k_index] = _spectral_trace(
                g_bulk, SPIN_INDICES["up"]
            )
            arrays["bulk_down"][energy_index, k_index] = _spectral_trace(
                g_bulk, SPIN_INDICES["down"]
            )
            iteration_counts[energy_index, k_index] = iterations
            residuals[energy_index, k_index] = residual

    # WannierTools' surfstat.f90 defines a surface-only quantity by subtracting
    # the principal-layer bulk DOS and clipping negative values.
    arrays["left_only"] = np.maximum(arrays["left_total"] - arrays["bulk_total"], 0.0)
    arrays["right_only"] = np.maximum(arrays["right_total"] - arrays["bulk_total"], 0.0)
    arrays["left_up_only"] = np.maximum(arrays["left_up"] - arrays["bulk_up"], 0.0)
    arrays["left_down_only"] = np.maximum(arrays["left_down"] - arrays["bulk_down"], 0.0)
    arrays["right_up_only"] = np.maximum(arrays["right_up"] - arrays["bulk_up"], 0.0)
    arrays["right_down_only"] = np.maximum(arrays["right_down"] - arrays["bulk_down"], 0.0)

    return {
        "sample_id": sample_id,
        "phase_label": str(sample["phase_label"]),
        "fermi_energy": float(fermi),
        "energy_window": float(window),
        "eta": float(eta),
        "kx": kx,
        "kx_over_pi": kx_over_pi,
        "energy_relative": energy_relative,
        "energy_absolute": energy_absolute,
        "iterations": iteration_counts,
        "residuals": residuals,
        **arrays,
    }


def write_wanniertools_style_surface_data(
    result: dict[str, np.ndarray | float | int | str],
    output_prefix: Path,
    write_text: bool,
) -> None:
    """Write NPZ plus text files resembling WannierTools surface outputs."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_prefix.with_suffix(".npz"), **result)
    if not write_text:
        return

    k_values = np.asarray(result["kx_over_pi"], dtype=float)
    energies = np.asarray(result["energy_relative"], dtype=float)
    file_specs = {
        "dos.dat_l": ("left_total", "left_only"),
        "dos.dat_r": ("right_total", "right_only"),
        "dos.dat_bulk": ("bulk_total",),
        "spindos.dat_l": ("left_up", "left_down"),
        "spindos.dat_r": ("right_up", "right_down"),
        "spindos_only.dat_l": ("left_up_only", "left_down_only"),
        "spindos_only.dat_r": ("right_up_only", "right_down_only"),
    }
    data_dir = output_prefix.parent / f"{output_prefix.name}_wanniertools_dat"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, columns in file_specs.items():
        path = data_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# kx/pi  E-EF  " + "  ".join(columns) + "\n")
            for k_index, kval in enumerate(k_values):
                for energy_index, energy in enumerate(energies):
                    values = [float(np.asarray(result[column])[energy_index, k_index]) for column in columns]
                    handle.write(
                        f"{kval: .10e} {energy: .10e} "
                        + " ".join(f"{value: .10e}" for value in values)
                        + "\n"
                    )
                handle.write("\n")


def _log_normalize(intensity: np.ndarray, contrast: float) -> np.ndarray:
    intensity = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    maximum = float(np.max(intensity))
    if maximum <= 0.0:
        return np.zeros_like(intensity)
    scaled = intensity / maximum
    return np.log1p(contrast * scaled) / np.log1p(contrast)


def _wanniertools_surface_cmap() -> LinearSegmentedColormap:
    # Matches the characteristic WannierTools ``surface-only`` palette:
    # white -> red -> black.
    return LinearSegmentedColormap.from_list(
        "wanniertools_surface",
        [(1.0, 1.0, 1.0), (1.0, 0.05, 0.00), (0.0, 0.0, 0.0)],
        N=256,
    )


def _spin_rgba(
    up: np.ndarray,
    down: np.ndarray,
    contrast: float,
) -> np.ndarray:
    total = np.maximum(up + down, 0.0)
    alpha = _log_normalize(total, contrast) ** 0.75
    polarization = (up - down) / (total + EPS)
    rgba = plt.get_cmap("bwr")((polarization + 1.0) / 2.0)
    rgba[..., 3] = alpha
    return rgba


def plot_wanniertools_surface_spectrum(
    result: dict[str, np.ndarray | float | int | str],
    sample: pd.Series,
    figures_dir: Path,
    contrast: float,
) -> None:
    """Plot left/right surface LDOS in the visual language of WannierTools."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(result["sample_id"])
    k_values = np.asarray(result["kx_over_pi"], dtype=float)
    energies = np.asarray(result["energy_relative"], dtype=float)
    extent = [float(k_values[0]), float(k_values[-1]), float(energies[0]), float(energies[-1])]
    cmap = _wanniertools_surface_cmap()

    left = _log_normalize(np.asarray(result["left_only"]), contrast)
    right = _log_normalize(np.asarray(result["right_only"]), contrast)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    image = None
    for axis, intensity, title in zip(
        axes,
        (left, right),
        ("Left semi-infinite edge", "Right semi-infinite edge"),
    ):
        image = axis.imshow(
            intensity,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.axvline(0.0, color="0.55", linewidth=0.5, linestyle=":")
        axis.set_xticks([-1.0, 0.0, 1.0], [r"$-X$", r"$\Gamma$", r"$X$"])
        axis.set_xlabel(r"edge momentum $k_x$")
        axis.set_title(title)
    axes[0].set_ylabel(r"$E-E_F$")
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.02, label="surface spectral weight")
    fig.suptitle(
        f"{sample_id} | {sample['phase_label']} | iterative surface Green function"
    )
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.14, top=0.86, wspace=0.08)
    save_figure(fig, figures_dir / f"{sample_id}_wanniertools_surfdos_left_right")
    plt.close(fig)

    # Spin-resolved map. Red denotes spin-up and blue denotes spin-down; opacity
    # is the local surface-only spectral intensity.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for axis, up_key, down_key, title in (
        (axes[0], "left_up_only", "left_down_only", "Left edge"),
        (axes[1], "right_up_only", "right_down_only", "Right edge"),
    ):
        rgba = _spin_rgba(
            np.asarray(result[up_key], dtype=float),
            np.asarray(result[down_key], dtype=float),
            contrast,
        )
        axis.imshow(
            rgba,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="bilinear",
        )
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.axvline(0.0, color="0.55", linewidth=0.5, linestyle=":")
        axis.set_xticks([-1.0, 0.0, 1.0], [r"$-X$", r"$\Gamma$", r"$X$"])
        axis.set_xlabel(r"edge momentum $k_x$")
        axis.set_title(title)
    axes[0].set_ylabel(r"$E-E_F$")
    legend = [
        Line2D([0], [0], linewidth=5, color=(0.85, 0.05, 0.00), label=r"spin $\uparrow$"),
        Line2D([0], [0], linewidth=5, color=(0.00, 0.20, 0.85), label=r"spin $\downarrow$"),
    ]
    axes[1].legend(handles=legend, frameon=False, loc="upper right")
    fig.suptitle(f"Spin-resolved surface spectrum | {sample_id}")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.86, wspace=0.08)
    save_figure(fig, figures_dir / f"{sample_id}_wanniertools_surfdos_spin_resolved")
    plt.close(fig)


def surface_spectrum_metrics(
    result: dict[str, np.ndarray | float | int | str],
    sample: pd.Series,
) -> dict[str, float | int]:
    energies = np.asarray(result["energy_relative"], dtype=float)
    left = np.asarray(result["left_only"], dtype=float)
    right = np.asarray(result["right_only"], dtype=float)
    fermi_index = int(np.argmin(np.abs(energies)))
    fermi = float(result["fermi_energy"])
    vbm_rel = float(sample["refined_vbm"]) - fermi
    cbm_rel = float(sample["refined_cbm"]) - fermi
    margin = 3.0 * float(result["eta"])
    lower = vbm_rel + margin
    upper = cbm_rel - margin
    gap_mask = (energies >= lower) & (energies <= upper) if lower < upper else np.zeros_like(energies, dtype=bool)
    max_residual = float(np.max(np.asarray(result["residuals"], dtype=float)))
    max_iterations = int(np.max(np.asarray(result["iterations"], dtype=int)))
    return {
        "surface_eta": float(result["eta"]),
        "energy_window": float(result["energy_window"]),
        "left_weight_at_fermi": float(np.max(left[fermi_index, :])),
        "right_weight_at_fermi": float(np.max(right[fermi_index, :])),
        "left_max_weight_in_bulk_gap": float(np.max(left[gap_mask, :])) if np.any(gap_mask) else float("nan"),
        "right_max_weight_in_bulk_gap": float(np.max(right[gap_mask, :])) if np.any(gap_mask) else float("nan"),
        "max_green_iterations": max_iterations,
        "max_green_residual": max_residual,
    }


# -----------------------------------------------------------------------------
# Finite-ribbon cross-check, corresponding to WannierTools ek_ribbon.f90
# -----------------------------------------------------------------------------


def ribbon_hamiltonian(kx: float, params: pd.Series | dict, ny: int) -> np.ndarray:
    onsite, hopping = ribbon_blocks(kx, params)
    dimension = 4 * ny
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for cell in range(ny):
        block = slice(4 * cell, 4 * (cell + 1))
        hamiltonian[block, block] = onsite
        if cell < ny - 1:
            next_block = slice(4 * (cell + 1), 4 * (cell + 2))
            hamiltonian[block, next_block] = hopping
            hamiltonian[next_block, block] = hopping.conj().T
    return hamiltonian


def ribbon_spin_block(
    kx: float, params: pd.Series | dict, ny: int, spin: str
) -> np.ndarray:
    full = ribbon_hamiltonian(kx, params, ny)
    within_cell = SPIN_INDICES[spin]
    indices = [4 * cell + orbital for cell in range(ny) for orbital in within_cell]
    return full[np.ix_(indices, indices)]


def calculate_spin_resolved_ribbon(
    sample: pd.Series,
    ny: int,
    nk: int,
    edge_cells: int,
) -> pd.DataFrame:
    sample_id = str(sample.get("sample_id", sample.name))
    k_values = np.linspace(-np.pi, np.pi, nk)
    rows = []
    block_dimension = 2 * ny
    left_indices = np.arange(0, 2 * edge_cells)
    right_indices = np.arange(2 * (ny - edge_cells), 2 * ny)
    for k_index, kx in enumerate(k_values):
        states = []
        for spin_name, spin_value in (("up", 1), ("down", -1)):
            energies, eigenvectors = eigh(
                ribbon_spin_block(float(kx), sample, ny, spin_name)
            )
            probabilities = np.abs(eigenvectors) ** 2
            left_weight = probabilities[left_indices, :].sum(axis=0)
            right_weight = probabilities[right_indices, :].sum(axis=0)
            for local_band in range(block_dimension):
                states.append(
                    {
                        "energy": float(energies[local_band]),
                        "spin": spin_name,
                        "spin_value": spin_value,
                        "left_weight": float(left_weight[local_band]),
                        "right_weight": float(right_weight[local_band]),
                        "total_edge_weight": float(
                            left_weight[local_band] + right_weight[local_band]
                        ),
                    }
                )
        states.sort(key=lambda row: row["energy"])
        for band_index, state in enumerate(states):
            total_edge = state["total_edge_weight"]
            edge_polarization = (
                (state["left_weight"] - state["right_weight"]) / total_edge
                if total_edge > EPS
                else 0.0
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "phase_label": sample["phase_label"],
                    "ny": ny,
                    "k_index": k_index,
                    "kx": float(kx),
                    "kx_over_pi": float(kx / np.pi),
                    "band_index": band_index,
                    **state,
                    "edge_polarization": float(edge_polarization),
                }
            )
    spectrum = pd.DataFrame(rows)
    fermi = 0.5 * (float(sample["refined_vbm"]) + float(sample["refined_cbm"]))
    spectrum["fermi_energy"] = fermi
    spectrum["energy_relative_to_fermi"] = spectrum["energy"] - fermi
    return spectrum


def plot_finite_ribbon_wanniertools_style(
    spectrum: pd.DataFrame,
    sample: pd.Series,
    ny: int,
    figures_dir: Path,
) -> None:
    """Exact ribbon eigenvalues colored by edge weight, like ribbonek.gnu."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(sample.get("sample_id", sample.name))
    gap = float(sample["final_indirect_gap"])
    energy_window = max(0.22, min(0.60, 2.2 * gap + 0.08))
    edge_cmap = LinearSegmentedColormap.from_list(
        "wanniertools_ribbon",
        [(0.0, 0.45, 0.0), (1.0, 0.85, 0.0), (0.85, 0.0, 0.0)],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    image = ax.scatter(
        spectrum["kx_over_pi"],
        spectrum["energy_relative_to_fermi"],
        c=np.clip(spectrum["total_edge_weight"], 0.0, 1.0),
        cmap=edge_cmap,
        vmin=0.0,
        vmax=1.0,
        s=2.2,
        linewidths=0.0,
        rasterized=True,
    )
    ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-energy_window, energy_window)
    ax.set_xticks([-1.0, 0.0, 1.0], [r"$-X$", r"$\Gamma$", r"$X$"])
    ax.set_xlabel(r"edge momentum $k_x$")
    ax.set_ylabel(r"$E-E_F$")
    ax.set_title(f"Finite ribbon | {sample_id} | Ny={ny}")
    fig.colorbar(image, ax=ax, label="left + right edge weight")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{sample_id}_Ny{ny}_wanniertools_ribbonek")
    plt.close(fig)


def ribbon_width_metrics(
    spectrum: pd.DataFrame,
    edge_threshold: float,
) -> dict[str, float]:
    edge = spectrum[spectrum["total_edge_weight"] >= edge_threshold]
    if edge.empty:
        return {
            "n_edge_states": 0,
            "min_abs_edge_energy": float("nan"),
            "apparent_edge_gap": float("nan"),
            "max_edge_weight_near_fermi": 0.0,
            "n_edge_states_within_0p01": 0,
        }
    min_abs = float(edge["energy_relative_to_fermi"].abs().min())
    near = spectrum[spectrum["energy_relative_to_fermi"].abs() <= 0.01]
    return {
        "n_edge_states": int(len(edge)),
        "min_abs_edge_energy": min_abs,
        "apparent_edge_gap": 2.0 * min_abs,
        "max_edge_weight_near_fermi": (
            float(near["total_edge_weight"].max()) if len(near) else 0.0
        ),
        "n_edge_states_within_0p01": int(
            (edge["energy_relative_to_fermi"].abs() <= 0.01).sum()
        ),
    }


def run_edge_analysis(
    master: pd.DataFrame,
    selection_file: Path,
    settings: Settings,
    output_dir: Path,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv(selection_file)
    ensure_columns(selection, ["role", "sample_id"], "Step 07 ribbon selection")
    master_index = master.set_index("sample_id")

    validation_sample = master_index.loc[selection.iloc[0]["sample_id"]]
    error = ribbon_reconstruction_error(validation_sample, settings.random_seed)
    write_json(
        output_dir / "step08_05_principal_layer_bloch_reconstruction.json",
        {"maximum_reconstruction_error": error, "passed": bool(error < 1.0e-10)},
    )
    if error >= 1.0e-10:
        raise RuntimeError(f"Principal-layer Bloch reconstruction failed: {error}")

    surface_metric_rows: list[dict] = []
    ribbon_metric_rows: list[dict] = []

    if settings.edge_method in {"green", "both"}:
        surface_dir = output_dir / "surface_green_data"
        surface_figures = figures_dir / "wanniertools_surface_green"
        surface_dir.mkdir(parents=True, exist_ok=True)
        surface_figures.mkdir(parents=True, exist_ok=True)
        for row in selection.itertuples(index=False):
            sample = master_index.loc[row.sample_id]
            result = calculate_surface_spectral_function(
                sample,
                nk=settings.surface_nk,
                energy_points=settings.surface_energy_points,
                eta=settings.surface_eta,
                tolerance=settings.surface_tol,
                max_iter=settings.surface_max_iter,
                energy_window=settings.surface_energy_window,
            )
            prefix = surface_dir / f"step08_05_{row.sample_id}_surface_green"
            write_wanniertools_style_surface_data(
                result, prefix, write_text=settings.write_surface_text
            )
            plot_wanniertools_surface_spectrum(
                result,
                sample,
                surface_figures,
                contrast=settings.surface_log_contrast,
            )
            surface_metric_rows.append(
                {
                    "role": row.role,
                    "sample_id": row.sample_id,
                    "phase_label": sample["phase_label"],
                    "nk": settings.surface_nk,
                    "energy_points": settings.surface_energy_points,
                    **surface_spectrum_metrics(result, sample),
                }
            )

    if settings.edge_method in {"ribbon", "both"}:
        ribbon_dir = output_dir / "ribbon_data"
        ribbon_figures = figures_dir / "wanniertools_finite_ribbon"
        ribbon_dir.mkdir(parents=True, exist_ok=True)
        ribbon_figures.mkdir(parents=True, exist_ok=True)
        largest_width = max(settings.ribbon_widths)
        for row in selection.itertuples(index=False):
            sample = master_index.loc[row.sample_id]
            for ny in settings.ribbon_widths:
                edge_cells = min(settings.edge_cells, max(1, ny // 6))
                spectrum = calculate_spin_resolved_ribbon(
                    sample,
                    ny=ny,
                    nk=settings.ribbon_nk,
                    edge_cells=edge_cells,
                )
                spectrum.to_csv(
                    ribbon_dir
                    / f"step08_05_{row.sample_id}_Ny{ny}_spin_left_right_spectrum.csv",
                    index=False,
                )
                ribbon_metric_rows.append(
                    {
                        "role": row.role,
                        "sample_id": row.sample_id,
                        "phase_label": sample["phase_label"],
                        "ny": ny,
                        "nk": settings.ribbon_nk,
                        "edge_cells": edge_cells,
                        **ribbon_width_metrics(spectrum, settings.edge_threshold),
                    }
                )
                if ny in {40, largest_width}:
                    plot_finite_ribbon_wanniertools_style(
                        spectrum, sample, ny, ribbon_figures
                    )

    surface_metrics = pd.DataFrame(surface_metric_rows)
    ribbon_metrics = pd.DataFrame(ribbon_metric_rows)
    surface_metrics.to_csv(
        output_dir / "step08_05_surface_green_metrics.csv", index=False
    )
    ribbon_metrics.to_csv(
        output_dir / "step08_05_ribbon_width_convergence.csv", index=False
    )

    if len(ribbon_metrics):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for role, part in ribbon_metrics.groupby("role"):
            ax.plot(part["ny"], part["apparent_edge_gap"], marker="o", label=role)
        ax.set_xlabel(r"ribbon width $N_y$")
        ax.set_ylabel(r"$2\min|E_{\rm edge}-E_F|$")
        ax.set_title("Finite-width convergence of the edge crossing")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, figures_dir / "step08_05_ribbon_width_convergence")
        plt.close(fig)

    return surface_metrics, ribbon_metrics


def write_current_phase_taxonomy(master: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    """Record exactly what Step 01--08 has and has not classified.

    The present labels are first-order spin-Chern labels.  The two signs of the
    spin Chern number are retained as distinct sectors instead of being hidden
    inside one binary type-II-QSH target.  Real-Chern / Stiefel--Whitney and
    higher-order labels are deliberately marked as not yet computed.
    """
    labels = []
    for row in master.itertuples(index=False):
        cup = int(row.chern_up_int)
        cdn = int(row.chern_down_int)
        ctot = int(row.chern_total_int)
        if (cup, cdn, ctot) == (0, 0, 0):
            labels.append("trivial_spin_chern")
        elif (cup, cdn, ctot) == (1, -1, 0):
            labels.append("typeII_QSH_Cup_plus")
        elif (cup, cdn, ctot) == (-1, 1, 0):
            labels.append("typeII_QSH_Cup_minus")
        else:
            labels.append(f"other_Cup_{cup}_Cdown_{cdn}_Ctotal_{ctot}")
    table = master[["sample_id", "phase_label", "chern_up_int", "chern_down_int", "chern_total_int"]].copy()
    table["first_order_sector"] = labels
    table.to_csv(output_dir / "step08_07_current_first_order_phase_taxonomy.csv", index=False)
    counts = table["first_order_sector"].value_counts().rename_axis("first_order_sector").reset_index(name="count")
    counts.to_csv(output_dir / "step08_07_current_first_order_phase_counts.csv", index=False)

    scope = {
        "current_model_space": "four-band spin-conserving MSG 123.342 Lieb Hamiltonian at half filling",
        "current_labels_are": [
            "full-BZ metal/boundary/insulator audit",
            "spin-resolved Chern numbers C_up and C_down",
            "total Chern number",
            "trivial versus type-II QSH",
            "two opposite spin-Chern chirality sectors",
        ],
        "not_yet_computed": [
            "C2zT real-Chern / second Stiefel-Whitney invariant",
            "mirror-resolved real topology",
            "symmetry eigenvalue or magnetic-TQC indicator",
            "nested Wilson loop or quantized multipole invariant",
            "finite-square corner spectrum and corner localization",
            "boundary-obstructed versus intrinsic higher-order diagnosis",
        ],
        "important_rule": (
            "Do not relabel the current binary spin-Chern data as a generic classifier "
            "for all topological insulators or higher-order topological insulators."
        ),
        "first_order_sector_counts": {
            str(row.first_order_sector): int(row.count) for row in counts.itertuples(index=False)
        },
    }
    write_json(output_dir / "step08_07_topology_scope_manifest.json", scope)
    return scope


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = parse_args()
    context = resolve_step07_context(args.step07_dir)

    inherited_seed = infer_seed(context)
    inherited_pairs = infer_interpolation_pairs(context)
    inherited_widths = infer_ribbon_widths(context)
    random_seed = args.random_seed if args.random_seed is not None else inherited_seed
    interpolation_pairs = (
        args.interpolation_pairs
        if args.interpolation_pairs is not None
        else inherited_pairs
    )
    ribbon_widths = (
        tuple(sorted(set(args.ribbon_widths)))
        if args.ribbon_widths is not None
        else inherited_widths
    )
    if any(width < 4 for width in ribbon_widths):
        raise ValueError(f"All ribbon widths must be >= 4: {ribbon_widths}")

    if args.quick:
        settings = Settings(
            random_seed=random_seed,
            n_splits=5,
            test_size=DEFAULT_TEST_SIZE,
            interpolation_pairs=min(2, interpolation_pairs),
            root_grid=501,
            bz_nk=80,
            chern_grids=(15, 21),
            chern_shifts=((0.0, 0.0), (0.5, 0.5)),
            edge_method=("green" if args.skip_ribbon and args.edge_method == "both" else args.edge_method),
            surface_nk=(args.surface_nk or 61),
            surface_energy_points=(args.surface_energy_points or 201),
            surface_eta=(args.surface_eta or 0.006),
            surface_max_iter=DEFAULT_SURFACE_MAX_ITER,
            surface_tol=1.0e-12,
            surface_log_contrast=DEFAULT_SURFACE_LOG_CONTRAST,
            surface_energy_window=args.surface_energy_window,
            write_surface_text=args.write_surface_text,
            ribbon_widths=tuple(ribbon_widths[:2]),
            ribbon_nk=61,
            edge_cells=2,
            edge_threshold=DEFAULT_EDGE_THRESHOLD,
            quick=True,
        )
    else:
        settings = Settings(
            random_seed=random_seed,
            n_splits=DEFAULT_N_SPLITS,
            test_size=DEFAULT_TEST_SIZE,
            interpolation_pairs=interpolation_pairs,
            root_grid=DEFAULT_ROOT_GRID,
            bz_nk=DEFAULT_BZ_NK,
            chern_grids=DEFAULT_CHERN_GRIDS,
            chern_shifts=DEFAULT_CHERN_SHIFTS,
            edge_method=("green" if args.skip_ribbon and args.edge_method == "both" else args.edge_method),
            surface_nk=(args.surface_nk or DEFAULT_SURFACE_NK),
            surface_energy_points=(args.surface_energy_points or DEFAULT_SURFACE_ENERGY_POINTS),
            surface_eta=(args.surface_eta or DEFAULT_SURFACE_ETA),
            surface_max_iter=DEFAULT_SURFACE_MAX_ITER,
            surface_tol=DEFAULT_SURFACE_TOL,
            surface_log_contrast=DEFAULT_SURFACE_LOG_CONTRAST,
            surface_energy_window=args.surface_energy_window,
            write_surface_text=args.write_surface_text,
            ribbon_widths=ribbon_widths,
            ribbon_nk=DEFAULT_RIBBON_NK,
            edge_cells=DEFAULT_EDGE_CELLS,
            edge_threshold=DEFAULT_EDGE_THRESHOLD,
            quick=False,
        )

    output_name = (
        "outputs_step08_analytic_boundary_wanniertools_edge_"
        f"{context.run_tag}{'_quick' if settings.quick else ''}"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (context.source_parent / output_name).resolve()
    )
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    master_file = context.require(
        "analysis_master",
        "step07_01_consensus_and_engineered_feature_master__*.csv",
    )
    pair_file = context.require(
        "nearest_pairs",
        "step07_04_nearest_topology_trivial_parameter_pairs__*.csv",
    )
    required_files = {
        "analysis_master": master_file,
        "nearest_pairs": pair_file,
    }

    master = pd.read_csv(master_file)
    pair_table = pd.read_csv(pair_file)
    ensure_columns(
        master,
        [
            "sample_id",
            "phase_label",
            "target_typeII",
            "parameter_cluster",
            "final_direct_gap",
            "final_indirect_gap",
            "refined_vbm",
            "refined_cbm",
            *PHYS7,
            *RAW8,
            "spin_splitting_1",
            "spin_splitting_2",
        ],
        "Step 07 master",
    )
    if len(master) == 0:
        raise ValueError("Step 07 analysis master is empty.")
    if master["sample_id"].duplicated().any():
        duplicated = master.loc[master["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values in Step 07 master: {duplicated[:10]}")
    ensure_columns(
        pair_table,
        ["topology_sample_id", "trivial_sample_id", "standardized_distance"],
        "Step 07 nearest-pair table",
    )
    if args.skip_edge:
        ribbon_selection_file = context.files.get("ribbon_selection")
        edge_selection_source = "not_required_skip_edge"
    else:
        ribbon_selection_file, edge_selection_source = resolve_edge_selection_file(
            context, master, pair_table, output_dir
        )
        required_files["edge_pair_selection"] = ribbon_selection_file
    write_input_compatibility_report(context, output_dir, required_files)

    analytic = add_analytic_features(master)
    analytic.to_csv(output_dir / "step08_01_analytic_feature_master.csv", index=False)
    exact_rule_metrics, exact_rule_table = audit_exact_mass_product_rule(analytic, output_dir)
    topology_scope = write_current_phase_taxonomy(analytic, output_dir)
    plot_mechanism_overview(analytic, figures_dir)
    stats = mass_boundary_statistics(analytic, output_dir)
    split_df, model_summary, coefficient_summary = grouped_logistic_comparison(
        analytic, settings, output_dir
    )

    roots, paths = refine_transition_paths(
        analytic,
        pair_table,
        settings,
        output_dir,
        figures_dir,
        skip_chern=args.skip_chern,
    )

    if args.skip_edge:
        surface_metrics = pd.DataFrame()
        ribbon_metrics = pd.DataFrame()
    else:
        surface_metrics, ribbon_metrics = run_edge_analysis(
            analytic,
            ribbon_selection_file,
            settings,
            output_dir,
            figures_dir,
        )

    targeted_table, targeted_summary = targeted_boundary_validation(
        args.targeted_samples,
        args.targeted_width,
        settings,
        output_dir,
    )

    summary = {
        "step07_dir": str(context.directory),
        "step07_input_mode": "existing_directory_only",
        "step07_run_tag": context.run_tag,
        "zip_reading_enabled": False,
        "edge_selection_source": edge_selection_source,
        "output_dir": str(output_dir),
        "settings": settings.__dict__,
        "n_high_confidence_insulators": int(len(analytic)),
        "n_typeII": int(analytic["target_typeII"].sum()),
        "n_trivial": int((1 - analytic["target_typeII"]).sum()),
        "analytic_mass_boundaries": [
            "M_plus = m_e + d1 + d2 = 0",
            "M_minus = -m_e + d1 + d2 = 0",
        ],
        "n_refined_root_candidates": int(len(roots)),
        "exact_mass_product_rule": exact_rule_metrics,
        "topology_scope": topology_scope,
        "targeted_validation": targeted_summary,
        "model_summary": model_summary.to_dict(orient="records"),
        "surface_green_metrics": surface_metrics.to_dict(orient="records"),
        "ribbon_width_metrics": ribbon_metrics.to_dict(orient="records"),
        "important_interpretation": [
            "Delta_scale and Delta_log_anisotropy replace the redundant Delta_min/product/imbalance trio.",
            "M_plus and M_minus are two distinct X/Y mass branches.",
            "A zero of t1 or t2 is only a candidate boundary; the axis d_z condition must also be satisfied.",
            "The primary edge calculation is the semi-infinite iterative surface Green function used by WannierTools.",
            "Finite-ribbon diagonalization is retained as an independent width-convergence check.",
            "Surface-only maps use surface spectral weight minus the bulk principal-layer contribution, following surfstat.f90.",
            "The full-BZ grid is always even and contains Gamma, X, Y, and M exactly.",
            "Chern numbers beside a closing are evaluated at the largest safe insulating offset, not arbitrarily close to the singular point.",
            "Current labels diagnose first-order spin-Chern topology only; higher-order and real topology require new invariants and finite-square calculations.",
        ],
    }
    write_json(output_dir / "step08_00_run_summary.json", summary)
    write_step08_registry(output_dir, context.run_tag)

    print("\nStep 08 v5 completed successfully.")
    print(f"Input : {context.directory}")
    print(f"Run tag: {context.run_tag}")
    print(f"Output: {output_dir}")
    print("\nExact audited-insulator rule:")
    print(json.dumps(exact_rule_metrics, indent=2, ensure_ascii=False))
    print("\nModel comparison:")
    print(model_summary.to_string(index=False))
    print("\nRefined root candidates:", len(roots))
    if len(surface_metrics):
        print("\nSemi-infinite surface Green-function metrics:")
        print(surface_metrics.to_string(index=False))
    if len(ribbon_metrics):
        print("\nFinite-ribbon width convergence:")
        print(ribbon_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
