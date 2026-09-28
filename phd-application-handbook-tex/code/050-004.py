from __future__ import annotations

"""
TTS Step13M — adaptive Wilson-loop certification of the narrow C_up=+2 sector
================================================================================

This workflow is deliberately local.  It does not rescan the seven-dimensional
parameter space.  It takes the two Step12M closures on the unresolved r3=0.060
edge and independently certifies the very narrow intermediate spin-up Chern
sector with non-Abelian Wilson loops of the occupied spin-up subspace.

Target sequence
---------------
    C_up = -2  -- generic four-valley charge +4 -->  +2
               -- Sigma' two-valley charge -2 -->    0

The Wilson-loop sign convention is calibrated against the independently strict
Step12M pre-closure control C_up=-2.  The post-closure C_up=0 control is then an
out-of-sample validation of that calibration.
"""

import argparse
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize

import TTS_step12M_junction_multiclosure_repair as step12
import TTS_step11M_lieb_aligned_fixed_slice as step11


CODE_VERSION = "TTS_STEP13M_ADAPTIVE_WILSON_INTERMEDIATE_CHERN_V1_20260724"
TARGET_EDGE_ID = "edge_03_03__03_04"
TWO_PI = 2.0 * math.pi

PHASE_COLORS = {
    -2: "#3B6FB6",
    -1: "#7FB6D6",
     0: "#B8B8B8",
     1: "#F2B36D",
     2: "#C84A3A",
}


@dataclass
class Step13Config:
    output_dir: Path

    # Fractions inside the interval between the two Step12M closures.
    middle_fractions: tuple[float, ...] = (0.25, 0.40, 0.50, 0.65, 0.80)
    minimum_middle_certified_points: int = 3

    # Three independent Wilson-loop discretizations.  Pairwise entries are used
    # together: (nkx[i], initial_nky[i], shift[i]).
    wilson_nkx: tuple[int, ...] = (401, 601, 801)
    wilson_initial_nky: tuple[int, ...] = (81, 101, 121)
    wilson_shifts: tuple[tuple[float, float], ...] = (
        (0.00, 0.00),
        (0.50, 0.37),
        (0.25, 0.73),
    )
    adaptive_phase_step: float = 0.32 * math.pi
    adaptive_max_points: int = 801
    adaptive_max_rounds: int = 8
    integer_residual_tolerance: float = 0.08
    minimum_overlap_singular_value: float = 1.0e-4

    # Independent occupied-subspace gap audit at every probe.
    gap_grid_n: int = 101
    gap_local_starts: int = 16
    minimum_positive_spin_gap: float = 1.0e-8

    quick: bool = False
    force_recalculate: bool = False
    random_seed: int = 20260724

    def normalized(self) -> "Step13Config":
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["figures", "wilson_checkpoints", "wilson_traces", "_physics_core"]:
            (self.output_dir / sub).mkdir(exist_ok=True)
        if len(self.wilson_nkx) != len(self.wilson_initial_nky):
            raise ValueError("wilson_nkx and wilson_initial_nky must have equal length")
        if len(self.wilson_shifts) != len(self.wilson_nkx):
            raise ValueError("wilson_shifts must match the number of Wilson attempts")
        if not self.middle_fractions:
            raise ValueError("middle_fractions cannot be empty")
        if any(not (0.0 < x < 1.0) for x in self.middle_fractions):
            raise ValueError("middle_fractions must lie strictly inside (0,1)")
        if self.minimum_middle_certified_points < 1:
            raise ValueError("minimum_middle_certified_points must be positive")
        if self.quick:
            self.wilson_nkx = (201, 301)
            self.wilson_initial_nky = (61, 81)
            self.wilson_shifts = ((0.0, 0.0), (0.5, 0.37))
            self.adaptive_max_points = min(self.adaptive_max_points, 401)
            self.gap_grid_n = min(self.gap_grid_n, 61)
            self.gap_local_starts = min(self.gap_local_starts, 8)
        return self


# =============================================================================
# Safe I/O
# =============================================================================


def _select_member(zf: zipfile.ZipFile, token: str, suffix: str | None = None) -> str:
    matches = []
    for name in zf.namelist():
        if token not in Path(name).name and token not in name:
            continue
        if suffix and not name.lower().endswith(suffix.lower()):
            continue
        matches.append(name)
    if not matches:
        raise FileNotFoundError(f"No archive member containing {token!r}")
    return sorted(matches, key=lambda n: (len(Path(n).parts), len(n)))[0]


def read_csv_token(source: str | Path, token: str) -> pd.DataFrame:
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            member = _select_member(zf, token, ".csv")
            with zf.open(member) as stream:
                try:
                    return pd.read_csv(stream, low_memory=False)
                except pd.errors.EmptyDataError:
                    return pd.DataFrame()
    if source.is_dir():
        matches = sorted(source.rglob(f"*{token}*"))
        if not matches:
            raise FileNotFoundError(token)
        try:
            return pd.read_csv(matches[0], low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise FileNotFoundError(source)


def read_json_token(source: str | Path, token: str) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            member = _select_member(zf, token, ".json")
            with zf.open(member) as stream:
                return json.load(stream)
    if source.is_dir():
        matches = sorted(source.rglob(f"*{token}*"))
        if not matches:
            raise FileNotFoundError(token)
        return json.loads(matches[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(source)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def wrap_k(value: float) -> float:
    return float((float(value) + math.pi) % TWO_PI - math.pi)


def principal_angle(value: complex) -> float:
    return float(np.angle(value))


# =============================================================================
# Step12M reconstruction
# =============================================================================


def load_step12_inputs(step12_source: str | Path) -> dict[str, Any]:
    final_certificate = read_json_token(step12_source, "step12M_18_final_mechanism_certificate.json")
    edge = read_csv_token(step12_source, "step12M_01_target_edge_definition.csv").iloc[0].copy()
    closures = read_csv_token(step12_source, "step12M_06_critical_closure_parameters.csv")
    segment_labels = read_csv_token(step12_source, "step12M_05_strict_segment_chern_labels.csv")
    group_certificates = read_csv_token(step12_source, "step12M_13_closure_group_certificates.csv")
    berry = read_csv_token(step12_source, "step12M_12_berry_charge_consensus.csv")
    grid = read_csv_token(step12_source, "step12M_15_refined_junction_strict_grid.csv")
    gap_track = read_csv_token(step12_source, "step12M_04_target_edge_gap_tracks.csv")

    if str(edge.get("edge_id")) != TARGET_EDGE_ID:
        raise ValueError(f"Unexpected target edge: {edge.get('edge_id')}")
    background = {
        name: float(final_certificate["fixed_background_parameters"][name])
        for name in ["m_e", "t1", "t2", "r1", "r2"]
    }

    if "closure_group_id" in closures.columns:
        grouped = closures.groupby("closure_group_id", sort=False)["critical_lambda"].mean()
        closure_lambdas = sorted(float(x) for x in grouped.to_list())
    else:
        closure_lambdas = sorted(float(x) for x in closures["critical_lambda"].to_list())
    if len(closure_lambdas) != 2:
        raise RuntimeError(f"Step13M requires exactly two Step12M closure groups, found {closure_lambdas}")

    return {
        "final_certificate": final_certificate,
        "edge": edge,
        "closures": closures,
        "segment_labels": segment_labels,
        "group_certificates": group_certificates,
        "berry": berry,
        "grid": grid,
        "gap_track": gap_track,
        "background": background,
        "closure_lambdas": closure_lambdas,
    }


def build_probe_table(inputs: dict[str, Any], config: Step13Config) -> pd.DataFrame:
    lam1, lam2 = inputs["closure_lambdas"]
    segments = inputs["segment_labels"].sort_values("segment_index")
    pre_rows = segments[segments["segment_index"].astype(int).eq(0)]
    post_rows = segments[segments["segment_index"].astype(int).eq(2)]
    pre_lambda = float(pre_rows.iloc[0]["probe_lambda"]) if not pre_rows.empty else 0.5 * lam1
    post_lambda = float(post_rows.iloc[0]["probe_lambda"]) if not post_rows.empty else 0.5 * (lam2 + 1.0)

    rows = [
        {
            "probe_id": "control_pre_minus2",
            "probe_role": "pre_control",
            "lambda": pre_lambda,
            "expected_chern_up": -2,
            "middle_fraction": np.nan,
        }
    ]
    for index, fraction in enumerate(config.middle_fractions):
        rows.append(
            {
                "probe_id": f"middle_{index+1:02d}",
                "probe_role": "intermediate",
                "lambda": lam1 + float(fraction) * (lam2 - lam1),
                "expected_chern_up": 2,
                "middle_fraction": float(fraction),
            }
        )
    rows.append(
        {
            "probe_id": "control_post_zero",
            "probe_role": "post_control",
            "lambda": post_lambda,
            "expected_chern_up": 0,
            "middle_fraction": np.nan,
        }
    )
    edge = inputs["edge"]
    for row in rows:
        lam = float(row["lambda"])
        row["r3"] = float(edge["r3_0"] + lam * (edge["r3_1"] - edge["r3_0"]))
        row["r4"] = float(edge["r4_0"] + lam * (edge["r4_1"] - edge["r4_0"]))
    return pd.DataFrame(rows)


def closure_ky_anchors(inputs: dict[str, Any]) -> list[float]:
    valleys = read_csv_token_from_optional(inputs, "step12M_07_critical_valley_orbits.csv")
    anchors: list[float] = []
    if valleys.empty:
        return anchors
    ky_column = "critical_ky" if "critical_ky" in valleys.columns else ("ky" if "ky" in valleys.columns else None)
    if ky_column is None:
        return anchors
    selected = valleys.copy()
    if "spin" in selected.columns:
        selected = selected[selected["spin"].astype(str).eq("up")]
    if "is_active_valley" in selected.columns:
        selected = selected[selected["is_active_valley"].astype(int).eq(1)]
    for value in selected[ky_column].dropna().to_numpy(float):
        for offset in (0.0, -0.08, -0.04, -0.02, 0.02, 0.04, 0.08):
            anchors.append(wrap_k(float(value) + offset))
    return sorted(set(round(x, 12) for x in anchors))


def read_csv_token_from_optional(inputs: dict[str, Any], token: str) -> pd.DataFrame:
    source = inputs.get("_step12_source")
    if source is None:
        return pd.DataFrame()
    try:
        return read_csv_token(source, token)
    except FileNotFoundError:
        return pd.DataFrame()


# =============================================================================
# Physical core and occupied-subspace gap audit
# =============================================================================


def configure_physics(tts_archive: str | Path, config: Step13Config):
    step12_config = step12.Step12Config(
        output_dir=config.output_dir / "_physics_core",
        quick=bool(config.quick),
        force_recalculate=False,
        random_seed=int(config.random_seed),
    ).normalized()
    core, step4, step5, step3, strict_config, search_config = step12.configure_physics(
        tts_archive,
        step12_config,
    )
    return core, step4, step5, step3, strict_config, search_config


def spin_hamiltonian(core, raw: dict[str, float], kx: float, ky: float) -> np.ndarray:
    matrix = np.asarray(core.h_spin_block_periodic(float(kx), float(ky), raw, "up"), dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Invalid spin Hamiltonian shape {matrix.shape}")
    return 0.5 * (matrix + matrix.conj().T)


def occupied_frame(core, raw: dict[str, float], kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(spin_hamiltonian(core, raw, kx, ky))
    n_occ = int(core.N_OCC_SPIN)
    if not (0 < n_occ < len(values)):
        raise ValueError(f"Invalid N_OCC_SPIN={n_occ} for dimension {len(values)}")
    return values, vectors[:, :n_occ]


def spin_internal_gap(core, raw: dict[str, float], kx: float, ky: float) -> float:
    values = np.linalg.eigvalsh(spin_hamiltonian(core, raw, kx, ky))
    n_occ = int(core.N_OCC_SPIN)
    return float(values[n_occ] - values[n_occ - 1])


def audit_global_spin_gap(
    core,
    raw: dict[str, float],
    *,
    grid_n: int,
    local_starts: int,
) -> dict[str, Any]:
    grid_n = int(grid_n)
    ks = np.linspace(-math.pi, math.pi, grid_n, endpoint=False)
    candidates: list[tuple[float, float, float]] = []
    for kx in ks:
        for ky in ks:
            gap = spin_internal_gap(core, raw, float(kx), float(ky))
            candidates.append((gap, float(kx), float(ky)))
    candidates.sort(key=lambda item: item[0])

    best = {"gap": float(candidates[0][0]), "kx": candidates[0][1], "ky": candidates[0][2]}
    unique_starts: list[tuple[float, float]] = []
    for _, kx, ky in candidates:
        if all(math.hypot(wrap_k(kx-a), wrap_k(ky-b)) > 0.05 for a, b in unique_starts):
            unique_starts.append((kx, ky))
        if len(unique_starts) >= int(local_starts):
            break

    def objective(x: np.ndarray) -> float:
        gap = spin_internal_gap(core, raw, wrap_k(float(x[0])), wrap_k(float(x[1])))
        return float(gap * gap)

    for start in unique_starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Powell",
            bounds=[(-math.pi, math.pi), (-math.pi, math.pi)],
            options={"xtol": 1.0e-11, "ftol": 1.0e-22, "maxiter": 1200},
        )
        kx, ky = wrap_k(result.x[0]), wrap_k(result.x[1])
        gap = spin_internal_gap(core, raw, kx, ky)
        if gap < best["gap"]:
            best = {"gap": float(gap), "kx": float(kx), "ky": float(ky)}

    return {
        "spin_internal_gap": float(best["gap"]),
        "gap_kx": float(best["kx"]),
        "gap_ky": float(best["ky"]),
        "gap_grid_n": int(grid_n),
        "gap_local_starts": int(len(unique_starts)),
    }


# =============================================================================
# Non-Abelian Wilson loop
# =============================================================================


def polar_unitary(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh, float(np.min(singular))


def wilson_loop_at_ky(
    h_fn: Callable[[float, float], np.ndarray],
    n_occ: int,
    ky: float,
    *,
    nkx: int,
    shift_x: float,
) -> dict[str, Any]:
    kxs = -math.pi + TWO_PI * (np.arange(int(nkx), dtype=float) + float(shift_x)) / int(nkx)
    frames: list[np.ndarray] = []
    direct_gaps: list[float] = []
    for kx in kxs:
        matrix = h_fn(float(kx), float(ky))
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
        frames.append(vectors[:, :n_occ])
        direct_gaps.append(float(values[n_occ] - values[n_occ - 1]))

    wilson = np.eye(n_occ, dtype=complex)
    min_singular = 1.0
    for index in range(len(frames)):
        overlap = frames[index].conj().T @ frames[(index + 1) % len(frames)]
        link, singular = polar_unitary(overlap)
        min_singular = min(min_singular, singular)
        wilson = wilson @ link

    eigenvalues = np.linalg.eigvals(wilson)
    eigenphases = np.sort(np.angle(eigenvalues))
    determinant = np.linalg.det(wilson)
    determinant /= abs(determinant) if abs(determinant) else 1.0
    return {
        "ky": float(ky),
        "determinant": complex(determinant),
        "det_phase": float(np.angle(determinant)),
        "eigenphases": np.asarray(eigenphases, dtype=float),
        "min_overlap_singular": float(min_singular),
        "min_kx_direct_gap": float(np.min(direct_gaps)),
    }


def track_phase_branches(phase_rows: list[np.ndarray]) -> np.ndarray:
    if not phase_rows:
        return np.empty((0, 0), dtype=float)
    tracked = [np.sort(np.asarray(phase_rows[0], dtype=float))]
    for raw in phase_rows[1:]:
        raw = np.asarray(raw, dtype=float)
        previous = tracked[-1]
        previous_wrapped = (previous + math.pi) % TWO_PI - math.pi
        cost = np.abs(np.angle(np.exp(1j * (raw[None, :] - previous_wrapped[:, None]))))
        row_ind, col_ind = linear_sum_assignment(cost)
        ordered = np.empty_like(raw)
        ordered[row_ind] = raw[col_ind]
        delta = np.angle(np.exp(1j * (ordered - previous_wrapped)))
        tracked.append(previous + delta)
    return np.asarray(tracked, dtype=float)


def adaptive_wilson_winding(
    h_fn: Callable[[float, float], np.ndarray],
    n_occ: int,
    *,
    nkx: int,
    initial_nky: int,
    shift_x: float,
    shift_y: float,
    phase_step: float,
    max_points: int,
    max_rounds: int,
    ky_anchors: Iterable[float] = (),
) -> tuple[dict[str, Any], pd.DataFrame]:
    start = -math.pi + TWO_PI * float(shift_y) / int(initial_nky)
    coords = list(start + TWO_PI * np.arange(int(initial_nky), dtype=float) / int(initial_nky))
    for anchor in ky_anchors:
        value = float(anchor)
        while value < start:
            value += TWO_PI
        while value >= start + TWO_PI:
            value -= TWO_PI
        coords.append(value)
    coords = sorted(set(round(float(x), 14) for x in coords))

    cache: dict[float, dict[str, Any]] = {}

    def evaluate(coord: float) -> dict[str, Any]:
        key = round(float(coord), 14)
        if key not in cache:
            cache[key] = wilson_loop_at_ky(
                h_fn,
                n_occ,
                wrap_k(float(coord)),
                nkx=int(nkx),
                shift_x=float(shift_x),
            )
            cache[key]["ky_coordinate"] = float(coord)
        return cache[key]

    refinement_rounds = 0
    for round_index in range(int(max_rounds) + 1):
        coords = sorted(coords)
        rows = [evaluate(x) for x in coords]
        new_points: list[float] = []
        for index, left in enumerate(coords):
            right = coords[index + 1] if index + 1 < len(coords) else coords[0] + TWO_PI
            left_row = rows[index]
            right_row = rows[index + 1] if index + 1 < len(rows) else rows[0]
            increment = principal_angle(right_row["determinant"] * np.conj(left_row["determinant"]))
            if abs(increment) > float(phase_step):
                new_points.append(0.5 * (left + right))
        if not new_points or len(coords) >= int(max_points) or round_index == int(max_rounds):
            refinement_rounds = round_index
            break
        remaining = int(max_points) - len(coords)
        coords.extend(new_points[:remaining])
        coords = sorted(set(round(float(x), 14) for x in coords))
        refinement_rounds = round_index + 1

    coords = sorted(coords)
    rows = [evaluate(x) for x in coords]
    determinants = np.asarray([row["determinant"] for row in rows], dtype=complex)
    increments = []
    for index in range(len(rows)):
        next_index = (index + 1) % len(rows)
        increments.append(principal_angle(determinants[next_index] * np.conj(determinants[index])))
    winding_float = float(np.sum(increments) / TWO_PI)
    winding_int = int(np.rint(winding_float))
    integer_residual = float(abs(winding_float - winding_int))

    det_unwrapped = [float(rows[0]["det_phase"])]
    for index in range(1, len(rows)):
        det_unwrapped.append(det_unwrapped[-1] + principal_angle(determinants[index] * np.conj(determinants[index - 1])))
    branch_array = track_phase_branches([row["eigenphases"] for row in rows])

    trace_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = {
            "ky_coordinate": float(coords[index]),
            "ky_wrapped": float(wrap_k(coords[index])),
            "det_phase": float(row["det_phase"]),
            "det_phase_unwrapped": float(det_unwrapped[index]),
            "min_overlap_singular": float(row["min_overlap_singular"]),
            "min_kx_direct_gap": float(row["min_kx_direct_gap"]),
        }
        for branch in range(branch_array.shape[1]):
            item[f"wilson_phase_branch_{branch}"] = float(branch_array[index, branch])
        trace_rows.append(item)

    # Add a closing point for plotting the full winding.
    closing = dict(trace_rows[0])
    closing["ky_coordinate"] = float(coords[0] + TWO_PI)
    closing["ky_wrapped"] = float(trace_rows[0]["ky_wrapped"])
    closing["det_phase_unwrapped"] = float(det_unwrapped[0] + TWO_PI * winding_float)
    trace_rows.append(closing)

    summary = {
        "raw_winding_float": winding_float,
        "raw_winding_int": winding_int,
        "integer_residual": integer_residual,
        "n_ky_points_final": int(len(coords)),
        "adaptive_refinement_rounds": int(refinement_rounds),
        "max_abs_phase_increment": float(np.max(np.abs(increments))),
        "min_overlap_singular": float(min(row["min_overlap_singular"] for row in rows)),
        "min_kx_direct_gap": float(min(row["min_kx_direct_gap"] for row in rows)),
    }
    return summary, pd.DataFrame(trace_rows)


# =============================================================================
# Probe execution and consensus
# =============================================================================


def run_wilson_attempts(
    *,
    core,
    step5,
    pair: pd.Series,
    probes: pd.DataFrame,
    ky_anchors: list[float],
    config: Step13Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempt_rows: list[dict[str, Any]] = []
    trace_frames: list[pd.DataFrame] = []
    gap_rows: list[dict[str, Any]] = []

    for _, probe in probes.iterrows():
        probe_id = str(probe["probe_id"])
        probe_role = str(probe["probe_role"])
        probe_lambda = float(probe["lambda"])
        probe_r3 = float(probe["r3"])
        probe_r4 = float(probe["r4"])
        probe_expected = int(probe["expected_chern_up"])
        raw = step5.raw_on_path(pair, probe_lambda)
        gap_checkpoint = config.output_dir / "wilson_checkpoints" / f"{probe_id}_gap.json"
        if gap_checkpoint.is_file() and not config.force_recalculate:
            gap = json.loads(gap_checkpoint.read_text(encoding="utf-8"))
        else:
            gap = audit_global_spin_gap(
                core,
                raw,
                grid_n=int(config.gap_grid_n),
                local_starts=int(config.gap_local_starts),
            )
            atomic_json(gap, gap_checkpoint)
        gap_rows.append({
            "probe_id": probe_id,
            "probe_role": probe_role,
            "lambda": probe_lambda,
            "r3": probe_r3,
            "r4": probe_r4,
            "expected_chern_up": probe_expected,
            **gap,
        })

        h_fn = lambda kx, ky, raw=raw: spin_hamiltonian(core, raw, kx, ky)
        for attempt_index, (nkx, nky, shift) in enumerate(
            zip(config.wilson_nkx, config.wilson_initial_nky, config.wilson_shifts)
        ):
            checkpoint = config.output_dir / "wilson_checkpoints" / f"{probe_id}_attempt{attempt_index:02d}.json"
            trace_path = config.output_dir / "wilson_traces" / f"{probe_id}_attempt{attempt_index:02d}.csv"
            if checkpoint.is_file() and trace_path.is_file() and not config.force_recalculate:
                summary = json.loads(checkpoint.read_text(encoding="utf-8"))
                trace = pd.read_csv(trace_path, low_memory=False)
            else:
                summary, trace = adaptive_wilson_winding(
                    h_fn,
                    int(core.N_OCC_SPIN),
                    nkx=int(nkx),
                    initial_nky=int(nky),
                    shift_x=float(shift[0]),
                    shift_y=float(shift[1]),
                    phase_step=float(config.adaptive_phase_step),
                    max_points=int(config.adaptive_max_points),
                    max_rounds=int(config.adaptive_max_rounds),
                    ky_anchors=ky_anchors,
                )
                atomic_json(summary, checkpoint)
                atomic_csv(trace, trace_path)
            row = {
                "probe_id": probe_id,
                "probe_role": probe_role,
                "lambda": probe_lambda,
                "r3": probe_r3,
                "r4": probe_r4,
                "expected_chern_up": probe_expected,
                "attempt_index": int(attempt_index),
                "nkx": int(nkx),
                "initial_nky": int(nky),
                "shift_x": float(shift[0]),
                "shift_y": float(shift[1]),
                **summary,
            }
            attempt_rows.append(row)
            trace = trace.copy()
            trace.insert(0, "attempt_index", int(attempt_index))
            trace.insert(0, "probe_id", probe_id)
            trace_frames.append(trace)

    return pd.DataFrame(attempt_rows), pd.concat(trace_frames, ignore_index=True), pd.DataFrame(gap_rows)


def determine_orientation(attempts: pd.DataFrame, config: Step13Config) -> dict[str, Any]:
    control = attempts[attempts["probe_id"].astype(str).eq("control_pre_minus2")].copy()
    valid = control[
        control["integer_residual"].astype(float).le(float(config.integer_residual_tolerance))
        & control["min_overlap_singular"].astype(float).ge(float(config.minimum_overlap_singular_value))
        & control["max_abs_phase_increment"].astype(float).le(1.05 * float(config.adaptive_phase_step))
    ]
    raw_values = valid["raw_winding_int"].astype(int).to_list()
    if not raw_values or len(set(raw_values)) != 1 or abs(raw_values[0]) != 2:
        return {
            "orientation_calibrated": 0,
            "orientation_factor": np.nan,
            "raw_control_values": raw_values,
            "reason": "pre-control Wilson winding did not have a unique |C|=2 consensus",
        }
    factor = int(-2 // raw_values[0])
    return {
        "orientation_calibrated": 1,
        "orientation_factor": factor,
        "raw_control_values": raw_values,
        "reason": "calibrated against independently strict Step12M C_up=-2 control",
    }


def build_probe_consensus(
    attempts: pd.DataFrame,
    gaps: pd.DataFrame,
    orientation: dict[str, Any],
    config: Step13Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts = attempts.copy()
    factor = orientation.get("orientation_factor")
    if orientation.get("orientation_calibrated"):
        attempts["oriented_chern_up"] = attempts["raw_winding_int"].astype(int) * int(factor)
    else:
        attempts["oriented_chern_up"] = np.nan
    attempts["attempt_reliable"] = (
        attempts["integer_residual"].astype(float).le(float(config.integer_residual_tolerance))
        & attempts["min_overlap_singular"].astype(float).ge(float(config.minimum_overlap_singular_value))
        & attempts["max_abs_phase_increment"].astype(float).le(1.05 * float(config.adaptive_phase_step))
    ).astype(int)

    gap_map = gaps.set_index("probe_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for probe_id, group in attempts.groupby("probe_id", sort=False):
        reliable = group[group["attempt_reliable"].astype(int).eq(1)]
        values = reliable["oriented_chern_up"].dropna().astype(int).to_list()
        consensus = bool(values) and len(set(values)) == 1
        consensus_value = int(values[0]) if consensus else None
        expected = int(group.iloc[0]["expected_chern_up"])
        gap = gap_map[str(probe_id)]
        positive_gap = float(gap["spin_internal_gap"]) > float(config.minimum_positive_spin_gap)
        rows.append({
            "probe_id": str(probe_id),
            "probe_role": str(group.iloc[0]["probe_role"]),
            "lambda": float(group.iloc[0]["lambda"]),
            "r3": float(group.iloc[0]["r3"]),
            "r4": float(group.iloc[0]["r4"]),
            "expected_chern_up": expected,
            "wilson_attempt_count": int(len(group)),
            "wilson_reliable_attempt_count": int(len(reliable)),
            "wilson_unique_oriented_integer_count": int(len(set(values))),
            "wilson_chern_up_consensus": np.nan if consensus_value is None else consensus_value,
            "wilson_consensus": int(consensus),
            "spin_internal_gap": float(gap["spin_internal_gap"]),
            "positive_spin_gap": int(positive_gap),
            "expected_value_match": int(consensus and consensus_value == expected),
            "probe_certificate_pass": int(consensus and consensus_value == expected and positive_gap),
            "minimum_overlap_singular": float(group["min_overlap_singular"].min()),
            "maximum_integer_residual": float(group["integer_residual"].max()),
            "maximum_phase_increment": float(group["max_abs_phase_increment"].max()),
        })
    return attempts, pd.DataFrame(rows)


# =============================================================================
# Certificate update
# =============================================================================


def update_closure_certificates(
    original: pd.DataFrame,
    probe_consensus: pd.DataFrame,
) -> pd.DataFrame:
    frame = original.sort_values("mean_critical_lambda").reset_index(drop=True).copy()
    middle = probe_consensus[probe_consensus["probe_role"].astype(str).eq("intermediate")]
    middle_pass = int((middle["probe_certificate_pass"].astype(int) == 1).sum()) >= 1
    pre_pass = bool(
        probe_consensus.loc[
            probe_consensus["probe_id"].astype(str).eq("control_pre_minus2"),
            "probe_certificate_pass",
        ].astype(int).eq(1).all()
    )
    post_pass = bool(
        probe_consensus.loc[
            probe_consensus["probe_id"].astype(str).eq("control_post_zero"),
            "probe_certificate_pass",
        ].astype(int).eq(1).all()
    )
    if len(frame) >= 2 and pre_pass and middle_pass and post_pass:
        assignments = [(-2, 2), (2, 0)]
        for index, (left, right) in enumerate(assignments):
            frame.loc[index, "left_chern_up"] = left
            frame.loc[index, "right_chern_up"] = right
            frame.loc[index, "observed_delta_chern_up"] = right - left
            charge = int(frame.loc[index, "berry_charge_sum_up"])
            match = int(charge == right - left)
            frame.loc[index, "signed_charge_matches"] = match
            frame.loc[index, "closure_group_certificate_pass"] = int(
                match
                and int(frame.loc[index, "all_berry_charges_consensus"]) == 1
                and int(frame.loc[index, "all_kp_jacobians_rank3"]) == 1
            )
    return frame


def build_final_certificate(
    *,
    inputs: dict[str, Any],
    probes: pd.DataFrame,
    attempts: pd.DataFrame,
    probe_consensus: pd.DataFrame,
    orientation: dict[str, Any],
    updated_groups: pd.DataFrame,
    config: Step13Config,
) -> dict[str, Any]:
    middle = probe_consensus[probe_consensus["probe_role"].astype(str).eq("intermediate")]
    n_middle_pass = int(middle["probe_certificate_pass"].astype(int).sum())
    no_middle_contradiction = not bool(
        ((middle["wilson_consensus"].astype(int) == 1)
         & (middle["wilson_chern_up_consensus"].fillna(999).astype(int) != 2)).any()
    )
    pre_pass = bool(
        probe_consensus.loc[
            probe_consensus["probe_id"].astype(str).eq("control_pre_minus2"),
            "probe_certificate_pass",
        ].astype(int).eq(1).all()
    )
    post_pass = bool(
        probe_consensus.loc[
            probe_consensus["probe_id"].astype(str).eq("control_post_zero"),
            "probe_certificate_pass",
        ].astype(int).eq(1).all()
    )
    intermediate_pass = (
        n_middle_pass >= int(config.minimum_middle_certified_points)
        and no_middle_contradiction
    )
    groups_pass = bool(len(updated_groups) == 2 and updated_groups["closure_group_certificate_pass"].astype(int).eq(1).all())

    original = inputs["final_certificate"]
    target = original["target_edge_certificate"]
    charge_total = int(target["berry_charge_sum_over_all_groups"])
    charge_match = bool(target["signed_total_charge_matches_grid_delta"])
    complete = bool(
        orientation.get("orientation_calibrated")
        and pre_pass
        and post_pass
        and intermediate_pass
        and groups_pass
        and charge_total == 2
        and charge_match
    )

    return {
        "code_version": CODE_VERSION,
        "research_design": "adaptive_non_abelian_wilson_loop_intermediate_sector_certification",
        "fixed_background_parameters": inputs["background"],
        "target_edge_id": TARGET_EDGE_ID,
        "closure_lambdas": inputs["closure_lambdas"],
        "orientation_calibration": orientation,
        "control_pre_minus2_pass": int(pre_pass),
        "control_post_zero_pass": int(post_pass),
        "n_intermediate_probe_points": int(len(middle)),
        "n_intermediate_points_certified_as_plus2": n_middle_pass,
        "minimum_required_intermediate_points": int(config.minimum_middle_certified_points),
        "no_reliable_middle_contradiction": int(no_middle_contradiction),
        "intermediate_Cup_plus2_certified": int(intermediate_pass),
        "updated_strict_segment_chern_sequence": [-2, 2, 0] if intermediate_pass else [-2, None, 0],
        "generic_four_valley_charge": 4,
        "SigmaPrime_two_valley_charge": -2,
        "berry_charge_sum_over_all_groups": charge_total,
        "signed_total_charge_matches_endpoint_delta": int(charge_match),
        "all_updated_closure_groups_certified": int(groups_pass),
        "target_sequence_minus2_to_plus2_to_zero": int(intermediate_pass and groups_pass),
        "edge_multiclosure_certificate_pass": int(complete),
        "step13M_research_complete": bool(complete),
        "valid_claim_if_pass": (
            "Along the fixed-background r3=0.060 junction path, adaptive non-Abelian Wilson loops "
            "independently certify a narrow C_up=+2 sector between a generic four-valley +4 closure "
            "and a Sigma' two-valley -2 closure, completing the strict sequence -2 -> +2 -> 0."
        ),
        "not_claimed": (
            "This certification is local to the fixed-background target path. It does not fill the entire "
            "two-dimensional uncertain band with C_up=+2 and does not define a universal seven-dimensional rule."
        ),
        "probe_summary": probe_consensus.to_dict("records"),
    }


# =============================================================================
# Figures
# =============================================================================


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_wilson_flows(
    traces: pd.DataFrame,
    consensus: pd.DataFrame,
    orientation: dict[str, Any],
    output_dir: Path,
) -> None:
    selected_ids = ["control_pre_minus2", "middle_03", "control_post_zero"]
    available = [pid for pid in selected_ids if pid in set(traces["probe_id"].astype(str))]
    if not available:
        return
    factor = int(orientation.get("orientation_factor", 1)) if orientation.get("orientation_calibrated") else 1
    fig, axes = plt.subplots(1, len(available), figsize=(4.4 * len(available), 3.8), squeeze=False)
    for ax, probe_id in zip(axes[0], available):
        group = traces[traces["probe_id"].astype(str).eq(probe_id)]
        best_attempt = int(group["attempt_index"].max())
        curve = group[group["attempt_index"].astype(int).eq(best_attempt)].sort_values("ky_coordinate")
        x = (curve["ky_coordinate"].to_numpy(float) - curve["ky_coordinate"].min()) / TWO_PI
        y = factor * (curve["det_phase_unwrapped"].to_numpy(float) - curve["det_phase_unwrapped"].iloc[0]) / TWO_PI
        ax.plot(x, y, linewidth=1.6)
        row = consensus[consensus["probe_id"].astype(str).eq(probe_id)].iloc[0]
        value = row["wilson_chern_up_consensus"]
        title_value = "unresolved" if pd.isna(value) else f"C↑ = {int(value):+d}"
        ax.set_title(f"{probe_id}\n{title_value}")
        ax.set_xlabel(r"$k_y$ cycle")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(r"oriented $\Delta\arg\det W_x/2\pi$")
    fig.suptitle("Adaptive occupied-subspace Wilson-loop winding")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "step13M_wilson_determinant_winding")


def plot_probe_consensus(
    attempts: pd.DataFrame,
    consensus: pd.DataFrame,
    closure_lambdas: list[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    reliable = attempts[attempts["attempt_reliable"].astype(int).eq(1)]
    for attempt_index, group in reliable.groupby("attempt_index"):
        ax.scatter(
            group["lambda"], group["oriented_chern_up"],
            s=42, alpha=0.65, label=f"Wilson discretization {int(attempt_index)+1}",
        )
    passed = consensus[consensus["probe_certificate_pass"].astype(int).eq(1)]
    ax.scatter(
        passed["lambda"], passed["wilson_chern_up_consensus"],
        marker="o", facecolors="none", edgecolors="black", s=110, linewidths=1.4,
        label="certified probe",
    )
    lam1, lam2 = closure_lambdas
    ax.axvline(lam1, linestyle="--", linewidth=1.2)
    ax.axvline(lam2, linestyle="--", linewidth=1.2)
    ax.plot([0, lam1, lam1, lam2, lam2, 1], [-2, -2, 2, 2, 0, 0], linewidth=1.2, alpha=0.7, label="target sequence")
    ax.set_xlim(0, 1)
    ax.set_ylim(-2.6, 2.6)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r"path coordinate $\lambda$")
    ax.set_ylabel(r"spin-up Chern number $C_\uparrow$")
    ax.set_title("Direct Wilson-loop certification of the narrow intermediate sector")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "step13M_intermediate_wilson_consensus")


def plot_target_path_final(
    gap_track: pd.DataFrame,
    consensus: pd.DataFrame,
    closure_lambdas: list[float],
    output_dir: Path,
) -> None:
    lam1, lam2 = closure_lambdas
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})
    ax = axes[0]
    ax.semilogy(gap_track["lambda"], np.maximum(gap_track["tracked_min_gap"], 1e-16), linewidth=1.4)
    ax.axvline(lam1, linestyle="--", linewidth=1.2)
    ax.axvline(lam2, linestyle="--", linewidth=1.2)
    ax.set_ylabel("tracked spin-up internal gap")
    ax.set_title("Resolved multi-closure path and directly certified Chern sequence")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot([0, lam1, lam1, lam2, lam2, 1], [-2, -2, 2, 2, 0, 0], linewidth=2.0)
    for _, row in consensus.iterrows():
        if int(row["probe_certificate_pass"]) == 1:
            c = int(row["wilson_chern_up_consensus"])
            ax.scatter(float(row["lambda"]), c, s=65, color=PHASE_COLORS[c], edgecolor="black", linewidth=0.6, zorder=5)
        else:
            ax.scatter(float(row["lambda"]), 0, marker="x", s=55, color="black", zorder=5)
    ax.axvline(lam1, linestyle="--", linewidth=1.2)
    ax.axvline(lam2, linestyle="--", linewidth=1.2)
    ax.text(lam1, 2.35, "generic four-valley\ncharge +4", ha="center", va="bottom", fontsize=8)
    ax.text(lam2, 2.35, r"$\Sigma'$ two-valley" + "\ncharge -2", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(r"path coordinate $\lambda$")
    ax.set_ylabel(r"$C_\uparrow$")
    ax.set_ylim(-2.7, 2.8)
    ax.set_yticks([-2, 0, 2])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "step13M_final_target_path_atlas")


def plot_junction_zoom(
    grid: pd.DataFrame,
    consensus: pd.DataFrame,
    edge: pd.Series,
    closure_lambdas: list[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.7))
    reliable = grid[grid["paper_phase_code"].isin([-2, -1, 0, 1, 2])]
    unreliable = grid[~grid["paper_phase_code"].isin([-2, -1, 0, 1, 2])]
    for phase, group in reliable.groupby("paper_phase_code"):
        phase = int(phase)
        ax.scatter(group["r3"], group["r4"], s=31, color=PHASE_COLORS[phase], label=rf"strict $C_\uparrow={phase:+d}$")
    if not unreliable.empty:
        ax.scatter(unreliable["r3"], unreliable["r4"], marker="x", s=26, color="black", alpha=0.55, label="uniform-grid boundary/unreliable")

    r3 = float(edge["r3_0"])
    r4_start, r4_end = float(edge["r4_0"]), float(edge["r4_1"])
    lam1, lam2 = closure_lambdas
    r4_1 = r4_start + lam1 * (r4_end-r4_start)
    r4_2 = r4_start + lam2 * (r4_end-r4_start)
    ax.plot([r3, r3], [r4_start, r4_1], color=PHASE_COLORS[-2], linewidth=3.0)
    ax.plot([r3, r3], [r4_1, r4_2], color=PHASE_COLORS[2], linewidth=3.0)
    ax.plot([r3, r3], [r4_2, r4_end], color=PHASE_COLORS[0], linewidth=3.0)
    ax.scatter([r3, r3], [r4_1, r4_2], marker="*", s=150, color="black", zorder=7, label="certified closure")

    passed = consensus[consensus["probe_certificate_pass"].astype(int).eq(1)]
    for _, row in passed.iterrows():
        c = int(row["wilson_chern_up_consensus"])
        ax.scatter(float(row["r3"]), float(row["r4"]), s=80, color=PHASE_COLORS[c], edgecolor="black", linewidth=0.8, zorder=8)

    ax.set_xlabel(r"$r_3$")
    ax.set_ylabel(r"$r_4$")
    ax.set_title("Right-junction zoom with path-resolved Wilson certification")
    ax.grid(alpha=0.20)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "step13M_right_junction_wilson_certified")


# =============================================================================
# Main workflow
# =============================================================================


def run_step13m(
    *,
    tts_archive: str | Path,
    step12_source: str | Path,
    output_dir: str | Path | None = None,
    config: Step13Config | None = None,
    run_physics: bool = True,
    run_plots: bool = True,
) -> dict[str, Any]:
    if config is None:
        if output_dir is None:
            output_dir = Path.cwd() / "outputs_tts_step13M_adaptive_wilson_intermediate_chern"
        config = Step13Config(output_dir=Path(output_dir))
    elif output_dir is not None:
        config.output_dir = Path(output_dir)
    config = config.normalized()

    print("[1/5] Load Step12M evidence and define Wilson probes")
    inputs = load_step12_inputs(step12_source)
    inputs["_step12_source"] = str(Path(step12_source).expanduser().resolve())
    probes = build_probe_table(inputs, config)
    atomic_csv(probes, config.output_dir / "step13M_01_probe_definitions.csv")
    audit = {
        "code_version": CODE_VERSION,
        "tts_archive": str(Path(tts_archive).expanduser().resolve()),
        "step12_source": str(Path(step12_source).expanduser().resolve()),
        "fixed_background_parameters": inputs["background"],
        "target_edge": inputs["edge"].to_dict(),
        "closure_lambdas": inputs["closure_lambdas"],
        "config": asdict(config),
    }
    atomic_json(audit, config.output_dir / "step13M_00_input_audit.json")

    if not run_physics:
        return {"code_version": CODE_VERSION, "physics_run": False, "probe_count": len(probes)}

    print("[2/5] Configure frozen TTS Hamiltonian and audit occupied-subspace gaps")
    core, step4, step5, step3, strict_config, search_config = configure_physics(tts_archive, config)
    pair = step11.build_pair_from_edge(inputs["edge"], inputs["background"])
    ky_anchors = closure_ky_anchors(inputs)

    print("[3/5] Adaptive non-Abelian Wilson loops")
    attempts, traces, gaps = run_wilson_attempts(
        core=core,
        step5=step5,
        pair=pair,
        probes=probes,
        ky_anchors=ky_anchors,
        config=config,
    )
    atomic_csv(gaps, config.output_dir / "step13M_02_spin_gap_audit.csv")
    atomic_csv(attempts, config.output_dir / "step13M_03_wilson_attempts_raw.csv")
    atomic_csv(traces, config.output_dir / "step13M_05_wilson_phase_traces.csv")

    orientation = determine_orientation(attempts, config)
    attempts_oriented, consensus = build_probe_consensus(attempts, gaps, orientation, config)
    atomic_csv(attempts_oriented, config.output_dir / "step13M_03_wilson_attempts.csv")
    atomic_csv(consensus, config.output_dir / "step13M_04_wilson_probe_consensus.csv")
    atomic_json(orientation, config.output_dir / "step13M_04b_orientation_calibration.json")

    print("[4/5] Update closure-group and final mechanism certificates")
    updated_groups = update_closure_certificates(inputs["group_certificates"], consensus)
    atomic_csv(updated_groups, config.output_dir / "step13M_07_updated_closure_group_certificates.csv")
    certificate = build_final_certificate(
        inputs=inputs,
        probes=probes,
        attempts=attempts_oriented,
        probe_consensus=consensus,
        orientation=orientation,
        updated_groups=updated_groups,
        config=config,
    )
    atomic_json(certificate, config.output_dir / "step13M_08_final_intermediate_chern_certificate.json")

    summary = pd.DataFrame([
        {
            "updated_segment_0_chern": -2,
            "updated_segment_1_chern": 2 if certificate["intermediate_Cup_plus2_certified"] else np.nan,
            "updated_segment_2_chern": 0,
            "generic_four_valley_charge": 4,
            "SigmaPrime_two_valley_charge": -2,
            "total_charge": 2,
            "edge_multiclosure_certificate_pass": certificate["edge_multiclosure_certificate_pass"],
        }
    ])
    atomic_csv(summary, config.output_dir / "step13M_06_updated_strict_segment_sequence.csv")

    print("[5/5] Publication figures")
    if run_plots:
        plot_wilson_flows(traces, consensus, orientation, config.output_dir)
        plot_probe_consensus(attempts_oriented, consensus, inputs["closure_lambdas"], config.output_dir)
        plot_target_path_final(inputs["gap_track"], consensus, inputs["closure_lambdas"], config.output_dir)
        plot_junction_zoom(inputs["grid"], consensus, inputs["edge"], inputs["closure_lambdas"], config.output_dir)

    print(json.dumps(certificate, ensure_ascii=False, indent=2, default=_json_default))
    return certificate


# =============================================================================
# Synthetic validation of the Wilson engine
# =============================================================================


def _qwz_hamiltonian(mass: float) -> Callable[[float, float], np.ndarray]:
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    def h(kx: float, ky: float) -> np.ndarray:
        return math.sin(kx)*sx + math.sin(ky)*sy + (mass + math.cos(kx) + math.cos(ky))*sz
    return h


def synthetic_wilson_validation() -> dict[str, Any]:
    topological, _ = adaptive_wilson_winding(
        _qwz_hamiltonian(-1.0), 1,
        nkx=121, initial_nky=61, shift_x=0.0, shift_y=0.0,
        phase_step=0.35*math.pi, max_points=301, max_rounds=6,
    )
    trivial, _ = adaptive_wilson_winding(
        _qwz_hamiltonian(3.0), 1,
        nkx=121, initial_nky=61, shift_x=0.0, shift_y=0.0,
        phase_step=0.35*math.pi, max_points=301, max_rounds=6,
    )
    return {
        "topological_abs_winding": abs(int(topological["raw_winding_int"])),
        "trivial_winding": int(trivial["raw_winding_int"]),
        "synthetic_validation_pass": bool(abs(int(topological["raw_winding_int"])) == 1 and int(trivial["raw_winding_int"]) == 0),
        "topological_summary": topological,
        "trivial_summary": trivial,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-archive", required=True, type=Path)
    parser.add_argument("--step12-source", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_tts_step13M_adaptive_wilson_intermediate_chern"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-recalculate", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Step13Config(
        output_dir=args.output_dir,
        quick=bool(args.quick),
        force_recalculate=bool(args.force_recalculate),
    )
    run_step13m(
        tts_archive=args.tts_archive,
        step12_source=args.step12_source,
        config=config,
        run_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
