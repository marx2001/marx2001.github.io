from __future__ import annotations

"""
TTS Step12M — right-junction multi-closure repair and publication phase map
============================================================================

Purpose
-------
Resolve the only failed Step11M edge certificate:

    edge_03_03__03_04:
    (r3, r4) = (0.060, 0.128) -> (0.060, 0.132)
    C_up      = -2             -> 0

Step11M found a certified Sigma' two-valley closing with Berry charge -2, but
that charge did not equal the endpoint Chern change +2.  Step12M therefore
forces a separate full-Brillouin-zone search for a preceding generic four-valley
closing.  The target hypothesis is

    C_up: -2 --(four generic valleys, +4)--> +2
              --(two Sigma' valleys, -2)--> 0.

The workflow also recomputes a refined strict 13x13 junction map and produces
publication-oriented figures with phase colors, unreliable-point markers,
legends, a junction zoom, the target path, and certified closure annotations.

Scope
-----
All background parameters (m_e, t1, t2, r1, r2) remain fixed.  Only the same
Lieb-aligned r3-r4 plane is studied.  No universal seven-dimensional boundary
is claimed.
"""

import argparse
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

import TTS_step09M_minus2_local_kp_refinement as base
import TTS_step09M_minus2_linear_signed_v2 as linear_v2
import TTS_step09M_D_multiclosure_path_atlas as step09d
import TTS_step10M_generic_four_valley_boundary_atlas as step10
import TTS_step11M_lieb_aligned_fixed_slice as step11


CODE_VERSION = "TTS_STEP12M_JUNCTION_MULTICLOSURE_REPAIR_V1_2_FORCE_SOURCE_LOAD_20260724"
REDUCED7 = list(base.REDUCED7)
FIXED5 = ["m_e", "t1", "t2", "r1", "r2"]
TARGET_EDGE_ID = "edge_03_03__03_04"

PHASE_COLORS = {
    -2: "#3B6FB6",
    -1: "#7FB6D6",
     0: "#F2F2F2",
     1: "#F2B36D",
     2: "#C84A3A",
}
PHASE_ORDER = [-2, -1, 0, 1, 2]


@dataclass
class Step12Config:
    output_dir: Path

    # Target edge is fixed by the Step11M junction grid.
    target_edge_id: str = TARGET_EDGE_ID

    # Targeted generic four-valley search.  The expected direct boundary from
    # the certified right-upper branch intersects r3=0.060 near lambda~0.624.
    generic_lambda_bracket: tuple[float, float] = (0.42, 0.80)
    sigma_prime_lambda_bracket: tuple[float, float] = (0.80, 1.00)
    generic_predicted_lambda: float = 0.6244
    generic_k_box_half_width: float = 0.90
    generic_powell_restarts: int = 12
    generic_de_repeats: int = 4
    closure_gap_tol: float = 2.0e-8
    closure_lambda_dedup_tol: float = 8.0e-4

    # Diagnostic gap tracks along the edge.
    gap_track_lambda_points: int = 241
    sigma_prime_kappa_points: int = 241

    # Refined strict junction map.  This range contains r3=0.060 and both
    # endpoints r4=0.128, 0.132, while extending above the second closing.
    refined_r3_range: tuple[float, float] = (0.056, 0.064)
    refined_r4_range: tuple[float, float] = (0.128, 0.134)
    refined_grid_n: int = 13

    strict_gap_grids: tuple[int, ...] = (71, 101)
    strict_chern_grids: tuple[int, ...] = (61, 81, 101)
    strict_chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )

    quick: bool = False
    force_recalculate: bool = False
    random_seed: int = 20260724

    def normalized(self) -> "Step12Config":
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in [
            "figures",
            "refined_junction_points",
            "_search_core",
            "_strict_core",
        ]:
            (self.output_dir / sub).mkdir(exist_ok=True)
        if self.refined_grid_n < 5:
            raise ValueError("refined_grid_n must be >= 5")
        if self.gap_track_lambda_points < 61:
            raise ValueError("gap_track_lambda_points must be >= 61")
        lo, hi = sorted(self.generic_lambda_bracket)
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("generic_lambda_bracket must lie in [0,1]")
        return self


# =============================================================================
# Generic I/O
# =============================================================================


def _select_member(zf: zipfile.ZipFile, token: str, suffix: str | None = None) -> str:
    matches = []
    for name in zf.namelist():
        if token not in Path(name).name and token not in name:
            continue
        if suffix is not None and not name.lower().endswith(suffix.lower()):
            continue
        matches.append(name)
    if not matches:
        raise FileNotFoundError(f"No member containing {token!r}")
    return sorted(matches, key=lambda x: (len(Path(x).parts), len(x)))[0]


def read_csv_token(source: str | Path, token: str) -> pd.DataFrame:
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            member = _select_member(zf, token, ".csv")
            with zf.open(member) as stream:
                return pd.read_csv(stream, low_memory=False)
    if source.is_dir():
        matches = sorted(source.rglob(f"*{token}*"))
        if not matches:
            raise FileNotFoundError(token)
        return pd.read_csv(matches[0], low_memory=False)
    raise FileNotFoundError(source)


def safe_read_csv_path(
    path: str | Path,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    required: bool = False,
) -> pd.DataFrame:
    """Read a CSV without failing on a valid zero-row/zero-byte checkpoint.

    Some Step12M diagnostics, especially the refined transition-edge table, may
    legitimately contain zero rows. Older code wrote such a DataFrame without
    column names, which creates an empty CSV and makes pandas raise
    ``EmptyDataError`` during the plotting-only reload. Returning an empty frame
    with a stable schema keeps the physical result intact and lets plotting run.
    """
    path = Path(path).expanduser().resolve()
    schema = list(columns) if columns is not None else None
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame(columns=schema)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=schema)
    if schema is not None:
        for column in schema:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        frame = frame.reindex(columns=schema + [c for c in frame.columns if c not in schema])
    return frame


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


def safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


# =============================================================================
# Input reconstruction
# =============================================================================


def load_background(step11_source: str | Path) -> dict[str, float]:
    payload = read_json_token(step11_source, "step11M_00_fixed_background.json")
    if "fixed_background_parameters" in payload:
        payload = payload["fixed_background_parameters"]
    background = {name: float(payload[name]) for name in FIXED5}
    return background


def load_target_edge(step11_source: str | Path, edge_id: str) -> pd.Series:
    edges = read_csv_token(step11_source, "step11M_04_right_junction_boundary_edges.csv")
    selected = edges[edges["edge_id"].astype(str).eq(edge_id)]
    if selected.empty:
        # Fallback to selected edge table.
        selected_edges = read_csv_token(step11_source, "step11M_05_selected_junction_edges.csv")
        selected = selected_edges[selected_edges["edge_id"].astype(str).eq(edge_id)]
    if selected.empty:
        raise KeyError(f"Target edge {edge_id!r} not found in Step11M output")
    return selected.iloc[0].copy()


def load_right_upper_seed(step11_source: str | Path, edge: pd.Series) -> pd.Series:
    points = read_csv_token(step11_source, "step11M_07_completed_certified_boundary_points.csv")
    selected = points[
        points["branch_hint"].astype(str).eq("generic_right_upper")
        & points["point_certificate_pass"].astype(int).eq(1)
    ].copy()
    if selected.empty:
        raise RuntimeError("No certified generic_right_upper seed in Step11M output")
    target = np.array([
        0.5 * (float(edge["r3_0"]) + float(edge["r3_1"])),
        0.5 * (float(edge["r4_0"]) + float(edge["r4_1"])),
    ])
    distance = np.hypot(
        selected["r3"].to_numpy(float) - target[0],
        selected["r4"].to_numpy(float) - target[1],
    )
    return selected.iloc[int(np.argmin(distance))].copy()


def validate_fixed_background(
    background: dict[str, float],
    edge: pd.Series,
    seed: pd.Series,
    tolerance: float = 1.0e-10,
) -> None:
    for name in FIXED5:
        expected = float(background[name])
        if name in seed and abs(float(seed[name]) - expected) > tolerance:
            raise ValueError(f"Seed background mismatch in {name}")
    if int(edge["chern_0"]) != -2 or int(edge["chern_1"]) != 0:
        raise ValueError(
            f"Expected target edge -2 -> 0, got {edge['chern_0']} -> {edge['chern_1']}"
        )


# =============================================================================
# Frozen physical core
# =============================================================================


def configure_physics(
    tts_archive: str | Path,
    config: Step12Config,
):
    core, step4, step5, source_dir = base.import_reference_modules(tts_archive)
    linear_v2.patch_step5_linear_path(step5, core)
    step3 = linear_v2.import_step3(source_dir)

    continuation = step10.ContinuationConfig(
        output_dir=config.output_dir / "_search_core",
        strict_gap_grids=tuple(config.strict_gap_grids),
        strict_chern_grids=tuple(config.strict_chern_grids),
        strict_chern_shifts=tuple(config.strict_chern_shifts),
        closure_gap_tol=float(config.closure_gap_tol),
        quick=bool(config.quick),
        random_seed=int(config.random_seed),
    ).normalized()
    strict_config = step10.strict_step3_config(
        step3,
        config.output_dir / "_strict_core",
        continuation,
    )
    search_config = step10.step5_config(
        step5,
        config.output_dir / "_search_core",
        continuation,
    )
    # Explicitly strengthen full-BZ search for the missed generic orbit.
    if config.quick:
        search_config.full_bz_de_popsize = 10
        search_config.full_bz_de_maxiter = 160
        search_config.sphere_k_radii = (0.03, 0.06)
    else:
        search_config.full_bz_de_popsize = 20
        search_config.full_bz_de_maxiter = 520
        search_config.local_starts_per_manifold = 14
        search_config.powell_maxiter = 1400
        search_config.sphere_k_radii = (0.02, 0.04, 0.08)
    search_config.spin_gap_accept_tol = float(config.closure_gap_tol)
    search_config.active_spin_gap_tol = max(2.0e-6, 20.0 * config.closure_gap_tol)
    search_config.random_seed = int(config.random_seed)
    step5._step10m_config = search_config
    return core, step4, step5, step3, strict_config, search_config


# =============================================================================
# Targeted full-BZ closure search
# =============================================================================


def wrap_k(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def d4_start_points(step4, kx: float, ky: float) -> list[tuple[float, float]]:
    orbit = step4.d4_orbit(wrap_k(kx), wrap_k(ky))
    starts: list[tuple[float, float]] = []
    for x, y in orbit:
        point = (wrap_k(x), wrap_k(y))
        if all(math.hypot(wrap_k(point[0] - a), wrap_k(point[1] - b)) > 1.0e-6 for a, b in starts):
            starts.append(point)
    return starts


def full_bz_gap_objective(step5, pair: pd.Series, x: np.ndarray) -> float:
    lam = float(np.clip(x[0], 0.0, 1.0))
    kx = wrap_k(float(x[1]))
    ky = wrap_k(float(x[2]))
    gap = step5.spin_middle_gap(kx, ky, pair, lam, "up")
    return float(gap * gap)


def _candidate_row(
    *,
    method: str,
    result,
    pair: pd.Series,
    step5,
    source_bracket: str,
    run_index: int,
) -> dict[str, Any]:
    lam, kx, ky = [float(v) for v in result.x]
    kx = wrap_k(kx)
    ky = wrap_k(ky)
    gap = float(step5.spin_middle_gap(kx, ky, pair, lam, "up"))
    return {
        "search_method": method,
        "source_bracket": source_bracket,
        "run_index": int(run_index),
        "critical_lambda": lam,
        "critical_kx": kx,
        "critical_ky": ky,
        "spin_up_gap": gap,
        "optimizer_success": int(bool(getattr(result, "success", True))),
        "optimizer_message": str(getattr(result, "message", "")),
        "optimizer_nfev": int(getattr(result, "nfev", -1)),
    }


def search_generic_four_valley(
    *,
    step4,
    step5,
    pair: pd.Series,
    seed: pd.Series,
    config: Step12Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lo, hi = sorted(config.generic_lambda_bracket)
    seed_kx = float(seed["critical_kx_representative"])
    seed_ky = float(seed["critical_ky_representative"])
    orbit_starts = d4_start_points(step4, seed_kx, seed_ky)

    # The local optimizers are initialized at the Step10M continuation valley.
    # A few nearby lambda starts prevent a flat local basin from hiding the zero.
    lambda_starts = np.linspace(
        max(lo, config.generic_predicted_lambda - 0.10),
        min(hi, config.generic_predicted_lambda + 0.10),
        max(3, min(7, config.generic_powell_restarts // 2)),
    )
    starts: list[np.ndarray] = []
    for lam in lambda_starts:
        for kx, ky in orbit_starts[:4]:
            starts.append(np.array([lam, kx, ky], dtype=float))
    starts = starts[: max(4, int(config.generic_powell_restarts))]

    candidates: list[dict[str, Any]] = []
    bounds = [(lo, hi), (-math.pi, math.pi), (-math.pi, math.pi)]
    for index, start in enumerate(starts):
        result = minimize(
            lambda x: full_bz_gap_objective(step5, pair, x),
            start,
            method="Powell",
            bounds=bounds,
            options={
                "xtol": 1.0e-12,
                "ftol": 1.0e-22,
                "maxiter": (700 if config.quick else 1800),
            },
        )
        candidates.append(
            _candidate_row(
                method="generic_seeded_Powell",
                result=result,
                pair=pair,
                step5=step5,
                source_bracket="generic",
                run_index=index,
            )
        )

    # Always perform several independent full-BZ DE searches.  Step11M only
    # invoked full-BZ fallback when no diagonal zero existed, which is why the
    # earlier generic closing could be missed after Sigma' had already closed.
    de_repeats = 2 if config.quick else int(config.generic_de_repeats)
    for index in range(de_repeats):
        result = differential_evolution(
            lambda x: full_bz_gap_objective(step5, pair, x),
            bounds=bounds,
            seed=int(config.random_seed + 1009 * (index + 1)),
            popsize=(10 if config.quick else 22),
            maxiter=(180 if config.quick else 600),
            tol=1.0e-12,
            polish=True,
            workers=1,
            updating="immediate",
        )
        candidates.append(
            _candidate_row(
                method="generic_full_BZ_DE",
                result=result,
                pair=pair,
                step5=step5,
                source_bracket="generic",
                run_index=index,
            )
        )

    accepted: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda r: float(r["spin_up_gap"])):
        if float(row["spin_up_gap"]) > float(config.closure_gap_tol):
            continue
        closure = {**row, "search_manifold": row["search_method"]}
        valleys = step5.enumerate_spin_valleys(closure, pair, step5._step10m_config)
        active = base.unique_active_spin_up_valleys(valleys, step5)
        row["n_active_spin_up_valleys"] = int(len(active))
        row["active_k_regions"] = ";".join(sorted(set(active["critical_k_region"].astype(str))))
        if len(active) != 4:
            continue
        if not active["critical_k_region"].astype(str).eq("generic").all():
            continue
        accepted.append(closure)

    accepted = step09d.global_deduplicate_closures(
        step5,
        accepted,
        float(config.closure_lambda_dedup_tol),
    )
    return candidates, accepted


def search_sigma_prime_closure(
    *,
    step5,
    pair: pd.Series,
    config: Step12Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lo, hi = sorted(config.sigma_prime_lambda_bracket)
    bracket = pd.Series(
        {
            "transition_id": TARGET_EDGE_ID + "_sigma_prime",
            "path_id": TARGET_EDGE_ID,
            "lambda_left": lo,
            "lambda_right": hi,
            "chern_left": np.nan,
            "chern_right": np.nan,
        }
    )
    candidates, closures = step5.search_transition_closures(
        bracket,
        pair,
        step5._step10m_config,
        12000,
    )
    selected: list[dict[str, Any]] = []
    for closure in closures:
        valleys = step5.enumerate_spin_valleys(closure, pair, step5._step10m_config)
        active = base.unique_active_spin_up_valleys(valleys, step5)
        if len(active) != 2:
            continue
        regions = set(active["critical_k_region"].astype(str))
        if not any("SigmaPrime" in region for region in regions):
            continue
        selected.append(dict(closure))
    selected = step09d.global_deduplicate_closures(
        step5,
        selected,
        float(config.closure_lambda_dedup_tol),
    )
    return candidates, selected


def combine_closures(
    step5,
    generic: list[dict[str, Any]],
    sigma_prime: list[dict[str, Any]],
    config: Step12Config,
) -> list[dict[str, Any]]:
    closures = [dict(row) for row in generic + sigma_prime]
    return step09d.global_deduplicate_closures(
        step5,
        closures,
        float(config.closure_lambda_dedup_tol),
    )


# =============================================================================
# Gap tracks for diagnostics
# =============================================================================


def minimize_k_in_box(
    *,
    step5,
    pair: pd.Series,
    lam: float,
    center: tuple[float, float],
    half_width: float,
) -> dict[str, float]:
    cx, cy = center
    lower_x = max(-math.pi, cx - half_width)
    upper_x = min(math.pi, cx + half_width)
    lower_y = max(-math.pi, cy - half_width)
    upper_y = min(math.pi, cy + half_width)

    def objective(x: np.ndarray) -> float:
        gap = step5.spin_middle_gap(float(x[0]), float(x[1]), pair, float(lam), "up")
        return float(gap * gap)

    result = minimize(
        objective,
        np.array([np.clip(cx, lower_x, upper_x), np.clip(cy, lower_y, upper_y)]),
        method="Powell",
        bounds=[(lower_x, upper_x), (lower_y, upper_y)],
        options={"xtol": 1.0e-10, "ftol": 1.0e-18, "maxiter": 600},
    )
    kx, ky = [float(v) for v in result.x]
    return {
        "generic_gap": float(step5.spin_middle_gap(kx, ky, pair, float(lam), "up")),
        "generic_kx": kx,
        "generic_ky": ky,
    }


def sigma_prime_gap_at_lambda(
    *,
    step5,
    pair: pd.Series,
    lam: float,
    kappa_points: int,
) -> dict[str, float]:
    kappas = np.linspace(-math.pi, math.pi, int(kappa_points), endpoint=False)
    gaps = np.array(
        [
            step5.spin_middle_gap(kappa, -kappa, pair, float(lam), "up")
            for kappa in kappas
        ],
        dtype=float,
    )
    index = int(np.argmin(gaps))
    left = float(kappas[max(0, index - 1)])
    right = float(kappas[min(len(kappas) - 1, index + 1)])
    if right > left:
        result = minimize_scalar(
            lambda kap: step5.spin_middle_gap(float(kap), -float(kap), pair, float(lam), "up") ** 2,
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1.0e-12, "maxiter": 600},
        )
        kappa = float(result.x)
    else:
        kappa = float(kappas[index])
    return {
        "sigma_prime_gap": float(
            step5.spin_middle_gap(kappa, -kappa, pair, float(lam), "up")
        ),
        "sigma_prime_kappa": kappa,
    }


def compute_gap_tracks(
    *,
    step5,
    pair: pd.Series,
    seed: pd.Series,
    config: Step12Config,
) -> pd.DataFrame:
    lambdas = np.linspace(0.0, 1.0, int(config.gap_track_lambda_points))
    generic_center = (
        float(seed["critical_kx_representative"]),
        float(seed["critical_ky_representative"]),
    )
    rows: list[dict[str, Any]] = []
    for index, lam in enumerate(lambdas):
        row: dict[str, Any] = {"lambda": float(lam), "lambda_index": int(index)}
        if config.generic_lambda_bracket[0] - 0.05 <= lam <= config.generic_lambda_bracket[1] + 0.05:
            result = minimize_k_in_box(
                step5=step5,
                pair=pair,
                lam=float(lam),
                center=generic_center,
                half_width=float(config.generic_k_box_half_width),
            )
            row.update(result)
            generic_center = (result["generic_kx"], result["generic_ky"])
        else:
            row.update({"generic_gap": np.nan, "generic_kx": np.nan, "generic_ky": np.nan})
        row.update(
            sigma_prime_gap_at_lambda(
                step5=step5,
                pair=pair,
                lam=float(lam),
                kappa_points=(121 if config.quick else int(config.sigma_prime_kappa_points)),
            )
        )
        finite = [
            value for value in [row["generic_gap"], row["sigma_prime_gap"]]
            if np.isfinite(value)
        ]
        row["tracked_min_gap"] = min(finite) if finite else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Strict segment labels and closure certificates
# =============================================================================


def strict_segment_probes(
    *,
    step3,
    strict_config,
    pair: pd.Series,
    closures: list[dict[str, Any]],
    path_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = step09d.closure_groups(closures)
    centres = [
        float(np.mean([float(row["critical_lambda"]) for row in group]))
        for group in groups
    ]
    boundaries = [0.0, *centres, 1.0]
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        probe = step09d.find_reliable_segment_probe(
            step3,
            strict_config,
            pair,
            path_id,
            float(left),
            float(right),
            (0.5, 0.35, 0.65, 0.2, 0.8),
            index,
        )
        results.append(probe)
        summary = probe["summary"]
        rows.append(
            {
                "path_id": path_id,
                "segment_index": index,
                "lambda_left_bound": float(left),
                "lambda_right_bound": float(right),
                "probe_lambda": summary.get("lambda"),
                "strict_chern_up": step09d.reliable_chern(summary),
                "strict_phase_label": summary.get("phase_label"),
                "min_direct_gap": summary.get("min_direct_gap"),
                "indirect_gap": summary.get("indirect_gap"),
                "strict_gap_verified": summary.get("strict_gap_verified"),
                "strict_chern_verified": summary.get("strict_chern_verified"),
            }
        )
    return results, rows


def certify_target_edge(
    *,
    edge: pd.Series,
    pair: pd.Series,
    closures: list[dict[str, Any]],
    step5,
    search_config,
    segment_results: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = step09d.closure_groups(closures)
    closure_rows: list[dict[str, Any]] = []
    valley_rows: list[dict[str, Any]] = []
    kp_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    berry_attempt_rows: list[dict[str, Any]] = []
    berry_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups):
        group_id = f"{TARGET_EDGE_ID}_step12_group{group_index:02d}"
        centre = float(np.mean([float(row["critical_lambda"]) for row in group]))
        group_charge = 0
        all_consensus = True
        all_rank3 = True
        valley_count = 0
        region_set: set[str] = set()
        for closure_index, closure in enumerate(group):
            closure_id = f"{group_id}_closure{closure_index:02d}"
            reduced = linear_v2.linear_reduced_on_path(pair, float(closure["critical_lambda"]))
            closure_rows.append(
                {
                    "edge_id": TARGET_EDGE_ID,
                    "closure_group_id": group_id,
                    "closure_id": closure_id,
                    **closure,
                    **reduced,
                }
            )
            analysis = step09d.analyse_closure(
                step5,
                pair,
                search_config,
                TARGET_EDGE_ID,
                group_id,
                closure_id,
                closure,
            )
            valleys = analysis["valleys"].copy()
            if len(valleys):
                valley_rows.extend(valleys.to_dict("records"))
                region_set.update(valleys["critical_k_region"].astype(str))
            kp_rows.extend(analysis["kp_rows"])
            derivative_rows.extend(analysis["derivative_rows"])
            gradient_rows.extend(analysis["gradient_rows"])
            berry_attempt_rows.extend(analysis["berry_attempt_rows"])
            berry_rows.extend(analysis["berry_summary_rows"])
            group_charge += int(analysis["charge_sum"])
            all_consensus = all_consensus and bool(analysis["all_consensus"])
            all_rank3 = all_rank3 and bool(analysis["all_rank3"])
            valley_count += int(analysis["n_active_valleys"])

        left_chern = step09d.reliable_chern(segment_results[group_index]["summary"])
        right_chern = step09d.reliable_chern(segment_results[group_index + 1]["summary"])
        delta = None if left_chern is None or right_chern is None else int(right_chern - left_chern)
        signed_match = (
            delta is not None
            and all_consensus
            and int(group_charge) == int(delta)
        )
        if valley_count == 4 and all("generic" in region for region in region_set):
            mechanism = "generic_four_valley"
        elif valley_count == 2 and any("SigmaPrime" in region for region in region_set):
            mechanism = "SigmaPrime_two_valley"
        else:
            mechanism = "other_or_mixed"
        group_rows.append(
            {
                "edge_id": TARGET_EDGE_ID,
                "closure_group_id": group_id,
                "mean_critical_lambda": centre,
                "mechanism": mechanism,
                "n_closure_orbits": len(group),
                "n_active_spin_up_valleys": valley_count,
                "left_chern_up": left_chern,
                "right_chern_up": right_chern,
                "observed_delta_chern_up": delta,
                "berry_charge_sum_up": int(group_charge),
                "all_berry_charges_consensus": int(all_consensus),
                "all_kp_jacobians_rank3": int(all_rank3),
                "signed_charge_matches": int(signed_match),
                "closure_group_certificate_pass": int(signed_match and all_rank3),
            }
        )

    endpoint_delta = int(edge["chern_1"] - edge["chern_0"])
    total_charge = int(sum(row["berry_charge_sum_up"] for row in group_rows))
    all_groups_pass = bool(
        len(group_rows)
        and all(int(row["closure_group_certificate_pass"]) == 1 for row in group_rows)
    )
    sequence = [step09d.reliable_chern(item["summary"]) for item in segment_results]
    expected_sequence = sequence == [-2, 2, 0]
    edge_certificate = {
        "code_version": CODE_VERSION,
        "edge_id": TARGET_EDGE_ID,
        "grid_chern_0": int(edge["chern_0"]),
        "grid_chern_1": int(edge["chern_1"]),
        "grid_endpoint_delta_chern_up": endpoint_delta,
        "strict_segment_chern_sequence": sequence,
        "target_sequence_minus2_to_plus2_to_zero": int(expected_sequence),
        "n_closure_groups": len(groups),
        "n_closure_orbits": len(closures),
        "n_active_spin_up_valleys_total": int(
            sum(row["n_active_spin_up_valleys"] for row in group_rows)
        ),
        "berry_charge_sum_over_all_groups": total_charge,
        "signed_total_charge_matches_grid_delta": int(total_charge == endpoint_delta),
        "all_closure_groups_certified": int(all_groups_pass),
        "edge_multiclosure_certificate_pass": int(
            expected_sequence and all_groups_pass and total_charge == endpoint_delta
        ),
        "interpretation": (
            "Pass requires two separately certified closure groups with strict "
            "segment sequence -2 -> +2 -> 0, signed Berry charges +4 and -2, "
            "and total charge +2 equal to the endpoint Chern change."
        ),
    }
    return {
        "closure_rows": closure_rows,
        "valley_rows": valley_rows,
        "kp_rows": kp_rows,
        "derivative_rows": derivative_rows,
        "gradient_rows": gradient_rows,
        "berry_attempt_rows": berry_attempt_rows,
        "berry_rows": berry_rows,
        "group_rows": group_rows,
        "edge_certificate": edge_certificate,
    }


# =============================================================================
# Refined junction strict map
# =============================================================================


def refined_checkpoint(config: Step12Config, ix: int, iy: int) -> Path:
    return config.output_dir / "refined_junction_points" / f"refined_{ix:02d}_{iy:02d}.json"


def phase_code_from_summary(summary: dict[str, Any]) -> int:
    label = str(summary.get("phase_label", ""))
    cup = summary.get("chern_up_int")
    strict_gap = int(summary.get("strict_gap_verified", 0) or 0)
    strict_chern = int(summary.get("strict_chern_verified", 0) or 0)
    if strict_gap == 1 and strict_chern == 1 and cup is not None and not pd.isna(cup):
        value = int(cup)
        if label in {"spin_chern_TI_candidate", "trivial_insulator"} and value in PHASE_ORDER:
            return value
    return 99


def run_refined_junction_grid(
    *,
    background: dict[str, float],
    step3,
    strict_config,
    config: Step12Config,
) -> pd.DataFrame:
    r3_values = np.linspace(*config.refined_r3_range, int(config.refined_grid_n))
    r4_values = np.linspace(*config.refined_r4_range, int(config.refined_grid_n))
    rows: list[dict[str, Any]] = []
    for iy, r4 in enumerate(r4_values):
        for ix, r3 in enumerate(r3_values):
            path = refined_checkpoint(config, ix, iy)
            if path.is_file() and not config.force_recalculate:
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                reduced = {**background, "r3": float(r3), "r4": float(r4)}
                result = step10.evaluate_strict(
                    step3,
                    strict_config,
                    reduced,
                    f"step12M_refined_{ix:02d}_{iy:02d}",
                )
                payload = result
                atomic_json(payload, path)
            summary = dict(payload["summary"])
            summary.update(
                {
                    "refined_ix": int(ix),
                    "refined_iy": int(iy),
                    "paper_phase_code": phase_code_from_summary(summary),
                }
            )
            rows.append(summary)
            print(
                f"refined junction {iy * len(r3_values) + ix + 1:3d}/"
                f"{len(r3_values) * len(r4_values)} | r3={r3:.6f} r4={r4:.6f} "
                f"C={summary['paper_phase_code']}"
            )
    frame = pd.DataFrame(rows)
    atomic_csv(frame, config.output_dir / "step12M_10_refined_junction_strict_grid.csv")
    return frame


def combine_junction_grids(
    step11_grid: pd.DataFrame,
    refined: pd.DataFrame,
) -> pd.DataFrame:
    old = step11_grid.copy()
    old["grid_source"] = "step11M_7x7"
    old["grid_resolution_priority"] = 0
    new = refined.copy()
    new["grid_source"] = "step12M_13x13"
    new["grid_resolution_priority"] = 1
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    combined["r3_round"] = combined["r3"].astype(float).round(10)
    combined["r4_round"] = combined["r4"].astype(float).round(10)
    combined = (
        combined.sort_values("grid_resolution_priority")
        .drop_duplicates(["r3_round", "r4_round"], keep="last")
        .drop(columns=["r3_round", "r4_round"])
        .sort_values(["r4", "r3"])
        .reset_index(drop=True)
    )
    return combined


REFINED_TRANSITION_EDGE_COLUMNS = [
    "ix0", "iy0", "ix1", "iy1",
    "r3_0", "r4_0", "chern_0",
    "r3_1", "r4_1", "chern_1",
    "mid_r3", "mid_r4", "delta_chern_up", "orientation",
]


def refined_transition_edges(refined: pd.DataFrame) -> pd.DataFrame:
    by_key = {
        (int(row.refined_ix), int(row.refined_iy)): row
        for row in refined.itertuples()
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for key, row in by_key.items():
        if int(row.paper_phase_code) not in {-2, 0, 2}:
            continue
        ix, iy = key
        for other_key in [(ix + 1, iy), (ix, iy + 1)]:
            if other_key not in by_key:
                continue
            other = by_key[other_key]
            if int(other.paper_phase_code) not in {-2, 0, 2}:
                continue
            if int(row.paper_phase_code) == int(other.paper_phase_code):
                continue
            edge_key = tuple(sorted((key, other_key)))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            rows.append(
                {
                    "ix0": ix,
                    "iy0": iy,
                    "ix1": other_key[0],
                    "iy1": other_key[1],
                    "r3_0": float(row.r3),
                    "r4_0": float(row.r4),
                    "chern_0": int(row.paper_phase_code),
                    "r3_1": float(other.r3),
                    "r4_1": float(other.r4),
                    "chern_1": int(other.paper_phase_code),
                    "mid_r3": 0.5 * (float(row.r3) + float(other.r3)),
                    "mid_r4": 0.5 * (float(row.r4) + float(other.r4)),
                    "delta_chern_up": int(other.paper_phase_code - row.paper_phase_code),
                    "orientation": "vertical_boundary" if ix != other_key[0] else "horizontal_boundary",
                }
            )
    return pd.DataFrame(rows, columns=REFINED_TRANSITION_EDGE_COLUMNS)


# =============================================================================
# Publication plotting
# =============================================================================


def publication_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def phase_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap([PHASE_COLORS[value] for value in PHASE_ORDER])
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


def formal_phase_grid(formal_scan_source: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = base.load_lower_grid(formal_scan_source)
    frame["paper_phase_code"] = frame.apply(base.phase_code, axis=1)
    x = np.sort(frame["scan_x"].unique().astype(float))
    y = np.sort(frame["scan_y"].unique().astype(float))
    matrix = np.full((len(y), len(x)), np.nan)
    x_index = {round(value, 12): i for i, value in enumerate(x)}
    y_index = {round(value, 12): i for i, value in enumerate(y)}
    for row in frame.itertuples():
        value = int(row.paper_phase_code)
        if value not in PHASE_ORDER:
            continue
        matrix[y_index[round(float(row.scan_y), 12)], x_index[round(float(row.scan_x), 12)]] = value
    return x, y, matrix


def plot_main_phase_map(
    *,
    formal_scan_source: str | Path,
    completed_points: pd.DataFrame,
    fits: pd.DataFrame,
    combined_grid: pd.DataFrame,
    config: Step12Config,
) -> None:
    publication_rc()
    cmap, norm = phase_cmap()
    x, y, matrix = formal_phase_grid(formal_scan_source)
    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 10.5 / 2.54))
    ax.imshow(
        matrix,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.92,
    )

    markers = {"generic_left_lower": "s", "generic_right_upper": "^"}
    for branch, group in completed_points.groupby("branch_hint"):
        ax.scatter(
            group["r3"],
            group["r4"],
            s=27,
            marker=markers.get(str(branch), "o"),
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )

    for row in fits.itertuples():
        branch = str(row.branch_hint)
        degree = int(row.selected_degree)
        coefficients = [float(getattr(row, f"coefficient_{i}")) for i in range(degree + 1)]
        if branch == "generic_left_lower":
            xx = np.linspace(float(row.x_min), float(row.x_max), 300)
            yy = np.polyval(coefficients, xx)
            ax.plot(xx, yy, color="black", linewidth=1.1, zorder=5)
        else:
            yy = np.linspace(float(row.x_min), float(row.x_max), 300)
            xx = np.polyval(coefficients, yy)
            ax.plot(xx, yy, color="black", linewidth=1.1, zorder=5)

    reliable = combined_grid[combined_grid["paper_phase_code"].isin([-2, 0, 2])]
    unreliable = combined_grid[~combined_grid["paper_phase_code"].isin([-2, 0, 2])]
    for phase in [-2, 0, 2]:
        group = reliable[reliable["paper_phase_code"].astype(int).eq(phase)]
        if len(group):
            ax.scatter(
                group["r3"], group["r4"],
                s=18, marker="s",
                c=PHASE_COLORS[phase], edgecolors="black", linewidths=0.25,
                zorder=7,
            )
    if len(unreliable):
        ax.scatter(
            unreliable["r3"], unreliable["r4"],
            s=17, marker="x", color="#555555", linewidths=0.65, zorder=8,
        )

    plot_xmin = min(float(x.min()), float(combined_grid["r3"].min()))
    plot_xmax = max(float(x.max()), float(combined_grid["r3"].max()))
    plot_ymin = min(float(y.min()), float(combined_grid["r4"].min()))
    plot_ymax = max(float(y.max()), float(combined_grid["r4"].max())) + 0.004

    r3_lo, r3_hi = config.refined_r3_range
    r4_lo, r4_hi = config.refined_r4_range
    ax.add_patch(
        Rectangle(
            (r3_lo, r4_lo), r3_hi - r3_lo, r4_hi - r4_lo,
            fill=False, edgecolor="black", linewidth=0.9, linestyle="--", zorder=9,
        )
    )
    ax.annotate(
        "right junction\n(refined in Step12M)",
        xy=(0.5 * (r3_lo + r3_hi), 0.5 * (r4_lo + r4_hi)),
        xytext=(r3_lo - 0.020, r4_hi + 0.006),
        arrowprops={"arrowstyle": "-", "linewidth": 0.7},
        fontsize=7.3, ha="right", va="top",
    )

    handles: list[Any] = [
        Patch(facecolor=PHASE_COLORS[value], edgecolor="black", linewidth=0.3, label=rf"$C_\uparrow={value}$")
        for value in PHASE_ORDER
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="black", label="certified left branch"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor="black", label="certified right branch"),
            Line2D([0], [0], marker="x", color="#555555", linestyle="none", label="boundary / unreliable"),
        ]
    )
    ax.legend(
        handles=handles, frameon=True, facecolor="white", edgecolor="none",
        framealpha=0.86, ncol=2, fontsize=6.8, loc="upper left",
        borderpad=0.35, handletextpad=0.45, columnspacing=0.8,
    )
    ax.set_xlabel(r"$r_3$")
    ax.set_ylabel(r"$r_4$")
    ax.set_title("TTS fixed-background topological phase map")
    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)
    fig.tight_layout()
    out = config.output_dir / "figures" / "step12M_final_fixed_slice_phase_map"
    fig.savefig(out.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _grid_spacing(frame: pd.DataFrame) -> tuple[float, float]:
    x = np.sort(frame["r3"].unique().astype(float))
    y = np.sort(frame["r4"].unique().astype(float))
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.001
    dy = float(np.median(np.diff(y))) if len(y) > 1 else 0.001
    return dx, dy


def plot_junction_zoom(
    *,
    refined: pd.DataFrame,
    transition_edges: pd.DataFrame,
    closure_rows: pd.DataFrame,
    edge: pd.Series,
    right_fit: pd.Series | None,
    config: Step12Config,
) -> None:
    publication_rc()
    fig, ax = plt.subplots(figsize=(12.5 / 2.54, 10.0 / 2.54))
    reliable = refined[refined["paper_phase_code"].isin([-2, 0, 2])]
    unreliable = refined[~refined["paper_phase_code"].isin([-2, 0, 2])]
    dx, dy = _grid_spacing(refined)
    marker_size = 135
    for phase in [-2, 0, 2]:
        group = reliable[reliable["paper_phase_code"].astype(int).eq(phase)]
        ax.scatter(
            group["r3"], group["r4"],
            s=marker_size, marker="s", c=PHASE_COLORS[phase],
            edgecolors="white", linewidths=0.45, zorder=2,
            label=rf"$C_\uparrow={phase}$",
        )
    if len(unreliable):
        ax.scatter(
            unreliable["r3"], unreliable["r4"],
            s=60, marker="x", color="#4F4F4F", linewidths=0.9,
            zorder=5, label="boundary / unreliable",
        )

    # Pixel-edge boundary segments inferred only from reliable neighbouring points.
    for row in transition_edges.itertuples():
        if str(row.orientation) == "vertical_boundary":
            ax.plot(
                [row.mid_r3, row.mid_r3],
                [row.mid_r4 - 0.48 * dy, row.mid_r4 + 0.48 * dy],
                color="black", linewidth=1.0, zorder=4,
            )
        else:
            ax.plot(
                [row.mid_r3 - 0.48 * dx, row.mid_r3 + 0.48 * dx],
                [row.mid_r4, row.mid_r4],
                color="black", linewidth=1.0, zorder=4,
            )

    # Target path.
    ax.plot(
        [float(edge["r3_0"]), float(edge["r3_1"])],
        [float(edge["r4_0"]), float(edge["r4_1"])],
        color="black", linewidth=1.1, linestyle="--", zorder=6,
        label="target edge path",
    )
    ax.scatter(
        [float(edge["r3_0"]), float(edge["r3_1"])],
        [float(edge["r4_0"]), float(edge["r4_1"])],
        s=36, facecolors="white", edgecolors="black", zorder=7,
    )

    if len(closure_rows):
        for index, row in closure_rows.sort_values("critical_lambda").reset_index(drop=True).iterrows():
            lam = float(row["critical_lambda"])
            r3 = (1.0 - lam) * float(edge["r3_0"]) + lam * float(edge["r3_1"])
            r4 = (1.0 - lam) * float(edge["r4_0"]) + lam * float(edge["r4_1"])
            if "full_BZ" in str(row.get("search_manifold", "")) or "generic" in str(row.get("search_manifold", "")):
                marker, text = "*", "four-valley"
                size = 120
            else:
                marker, text = "D", r"two-valley $\Sigma'$"
                size = 48
            ax.scatter([r3], [r4], marker=marker, s=size, c="black", zorder=9)
            ax.annotate(
                text,
                xy=(r3, r4), xytext=(5, 6 if index == 0 else -13),
                textcoords="offset points", fontsize=7.2,
                ha="left", va="bottom" if index == 0 else "top",
            )

    # Certified right-upper fit is solid only in its original validated domain;
    # the extension toward the newly certified generic closure is dashed.
    if right_fit is not None:
        degree = int(right_fit["selected_degree"])
        coeff = [float(right_fit[f"coefficient_{i}"]) for i in range(degree + 1)]
        y_valid = np.linspace(float(right_fit["x_min"]), float(right_fit["x_max"]), 120)
        x_valid = np.polyval(coeff, y_valid)
        ax.plot(x_valid, y_valid, color="black", linewidth=1.0, zorder=6)
        y_ext = np.linspace(float(right_fit["x_max"]), config.refined_r4_range[1], 100)
        x_ext = np.polyval(coeff, y_ext)
        ax.plot(x_ext, y_ext, color="black", linewidth=0.8, linestyle=":", zorder=6)

    ax.set_xlim(config.refined_r3_range[0] - 0.0004, config.refined_r3_range[1] + 0.0004)
    ax.set_ylim(config.refined_r4_range[0] - 0.0003, config.refined_r4_range[1] + 0.0003)
    ax.set_xlabel(r"$r_3$")
    ax.set_ylabel(r"$r_4$")
    ax.set_title("Right-upper junction: strict Chern map")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    fig.tight_layout()
    out = config.output_dir / "figures" / "step12M_right_junction_zoom"
    fig.savefig(out.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_target_path_atlas(
    *,
    gap_tracks: pd.DataFrame,
    closure_rows: pd.DataFrame,
    segment_rows: pd.DataFrame,
    config: Step12Config,
) -> None:
    publication_rc()
    fig, ax = plt.subplots(figsize=(14.0 / 2.54, 8.5 / 2.54))
    if "generic_gap" in gap_tracks:
        ax.semilogy(
            gap_tracks["lambda"], np.maximum(gap_tracks["generic_gap"], 1.0e-16),
            linewidth=1.1, label="generic full-BZ valley track",
        )
    ax.semilogy(
        gap_tracks["lambda"], np.maximum(gap_tracks["sigma_prime_gap"], 1.0e-16),
        linewidth=1.0, label=r"$\Sigma'$ valley track",
    )
    for index, row in closure_rows.sort_values("critical_lambda").reset_index(drop=True).iterrows():
        lam = float(row["critical_lambda"])
        ax.axvline(lam, color="black", linewidth=0.8, linestyle="--")
        ax.text(
            lam, 2.0e-15 if index == 0 else 7.0e-15,
            "4 valleys" if index == 0 else "2 valleys",
            rotation=90, fontsize=7.0, ha="right", va="bottom",
        )
    ax.set_xlabel(r"path coordinate $\lambda$")
    ax.set_ylabel(r"spin-up internal gap")
    ax.set_ylim(5.0e-16, max(0.05, float(np.nanmax(gap_tracks[["generic_gap", "sigma_prime_gap"]].to_numpy())) * 1.3))
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")

    ax2 = ax.twinx()
    for row in segment_rows.itertuples():
        if pd.isna(row.strict_chern_up):
            continue
        ax2.hlines(
            int(row.strict_chern_up),
            float(row.lambda_left_bound),
            float(row.lambda_right_bound),
            color="#333333", linewidth=2.0, alpha=0.75,
        )
        ax2.scatter([float(row.probe_lambda)], [int(row.strict_chern_up)], marker="s", s=18, color="#333333")
    ax2.set_ylabel(r"strict $C_\uparrow$")
    ax2.set_yticks([-2, 0, 2])
    ax2.set_ylim(-2.7, 2.7)
    fig.tight_layout()
    out = config.output_dir / "figures" / "step12M_target_edge_multiclosure_atlas"
    fig.savefig(out.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_mass_sign_mapping(
    *,
    step11_source: str | Path,
    config: Step12Config,
) -> None:
    predictions = read_csv_token(step11_source, "step11M_11_spatial_holdout_predictions.csv")
    definitions = read_csv_token(step11_source, "step11M_10_local_signed_mass_definitions.csv")
    publication_rc()
    fig, ax = plt.subplots(figsize=(12.5 / 2.54, 7.5 / 2.54))
    markers = {"generic_left_lower": "s", "generic_right_upper": "^"}
    for branch, group in predictions.groupby("branch_hint"):
        for phase in [-2, 2]:
            subset = group[group["paper_phase_code"].astype(int).eq(phase)]
            if len(subset):
                ax.scatter(
                    subset["signed_mass"],
                    np.full(len(subset), phase),
                    s=26,
                    marker=markers.get(str(branch), "o"),
                    c=PHASE_COLORS[phase],
                    edgecolors="black", linewidths=0.35,
                    alpha=0.85,
                    label=f"{branch.replace('generic_', '')}: C={phase}",
                )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("local signed mass coordinate")
    ax.set_ylabel(r"$C_\uparrow$")
    ax.set_yticks([-2, 2])
    ax.set_ylim(-2.8, 2.8)
    ax.set_title("Local mass sign and topological sector")
    # Explicit sign mapping inferred from the validated Step11M predictions.
    text_lines = []
    for branch, group in predictions.groupby("branch_hint"):
        mapping = (
            group.groupby("mass_sign")["paper_phase_code"]
            .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
            .to_dict()
        )
        if -1 in mapping and 1 in mapping:
            short = str(branch).replace("generic_", "").replace("_", "-")
            text_lines.append(
                f"{short}:  M<0 → C={mapping[-1]},   M>0 → C={mapping[1]}"
            )
    if text_lines:
        ax.text(
            0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=7.1,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="black", label="left-lower branch"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor="black", label="right-upper branch"),
        Patch(facecolor=PHASE_COLORS[-2], edgecolor="black", linewidth=0.35, label=r"$C_\uparrow=-2$"),
        Patch(facecolor=PHASE_COLORS[2], edgecolor="black", linewidth=0.35, label=r"$C_\uparrow=+2$"),
    ]
    ax.legend(
        handles=legend_handles, frameon=False, fontsize=6.8,
        loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0,
    )
    fig.tight_layout()
    out = config.output_dir / "figures" / "step12M_local_mass_sign_mapping"
    fig.savefig(out.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Aggregation and orchestration
# =============================================================================


def save_analysis_tables(
    *,
    config: Step12Config,
    edge: pd.Series,
    generic_candidates: list[dict[str, Any]],
    sigma_candidates: list[dict[str, Any]],
    closures: list[dict[str, Any]],
    gap_tracks: pd.DataFrame,
    segment_rows: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> None:
    atomic_csv(pd.DataFrame([edge.to_dict()]), config.output_dir / "step12M_01_target_edge_definition.csv")
    candidates = pd.DataFrame(
        [
            {"candidate_family": "generic", **row}
            for row in generic_candidates
        ]
        + [
            {"candidate_family": "SigmaPrime", **row}
            for row in sigma_candidates
        ]
    )
    atomic_csv(candidates, config.output_dir / "step12M_02_full_bz_search_candidates.csv")
    atomic_csv(pd.DataFrame(closures), config.output_dir / "step12M_03_accepted_closure_orbits.csv")
    atomic_csv(gap_tracks, config.output_dir / "step12M_04_target_edge_gap_tracks.csv")
    atomic_csv(pd.DataFrame(segment_rows), config.output_dir / "step12M_05_strict_segment_chern_labels.csv")
    mapping = {
        "closure_rows": "step12M_06_critical_closure_parameters.csv",
        "valley_rows": "step12M_07_critical_valley_orbits.csv",
        "kp_rows": "step12M_08_local_kp_summaries.csv",
        "derivative_rows": "step12M_09_local_kp_derivatives.csv",
        "gradient_rows": "step12M_10_local_mass_gradients.csv",
        "berry_attempt_rows": "step12M_11_berry_sphere_attempts.csv",
        "berry_rows": "step12M_12_berry_charge_consensus.csv",
        "group_rows": "step12M_13_closure_group_certificates.csv",
    }
    for key, filename in mapping.items():
        atomic_csv(pd.DataFrame(analysis[key]), config.output_dir / filename)
    atomic_json(analysis["edge_certificate"], config.output_dir / "step12M_14_target_edge_multiclosure_certificate.json")


def build_final_certificate(
    *,
    config: Step12Config,
    background: dict[str, float],
    edge_certificate: dict[str, Any],
    refined: pd.DataFrame,
    transition_edges: pd.DataFrame,
) -> dict[str, Any]:
    reliable = refined[refined["paper_phase_code"].isin([-2, 0, 2])]
    counts = reliable["paper_phase_code"].astype(int).value_counts().sort_index().to_dict()
    return {
        "code_version": CODE_VERSION,
        "research_design": "targeted_fixed_slice_right_junction_repair",
        "fixed_background_parameters": background,
        "free_plane_parameters": ["r3", "r4"],
        "target_edge_id": config.target_edge_id,
        "target_edge_certificate": edge_certificate,
        "refined_junction_grid": {
            "r3_range": list(config.refined_r3_range),
            "r4_range": list(config.refined_r4_range),
            "grid_n": int(config.refined_grid_n),
            "n_total_points": int(len(refined)),
            "n_reliable_points": int(len(reliable)),
            "phase_counts": {str(k): int(v) for k, v in counts.items()},
            "n_reliable_transition_edges": int(len(transition_edges)),
        },
        "all_background_parameters_remain_fixed": True,
        "step12M_research_complete": bool(
            int(edge_certificate.get("edge_multiclosure_certificate_pass", 0)) == 1
        ),
        "valid_claim_if_pass": (
            "Along the formerly unresolved r3=0.060 junction edge, the strict "
            "Chern sequence is -2 -> +2 -> 0.  A generic four-valley closure "
            "carries +4 and a later Sigma' two-valley closure carries -2, so "
            "the total signed charge +2 equals the endpoint Chern change."
        ),
        "not_claimed": (
            "The result is local to the fixed-background r3-r4 plane and does "
            "not define a universal seven-dimensional mass formula."
        ),
    }


def make_plots_from_outputs(
    *,
    formal_scan_source: str | Path,
    step11_source: str | Path,
    config: Step12Config,
) -> None:
    completed = read_csv_token(step11_source, "step11M_07_completed_certified_boundary_points.csv")
    fits = read_csv_token(step11_source, "step11M_08_fixed_slice_boundary_curve_fits.csv")
    step11_grid = read_csv_token(step11_source, "step11M_03_right_junction_strict_grid.csv")
    refined_path = config.output_dir / "step12M_15_refined_junction_strict_grid.csv"
    if not refined_path.is_file():
        refined_path = config.output_dir / "step12M_10_refined_junction_strict_grid.csv"
    refined = safe_read_csv_path(refined_path, required=True)
    combined_path = config.output_dir / "step12M_16_combined_junction_grid.csv"
    combined = safe_read_csv_path(combined_path) if combined_path.is_file() else combine_junction_grids(step11_grid, refined)
    transitions_path = config.output_dir / "step12M_17_refined_transition_edges.csv"
    if transitions_path.is_file():
        transitions = safe_read_csv_path(
            transitions_path, columns=REFINED_TRANSITION_EDGE_COLUMNS
        )
    else:
        transitions = refined_transition_edges(refined)
    closure_path = config.output_dir / "step12M_06_critical_closure_parameters.csv"
    closure_rows = safe_read_csv_path(closure_path) if closure_path.is_file() else pd.DataFrame()
    edge = load_target_edge(step11_source, config.target_edge_id)
    right = fits[fits["branch_hint"].astype(str).eq("generic_right_upper")]
    right_fit = right.iloc[0] if len(right) else None

    plot_main_phase_map(
        formal_scan_source=formal_scan_source,
        completed_points=completed,
        fits=fits,
        combined_grid=combined,
        config=config,
    )
    plot_junction_zoom(
        refined=refined,
        transition_edges=transitions,
        closure_rows=closure_rows,
        edge=edge,
        right_fit=right_fit,
        config=config,
    )
    gap_path = config.output_dir / "step12M_04_target_edge_gap_tracks.csv"
    segment_path = config.output_dir / "step12M_05_strict_segment_chern_labels.csv"
    if gap_path.is_file() and segment_path.is_file() and len(closure_rows):
        plot_target_path_atlas(
            gap_tracks=safe_read_csv_path(gap_path, required=True),
            closure_rows=closure_rows,
            segment_rows=safe_read_csv_path(segment_path, required=True),
            config=config,
        )
    plot_mass_sign_mapping(step11_source=step11_source, config=config)


def run_step12m(
    *,
    tts_archive: str | Path,
    formal_scan_source: str | Path,
    step11_source: str | Path,
    output_dir: str | Path,
    config: Step12Config | None = None,
    run_physics: bool = True,
    run_refined_grid_flag: bool = True,
    run_plots: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step12Config(output_dir=Path(output_dir))
    config.output_dir = Path(output_dir)
    config = config.normalized()
    atomic_json(asdict(config), config.output_dir / "step12M_00_run_configuration.json")

    background = load_background(step11_source)
    edge = load_target_edge(step11_source, config.target_edge_id)
    seed = load_right_upper_seed(step11_source, edge)
    validate_fixed_background(background, edge, seed)
    atomic_json(
        {
            "code_version": CODE_VERSION,
            "fixed_background_parameters": background,
            "target_edge": edge.to_dict(),
            "nearest_certified_generic_seed": seed.to_dict(),
        },
        config.output_dir / "step12M_00_input_audit.json",
    )

    if not run_physics:
        if run_plots:
            make_plots_from_outputs(
                formal_scan_source=formal_scan_source,
                step11_source=step11_source,
                config=config,
            )
        return {"status": "plot_only", "output_dir": str(config.output_dir)}

    core, step4, step5, step3, strict_config, search_config = configure_physics(
        tts_archive,
        config,
    )
    pair = step11.build_pair_from_edge(edge, background)

    print("[1/5] Full-BZ generic four-valley search")
    generic_candidates, generic_closures = search_generic_four_valley(
        step4=step4,
        step5=step5,
        pair=pair,
        seed=seed,
        config=config,
    )
    print(f"      accepted generic closures = {len(generic_closures)}")

    print("[2/5] SigmaPrime two-valley search")
    sigma_candidates, sigma_closures = search_sigma_prime_closure(
        step5=step5,
        pair=pair,
        config=config,
    )
    print(f"      accepted SigmaPrime closures = {len(sigma_closures)}")

    closures = combine_closures(step5, generic_closures, sigma_closures, config)
    if len(closures) < 2:
        # Preserve all diagnostics before failing clearly.
        atomic_csv(pd.DataFrame(generic_candidates), config.output_dir / "step12M_02a_generic_search_candidates.csv")
        atomic_csv(pd.DataFrame(sigma_candidates), config.output_dir / "step12M_02b_sigma_search_candidates.csv")
        atomic_csv(pd.DataFrame(closures), config.output_dir / "step12M_03_accepted_closure_orbits.csv")
        raise RuntimeError(
            "Step12M did not find two distinct closure groups. Inspect step12M_02a/02b candidates; "
            "do not reinterpret the Step11M charge sign manually."
        )

    print("[3/5] Gap tracks, strict segment Chern labels, k.p and Berry charges")
    gap_tracks = compute_gap_tracks(step5=step5, pair=pair, seed=seed, config=config)
    segment_results, segment_rows = strict_segment_probes(
        step3=step3,
        strict_config=strict_config,
        pair=pair,
        closures=closures,
        path_id=config.target_edge_id,
    )
    analysis = certify_target_edge(
        edge=edge,
        pair=pair,
        closures=closures,
        step5=step5,
        search_config=search_config,
        segment_results=segment_results,
    )
    save_analysis_tables(
        config=config,
        edge=edge,
        generic_candidates=generic_candidates,
        sigma_candidates=sigma_candidates,
        closures=closures,
        gap_tracks=gap_tracks,
        segment_rows=segment_rows,
        analysis=analysis,
    )

    print("[4/5] Refined 13x13 strict junction map")
    if run_refined_grid_flag:
        refined = run_refined_junction_grid(
            background=background,
            step3=step3,
            strict_config=strict_config,
            config=config,
        )
    else:
        refined = read_csv_token(step11_source, "step11M_03_right_junction_strict_grid.csv")
        # Normalize names so plotting works, but this is not considered a Step12 refined map.
        refined = refined.rename(columns={"junction_ix": "refined_ix", "junction_iy": "refined_iy"})
    atomic_csv(refined, config.output_dir / "step12M_15_refined_junction_strict_grid.csv")
    step11_grid = read_csv_token(step11_source, "step11M_03_right_junction_strict_grid.csv")
    combined = combine_junction_grids(step11_grid, refined)
    transitions = refined_transition_edges(refined)
    atomic_csv(combined, config.output_dir / "step12M_16_combined_junction_grid.csv")
    atomic_csv(transitions, config.output_dir / "step12M_17_refined_transition_edges.csv")

    certificate = build_final_certificate(
        config=config,
        background=background,
        edge_certificate=analysis["edge_certificate"],
        refined=refined,
        transition_edges=transitions,
    )
    atomic_json(certificate, config.output_dir / "step12M_18_final_mechanism_certificate.json")

    print("[5/5] Publication figures")
    if run_plots:
        make_plots_from_outputs(
            formal_scan_source=formal_scan_source,
            step11_source=step11_source,
            config=config,
        )

    print(json.dumps(certificate, ensure_ascii=False, indent=2, default=_json_default))
    return certificate


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair the unresolved TTS Step11M right-junction edge and redraw the fixed-slice phase map."
    )
    parser.add_argument("--tts-archive", required=True, type=Path)
    parser.add_argument("--formal-scan", required=True, type=Path)
    parser.add_argument("--step11-source", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs_tts_step12M_junction_multiclosure_repair"), type=Path)
    parser.add_argument("--quick", action="store_true", help="Reduced numerical settings for code testing only.")
    parser.add_argument("--force", action="store_true", help="Recalculate existing refined-grid checkpoints.")
    parser.add_argument("--skip-refined-grid", action="store_true", help="Run only the target edge certificate and plots.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from existing Step12M CSV files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = Step12Config(
        output_dir=args.output_dir,
        quick=bool(args.quick),
        force_recalculate=bool(args.force),
    )
    run_step12m(
        tts_archive=args.tts_archive,
        formal_scan_source=args.formal_scan,
        step11_source=args.step11_source,
        output_dir=args.output_dir,
        config=config,
        run_physics=not bool(args.plot_only),
        run_refined_grid_flag=not bool(args.skip_refined_grid),
        run_plots=not bool(args.skip_plots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
