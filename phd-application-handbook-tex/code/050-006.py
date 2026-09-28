from __future__ import annotations

"""
TTS Step15M — exhaustive reclassification of every previously unresolved point
===============================================================================

Purpose
-------
Re-evaluate every point marked ``boundary / unreliable`` in the fixed-background
r3-r4 map and replace the ambiguous plotting label with a physically definite
status whenever possible:

1. strict gapped phase with a certified spin Chern number;
2. indirect-overlap metal;
3. true full-gap closing / phase-boundary point;
4. spin-sector internal closing or spin-filling transition;
5. numeric_unresolved (retained only if all escalated methods fail).

A true gap-closing point is *not* assigned a Chern number, because the occupied
subspace is not globally isolated there.  Gapped points are first tested with
multi-grid Fukui calculations and, if necessary, escalated to adaptive
non-Abelian Wilson loops.
"""

import argparse
import importlib.util
import io
import json
import math
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.interpolate import NearestNDInterpolator

CODE_VERSION = "TTS_STEP15M_RECLASSIFY_ALL_BOUNDARY_POINTS_V1_20260724"
TWO_PI = 2.0 * math.pi
REDUCED7 = ("m_e", "t1", "t2", "r1", "r2", "r3", "r4")

PHASE_COLORS = {
    -2: "#4F81BD",
    -1: "#8FBAD9",
     0: "#E6E6E6",
     1: "#E8B36A",
     2: "#CF5A47",
}
STATUS_COLORS = {
    "phase_boundary_gap_closing": "#111111",
    "indirect_overlap_metal": "#8C6D31",
    "spin_sector_gap_closing": "#7A3E9D",
    "spin_filling_transition": "#17A589",
    "indirect_overlap_boundary": "#B9770E",
    "numeric_unresolved": "#666666",
}


@dataclass
class Step15Config:
    output_dir: Path = Path("outputs_tts_step15M_reclassify_all_boundary_points")

    # Global gap audit.
    gap_grid_n: int = 101
    gap_local_starts: int = 14
    gap_powell_maxiter: int = 1200
    physical_zero_tolerance: float = 1.0e-9
    positive_gap_tolerance: float = 5.0e-9
    indirect_metal_tolerance: float = 5.0e-9

    # Fukui hierarchy.
    fukui_grids: tuple[int, ...] = (61, 81, 101, 121)
    fukui_shifts: tuple[tuple[float, float], ...] = (
        (0.00, 0.00),
        (0.50, 0.50),
        (0.25, 0.73),
    )
    fukui_integer_tolerance: float = 0.055
    fukui_min_link: float = 1.0e-9
    fukui_sum_rule_tolerance: float = 0.06
    fukui_min_consensus: int = 3
    fukui_min_dominant_fraction: float = 0.75

    # Adaptive Wilson fallback.
    wilson_nkx: tuple[int, ...] = (401, 601, 801)
    wilson_initial_nky: tuple[int, ...] = (81, 101, 121)
    wilson_shifts: tuple[tuple[float, float], ...] = (
        (0.00, 0.00),
        (0.50, 0.37),
        (0.25, 0.73),
    )
    wilson_phase_step: float = 0.30 * math.pi
    wilson_max_points: int = 901
    wilson_max_rounds: int = 9
    wilson_integer_tolerance: float = 0.06
    wilson_min_overlap: float = 5.0e-5

    checkpoint_every: int = 1
    force_recalculate: bool = False
    quick: bool = False
    random_seed: int = 20260724

    def normalized(self) -> "Step15Config":
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ["checkpoints", "wilson_traces", "figures", "_physics_source"]:
            (self.output_dir / name).mkdir(exist_ok=True)
        if self.quick:
            self.gap_grid_n = min(self.gap_grid_n, 51)
            self.gap_local_starts = min(self.gap_local_starts, 6)
            self.fukui_grids = (41, 61)
            self.fukui_shifts = ((0.0, 0.0), (0.5, 0.5))
            self.fukui_min_consensus = 2
            self.wilson_nkx = (201, 301)
            self.wilson_initial_nky = (61, 81)
            self.wilson_shifts = ((0.0, 0.0), (0.5, 0.37))
            self.wilson_max_points = min(self.wilson_max_points, 401)
        if len(self.wilson_nkx) != len(self.wilson_initial_nky):
            raise ValueError("wilson_nkx and wilson_initial_nky must have equal length")
        if len(self.wilson_nkx) != len(self.wilson_shifts):
            raise ValueError("wilson_shifts must match Wilson attempts")
        return self


# =============================================================================
# Safe archive I/O
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
    return sorted(matches, key=lambda x: (len(Path(x).parts), len(x)))[0]


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


# =============================================================================
# Frozen Hamiltonian source
# =============================================================================


def extract_tts_core(tts_archive: str | Path, destination: Path) -> Path:
    archive = Path(tts_archive).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    marker = destination / "TTS_step01_model_and_label_audit_v2.py"
    if marker.is_file():
        return destination
    with zipfile.ZipFile(archive) as zf:
        candidates = [n for n in zf.namelist() if Path(n).name == "TTS_step01_model_and_label_audit_v2.py"]
        destination.mkdir(parents=True, exist_ok=True)
        if candidates:
            member = sorted(candidates, key=lambda x: (len(Path(x).parts), len(x)))[0]
            with zf.open(member) as src, marker.open("wb") as dst:
                dst.write(src.read())
            return destination

        nested = [n for n in zf.namelist() if Path(n).name.lower() in {"tts(1).zip", "tts.zip"}]
        if not nested:
            raise FileNotFoundError("Neither Step01 source nor a nested tts(1).zip was found")
        nested_member = sorted(nested, key=lambda x: (len(Path(x).parts), len(x)))[0]
        nested_bytes = zf.read(nested_member)
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
            inner_candidates = [n for n in inner.namelist() if Path(n).name == "TTS_step01_model_and_label_audit_v2.py"]
            if not inner_candidates:
                raise FileNotFoundError("TTS_step01_model_and_label_audit_v2.py not found in nested tts archive")
            member = sorted(inner_candidates, key=lambda x: (len(Path(x).parts), len(x)))[0]
            with inner.open(member) as src, marker.open("wb") as dst:
                dst.write(src.read())
    return destination


def import_module_from_path(name: str, path: Path):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_core(tts_archive: str | Path, config: Step15Config):
    root = extract_tts_core(tts_archive, config.output_dir / "_physics_source")
    return import_module_from_path("tts_step15_core", root / "TTS_step01_model_and_label_audit_v2.py")


# =============================================================================
# Gap audit
# =============================================================================


def wrap_k(value: float) -> float:
    return float((float(value) + math.pi) % TWO_PI - math.pi)


def raw_parameters(background: dict[str, float], row: pd.Series, core) -> dict[str, float]:
    reduced = {**background, "r3": float(row["r3"]), "r4": float(row["r4"])}
    return core.raw8_from_reduced7(reduced, e0=0.0)


def spectral_values(core, raw: dict[str, float], kx: float, ky: float) -> dict[str, Any]:
    full = np.linalg.eigvalsh(np.asarray(core.h_tts_periodic(kx, ky, raw), dtype=complex))
    up = np.linalg.eigvalsh(np.asarray(core.h_spin_block_periodic(kx, ky, raw, "up"), dtype=complex))
    down = np.linalg.eigvalsh(np.asarray(core.h_spin_block_periodic(kx, ky, raw, "down"), dtype=complex))
    n_total = int(core.N_OCC_TOTAL)
    n_spin = int(core.N_OCC_SPIN)
    return {
        "direct": float(full[n_total] - full[n_total - 1]),
        "valence": float(full[n_total - 1]),
        "conduction": float(full[n_total]),
        "spin_up": float(up[n_spin] - up[n_spin - 1]),
        "spin_down": float(down[n_spin] - down[n_spin - 1]),
        "balanced": float(min(up[n_spin], down[n_spin]) - max(up[n_spin - 1], down[n_spin - 1])),
    }


def unique_starts(candidates: list[tuple[float, float, float]], n: int, distance: float = 0.08) -> list[tuple[float, float]]:
    starts: list[tuple[float, float]] = []
    for _, kx, ky in sorted(candidates, key=lambda x: x[0]):
        if all(math.hypot(wrap_k(kx-a), wrap_k(ky-b)) > distance for a, b in starts):
            starts.append((kx, ky))
        if len(starts) >= n:
            break
    return starts


def optimize_periodic(
    objective: Callable[[float, float], float],
    starts: list[tuple[float, float]],
    *,
    maxiter: int,
) -> tuple[float, float, float]:
    best = (float("inf"), float(starts[0][0]), float(starts[0][1]))
    for start in starts:
        result = minimize(
            lambda x: float(objective(wrap_k(x[0]), wrap_k(x[1]))),
            np.asarray(start, dtype=float),
            method="Powell",
            bounds=[(-math.pi, math.pi), (-math.pi, math.pi)],
            options={"xtol": 1.0e-11, "ftol": 1.0e-20, "maxiter": int(maxiter)},
        )
        kx, ky = wrap_k(result.x[0]), wrap_k(result.x[1])
        value = float(objective(kx, ky))
        if value < best[0]:
            best = (value, kx, ky)
    return best


def adaptive_gap_audit(core, raw: dict[str, float], config: Step15Config) -> dict[str, Any]:
    ks = np.linspace(-math.pi, math.pi, int(config.gap_grid_n), endpoint=False)
    candidates = {key: [] for key in ["direct", "spin_up", "spin_down", "balanced", "neg_valence", "conduction"]}
    for kx in ks:
        for ky in ks:
            s = spectral_values(core, raw, float(kx), float(ky))
            candidates["direct"].append((s["direct"], float(kx), float(ky)))
            candidates["spin_up"].append((s["spin_up"], float(kx), float(ky)))
            candidates["spin_down"].append((s["spin_down"], float(kx), float(ky)))
            candidates["balanced"].append((s["balanced"], float(kx), float(ky)))
            candidates["neg_valence"].append((-s["valence"], float(kx), float(ky)))
            candidates["conduction"].append((s["conduction"], float(kx), float(ky)))

    starts = {k: unique_starts(v, int(config.gap_local_starts)) for k, v in candidates.items()}

    direct = optimize_periodic(lambda x, y: spectral_values(core, raw, x, y)["direct"], starts["direct"], maxiter=config.gap_powell_maxiter)
    spin_up = optimize_periodic(lambda x, y: spectral_values(core, raw, x, y)["spin_up"], starts["spin_up"], maxiter=config.gap_powell_maxiter)
    spin_down = optimize_periodic(lambda x, y: spectral_values(core, raw, x, y)["spin_down"], starts["spin_down"], maxiter=config.gap_powell_maxiter)
    balanced = optimize_periodic(lambda x, y: spectral_values(core, raw, x, y)["balanced"], starts["balanced"], maxiter=config.gap_powell_maxiter)
    neg_v = optimize_periodic(lambda x, y: -spectral_values(core, raw, x, y)["valence"], starts["neg_valence"], maxiter=config.gap_powell_maxiter)
    cond = optimize_periodic(lambda x, y: spectral_values(core, raw, x, y)["conduction"], starts["conduction"], maxiter=config.gap_powell_maxiter)

    max_valence = -neg_v[0]
    min_conduction = cond[0]
    indirect = min_conduction - max_valence
    return {
        "min_direct_gap": direct[0], "direct_gap_kx": direct[1], "direct_gap_ky": direct[2],
        "min_spin_gap_up": spin_up[0], "spin_up_gap_kx": spin_up[1], "spin_up_gap_ky": spin_up[2],
        "min_spin_gap_down": spin_down[0], "spin_down_gap_kx": spin_down[1], "spin_down_gap_ky": spin_down[2],
        "min_balanced_sector_gap": balanced[0], "balanced_gap_kx": balanced[1], "balanced_gap_ky": balanced[2],
        "max_valence": max_valence, "vbm_kx": neg_v[1], "vbm_ky": neg_v[2],
        "min_conduction": min_conduction, "cbm_kx": cond[1], "cbm_ky": cond[2],
        "indirect_gap": indirect,
        "gap_grid_n": int(config.gap_grid_n),
        "gap_local_starts": int(config.gap_local_starts),
    }


def preliminary_status(gap: dict[str, Any], config: Step15Config) -> str:
    z = float(config.physical_zero_tolerance)
    if float(gap["min_direct_gap"]) <= z:
        return "phase_boundary_gap_closing"
    if float(gap["indirect_gap"]) < -float(config.indirect_metal_tolerance):
        return "indirect_overlap_metal"
    if float(gap["indirect_gap"]) <= z:
        return "indirect_overlap_boundary"
    if float(gap["min_balanced_sector_gap"]) <= z:
        return "spin_filling_transition"
    if min(float(gap["min_spin_gap_up"]), float(gap["min_spin_gap_down"])) <= z:
        return "spin_sector_gap_closing"
    # Any strictly positive isolated occupied subspace is sent to topology
    # certification, even when the gap is extremely small.
    return "strict_gapped_candidate"


# =============================================================================
# Fukui classification
# =============================================================================


def integer_if_close(value: float, tolerance: float) -> int | None:
    if not np.isfinite(value):
        return None
    nearest = int(np.rint(value))
    return nearest if abs(float(value) - nearest) <= float(tolerance) else None


def fukui_attempt(core, raw: dict[str, float], nk: int, shift: tuple[float, float], config: Step15Config) -> dict[str, Any]:
    row: dict[str, Any] = {
        "nk": int(nk), "shift_x": float(shift[0]), "shift_y": float(shift[1]),
        "attempt_reliable": 0, "attempt_error": "",
    }
    try:
        result = core.calculate_spin_cherns(raw, int(nk), shift)
        row.update(result)
        cu = integer_if_close(float(result["chern_up"]), config.fukui_integer_tolerance)
        cd = integer_if_close(float(result["chern_down"]), config.fukui_integer_tolerance)
        ct = integer_if_close(float(result["chern_total"]), config.fukui_integer_tolerance)
        row.update({"chern_up_int": cu, "chern_down_int": cd, "chern_total_int": ct})
        reliable = (
            cu is not None and cd is not None and ct is not None
            and min(float(result["min_det_up"]), float(result["min_det_down"]), float(result["min_det_total"])) > float(config.fukui_min_link)
            and float(result["sum_rule_error"]) <= float(config.fukui_sum_rule_tolerance)
        )
        row["attempt_reliable"] = int(reliable)
    except Exception as exc:
        row["attempt_error"] = repr(exc)
    return row


def fukui_consensus(attempts: list[dict[str, Any]], config: Step15Config) -> dict[str, Any]:
    reliable = [r for r in attempts if int(r.get("attempt_reliable", 0)) == 1]
    if not reliable:
        return {"fukui_consensus": 0, "fukui_chern_up": None, "fukui_chern_down": None, "fukui_chern_total": None, "fukui_consensus_count": 0}
    values = [(int(r["chern_up_int"]), int(r["chern_down_int"]), int(r["chern_total_int"])) for r in reliable]
    counts = pd.Series(values, dtype="object").value_counts()
    best = counts.index[0]
    count = int(counts.iloc[0])
    dominant_fraction = count / max(len(reliable), 1)
    ok = (
        count >= int(config.fukui_min_consensus)
        and dominant_fraction >= float(config.fukui_min_dominant_fraction)
    )
    return {
        "fukui_consensus": int(ok),
        "fukui_chern_up": int(best[0]) if ok else None,
        "fukui_chern_down": int(best[1]) if ok else None,
        "fukui_chern_total": int(best[2]) if ok else None,
        "fukui_consensus_count": count,
        "fukui_reliable_attempt_count": int(len(reliable)),
        "fukui_dominant_fraction": float(dominant_fraction),
        "fukui_unique_reliable_tuples": int(len(counts)),
    }


# =============================================================================
# Adaptive Wilson loop
# =============================================================================


def polar_unitary(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh, float(np.min(singular))


def spin_hamiltonian(core, raw: dict[str, float], spin: str, kx: float, ky: float) -> np.ndarray:
    h = np.asarray(core.h_spin_block_periodic(float(kx), float(ky), raw, spin), dtype=complex)
    return 0.5 * (h + h.conj().T)


def wilson_loop_at_ky(
    h_fn: Callable[[float, float], np.ndarray], n_occ: int, ky: float, *, nkx: int, shift_x: float,
) -> dict[str, Any]:
    kxs = -math.pi + TWO_PI * (np.arange(int(nkx), dtype=float) + float(shift_x)) / int(nkx)
    frames: list[np.ndarray] = []
    direct_gaps: list[float] = []
    for kx in kxs:
        values, vectors = np.linalg.eigh(h_fn(float(kx), float(ky)))
        frames.append(vectors[:, :n_occ])
        direct_gaps.append(float(values[n_occ] - values[n_occ - 1]))
    wilson = np.eye(n_occ, dtype=complex)
    min_singular = 1.0
    for i in range(len(frames)):
        overlap = frames[i].conj().T @ frames[(i + 1) % len(frames)]
        link, singular = polar_unitary(overlap)
        min_singular = min(min_singular, singular)
        wilson = wilson @ link
    determinant = np.linalg.det(wilson)
    determinant /= abs(determinant) if abs(determinant) else 1.0
    return {
        "det_phase": float(np.angle(determinant)),
        "min_overlap_singular": float(min_singular),
        "min_kx_direct_gap": float(np.min(direct_gaps)),
    }


def adaptive_wilson_winding(
    h_fn: Callable[[float, float], np.ndarray], n_occ: int, *, nkx: int, initial_nky: int,
    shift_x: float, shift_y: float, phase_step: float, max_points: int, max_rounds: int,
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
            row = wilson_loop_at_ky(h_fn, n_occ, wrap_k(coord), nkx=int(nkx), shift_x=float(shift_x))
            row["ky_coordinate"] = float(coord)
            cache[key] = row
        return cache[key]

    rounds = 0
    for round_index in range(int(max_rounds) + 1):
        coords = sorted(coords)
        rows = [evaluate(x) for x in coords]
        new_points: list[float] = []
        for i, left in enumerate(coords):
            right = coords[i + 1] if i + 1 < len(coords) else coords[0] + TWO_PI
            phase_left = rows[i]["det_phase"]
            phase_right = rows[(i + 1) % len(rows)]["det_phase"]
            delta = float(np.angle(np.exp(1j * (phase_right - phase_left))))
            if abs(delta) > float(phase_step) and len(coords) + len(new_points) < int(max_points):
                new_points.append(0.5 * (left + right))
        if not new_points:
            rounds = round_index
            break
        coords.extend(new_points)
        rounds = round_index + 1

    coords = sorted(coords)
    rows = [evaluate(x) for x in coords]
    phases = np.asarray([r["det_phase"] for r in rows], dtype=float)
    increments = np.angle(np.exp(1j * (np.roll(phases, -1) - phases)))
    raw = float(np.sum(increments) / TWO_PI)
    raw_int = int(np.rint(raw))
    trace = pd.DataFrame({
        "ky_coordinate": coords,
        "ky_wrapped": [wrap_k(x) for x in coords],
        "det_phase": phases,
        "min_overlap_singular": [r["min_overlap_singular"] for r in rows],
        "min_kx_direct_gap": [r["min_kx_direct_gap"] for r in rows],
    })
    summary = {
        "raw_winding": raw,
        "raw_winding_int": raw_int,
        "integer_residual": abs(raw - raw_int),
        "min_overlap_singular": float(trace["min_overlap_singular"].min()),
        "max_abs_phase_increment": float(np.max(np.abs(increments))),
        "final_nky": int(len(coords)),
        "refinement_rounds": int(rounds),
    }
    return summary, trace


def wilson_attempts_for_spin(
    *, core, raw: dict[str, float], spin: str, orientation_factor: int, gap: dict[str, Any],
    config: Step15Config, sample_id: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    ky0 = float(gap[f"spin_{spin}_gap_ky"])
    anchors = [wrap_k(ky0 + d) for d in (0.0, -0.08, -0.04, -0.02, 0.02, 0.04, 0.08)]
    attempts: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    h_fn = lambda kx, ky: spin_hamiltonian(core, raw, spin, kx, ky)
    for index, (nkx, nky, shift) in enumerate(zip(config.wilson_nkx, config.wilson_initial_nky, config.wilson_shifts)):
        summary, trace = adaptive_wilson_winding(
            h_fn, int(core.N_OCC_SPIN), nkx=int(nkx), initial_nky=int(nky),
            shift_x=float(shift[0]), shift_y=float(shift[1]), phase_step=float(config.wilson_phase_step),
            max_points=int(config.wilson_max_points), max_rounds=int(config.wilson_max_rounds), ky_anchors=anchors,
        )
        oriented = int(summary["raw_winding_int"]) * int(orientation_factor)
        reliable = (
            float(summary["integer_residual"]) <= float(config.wilson_integer_tolerance)
            and float(summary["min_overlap_singular"]) >= float(config.wilson_min_overlap)
            and float(summary["max_abs_phase_increment"]) <= 1.10 * float(config.wilson_phase_step)
        )
        attempts.append({
            "sample_id": sample_id, "spin": spin, "attempt_index": index,
            "nkx": int(nkx), "initial_nky": int(nky), "shift_x": shift[0], "shift_y": shift[1],
            **summary, "oriented_chern": oriented, "attempt_reliable": int(reliable),
        })
        trace = trace.copy()
        trace.insert(0, "attempt_index", index)
        trace.insert(0, "spin", spin)
        trace.insert(0, "sample_id", sample_id)
        traces.append(trace)
    return attempts, traces


def wilson_consensus(attempts: list[dict[str, Any]]) -> tuple[bool, int | None, int]:
    values = [int(x["oriented_chern"]) for x in attempts if int(x["attempt_reliable"]) == 1]
    if not values:
        return False, None, 0
    counts = pd.Series(values).value_counts()
    value = int(counts.index[0])
    count = int(counts.iloc[0])
    return bool(count >= 2 and len(counts) == 1), value, count


# =============================================================================
# Final phase label
# =============================================================================


def phase_label_from_cherns(cu: int, cd: int, ct: int, indirect_gap: float) -> tuple[str, int | None]:
    if indirect_gap <= 0:
        return "indirect_overlap_metal", None
    if cu == -cd and ct == 0 and abs(cu) >= 1:
        return "spin_chern_TI", cu
    if cu == 0 and cd == 0 and ct == 0:
        return "trivial_insulator", 0
    if ct != 0:
        return "chern_insulator", cu
    return "other_gapped_phase", cu


def classify_one(
    *, core, background: dict[str, float], row: pd.Series, orientation_factor: int,
    config: Step15Config,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[pd.DataFrame]]:
    sample_id = str(row["sample_id"])
    raw = raw_parameters(background, row, core)
    gap = adaptive_gap_audit(core, raw, config)
    status = preliminary_status(gap, config)
    result: dict[str, Any] = {
        "sample_id": sample_id, "r3": float(row["r3"]), "r4": float(row["r4"]),
        "original_source_type": str(row.get("source_type", "")),
        "original_source_region": str(row.get("source_region", "")),
        **gap, "preliminary_status": status,
        "final_status": status, "final_chern_up": np.nan, "final_chern_down": np.nan,
        "final_chern_total": np.nan, "topology_method": "gap_audit_only",
        "strict_phase_code": 99, "definite_physical_conclusion": 1,
    }
    fukui_rows: list[dict[str, Any]] = []
    wilson_rows: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []

    if status not in {"strict_gapped_candidate"}:
        return result, fukui_rows, wilson_rows, traces

    attempts: list[dict[str, Any]] = []
    for nk in config.fukui_grids:
        for shift in config.fukui_shifts:
            item = fukui_attempt(core, raw, int(nk), shift, config)
            item["sample_id"] = sample_id
            attempts.append(item)
    fukui_rows.extend(attempts)
    fcons = fukui_consensus(attempts, config)
    result.update(fcons)
    if int(fcons["fukui_consensus"]) == 1:
        cu, cd, ct = int(fcons["fukui_chern_up"]), int(fcons["fukui_chern_down"]), int(fcons["fukui_chern_total"])
        label, code = phase_label_from_cherns(cu, cd, ct, float(gap["indirect_gap"]))
        result.update({
            "final_status": label, "final_chern_up": cu, "final_chern_down": cd,
            "final_chern_total": ct, "topology_method": "multigrid_fukui_consensus",
            "strict_phase_code": 99 if code is None else int(code),
        })
        return result, fukui_rows, wilson_rows, traces

    # Wilson fallback for both spin sectors.
    up_attempts, up_traces = wilson_attempts_for_spin(
        core=core, raw=raw, spin="up", orientation_factor=orientation_factor,
        gap=gap, config=config, sample_id=sample_id,
    )
    down_attempts, down_traces = wilson_attempts_for_spin(
        core=core, raw=raw, spin="down", orientation_factor=orientation_factor,
        gap=gap, config=config, sample_id=sample_id,
    )
    wilson_rows.extend(up_attempts + down_attempts)
    traces.extend(up_traces + down_traces)
    up_ok, cu, up_count = wilson_consensus(up_attempts)
    down_ok, cd, down_count = wilson_consensus(down_attempts)
    result.update({
        "wilson_up_consensus": int(up_ok), "wilson_down_consensus": int(down_ok),
        "wilson_up_consensus_count": up_count, "wilson_down_consensus_count": down_count,
    })
    if up_ok and down_ok and cu is not None and cd is not None:
        ct = int(cu + cd)
        label, code = phase_label_from_cherns(cu, cd, ct, float(gap["indirect_gap"]))
        result.update({
            "final_status": label, "final_chern_up": cu, "final_chern_down": cd,
            "final_chern_total": ct, "topology_method": "adaptive_wilson_consensus",
            "strict_phase_code": 99 if code is None else int(code),
        })
    else:
        result.update({
            "final_status": "numeric_unresolved", "topology_method": "fukui_and_wilson_failed",
            "definite_physical_conclusion": 0,
        })
    return result, fukui_rows, wilson_rows, traces


# =============================================================================
# Plotting
# =============================================================================


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })


def plot_reclassified_region(
    revised: pd.DataFrame, boundary_results: pd.DataFrame, *, xlim: tuple[float, float], ylim: tuple[float, float],
    title: str, output_stem: Path,
) -> None:
    configure_plot_style()
    strict = revised[revised["strict_phase_code"].isin([-2, -1, 0, 1, 2])].copy()
    if strict.empty:
        return
    interp = NearestNDInterpolator(strict[["r3", "r4"]].to_numpy(float), strict["strict_phase_code"].to_numpy(int))
    x = np.linspace(xlim[0], xlim[1], 700)
    y = np.linspace(ylim[0], ylim[1], 700)
    X, Y = np.meshgrid(x, y)
    Z = interp(X, Y)
    rgba = np.zeros((len(y), len(x), 4), dtype=float)
    for code, color in PHASE_COLORS.items():
        rgba[Z == code] = plt.matplotlib.colors.to_rgba(color)

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.imshow(rgba, origin="lower", extent=[*xlim, *ylim], interpolation="nearest", aspect="auto")
    local = boundary_results[
        boundary_results["r3"].between(*xlim) & boundary_results["r4"].between(*ylim)
    ]
    markers = {
        "phase_boundary_gap_closing": ("x", STATUS_COLORS["phase_boundary_gap_closing"]),
        "indirect_overlap_metal": ("P", STATUS_COLORS["indirect_overlap_metal"]),
        "spin_sector_gap_closing": ("D", STATUS_COLORS["spin_sector_gap_closing"]),
        "spin_filling_transition": ("v", STATUS_COLORS["spin_filling_transition"]),
        "indirect_overlap_boundary": ("h", STATUS_COLORS["indirect_overlap_boundary"]),
        "numeric_unresolved": ("X", STATUS_COLORS["numeric_unresolved"]),
    }
    for status, (marker, color) in markers.items():
        rows = local[local["final_status"].eq(status)]
        if not rows.empty:
            ax.scatter(rows["r3"], rows["r4"], marker=marker, s=55, color=color, linewidths=1.2, zorder=6)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_box_aspect(1)
    ax.set_xlabel(r"$\boldsymbol{\mathit{r}}_{\mathrm{3}}$", fontsize=24)
    ax.set_ylabel(r"$\boldsymbol{\mathit{r}}_{\mathrm{4}}$", fontsize=24)
    ax.set_title(title, fontsize=23, pad=14)
    ax.tick_params(direction="in", top=True, right=True, length=7, width=1.7, labelsize=15)
    for spine in ax.spines.values(): spine.set_linewidth(1.7)
    handles = [Patch(facecolor=PHASE_COLORS[c], edgecolor="black", label=rf"$\boldsymbol{{\mathit{{C}}}}_{{\uparrow}}={c}$") for c in sorted(set(strict["strict_phase_code"]).intersection(PHASE_COLORS))]
    handles += [Line2D([0],[0], marker=m, linestyle="None", color=col, markersize=9, label=s.replace("_", " ")) for s,(m,col) in markers.items() if (local["final_status"] == s).any()]
    leg = ax.legend(handles=handles, loc="best", fontsize=11, frameon=True)
    for text in leg.get_texts(): text.set_fontweight("bold")
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output_stem.with_suffix("." + suffix), dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================


def run_step15m(
    *, tts_archive: str | Path, step11_source: str | Path, step13_source: str | Path,
    step14_source: str | Path, output_dir: str | Path | None = None,
    config: Step15Config | None = None,
) -> dict[str, Any]:
    if config is None:
        config = Step15Config(output_dir=Path(output_dir or "outputs_tts_step15M_reclassify_all_boundary_points"))
    elif output_dir is not None:
        config.output_dir = Path(output_dir)
    config = config.normalized()

    all_points = read_csv_token(step14_source, "step14M_01b_all_phase_points_with_uncertain.csv")
    unresolved = all_points[all_points["uncertain_or_boundary"].astype(int).eq(1)].copy().reset_index(drop=True)
    background = read_json_token(step11_source, "step11M_13_fixed_slice_certificate.json")["fixed_background_parameters"]
    orientation = read_json_token(step13_source, "step13M_04b_orientation_calibration.json")
    if int(orientation.get("orientation_calibrated", 0)) != 1:
        raise RuntimeError("Step13 Wilson orientation is not calibrated")
    orientation_factor = int(orientation["orientation_factor"])
    core = load_core(tts_archive, config)

    atomic_csv(unresolved, config.output_dir / "step15M_01_original_unresolved_points.csv")
    atomic_json({
        "code_version": CODE_VERSION,
        "n_all_points": int(len(all_points)),
        "n_original_unresolved": int(len(unresolved)),
        "fixed_background_parameters": background,
        "wilson_orientation_factor": orientation_factor,
        "config": asdict(config),
    }, config.output_dir / "step15M_00_run_audit.json")

    results: list[dict[str, Any]] = []
    fukui_all: list[dict[str, Any]] = []
    wilson_all: list[dict[str, Any]] = []
    start = time.time()
    for index, row in unresolved.iterrows():
        sample_id = str(row["sample_id"])
        checkpoint = config.output_dir / "checkpoints" / f"{sample_id}.json"
        if checkpoint.is_file() and not config.force_recalculate:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            results.append(payload["result"])
            fukui_all.extend(payload.get("fukui_attempts", []))
            wilson_all.extend(payload.get("wilson_attempts", []))
            continue
        print(f"[{index+1}/{len(unresolved)}] {sample_id}  r3={row['r3']:.8f} r4={row['r4']:.8f}")
        result, fukui_rows, wilson_rows, traces = classify_one(
            core=core, background=background, row=row, orientation_factor=orientation_factor, config=config,
        )
        payload = {"result": result, "fukui_attempts": fukui_rows, "wilson_attempts": wilson_rows}
        atomic_json(payload, checkpoint)
        for trace in traces:
            sid = str(trace.iloc[0]["sample_id"]); spin = str(trace.iloc[0]["spin"]); attempt = int(trace.iloc[0]["attempt_index"])
            atomic_csv(trace, config.output_dir / "wilson_traces" / f"{sid}__{spin}__attempt{attempt:02d}.csv")
        results.append(result); fukui_all.extend(fukui_rows); wilson_all.extend(wilson_rows)
        print("    ->", result["final_status"], result.get("final_chern_up"))

    result_df = pd.DataFrame(results)
    atomic_csv(result_df, config.output_dir / "step15M_05_final_boundary_reclassification.csv")
    atomic_csv(pd.DataFrame(fukui_all), config.output_dir / "step15M_03_fukui_attempts.csv")
    atomic_csv(pd.DataFrame(wilson_all), config.output_dir / "step15M_04_wilson_attempts.csv")

    revised = all_points.copy()
    revised["strict_phase_code"] = np.where(revised["uncertain_or_boundary"].astype(int).eq(0), revised["chern_up"], 99).astype(int)
    revised["final_status"] = np.where(revised["uncertain_or_boundary"].astype(int).eq(0), "previously_certified_phase", "original_unresolved")
    by_id = result_df.set_index("sample_id")
    for idx, row in revised[revised["uncertain_or_boundary"].astype(int).eq(1)].iterrows():
        res = by_id.loc[str(row["sample_id"])]
        revised.loc[idx, "final_status"] = res["final_status"]
        revised.loc[idx, "strict_phase_code"] = int(res["strict_phase_code"])
        revised.loc[idx, "chern_up"] = res["final_chern_up"]
        revised.loc[idx, "uncertain_or_boundary"] = int(res["definite_physical_conclusion"] == 0)
    atomic_csv(revised, config.output_dir / "step15M_06_revised_all_phase_points.csv")

    counts = result_df["final_status"].value_counts(dropna=False).rename_axis("final_status").reset_index(name="count")
    counts["fraction"] = counts["count"] / max(len(result_df), 1)
    atomic_csv(counts, config.output_dir / "step15M_07_final_status_counts.csv")

    plot_reclassified_region(
        revised, result_df, xlim=(-0.67, -0.52), ylim=(-0.180, -0.135),
        title="TTS left-lower phase boundary after exhaustive reclassification",
        output_stem=config.output_dir / "figures" / "step15M_left_lower_reclassified",
    )
    plot_reclassified_region(
        revised, result_df, xlim=(0.054, 0.066), ylim=(0.128, 0.134),
        title="TTS right junction after exhaustive reclassification",
        output_stem=config.output_dir / "figures" / "step15M_right_junction_reclassified",
    )

    n_numeric = int(result_df["final_status"].eq("numeric_unresolved").sum())
    n_definite = int(result_df["definite_physical_conclusion"].astype(int).sum())
    certificate = {
        "code_version": CODE_VERSION,
        "research_design": "exhaustive_adaptive_reclassification_of_all_previous_boundary_points",
        "n_original_unresolved_points": int(len(unresolved)),
        "n_points_with_definite_physical_conclusion": n_definite,
        "n_strict_chern_classified": int(result_df["strict_phase_code"].isin([-2,-1,0,1,2]).sum()),
        "n_true_gap_closing_or_near_boundary": int(result_df["final_status"].eq("phase_boundary_gap_closing").sum()),
        "n_indirect_overlap_metals": int(result_df["final_status"].eq("indirect_overlap_metal").sum()),
        "n_spin_sector_gap_closings": int(result_df["final_status"].eq("spin_sector_gap_closing").sum()),
        "n_spin_filling_transitions": int(result_df["final_status"].eq("spin_filling_transition").sum()),
        "n_indirect_overlap_boundaries": int(result_df["final_status"].eq("indirect_overlap_boundary").sum()),
        "n_numeric_unresolved": n_numeric,
        "all_previous_boundary_points_physically_resolved": bool(n_numeric == 0 and n_definite == len(unresolved)),
        "elapsed_seconds": float(time.time() - start),
        "interpretation": {
            "strict_chern_phase": "A positive global insulating gap and isolated spin subspaces permit a spin-Chern label.",
            "true_boundary": "A gap-closing point is a definite phase-boundary conclusion but has no well-defined Chern number at the critical parameter.",
            "metal": "An indirect-overlap point is not a topological insulator even if direct gaps at fixed k remain positive.",
        },
    }
    atomic_json(certificate, config.output_dir / "step15M_08_final_reclassification_certificate.json")
    print(json.dumps(certificate, ensure_ascii=False, indent=2, default=_json_default))
    return certificate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tts-archive", required=True)
    p.add_argument("--step11-source", required=True)
    p.add_argument("--step13-source", required=True)
    p.add_argument("--step14-source", required=True)
    p.add_argument("--output-dir", default="outputs_tts_step15M_reclassify_all_boundary_points")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force-recalculate", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = Step15Config(output_dir=Path(args.output_dir), quick=bool(args.quick), force_recalculate=bool(args.force_recalculate))
    run_step15m(
        tts_archive=args.tts_archive, step11_source=args.step11_source,
        step13_source=args.step13_source, step14_source=args.step14_source,
        output_dir=args.output_dir, config=cfg,
    )


if __name__ == "__main__":
    main()
