#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FES Step 04: transformed-coordinate phase-boundary scan.

Purpose
-------
Starting from the best strictly verified spin-Chern band-metal point obtained in
Step 03, scan the transformed coordinates

    t_s = (t1 + t2) / 2,   t_d = (t1 - t2) / 2,
    r_s = (r1 + r2) / 2,   r_d = (r1 - r2) / 2,

and determine whether the indirect-gap boundary E_g^ind = 0 overlaps a finite
region with the spin-Chern phase C_up = -C_down = +/-1.

The default workflow contains three complementary calculations:
1. a two-dimensional (m_e, r_d) plane around the best topological point;
2. a two-dimensional (r_s, r_d) plane around the same point;
3. a five-dimensional straight path from the best topological point to the
   nearest strictly audited positive-indirect-gap point from Step 03.

Coarse maps are followed by multi-grid, multi-shift strict verification of
points close to either the indirect-gap boundary or the topological boundary.
A nonzero spin Chern number is labeled only as a candidate; type-II QSH still
requires later band-inversion, edge-state, and transport verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import FES_step02_sobol_scan as core
except ImportError as exc:
    raise ImportError(
        "FES_step02_sobol_scan.py must be in the same directory or on PYTHONPATH."
    ) from exc


CODE_VERSION = "FES_STEP04_BOUNDARY_V1_20260713"
RESULT_SCHEMA_VERSION = "fes_step04_boundary_schema_v1"
HALF_COORDS = ("m_e", "t_s_half", "t_d_half", "r_s_half", "r_d_half")
REDUCED5 = ("m_e", "t1", "t2", "r1", "r2")


@dataclass
class BoundaryConfig:
    step03_input: str = "outputs_fes6_step03_local_gap"
    output_dir: str = "outputs_fes6_step04_boundary"
    workers: int = max(1, min(8, os.cpu_count() or 1))

    planes: tuple[str, ...] = ("me-rd", "rs-rd")
    grid_size: int = 61
    path_points: int = 301

    # Automatic ranges use max(user span, anchor separation + margin).
    me_span: float = 0.040
    rs_span: float = 0.040
    rd_span: float = 0.030
    anchor_margin: float = 0.010

    coarse_gap_nk: int = 21
    coarse_chern_nk: int = 13
    gap_tol: float = 1.0e-3
    chern_tol: float = 0.08
    min_det_tol: float = 1.0e-7

    # Strict verification settings.
    verify_gap_grids: tuple[int, ...] = (51, 71)
    verify_gap_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    )
    verify_chern_grids: tuple[int, ...] = (21, 31, 41)
    verify_chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )

    indirect_boundary_window: float = 0.012
    direct_boundary_window: float = 0.012
    max_strict_verify: int = 48

    overwrite: bool = False
    resume: bool = False
    test_mode: bool = False


# -----------------------------------------------------------------------------
# Coordinate transformations
# -----------------------------------------------------------------------------

def reduced_to_half_coords(reduced: Dict[str, float]) -> Dict[str, float]:
    p = {name: float(reduced[name]) for name in REDUCED5}
    return {
        "m_e": p["m_e"],
        "t_s_half": 0.5 * (p["t1"] + p["t2"]),
        "t_d_half": 0.5 * (p["t1"] - p["t2"]),
        "r_s_half": 0.5 * (p["r1"] + p["r2"]),
        "r_d_half": 0.5 * (p["r1"] - p["r2"]),
    }


def half_coords_to_reduced(coords: Dict[str, float]) -> Dict[str, float]:
    c = {name: float(coords[name]) for name in HALF_COORDS}
    return {
        "m_e": c["m_e"],
        "t1": c["t_s_half"] + c["t_d_half"],
        "t2": c["t_s_half"] - c["t_d_half"],
        "r1": c["r_s_half"] + c["r_d_half"],
        "r2": c["r_s_half"] - c["r_d_half"],
    }


def reduced_to_raw6(reduced: Dict[str, float]) -> Dict[str, float]:
    return core.raw6_from_reduced5(reduced, e0=0.0)


def in_default_bounds(reduced: Dict[str, float]) -> bool:
    return all(-1.0 <= float(reduced[name]) <= 1.0 for name in REDUCED5)


# -----------------------------------------------------------------------------
# Step 03 input and anchor selection
# -----------------------------------------------------------------------------

def _find_result_root(root: Path) -> Path:
    direct = root / "fes_step03_strict_verified_results.csv"
    if direct.exists():
        return root
    matches = list(root.rglob("fes_step03_strict_verified_results.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Could not uniquely locate fes_step03_strict_verified_results.csv "
            f"under {root}. Found {len(matches)} matches."
        )
    return matches[0].parent


def locate_step03_directory(input_path: str | Path, output_dir: Path) -> Path:
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Step 03 input does not exist: {source}")
    if source.is_dir():
        return _find_result_root(source)
    if source.suffix.lower() != ".zip":
        raise ValueError("Step 03 input must be a directory or ZIP archive.")

    extract_dir = output_dir / "_step03_input_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as zf:
        zf.extractall(extract_dir)
    return _find_result_root(extract_dir)


def _row_reduced(row: pd.Series) -> Dict[str, float]:
    return {name: float(row[name]) for name in REDUCED5}


def select_anchor_points(step03_dir: Path) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    strict_path = step03_dir / "fes_step03_strict_verified_results.csv"
    strict_df = pd.read_csv(strict_path)
    if strict_df.empty:
        raise ValueError("Step 03 strict verification table is empty.")

    topo = strict_df[
        (strict_df["strict_phase_label"] == "spin_chern_band_metal")
        & (strict_df["verified_chern_reliable"].fillna(0).astype(int) == 1)
    ].copy()
    if topo.empty:
        raise ValueError("No strictly verified spin_chern_band_metal seed was found.")
    topo = topo.sort_values("verified_indirect_gap", ascending=False)
    topo_row = topo.iloc[0]
    topo_reduced = _row_reduced(topo_row)
    topo_half = reduced_to_half_coords(topo_reduced)

    positive = strict_df[
        (strict_df["verified_indirect_gap"] > 0.0)
        & (strict_df["verified_is_direct_gapped"].fillna(0).astype(int) == 1)
        & (strict_df["verified_is_balanced_spin_sector"].fillna(0).astype(int) == 1)
    ].copy()

    if positive.empty:
        # Fallback: use the non-topological point with the largest indirect gap.
        non_topo = strict_df[strict_df.index != topo_row.name].copy()
        if non_topo.empty:
            raise ValueError("No non-topological anchor is available.")
        positive = non_topo.sort_values("verified_indirect_gap", ascending=False).head(1)

    # Choose the nearest positive-gap point in transformed coordinates.
    distances = []
    for idx, row in positive.iterrows():
        h = reduced_to_half_coords(_row_reduced(row))
        distance = math.sqrt(sum((h[name] - topo_half[name]) ** 2 for name in HALF_COORDS))
        distances.append((idx, distance))
    positive_idx = min(distances, key=lambda item: item[1])[0]
    positive_row = strict_df.loc[positive_idx]
    positive_reduced = _row_reduced(positive_row)

    anchor_rows = []
    for role, row in (("best_topological", topo_row), ("positive_gap_anchor", positive_row)):
        reduced = _row_reduced(row)
        half = reduced_to_half_coords(reduced)
        anchor_rows.append(
            {
                "anchor_role": role,
                "candidate_id": str(row.get("candidate_id", "")),
                **reduced,
                **half,
                "strict_phase_label": str(row.get("strict_phase_label", "")),
                "verified_min_direct_gap": float(row.get("verified_min_direct_gap", np.nan)),
                "verified_indirect_gap": float(row.get("verified_indirect_gap", np.nan)),
                "verified_chern_up_int": row.get("verified_chern_up_int", np.nan),
                "verified_chern_down_int": row.get("verified_chern_down_int", np.nan),
                "verified_chern_reliable": int(row.get("verified_chern_reliable", 0)),
            }
        )
    return pd.DataFrame(anchor_rows), topo_reduced, positive_reduced


# -----------------------------------------------------------------------------
# Point diagnostics
# -----------------------------------------------------------------------------

def special_point_diagnostics(raw6: Dict[str, float]) -> Dict[str, object]:
    points = {
        "gamma": (0.0, 0.0),
        "M": (math.pi, math.pi),
        "X": (math.pi, 0.0),
        "Y": (0.0, math.pi),
    }
    result: Dict[str, object] = {}
    for name, (kx, ky) in points.items():
        up = np.linalg.eigvalsh(core.h_spin_block_atomic(kx, ky, raw6, "up"))
        down = np.linalg.eigvalsh(core.h_spin_block_atomic(kx, ky, raw6, "down"))
        energies = np.sort(np.concatenate([up, down]))
        result[f"{name}_v4"] = float(energies[3])
        result[f"{name}_c5"] = float(energies[4])
        result[f"{name}_local_gap"] = float(energies[4] - energies[3])

    result["delta_v_M_minus_gamma"] = result["M_v4"] - result["gamma_v4"]
    result["delta_c_M_minus_gamma"] = result["M_c5"] - result["gamma_c5"]
    result["cross_gap_gammaV_to_Mc"] = result["M_c5"] - result["gamma_v4"]
    result["cross_gap_MV_to_gammaC"] = result["gamma_c5"] - result["M_v4"]
    result["gamma_M_indirect_proxy"] = min(
        result["cross_gap_gammaV_to_Mc"], result["cross_gap_MV_to_gammaC"]
    )
    result["gamma_M_vbm_site"] = (
        "Gamma" if result["gamma_v4"] >= result["M_v4"] else "M"
    )
    result["gamma_M_cbm_site"] = (
        "Gamma" if result["gamma_c5"] <= result["M_c5"] else "M"
    )
    return result


def _empty_chern() -> Dict[str, object]:
    return {
        "chern_nk": np.nan,
        "chern_up": np.nan,
        "chern_down": np.nan,
        "chern_total_inferred": np.nan,
        "spin_chern": np.nan,
        "chern_up_int": np.nan,
        "chern_down_int": np.nan,
        "chern_total_int": np.nan,
        "min_det_up": np.nan,
        "min_det_down": np.nan,
        "chern_reliable": 0,
    }


def scan_boundary_point(task: tuple) -> Dict[str, object]:
    (
        point_id,
        source_kind,
        source_name,
        i_index,
        j_index,
        path_lambda,
        half_coords,
        gap_nk,
        chern_nk,
        gap_tol,
        chern_tol,
        min_det_tol,
    ) = task

    half = {name: float(half_coords[name]) for name in HALF_COORDS}
    reduced = half_coords_to_reduced(half)
    raw6 = reduced_to_raw6(reduced)

    row: Dict[str, object] = {
        "point_id": str(point_id),
        "source_kind": str(source_kind),
        "source_name": str(source_name),
        "i_index": int(i_index),
        "j_index": int(j_index),
        "path_lambda": float(path_lambda),
        **half,
        **reduced,
        **core.derived_features(raw6),
        **_empty_chern(),
        "phase_label": "numeric_error",
        "is_spin_chern_topological": 0,
        "is_spin_chern_TI_candidate": 0,
        "error": "",
    }

    if not in_default_bounds(reduced):
        row["phase_label"] = "outside_parameter_bounds"
        return row

    try:
        gap = core.scan_band_gaps_shifted(
            raw6, int(gap_nk), (0.0, 0.0), float(gap_tol)
        )
        row.update(gap)
        row.update(special_point_diagnostics(raw6))
    except Exception as exc:
        row["error"] = f"gap_failed: {exc!r}"
        return row

    if not int(row.get("is_direct_gapped", 0)):
        row["phase_label"] = "noninsulating_or_gap_closing"
        return row
    if not int(row.get("is_balanced_spin_sector", 0)):
        row["phase_label"] = "spin_sector_filling_mismatch"
        return row

    try:
        chern = core.calculate_coarse_spin_chern(
            raw6,
            int(chern_nk),
            float(chern_tol),
            float(min_det_tol),
        )
        row.update(chern)
    except Exception as exc:
        row["phase_label"] = "chern_unreliable"
        row["error"] = f"chern_failed: {exc!r}"
        return row

    label, candidate = core.classify_from_chern(row, row)
    cu = row.get("chern_up_int")
    cd = row.get("chern_down_int")
    ct = row.get("chern_total_int")
    spin_topo = bool(
        int(row.get("chern_reliable", 0))
        and pd.notna(cu)
        and pd.notna(cd)
        and pd.notna(ct)
        and int(cu) == -int(cd)
        and int(cu) != 0
        and int(ct) == 0
    )
    row["phase_label"] = label
    row["is_spin_chern_topological"] = int(spin_topo)
    row["is_spin_chern_TI_candidate"] = int(candidate)
    return row


# -----------------------------------------------------------------------------
# Point generation
# -----------------------------------------------------------------------------

def _auto_span(
    name: str,
    topo_half: Dict[str, float],
    positive_half: Dict[str, float],
    minimum_span: float,
    margin: float,
) -> float:
    return max(
        float(minimum_span),
        0.65 * abs(float(positive_half[name]) - float(topo_half[name])) + margin,
    )


def plane_definition(name: str) -> tuple[str, str]:
    definitions = {
        "me-rd": ("m_e", "r_d_half"),
        "rs-rd": ("r_s_half", "r_d_half"),
        "me-rs": ("m_e", "r_s_half"),
        "ts-td": ("t_s_half", "t_d_half"),
    }
    if name not in definitions:
        raise ValueError(f"Unsupported plane {name!r}; choose from {sorted(definitions)}")
    return definitions[name]


def generate_plane_tasks(
    plane_name: str,
    topo_reduced: Dict[str, float],
    positive_reduced: Dict[str, float],
    config: BoundaryConfig,
) -> tuple[list[tuple], Dict[str, object]]:
    x_name, y_name = plane_definition(plane_name)
    topo_half = reduced_to_half_coords(topo_reduced)
    positive_half = reduced_to_half_coords(positive_reduced)

    minimum_spans = {
        "m_e": config.me_span,
        "r_s_half": config.rs_span,
        "r_d_half": config.rd_span,
        "t_s_half": 0.04,
        "t_d_half": 0.04,
    }
    x_span = _auto_span(
        x_name, topo_half, positive_half, minimum_spans[x_name], config.anchor_margin
    )
    y_span = _auto_span(
        y_name, topo_half, positive_half, minimum_spans[y_name], config.anchor_margin
    )

    x_values = np.linspace(topo_half[x_name] - x_span, topo_half[x_name] + x_span, config.grid_size)
    y_values = np.linspace(topo_half[y_name] - y_span, topo_half[y_name] + y_span, config.grid_size)

    tasks: list[tuple] = []
    for i, x_value in enumerate(x_values):
        for j, y_value in enumerate(y_values):
            coords = dict(topo_half)
            coords[x_name] = float(x_value)
            coords[y_name] = float(y_value)
            point_id = f"{plane_name}_i{i:04d}_j{j:04d}"
            tasks.append(
                (
                    point_id,
                    "plane",
                    plane_name,
                    i,
                    j,
                    np.nan,
                    coords,
                    config.coarse_gap_nk,
                    config.coarse_chern_nk,
                    config.gap_tol,
                    config.chern_tol,
                    config.min_det_tol,
                )
            )

    metadata = {
        "plane_name": plane_name,
        "x_name": x_name,
        "y_name": y_name,
        "x_min": float(x_values.min()),
        "x_max": float(x_values.max()),
        "y_min": float(y_values.min()),
        "y_max": float(y_values.max()),
        "grid_size": int(config.grid_size),
        "x_span": float(x_span),
        "y_span": float(y_span),
    }
    return tasks, metadata


def generate_path_tasks(
    topo_reduced: Dict[str, float],
    positive_reduced: Dict[str, float],
    config: BoundaryConfig,
) -> list[tuple]:
    topo_half = reduced_to_half_coords(topo_reduced)
    positive_half = reduced_to_half_coords(positive_reduced)
    tasks: list[tuple] = []
    for index, lam in enumerate(np.linspace(0.0, 1.0, config.path_points)):
        coords = {
            name: (1.0 - lam) * topo_half[name] + lam * positive_half[name]
            for name in HALF_COORDS
        }
        tasks.append(
            (
                f"anchor_path_{index:05d}",
                "path",
                "topological_to_positive_gap",
                index,
                0,
                float(lam),
                coords,
                config.coarse_gap_nk,
                config.coarse_chern_nk,
                config.gap_tol,
                config.chern_tol,
                config.min_det_tol,
            )
        )
    return tasks


# -----------------------------------------------------------------------------
# Execution, boundary queue, and strict verification
# -----------------------------------------------------------------------------

def run_tasks(tasks: Sequence[tuple], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [scan_boundary_point(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(scan_boundary_point, tasks, chunksize=8))
    return pd.DataFrame(rows)


def _add_adjacent_transition_points(df: pd.DataFrame, selected_ids: set[str]) -> None:
    if df.empty or df["source_kind"].iloc[0] != "plane":
        return
    grid = df.set_index(["i_index", "j_index"])
    topo = df["is_spin_chern_topological"].fillna(0).astype(int)
    topo_by_index = dict(zip(zip(df["i_index"], df["j_index"]), topo))
    id_by_index = dict(zip(zip(df["i_index"], df["j_index"]), df["point_id"]))
    for (i, j), flag in topo_by_index.items():
        for neighbor in ((i + 1, j), (i, j + 1)):
            if neighbor in topo_by_index and topo_by_index[neighbor] != flag:
                selected_ids.add(str(id_by_index[(i, j)]))
                selected_ids.add(str(id_by_index[neighbor]))


def select_refine_queue(
    all_df: pd.DataFrame, config: BoundaryConfig
) -> pd.DataFrame:
    selected_ids: set[str] = set()

    # Always include any coarse TI candidate and the best topological points.
    selected_ids.update(
        all_df.loc[
            all_df["phase_label"] == "spin_chern_TI_candidate", "point_id"
        ].astype(str)
    )
    topological = all_df[all_df["is_spin_chern_topological"] == 1].copy()
    if not topological.empty:
        selected_ids.update(
            topological.nlargest(12, "indirect_gap")["point_id"].astype(str)
        )

    boundary_mask = (
        all_df["indirect_gap"].abs() <= config.indirect_boundary_window
    ) | (all_df["min_direct_gap"].abs() <= config.direct_boundary_window)
    selected_ids.update(all_df.loc[boundary_mask, "point_id"].astype(str))

    for _, group in all_df[all_df["source_kind"] == "plane"].groupby("source_name"):
        _add_adjacent_transition_points(group, selected_ids)

    queue = all_df[all_df["point_id"].astype(str).isin(selected_ids)].copy()
    if queue.empty:
        return queue

    queue["rank_TI"] = (queue["phase_label"] == "spin_chern_TI_candidate").astype(int)
    queue["rank_topological"] = queue["is_spin_chern_topological"].fillna(0).astype(int)
    queue["rank_indirect_distance"] = queue["indirect_gap"].abs()
    queue["rank_direct_distance"] = queue["min_direct_gap"].abs()
    queue = queue.sort_values(
        [
            "rank_TI",
            "rank_topological",
            "rank_indirect_distance",
            "rank_direct_distance",
        ],
        ascending=[False, False, True, True],
    )
    queue = queue.drop_duplicates("point_id").head(config.max_strict_verify)
    return queue


def strict_verify_one(row: Dict[str, object], config: BoundaryConfig, details_dir: Path) -> Dict[str, object]:
    reduced = {name: float(row[name]) for name in REDUCED5}
    raw6 = reduced_to_raw6(reduced)
    result = dict(row)
    result["strict_error"] = ""

    try:
        gap_summary, gap_runs = core.consensus_gap_audit(
            raw6,
            config.verify_gap_grids,
            config.verify_gap_shifts,
            config.gap_tol,
        )
        result.update(gap_summary)
        gap_runs.to_csv(details_dir / f"{row['point_id']}_gap_consensus.csv", index=False)
    except Exception as exc:
        result["strict_phase_label"] = "strict_gap_failed"
        result["strict_error"] = repr(exc)
        return result

    if not int(result.get("verified_is_direct_gapped", 0)):
        result["strict_phase_label"] = "noninsulating_or_gap_closing"
        return result
    if not int(result.get("verified_is_balanced_spin_sector", 0)):
        result["strict_phase_label"] = "spin_sector_filling_mismatch"
        return result

    try:
        chern_summary, chern_runs = core.consensus_chern_audit(
            raw6,
            config.verify_chern_grids,
            config.verify_chern_shifts,
            config.chern_tol,
            config.min_det_tol,
            include_total=True,
        )
        result.update(chern_summary)
        chern_runs.to_csv(details_dir / f"{row['point_id']}_chern_consensus.csv", index=False)
    except Exception as exc:
        result["strict_phase_label"] = "chern_unreliable"
        result["strict_error"] = repr(exc)
        return result

    verified_label, _ = core.verified_phase_label(result)
    result["strict_phase_label"] = verified_label.removeprefix("verified_")
    return result


def run_strict_verification(
    queue: pd.DataFrame,
    config: BoundaryConfig,
    output_dir: Path,
) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame()
    details_dir = output_dir / "strict_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    rows = [strict_verify_one(row.to_dict(), config, details_dir) for _, row in queue.iterrows()]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Boundary diagnostics and plots
# -----------------------------------------------------------------------------

def estimate_zero_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    crossings: list[float] = []
    for i in range(len(x) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0.0:
            crossings.append(float(x[i]))
        elif y0 * y1 < 0.0:
            fraction = -y0 / (y1 - y0)
            crossings.append(float(x[i] + fraction * (x[i + 1] - x[i])))
    return crossings


def path_boundary_summary(path_df: pd.DataFrame) -> Dict[str, object]:
    if path_df.empty:
        return {}
    path = path_df.sort_values("path_lambda")
    lam = path["path_lambda"].to_numpy(float)
    indirect = path["indirect_gap"].to_numpy(float)
    direct = path["min_direct_gap"].to_numpy(float)
    topo = path["is_spin_chern_topological"].fillna(0).astype(int).to_numpy()

    topo_transitions = []
    for i in range(len(lam) - 1):
        if topo[i] != topo[i + 1]:
            topo_transitions.append(float(0.5 * (lam[i] + lam[i + 1])))

    ti_mask = (topo == 1) & (indirect > 1.0e-3) & (direct > 1.0e-3)
    topo_mask = topo == 1
    return {
        "path_indirect_zero_lambdas": estimate_zero_crossings(lam, indirect),
        "path_direct_zero_lambdas": estimate_zero_crossings(lam, direct),
        "path_topology_transition_lambdas": topo_transitions,
        "path_n_coarse_topological": int(topo_mask.sum()),
        "path_n_coarse_TI_candidates": int(ti_mask.sum()),
        "path_best_indirect_gap_in_topological_sector": (
            float(np.max(indirect[topo_mask])) if topo_mask.any() else None
        ),
    }


def write_slice_diagnostics(df: pd.DataFrame, plane_meta: Dict[str, object], path: Path) -> None:
    x_name = str(plane_meta["x_name"])
    rows = []
    for x_value, group in df.groupby(x_name):
        topo = group[group["is_spin_chern_topological"] == 1]
        rows.append(
            {
                x_name: float(x_value),
                "n_points": int(len(group)),
                "n_topological": int(len(topo)),
                "n_TI_candidates": int((group["phase_label"] == "spin_chern_TI_candidate").sum()),
                "max_indirect_gap_all": float(group["indirect_gap"].max()),
                "max_indirect_gap_topological": (
                    float(topo["indirect_gap"].max()) if not topo.empty else np.nan
                ),
                "min_direct_gap_topological": (
                    float(topo["min_direct_gap"].min()) if not topo.empty else np.nan
                ),
            }
        )
    pd.DataFrame(rows).sort_values(x_name).to_csv(path, index=False)


def plot_plane_indirect(df: pd.DataFrame, meta: Dict[str, object], output_path: Path) -> None:
    x_name = str(meta["x_name"])
    y_name = str(meta["y_name"])
    pivot = df.pivot(index=y_name, columns=x_name, values="indirect_gap").sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    z = pivot.to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    mesh = ax.pcolormesh(x, y, z, shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"$E_g^{ind}$")
    if np.nanmin(z) <= 0.0 <= np.nanmax(z):
        ax.contour(x, y, z, levels=[0.0])
    topo_pivot = df.pivot(index=y_name, columns=x_name, values="is_spin_chern_topological").sort_index().sort_index(axis=1)
    topo_z = topo_pivot.to_numpy(float)
    if np.nanmin(topo_z) < 0.5 < np.nanmax(topo_z):
        ax.contour(x, y, topo_z, levels=[0.5])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{meta['plane_name']}: indirect gap; contours = gap zero and topology boundary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_plane_phase(df: pd.DataFrame, meta: Dict[str, object], output_path: Path) -> None:
    x_name = str(meta["x_name"])
    y_name = str(meta["y_name"])
    phase_code = {
        "noninsulating_or_gap_closing": 0,
        "spin_sector_filling_mismatch": 1,
        "chern_unreliable": 2,
        "indirect_overlap": 3,
        "trivial_insulator": 4,
        "spin_chern_band_metal": 5,
        "spin_chern_TI_candidate": 6,
        "other_gapped_phase": 7,
        "Chern_insulator": 8,
        "outside_parameter_bounds": -1,
        "numeric_error": -2,
    }
    work = df.copy()
    work["phase_code"] = work["phase_label"].map(phase_code).fillna(-3)
    pivot = work.pivot(index=y_name, columns=x_name, values="phase_code").sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    z = pivot.to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    mesh = ax.pcolormesh(x, y, z, shading="auto")
    fig.colorbar(mesh, ax=ax, label="phase code")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{meta['plane_name']}: coarse phase map")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_path(path_df: pd.DataFrame, output_path: Path) -> None:
    path = path_df.sort_values("path_lambda")
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.plot(path["path_lambda"], path["indirect_gap"], label=r"$E_g^{ind}$")
    ax.plot(path["path_lambda"], path["min_direct_gap"], label=r"$E_g^{dir}$")
    ax.axhline(0.0, linewidth=1.0)
    topo = path[path["is_spin_chern_topological"] == 1]
    if not topo.empty:
        ax.scatter(topo["path_lambda"], topo["indirect_gap"], s=14, label="spin-Chern topological")
    ax.set_xlabel(r"path coordinate $\lambda$")
    ax.set_ylabel("gap")
    ax.set_title("Path from best topological point to positive-gap anchor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Pipeline and command line
# -----------------------------------------------------------------------------

def prepare_output(config: BoundaryConfig) -> Path:
    output = Path(config.output_dir).expanduser().resolve()
    if output.exists() and config.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use --overwrite or --resume."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def script_sha256() -> str | None:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return None


def run_pipeline(config: BoundaryConfig) -> Dict[str, Path]:
    output = prepare_output(config)
    step03_dir = locate_step03_directory(config.step03_input, output)
    anchors_df, topo_reduced, positive_reduced = select_anchor_points(step03_dir)
    anchors_path = output / "fes_step04_anchor_points.csv"
    anchors_df.to_csv(anchors_path, index=False)

    metadata = {
        "code_version": CODE_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": script_sha256(),
        "step02_core_file": str(Path(core.__file__).resolve()),
        "step02_core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "step03_result_directory": str(step03_dir.resolve()),
        "config": asdict(config),
    }
    metadata_path = output / "fes_step04_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    all_frames: list[pd.DataFrame] = []
    plane_meta_rows: list[Dict[str, object]] = []
    output_paths: Dict[str, Path] = {
        "anchors": anchors_path,
        "metadata": metadata_path,
    }

    for plane_name in config.planes:
        tasks, meta = generate_plane_tasks(
            plane_name, topo_reduced, positive_reduced, config
        )
        plane_path = output / f"fes_step04_{plane_name}_scan.csv"
        if config.resume and plane_path.exists():
            plane_df = pd.read_csv(plane_path)
        else:
            plane_df = run_tasks(tasks, config.workers)
            plane_df.to_csv(plane_path, index=False)
        all_frames.append(plane_df)
        plane_meta_rows.append(meta)
        output_paths[f"plane_{plane_name}"] = plane_path

        plot_plane_indirect(
            plane_df,
            meta,
            output / f"fes_step04_{plane_name}_indirect_gap_map.png",
        )
        plot_plane_phase(
            plane_df,
            meta,
            output / f"fes_step04_{plane_name}_phase_map.png",
        )
        write_slice_diagnostics(
            plane_df,
            meta,
            output / f"fes_step04_{plane_name}_slice_diagnostics.csv",
        )

    plane_meta_path = output / "fes_step04_plane_metadata.csv"
    pd.DataFrame(plane_meta_rows).to_csv(plane_meta_path, index=False)
    output_paths["plane_metadata"] = plane_meta_path

    path_tasks = generate_path_tasks(topo_reduced, positive_reduced, config)
    path_scan_path = output / "fes_step04_anchor_path_scan.csv"
    if config.resume and path_scan_path.exists():
        path_df = pd.read_csv(path_scan_path)
    else:
        path_df = run_tasks(path_tasks, config.workers)
        path_df.to_csv(path_scan_path, index=False)
    all_frames.append(path_df)
    plot_path(path_df, output / "fes_step04_anchor_path_gaps.png")
    output_paths["path_scan"] = path_scan_path

    all_df = pd.concat(all_frames, ignore_index=True)
    all_scan_path = output / "fes_step04_all_coarse_scan.csv"
    all_df.to_csv(all_scan_path, index=False)
    output_paths["all_coarse"] = all_scan_path

    queue = select_refine_queue(all_df, config)
    queue_path = output / "fes_step04_strict_refine_queue.csv"
    queue.to_csv(queue_path, index=False)
    output_paths["refine_queue"] = queue_path

    strict_df = run_strict_verification(queue, config, output)
    strict_path = output / "fes_step04_strict_verified_results.csv"
    strict_df.to_csv(strict_path, index=False)
    output_paths["strict_results"] = strict_path

    if strict_df.empty:
        ti_df = strict_df.copy()
        topo_df = strict_df.copy()
    else:
        ti_df = strict_df[
            strict_df["strict_phase_label"] == "spin_chern_TI_candidate"
        ].copy()
        topo_df = strict_df[
            strict_df["strict_phase_label"].isin(
                ["spin_chern_TI_candidate", "spin_chern_band_metal"]
            )
        ].copy()

    ti_path = output / "fes_step04_spin_chern_TI_candidates.csv"
    topo_path = output / "fes_step04_verified_topological_points.csv"
    ti_df.to_csv(ti_path, index=False)
    topo_df.to_csv(topo_path, index=False)
    output_paths["ti_candidates"] = ti_path
    output_paths["verified_topological"] = topo_path

    summary: Dict[str, object] = {
        "n_plane_points": int(sum(len(frame) for frame in all_frames[:-1])),
        "n_path_points": int(len(path_df)),
        "n_coarse_points_total": int(len(all_df)),
        "coarse_phase_counts": all_df["phase_label"].value_counts().to_dict(),
        "n_strict_refine_queue": int(len(queue)),
        "n_strict_verified": int(len(strict_df)),
        "strict_phase_counts": (
            strict_df["strict_phase_label"].value_counts().to_dict()
            if not strict_df.empty
            else {}
        ),
        "n_strict_spin_chern_TI_candidates": int(len(ti_df)),
        "best_strict_topological_indirect_gap": (
            float(topo_df["verified_indirect_gap"].max()) if not topo_df.empty else None
        ),
        "path_boundary_analysis": path_boundary_summary(path_df),
    }
    summary_path = output / "fes_step04_boundary_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_paths["summary"] = summary_path

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step03-input", default="outputs_fes6_step03_local_gap")
    parser.add_argument("--output-dir", default="outputs_fes6_step04_boundary")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--path-points", type=int, default=301)
    parser.add_argument(
        "--planes",
        default="me-rd,rs-rd",
        help="Comma-separated planes: me-rd, rs-rd, me-rs, ts-td",
    )
    parser.add_argument("--max-strict-verify", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if args.test:
        config = BoundaryConfig(
            step03_input=args.step03_input,
            output_dir=args.output_dir,
            workers=min(args.workers, 2),
            planes=("me-rd",),
            grid_size=9,
            path_points=31,
            coarse_gap_nk=13,
            coarse_chern_nk=9,
            verify_gap_grids=(17, 21),
            verify_gap_shifts=((0.0, 0.0), (0.5, 0.5)),
            verify_chern_grids=(11, 15),
            verify_chern_shifts=((0.0, 0.0),),
            max_strict_verify=min(args.max_strict_verify, 8),
            overwrite=args.overwrite,
            resume=args.resume,
            test_mode=True,
        )
    else:
        planes = tuple(item.strip() for item in args.planes.split(",") if item.strip())
        config = BoundaryConfig(
            step03_input=args.step03_input,
            output_dir=args.output_dir,
            workers=max(1, args.workers),
            planes=planes,
            grid_size=max(9, args.grid_size),
            path_points=max(31, args.path_points),
            max_strict_verify=max(1, args.max_strict_verify),
            overwrite=args.overwrite,
            resume=args.resume,
        )

    run_pipeline(config)


if __name__ == "__main__":
    main()
