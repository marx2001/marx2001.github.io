from __future__ import annotations

"""
TTS Step11M — Lieb-aligned fixed-slice mechanism closure
=========================================================

This stage deliberately keeps the five background parameters

    (m_e, t1, t2, r1, r2)

fixed, exactly as a representative reduced-parameter phase diagram should do.

The workflow follows the same logic used in the Lieb study:
1. reduce the topology problem to a physically interpretable two-parameter plane;
2. complete and certify the phase boundaries in that plane;
3. define signed local mass coordinates from those certified boundaries;
4. validate the mass-sign rule with spatially blocked holdout tests;
5. resolve the only remaining local junction by strict Chern mapping and
   selected multi-closure edge certificates.

It does NOT claim a universal seven-dimensional mass formula.
"""

import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import TTS_step09M_minus2_local_kp_refinement as base
import TTS_step09M_minus2_linear_signed_v2 as linear_v2
import TTS_step09M_D_multiclosure_path_atlas as step09d
import TTS_step10M_generic_four_valley_boundary_atlas as step10


CODE_VERSION = "TTS_STEP11M_LIEB_ALIGNED_FIXED_SLICE_V1_20260724"
REDUCED7 = list(base.REDUCED7)
FIXED5 = ["m_e", "t1", "t2", "r1", "r2"]
PHASE_CODES = {-2, -1, 0, 1, 2}

PHASE_ORDER = [-2, -1, 0, 1, 2, 99]
PHASE_COLORS = {
    2: (242/255, 142/255, 139/255),
    1: (127/255, 203/255, 161/255),
    0: (178/255, 178/255, 178/255),
    -1: (138/255, 163/255, 205/255),
    -2: (63/255, 99/255, 173/255),
    99: (1.0, 1.0, 1.0),
}


@dataclass
class Step11Config:
    output_dir: Path

    # Left-lower branch completion.
    left_bridge_target_count: int = 2
    left_bridge_min_gap_in_r3: float = 0.020
    bridge_normal_half_width: float = 0.030

    # Right-upper junction strict map.
    junction_r3_range: tuple[float, float] = (0.052, 0.068)
    junction_r4_range: tuple[float, float] = (0.116, 0.140)
    junction_grid_n: int = 7
    junction_max_edge_certificates: int = 3

    # Strict physics.
    strict_gap_grids: tuple[int, ...] = (71, 101)
    strict_chern_grids: tuple[int, ...] = (61, 81, 101)
    strict_chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )

    # Local mass-coordinate validation.
    local_mass_band_half_width: float = 0.045
    boundary_exclusion_width: float = 0.004
    spatial_folds: int = 4

    quick: bool = False
    random_seed: int = 20260724

    def normalized(self) -> "Step11Config":
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "bridge_points").mkdir(exist_ok=True)
        (self.output_dir / "junction_points").mkdir(exist_ok=True)
        (self.output_dir / "junction_edges").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "_strict_core").mkdir(exist_ok=True)
        (self.output_dir / "_search_core").mkdir(exist_ok=True)
        if self.junction_grid_n < 3:
            raise ValueError("junction_grid_n must be >= 3")
        if self.spatial_folds < 2:
            raise ValueError("spatial_folds must be >= 2")
        return self


# -----------------------------------------------------------------------------
# Generic archive I/O
# -----------------------------------------------------------------------------


def read_csv_token(source: str | Path, token: str) -> pd.DataFrame:
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            matches = [
                name for name in archive.namelist()
                if token in Path(name).name or token in name
            ]
            matches = [name for name in matches if name.lower().endswith(".csv")]
            if not matches:
                raise FileNotFoundError(f"No CSV containing {token!r} in {source}")
            selected = sorted(matches, key=lambda x: (len(Path(x).parts), len(x)))[0]
            with archive.open(selected) as stream:
                return pd.read_csv(stream, low_memory=False)
    if source.is_dir():
        matches = sorted(source.rglob(f"*{token}*"))
        if not matches:
            raise FileNotFoundError(token)
        return pd.read_csv(matches[0], low_memory=False)
    raise FileNotFoundError(source)


def read_json_token(source: str | Path, token: str) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            matches = [
                name for name in archive.namelist()
                if token in Path(name).name or token in name
            ]
            matches = [name for name in matches if name.lower().endswith(".json")]
            if not matches:
                raise FileNotFoundError(f"No JSON containing {token!r} in {source}")
            selected = sorted(matches, key=lambda x: (len(Path(x).parts), len(x)))[0]
            with archive.open(selected) as stream:
                return json.load(stream)
    if source.is_dir():
        matches = sorted(source.rglob(f"*{token}*"))
        if not matches:
            raise FileNotFoundError(token)
        return json.loads(matches[0].read_text(encoding="utf-8"))
    raise FileNotFoundError(source)


def phase_code(row: pd.Series | dict[str, Any]) -> int:
    if isinstance(row, dict):
        get = row.get
    else:
        get = row.get
    value = get("paper_phase_code", np.nan)
    if pd.notna(value):
        value = int(value)
        return value if value in PHASE_CODES else 99
    label = str(get("phase_label", ""))
    cup = get("chern_up_int", np.nan)
    if label in {"spin_chern_TI_candidate", "trivial_insulator"} and pd.notna(cup):
        cup = int(cup)
        return cup if cup in PHASE_CODES else 99
    return 99


# -----------------------------------------------------------------------------
# Fixed-background and branch models
# -----------------------------------------------------------------------------


def load_step10_tables(step10_source: str | Path) -> dict[str, pd.DataFrame]:
    return {
        "points": read_csv_token(
            step10_source,
            "step10M_03_certified_generic_boundary_points.csv",
        ),
        "fits": read_csv_token(
            step10_source,
            "step10M_13_generic_boundary_curve_fits.csv",
        ),
        "mass_atlas": read_csv_token(
            step10_source,
            "step10M_14_generic_branch_mass_atlas.csv",
        ),
    }


def validate_fixed_background(points: pd.DataFrame, tolerance: float = 1.0e-10) -> dict[str, float]:
    background = {}
    for name in FIXED5:
        values = points[name].to_numpy(float)
        spread = float(np.max(values) - np.min(values))
        if spread > tolerance:
            raise RuntimeError(
                f"Step10 points do not belong to one fixed slice: "
                f"{name} spread={spread:.3e}"
            )
        background[name] = float(np.mean(values))
    return background


def fit_row_for_branch(fits: pd.DataFrame, branch: str) -> pd.Series:
    selected = fits[
        fits["branch_hint"].astype(str).eq(branch)
        & fits["fit_available"].astype(int).eq(1)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one available fit for {branch}, found {len(selected)}")
    return selected.iloc[0]


def polynomial_value(fit: pd.Series, x: float | np.ndarray) -> float | np.ndarray:
    return (
        float(fit["coefficient_2"]) * np.asarray(x) ** 2
        + float(fit["coefficient_1"]) * np.asarray(x)
        + float(fit["coefficient_0"])
    )


def polynomial_derivative(fit: pd.Series, x: float) -> float:
    return 2.0 * float(fit["coefficient_2"]) * float(x) + float(fit["coefficient_1"])


def branch_mass_vector(mass_atlas: pd.DataFrame, branch: str) -> np.ndarray:
    group = (
        mass_atlas[mass_atlas["branch_hint"].astype(str).eq(branch)]
        .set_index("parameter")
        .reindex(REDUCED7)
    )
    vector = group["mean_normalized_mass_gradient"].to_numpy(float)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-15)
    return vector


def signed_mass_definition(
    fits: pd.DataFrame,
    mass_atlas: pd.DataFrame,
    branch: str,
) -> dict[str, Any]:
    fit = fit_row_for_branch(fits, branch)
    vector = branch_mass_vector(mass_atlas, branch)
    independent = str(fit["independent_parameter"])
    dependent = str(fit["dependent_parameter"])

    x_mid = 0.5 * (float(fit["x_min"]) + float(fit["x_max"]))
    derivative = polynomial_derivative(fit, x_mid)

    if independent == "r3" and dependent == "r4":
        # F = r4 - f(r3)
        raw_gradient = np.array([-derivative, 1.0], dtype=float)
        raw_expression = (
            f"r4 - ({float(fit['coefficient_2']):+.10g}*r3^2 "
            f"{float(fit['coefficient_1']):+.10g}*r3 "
            f"{float(fit['coefficient_0']):+.10g})"
        )
    elif independent == "r4" and dependent == "r3":
        # F = r3 - f(r4)
        raw_gradient = np.array([1.0, -derivative], dtype=float)
        raw_expression = (
            f"r3 - ({float(fit['coefficient_2']):+.10g}*r4^2 "
            f"{float(fit['coefficient_1']):+.10g}*r4 "
            f"{float(fit['coefficient_0']):+.10g})"
        )
    else:
        raise RuntimeError(
            f"Unsupported branch fit coordinates: {independent}->{dependent}"
        )

    plane_mass = vector[-2:]
    orientation = 1.0 if float(np.dot(raw_gradient, plane_mass)) >= 0 else -1.0
    return {
        "branch_hint": branch,
        "independent_parameter": independent,
        "dependent_parameter": dependent,
        "selected_degree": int(fit["selected_degree"]),
        "coefficient_2": float(fit["coefficient_2"]),
        "coefficient_1": float(fit["coefficient_1"]),
        "coefficient_0": float(fit["coefficient_0"]),
        "x_min": float(fit["x_min"]),
        "x_max": float(fit["x_max"]),
        "orientation": float(orientation),
        "formula": f"{orientation:+.0f} * [{raw_expression}]",
        "mass_gradient_plane_r3": float(plane_mass[0]),
        "mass_gradient_plane_r4": float(plane_mass[1]),
    }


def evaluate_signed_mass(definition: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    a2 = float(definition["coefficient_2"])
    a1 = float(definition["coefficient_1"])
    a0 = float(definition["coefficient_0"])
    orientation = float(definition["orientation"])
    if definition["independent_parameter"] == "r3":
        raw = frame["r4"].to_numpy(float) - (
            a2 * frame["r3"].to_numpy(float) ** 2
            + a1 * frame["r3"].to_numpy(float)
            + a0
        )
    else:
        raw = frame["r3"].to_numpy(float) - (
            a2 * frame["r4"].to_numpy(float) ** 2
            + a1 * frame["r4"].to_numpy(float)
            + a0
        )
    return orientation * raw


# -----------------------------------------------------------------------------
# Left-lower branch completion
# -----------------------------------------------------------------------------


def prepare_left_bridge_targets(
    points: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    target_count: int,
    minimum_gap: float,
) -> pd.DataFrame:
    branch = "generic_left_lower"
    group = (
        points[points["branch_hint"].astype(str).eq(branch)]
        .sort_values("r3")
        .reset_index(drop=True)
    )
    fit = fit_row_for_branch(fits, branch)
    gaps = []
    for i in range(len(group) - 1):
        left = float(group.iloc[i]["r3"])
        right = float(group.iloc[i + 1]["r3"])
        gap = right - left
        if gap >= float(minimum_gap):
            gaps.append((gap, left, right))
    gaps.sort(reverse=True)

    target_values: list[float] = []
    for gap, left, right in gaps:
        remaining = max(1, int(round(gap / max(minimum_gap, 1.0e-6))) - 1)
        count = min(remaining, target_count - len(target_values))
        if count <= 0:
            break
        target_values.extend(
            np.linspace(left, right, count + 2, dtype=float)[1:-1].tolist()
        )
        if len(target_values) >= target_count:
            break

    rows = []
    for index, r3 in enumerate(sorted(target_values[:target_count])):
        r4 = float(polynomial_value(fit, r3))
        nearest_index = int(
            np.argmin(
                np.hypot(
                    group["r3"].to_numpy(float) - r3,
                    group["r4"].to_numpy(float) - r4,
                )
            )
        )
        nearest = group.iloc[nearest_index]
        rows.append({
            "target_id": f"left_bridge_target_{index+1:02d}",
            "branch_hint": branch,
            "target_r3": r3,
            "target_r4": r4,
            "nearest_point_id": str(nearest["point_id"]),
            "nearest_kx": float(nearest["critical_kx_representative"]),
            "nearest_ky": float(nearest["critical_ky_representative"]),
            "nearest_normal_r3": float(nearest["normal_r3"]),
            "nearest_normal_r4": float(nearest["normal_r4"]),
        })
    return pd.DataFrame(rows)


def run_left_bridge_completion(
    *,
    tts_archive: str | Path,
    step10_source: str | Path,
    output_dir: str | Path,
    config: Step11Config,
) -> pd.DataFrame:
    config = config.normalized()
    tables = load_step10_tables(step10_source)
    background = validate_fixed_background(tables["points"])
    targets = prepare_left_bridge_targets(
        tables["points"],
        tables["fits"],
        target_count=config.left_bridge_target_count,
        minimum_gap=config.left_bridge_min_gap_in_r3,
    )
    targets.to_csv(
        config.output_dir / "step11M_01_left_bridge_target_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if targets.empty:
        return pd.DataFrame()

    core, step4, step5, source_dir = base.import_reference_modules(tts_archive)
    linear_v2.patch_step5_linear_path(step5, core)
    step3 = linear_v2.import_step3(source_dir)

    continuation = step10.ContinuationConfig(
        output_dir=config.output_dir / "_bridge_internal",
        normal_half_width=config.bridge_normal_half_width,
        strict_gap_grids=tuple(config.strict_gap_grids),
        strict_chern_grids=tuple(config.strict_chern_grids),
        strict_chern_shifts=tuple(config.strict_chern_shifts),
        quick=config.quick,
        random_seed=config.random_seed,
    ).normalized()
    strict_config = step10.strict_step3_config(
        step3,
        config.output_dir / "_strict_core",
        continuation,
    )
    kp_config = step10.step5_config(
        step5,
        config.output_dir / "_search_core",
        continuation,
    )
    step5._step10m_config = kp_config

    fit = fit_row_for_branch(tables["fits"], "generic_left_lower")
    result_rows = []
    for target_index, target in targets.iterrows():
        point_id = str(target["target_id"])
        point_file = config.output_dir / "bridge_points" / f"{point_id}.json"
        if point_file.is_file():
            payload = json.loads(point_file.read_text(encoding="utf-8"))
            summary = payload.get("point_summary", {})
            if summary:
                result_rows.append(summary)
            continue

        r3 = float(target["target_r3"])
        r4 = float(target["target_r4"])
        derivative = polynomial_derivative(fit, r3)
        normal = np.array([-derivative, 1.0], dtype=float)
        nearest_normal = np.array(
            [float(target["nearest_normal_r3"]), float(target["nearest_normal_r4"])],
            dtype=float,
        )
        if float(np.dot(normal, nearest_normal)) < 0:
            normal *= -1.0
        normal /= max(float(np.linalg.norm(normal)), 1.0e-15)

        center = {
            **background,
            "r3": r3,
            "r4": r4,
        }
        previous_k = (
            float(target["nearest_kx"]),
            float(target["nearest_ky"]),
        )

        closure = None
        pair = None
        candidate_rows = []
        for width_index, width in enumerate(
            (
                config.bridge_normal_half_width,
                1.5 * config.bridge_normal_half_width,
                2.0 * config.bridge_normal_half_width,
            )
        ):
            pair = step10.build_pair(center, normal, float(width), point_id)
            closure, candidates = step10.search_generic_closure(
                step5,
                pair,
                previous_k,
                continuation,
                5000 + 100 * target_index + width_index,
            )
            candidates["point_id"] = point_id
            candidates["normal_half_width"] = float(width)
            candidate_rows.extend(candidates.to_dict("records"))
            if closure is not None:
                break

        if closure is None or pair is None:
            payload = {
                "point_summary": {
                    "point_id": point_id,
                    "branch_hint": "generic_left_lower",
                    "point_certificate_pass": 0,
                    "failure_reason": "no_generic_four_valley_closure",
                    "predictor_r3": r3,
                    "predictor_r4": r4,
                },
                "search_candidates": candidate_rows,
            }
        else:
            payload = step10.analyse_boundary_point(
                core=core,
                step5=step5,
                step3=step3,
                strict_config=strict_config,
                pair=pair,
                closure=closure,
                point_id=point_id,
                branch_hint="generic_left_lower",
                seed_id=str(target["nearest_point_id"]),
                trace_direction=0,
                step_index=int(target_index + 1),
                config=continuation,
            )
            payload["search_candidates"] = candidate_rows

        point_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary = payload.get("point_summary", {})
        if summary:
            result_rows.append(summary)

    result = pd.DataFrame(result_rows)
    result.to_csv(
        config.output_dir / "step11M_02_left_bridge_completion_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


# -----------------------------------------------------------------------------
# Strict junction grid
# -----------------------------------------------------------------------------


def junction_checkpoint_path(config: Step11Config, ix: int, iy: int) -> Path:
    return config.output_dir / "junction_points" / f"junction_{ix:02d}_{iy:02d}.json"


def run_junction_strict_grid(
    *,
    tts_archive: str | Path,
    step10_source: str | Path,
    output_dir: str | Path,
    config: Step11Config,
) -> pd.DataFrame:
    config = config.normalized()
    tables = load_step10_tables(step10_source)
    background = validate_fixed_background(tables["points"])

    core, step4, step5, source_dir = base.import_reference_modules(tts_archive)
    linear_v2.patch_step5_linear_path(step5, core)
    step3 = linear_v2.import_step3(source_dir)

    continuation = step10.ContinuationConfig(
        output_dir=config.output_dir / "_junction_internal",
        strict_gap_grids=tuple(config.strict_gap_grids),
        strict_chern_grids=tuple(config.strict_chern_grids),
        strict_chern_shifts=tuple(config.strict_chern_shifts),
        quick=config.quick,
        random_seed=config.random_seed,
    ).normalized()
    strict_config = step10.strict_step3_config(
        step3,
        config.output_dir / "_strict_core",
        continuation,
    )

    r3_values = np.linspace(
        config.junction_r3_range[0],
        config.junction_r3_range[1],
        int(config.junction_grid_n),
    )
    r4_values = np.linspace(
        config.junction_r4_range[0],
        config.junction_r4_range[1],
        int(config.junction_grid_n),
    )

    rows = []
    for iy, r4 in enumerate(r4_values):
        for ix, r3 in enumerate(r3_values):
            path = junction_checkpoint_path(config, ix, iy)
            if path.is_file():
                summary = json.loads(path.read_text(encoding="utf-8"))
            else:
                reduced = {**background, "r3": float(r3), "r4": float(r4)}
                result = step10.evaluate_strict(
                    step3,
                    strict_config,
                    reduced,
                    f"step11M_junction_{ix:02d}_{iy:02d}",
                )
                summary = result["summary"]
                summary.update({"junction_ix": ix, "junction_iy": iy})
                path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            summary = dict(summary)
            summary["junction_ix"] = ix
            summary["junction_iy"] = iy
            summary["paper_phase_code"] = phase_code(summary)
            rows.append(summary)

    frame = pd.DataFrame(rows)
    frame.to_csv(
        config.output_dir / "step11M_03_right_junction_strict_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return frame


def junction_boundary_edges(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (int(row.junction_ix), int(row.junction_iy)): row
        for row in frame.itertuples()
    }
    rows = []
    for (ix, iy), row in lookup.items():
        for neighbour in ((ix + 1, iy), (ix, iy + 1)):
            if neighbour not in lookup:
                continue
            other = lookup[neighbour]
            left_valid = (
                int(getattr(row, "strict_gap_verified", 0) or 0) == 1
                and int(getattr(row, "strict_chern_verified", 0) or 0) == 1
                and int(row.paper_phase_code) in PHASE_CODES
            )
            right_valid = (
                int(getattr(other, "strict_gap_verified", 0) or 0) == 1
                and int(getattr(other, "strict_chern_verified", 0) or 0) == 1
                and int(other.paper_phase_code) in PHASE_CODES
            )
            if not (left_valid and right_valid):
                continue
            c0 = int(row.paper_phase_code)
            c1 = int(other.paper_phase_code)
            if c0 == c1:
                continue
            delta = c1 - c0
            if abs(delta) == 4:
                candidate = "generic_four_valley_candidate"
            elif abs(delta) == 2:
                candidate = "two_valley_or_multiclosure_candidate"
            elif abs(delta) == 1:
                candidate = "single_valley_candidate"
            else:
                candidate = "multiclosure_candidate"
            rows.append({
                "edge_id": f"edge_{ix:02d}_{iy:02d}__{neighbour[0]:02d}_{neighbour[1]:02d}",
                "ix0": ix,
                "iy0": iy,
                "ix1": neighbour[0],
                "iy1": neighbour[1],
                "r3_0": float(row.r3),
                "r4_0": float(row.r4),
                "chern_0": c0,
                "r3_1": float(other.r3),
                "r4_1": float(other.r4),
                "chern_1": c1,
                "delta_chern_up": delta,
                "mid_r3": 0.5 * (float(row.r3) + float(other.r3)),
                "mid_r4": 0.5 * (float(row.r4) + float(other.r4)),
                "candidate_mechanism": candidate,
            })
    return pd.DataFrame(rows)


def select_representative_junction_edges(
    edges: pd.DataFrame,
    max_edges: int,
) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    selected_rows = []
    # First retain one edge for every distinct signed Chern transition.
    for transition, group in edges.groupby(["chern_0", "chern_1"]):
        group = group.copy()
        center = group[["mid_r3", "mid_r4"]].mean().to_numpy(float)
        distance = np.linalg.norm(
            group[["mid_r3", "mid_r4"]].to_numpy(float) - center,
            axis=1,
        )
        selected_rows.append(group.iloc[int(np.argmin(distance))])
    selected = pd.DataFrame(selected_rows)
    if len(selected) > max_edges:
        selected = selected.head(max_edges)
    elif len(selected) < max_edges:
        remaining = edges[~edges["edge_id"].isin(selected["edge_id"])].copy()
        while len(selected) < max_edges and len(remaining):
            selected_points = selected[["mid_r3", "mid_r4"]].to_numpy(float)
            candidates = remaining[["mid_r3", "mid_r4"]].to_numpy(float)
            min_dist = np.min(
                np.linalg.norm(
                    candidates[:, None, :] - selected_points[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            pick = int(np.argmax(min_dist))
            selected = pd.concat(
                [selected, remaining.iloc[[pick]]],
                ignore_index=True,
            )
            remaining = remaining.drop(remaining.index[pick])
    return selected.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Selected junction-edge multi-closure certificates
# -----------------------------------------------------------------------------


def build_pair_from_edge(
    edge: pd.Series,
    background: dict[str, float],
) -> pd.Series:
    start = pd.Series(
        {
            **background,
            "r3": float(edge["r3_0"]),
            "r4": float(edge["r4_0"]),
        }
    )
    end = pd.Series(
        {
            **background,
            "r3": float(edge["r3_1"]),
            "r4": float(edge["r4_1"]),
        }
    )
    return base.build_pair_series(start, end, path_id=str(edge["edge_id"]))


def analyse_junction_edge_multiclosure(
    *,
    edge: pd.Series,
    background: dict[str, float],
    core,
    step5,
    step3,
    strict_config,
    search_config,
    multi_config: step09d.MultiClosureConfig,
) -> dict[str, Any]:
    edge_id = str(edge["edge_id"])
    pair = build_pair_from_edge(edge, background)

    gap_scan = step09d.dense_gap_scan(step5, pair, multi_config, edge_id)
    coarse, coarse_attempts = step09d.coarse_chern_scan(
        core,
        step5,
        pair,
        multi_config,
        edge_id,
    )
    brackets = step09d.detect_candidate_brackets(
        gap_scan,
        coarse,
        multi_config,
    )

    all_closures = []
    search_rows = []
    for bracket_index, bracket_row in brackets.iterrows():
        bracket = pd.Series({
            "transition_id": f"{edge_id}_bracket{bracket_index:02d}",
            "path_id": edge_id,
            "lambda_left": float(bracket_row["lambda_left"]),
            "lambda_right": float(bracket_row["lambda_right"]),
            "chern_left": np.nan,
            "chern_right": np.nan,
        })
        candidates, closures = step5.search_transition_closures(
            bracket,
            pair,
            search_config,
            9000 + int(bracket_index),
        )
        search_rows.extend(
            [
                {
                    "edge_id": edge_id,
                    "bracket_index": int(bracket_index),
                    **row,
                }
                for row in candidates
            ]
        )
        for closure in closures:
            item = dict(closure)
            item["source_bracket_index"] = int(bracket_index)
            all_closures.append(item)

    unique = step09d.global_deduplicate_closures(
        step5,
        all_closures,
        multi_config.closure_lambda_dedup_tol,
    )
    groups = step09d.closure_groups(unique)
    centres = [
        float(np.mean([float(row["critical_lambda"]) for row in group]))
        for group in groups
    ]

    segment_results = []
    segment_rows = []
    boundaries = [0.0, *centres, 1.0]
    for segment_index, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        probe = step09d.find_reliable_segment_probe(
            step3,
            strict_config,
            pair,
            edge_id,
            float(left),
            float(right),
            multi_config.segment_probe_fractions,
            segment_index,
        )
        segment_results.append(probe)
        summary = probe["summary"]
        segment_rows.append({
            "edge_id": edge_id,
            "segment_index": segment_index,
            "lambda_left_bound": left,
            "lambda_right_bound": right,
            "lambda": summary.get("lambda"),
            "strict_chern_up": step09d.reliable_chern(summary),
            "strict_phase_label": summary.get("phase_label"),
            "min_direct_gap": summary.get("min_direct_gap"),
        })

    closure_rows = []
    kp_rows = []
    gradient_rows = []
    berry_rows = []
    group_certificates = []

    for group_index, group in enumerate(groups):
        group_id = f"{edge_id}_group{group_index:02d}"
        group_charge = 0
        all_consensus = True
        all_rank3 = True
        valley_count = 0
        for closure_index, closure in enumerate(group):
            closure_id = f"{group_id}_closure{closure_index:02d}"
            reduced = linear_v2.linear_reduced_on_path(
                pair,
                float(closure["critical_lambda"]),
            )
            closure_rows.append({
                "edge_id": edge_id,
                "closure_group_id": group_id,
                "closure_id": closure_id,
                **closure,
                **reduced,
            })
            analysis = step09d.analyse_closure(
                step5,
                pair,
                search_config,
                edge_id,
                group_id,
                closure_id,
                closure,
            )
            kp_rows.extend(analysis["kp_rows"])
            gradient_rows.extend(analysis["gradient_rows"])
            berry_rows.extend(analysis["berry_summary_rows"])
            group_charge += int(analysis["charge_sum"])
            all_consensus = all_consensus and bool(analysis["all_consensus"])
            all_rank3 = all_rank3 and bool(analysis["all_rank3"])
            valley_count += int(analysis["n_active_valleys"])

        left_chern = (
            step09d.reliable_chern(segment_results[group_index]["summary"])
            if group_index < len(segment_results) else None
        )
        right_chern = (
            step09d.reliable_chern(segment_results[group_index + 1]["summary"])
            if group_index + 1 < len(segment_results) else None
        )
        delta = (
            None
            if left_chern is None or right_chern is None
            else int(right_chern - left_chern)
        )
        signed_match = (
            delta is not None
            and all_consensus
            and int(group_charge) == int(delta)
        )
        group_certificates.append({
            "edge_id": edge_id,
            "closure_group_id": group_id,
            "mean_critical_lambda": centres[group_index],
            "n_closure_orbits": len(group),
            "n_active_spin_up_valleys": valley_count,
            "left_chern_up": left_chern,
            "right_chern_up": right_chern,
            "observed_delta_chern_up": delta,
            "berry_charge_sum_up": group_charge,
            "all_berry_charges_consensus": int(all_consensus),
            "all_kp_jacobians_rank3": int(all_rank3),
            "signed_charge_matches": int(signed_match),
            "closure_group_certificate_pass": int(signed_match and all_rank3),
        })

    endpoint_delta = int(edge["chern_1"] - edge["chern_0"])
    total_charge = int(sum(row["berry_charge_sum_up"] for row in group_certificates))
    all_groups_pass = bool(
        len(group_certificates)
        and all(
            int(row["closure_group_certificate_pass"]) == 1
            for row in group_certificates
        )
    )
    edge_certificate = {
        "edge_id": edge_id,
        "grid_chern_0": int(edge["chern_0"]),
        "grid_chern_1": int(edge["chern_1"]),
        "grid_endpoint_delta_chern_up": endpoint_delta,
        "n_closure_groups": len(groups),
        "n_closure_orbits": len(unique),
        "n_active_spin_up_valleys_total": int(
            sum(row["n_active_spin_up_valleys"] for row in group_certificates)
        ),
        "berry_charge_sum_over_all_groups": total_charge,
        "signed_total_charge_matches_grid_delta": int(
            total_charge == endpoint_delta
        ),
        "all_closure_groups_certified": int(all_groups_pass),
        "edge_multiclosure_certificate_pass": int(
            all_groups_pass and total_charge == endpoint_delta
        ),
    }
    return {
        "edge": edge.to_dict(),
        "gap_scan": gap_scan.to_dict("records"),
        "coarse_chern": coarse.to_dict("records"),
        "coarse_chern_attempts": coarse_attempts.to_dict("records"),
        "candidate_brackets": brackets.to_dict("records"),
        "search_candidates": search_rows,
        "closures": closure_rows,
        "segments": segment_rows,
        "kp_summaries": kp_rows,
        "mass_gradients": gradient_rows,
        "berry_summaries": berry_rows,
        "group_certificates": group_certificates,
        "edge_certificate": edge_certificate,
    }


def run_junction_edge_certificates(
    *,
    tts_archive: str | Path,
    step10_source: str | Path,
    junction_grid: pd.DataFrame,
    output_dir: str | Path,
    config: Step11Config,
) -> pd.DataFrame:
    config = config.normalized()
    tables = load_step10_tables(step10_source)
    background = validate_fixed_background(tables["points"])
    edges = junction_boundary_edges(junction_grid)
    edges.to_csv(
        config.output_dir / "step11M_04_right_junction_boundary_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected = select_representative_junction_edges(
        edges,
        config.junction_max_edge_certificates,
    )
    selected.to_csv(
        config.output_dir / "step11M_05_selected_junction_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if selected.empty:
        return pd.DataFrame()

    core, step4, step5, source_dir = base.import_reference_modules(tts_archive)
    linear_v2.patch_step5_linear_path(step5, core)
    step3 = linear_v2.import_step3(source_dir)

    multi_config = step09d.MultiClosureConfig(
        output_dir=config.output_dir / "_junction_multiclosure",
        lambda_points=(151 if config.quick else 301),
        diagonal_k_points=(81 if config.quick else 141),
        candidate_gap_ceiling=2.0e-2,
        candidate_log_prominence=0.25,
        candidate_min_index_distance=4,
        bracket_half_width_indices=6,
        coarse_chern_lambda_points=(11 if config.quick else 17),
        coarse_chern_grids=((21,) if config.quick else (21, 31)),
        coarse_chern_shifts=((0.0, 0.0),),
        strict_chern_grids=tuple(config.strict_chern_grids),
        strict_chern_shifts=tuple(config.strict_chern_shifts),
        strict_gap_grids=tuple(config.strict_gap_grids),
        full_search_quick=config.quick,
        random_seed=config.random_seed,
    ).normalized()

    strict_config = step09d.configure_strict_step3(
        step3,
        config.output_dir / "_strict_core",
        multi_config,
    )
    search_config = step09d.configure_step5_search(
        step5,
        config.output_dir / "_search_core",
        multi_config,
    )

    certificate_rows = []
    for _, edge in selected.iterrows():
        edge_id = str(edge["edge_id"])
        path = config.output_dir / "junction_edges" / f"{edge_id}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = analyse_junction_edge_multiclosure(
                edge=edge,
                background=background,
                core=core,
                step5=step5,
                step3=step3,
                strict_config=strict_config,
                search_config=search_config,
                multi_config=multi_config,
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        certificate_rows.append(payload["edge_certificate"])

    result = pd.DataFrame(certificate_rows)
    result.to_csv(
        config.output_dir / "step11M_06_junction_edge_certificates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


# -----------------------------------------------------------------------------
# Lieb-style local signed-mass validation
# -----------------------------------------------------------------------------


def load_formal_lower_grid(formal_scan_source: str | Path) -> pd.DataFrame:
    frame = base.load_lower_grid(formal_scan_source)
    frame = frame.copy()
    frame["paper_phase_code"] = frame.apply(phase_code, axis=1)
    if "r3" not in frame:
        frame["r3"] = frame["scan_x"]
    if "r4" not in frame:
        frame["r4"] = frame["scan_y"]
    return frame


def mapping_from_train(sign_values: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    mapping = {}
    for sign in (-1, 1):
        selected = labels[sign_values == sign]
        if len(selected) == 0:
            continue
        values, counts = np.unique(selected, return_counts=True)
        mapping[sign] = int(values[int(np.argmax(counts))])
    return mapping


def validate_one_branch_mass(
    grid: pd.DataFrame,
    definition: dict[str, Any],
    *,
    band_half_width: float,
    boundary_exclusion: float,
    spatial_folds: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = grid.copy()
    frame["signed_mass"] = evaluate_signed_mass(definition, frame)
    independent = definition["independent_parameter"]
    x = frame[independent].to_numpy(float)

    x_min = float(definition["x_min"])
    x_max = float(definition["x_max"])
    x_margin = max(0.02, 0.15 * (x_max - x_min))
    selected = frame[
        frame["paper_phase_code"].isin([-2, 2])
        & (frame["signed_mass"].abs() <= float(band_half_width))
        & (frame["signed_mass"].abs() >= float(boundary_exclusion))
        & (frame[independent] >= x_min - x_margin)
        & (frame[independent] <= x_max + x_margin)
    ].copy()
    if len(selected) < 8:
        return selected, {
            "branch_hint": definition["branch_hint"],
            "n_points": int(len(selected)),
            "n_folds_evaluated": 0,
            "balanced_accuracy": np.nan,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "status": "insufficient_local_grid_points",
        }

    selected["mass_sign"] = np.where(selected["signed_mass"] >= 0, 1, -1)
    edges = np.linspace(
        float(selected[independent].min()),
        float(selected[independent].max()),
        int(spatial_folds) + 1,
    )
    selected["spatial_fold"] = np.clip(
        np.digitize(selected[independent], edges[1:-1], right=False),
        0,
        spatial_folds - 1,
    )

    prediction_rows = []
    for fold in sorted(selected["spatial_fold"].unique()):
        train = selected[selected["spatial_fold"] != fold]
        test = selected[selected["spatial_fold"] == fold]
        if len(test) == 0:
            continue
        mapping = mapping_from_train(
            train["mass_sign"].to_numpy(int),
            train["paper_phase_code"].to_numpy(int),
        )
        if set(mapping) != {-1, 1}:
            continue
        predicted = np.array(
            [mapping[int(sign)] for sign in test["mass_sign"]],
            dtype=int,
        )
        part = test[
            ["r3", "r4", "signed_mass", "mass_sign", "paper_phase_code", "spatial_fold"]
        ].copy()
        part["prediction"] = predicted
        part["branch_hint"] = definition["branch_hint"]
        prediction_rows.append(part)

    if not prediction_rows:
        return selected, {
            "branch_hint": definition["branch_hint"],
            "n_points": int(len(selected)),
            "n_folds_evaluated": 0,
            "balanced_accuracy": np.nan,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "status": "no_valid_spatial_folds",
        }

    predictions = pd.concat(prediction_rows, ignore_index=True)
    y = predictions["paper_phase_code"].to_numpy(int)
    p = predictions["prediction"].to_numpy(int)
    metrics = {
        "branch_hint": definition["branch_hint"],
        "n_points": int(len(selected)),
        "n_predictions": int(len(predictions)),
        "n_folds_evaluated": int(predictions["spatial_fold"].nunique()),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "status": "evaluated",
    }
    return predictions, metrics


# -----------------------------------------------------------------------------
# Aggregation and figures
# -----------------------------------------------------------------------------


def collect_bridge_payloads(config: Step11Config) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((config.output_dir / "bridge_points").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def refit_completed_branches(
    step10_points: pd.DataFrame,
    bridge_payloads: list[dict[str, Any]],
    config: Step11Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [step10_points]
    new_rows = []
    for payload in bridge_payloads:
        summary = payload.get("point_summary", {})
        if int(summary.get("point_certificate_pass", 0) or 0) == 1:
            new_rows.append(summary)
    if new_rows:
        rows.append(pd.DataFrame(new_rows))
    points = pd.concat(rows, ignore_index=True)
    points = step10.deduplicate_points(points, distance=2.5e-3)

    continuation = step10.ContinuationConfig(
        output_dir=config.output_dir / "_fit_internal",
        min_branch_points_for_fit=4,
    )
    fit_rows = [
        step10.fit_branch(group, branch, 4)
        for branch, group in points.groupby("branch_hint")
    ]
    fits = pd.DataFrame(fit_rows)
    mass_atlas = step10.branch_mass_atlas(
        points,
        bootstrap_repeats=(200 if config.quick else 1000),
        seed=config.random_seed,
    )
    return points, fits, mass_atlas


def plot_fixed_slice_phase_map(
    formal_grid: pd.DataFrame,
    certified_points: pd.DataFrame,
    fits: pd.DataFrame,
    junction_grid: pd.DataFrame,
    output_path: Path,
) -> None:
    cmap = ListedColormap([PHASE_COLORS[value] for value in PHASE_ORDER])
    norm = BoundaryNorm(
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 99.5],
        cmap.N,
    )

    pivot = formal_grid.pivot(
        index="scan_iy",
        columns="scan_ix",
        values="paper_phase_code",
    ).sort_index()
    x = formal_grid.pivot(
        index="scan_iy",
        columns="scan_ix",
        values="scan_x",
    ).sort_index().iloc[0].to_numpy(float)
    y = formal_grid.pivot(
        index="scan_iy",
        columns="scan_ix",
        values="scan_y",
    ).sort_index().iloc[:, 0].to_numpy(float)

    def edges(values: np.ndarray) -> np.ndarray:
        mids = 0.5 * (values[:-1] + values[1:])
        return np.r_[
            values[0] - (mids[0] - values[0]),
            mids,
            values[-1] + (values[-1] - mids[-1]),
        ]

    fig, ax = plt.subplots(figsize=(16.0 / 2.54, 10.0 / 2.54))
    ax.pcolormesh(
        edges(x),
        edges(y),
        pivot.to_numpy(float),
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    markers = {
        "generic_left_lower": "s",
        "generic_right_upper": "^",
    }
    for branch, group in certified_points.groupby("branch_hint"):
        ax.scatter(
            group["r3"],
            group["r4"],
            s=25,
            marker=markers.get(branch, "D"),
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            label=branch.replace("generic_", "") + " certified",
        )
        fit = fits[
            fits["branch_hint"].astype(str).eq(branch)
            & fits["fit_available"].astype(int).eq(1)
        ]
        if len(fit):
            row = fit.iloc[0]
            xx = np.linspace(float(row["x_min"]), float(row["x_max"]), 300)
            yy = polynomial_value(row, xx)
            if row["independent_parameter"] == "r3":
                ax.plot(xx, yy, "k-", linewidth=1.0)
            else:
                ax.plot(yy, xx, "k-", linewidth=1.0)

    if len(junction_grid):
        reliable = junction_grid[
            junction_grid["paper_phase_code"].isin([-2, 0, 2])
        ]
        ax.scatter(
            reliable["r3"],
            reliable["r4"],
            s=10,
            marker=".",
            color="black",
            label="strict junction grid",
        )

    ax.set_xlabel(r"$r_3$")
    ax.set_ylabel(r"$r_4$")
    ax.set_xlim(-0.82, 0.11)
    ax.set_ylim(-0.21, 0.16)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=1200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_mass_coordinate_validation(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    if predictions.empty:
        return
    fig, ax = plt.subplots(figsize=(12.0 / 2.54, 7.0 / 2.54))
    markers = {
        "generic_left_lower": "s",
        "generic_right_upper": "^",
    }
    for branch, group in predictions.groupby("branch_hint"):
        ax.scatter(
            group["signed_mass"],
            group["paper_phase_code"],
            s=20,
            marker=markers.get(branch, "o"),
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            label=branch.replace("generic_", ""),
        )
    ax.axvline(0.0, linestyle="--", linewidth=0.8, color="black")
    ax.set_xlabel("local signed mass coordinate")
    ax.set_ylabel(r"$C_\uparrow$")
    ax.set_yticks([-2, 2])
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=1200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def aggregate_step11(
    *,
    formal_scan_source: str | Path,
    step10_source: str | Path,
    output_dir: str | Path,
    config: Step11Config,
) -> dict[str, Any]:
    config = config.normalized()
    tables = load_step10_tables(step10_source)
    background = validate_fixed_background(tables["points"])
    bridge_payloads = collect_bridge_payloads(config)
    points, fits, mass_atlas = refit_completed_branches(
        tables["points"],
        bridge_payloads,
        config,
    )

    points.to_csv(
        config.output_dir / "step11M_07_completed_certified_boundary_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fits.to_csv(
        config.output_dir / "step11M_08_fixed_slice_boundary_curve_fits.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mass_atlas.to_csv(
        config.output_dir / "step11M_09_fixed_slice_local_mass_atlas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    definitions = []
    for branch in sorted(points["branch_hint"].dropna().unique()):
        definitions.append(
            signed_mass_definition(fits, mass_atlas, str(branch))
        )
    definitions_df = pd.DataFrame(definitions)
    definitions_df.to_csv(
        config.output_dir / "step11M_10_local_signed_mass_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    formal_grid = load_formal_lower_grid(formal_scan_source)
    prediction_frames = []
    metric_rows = []
    for definition in definitions:
        predictions, metrics = validate_one_branch_mass(
            formal_grid,
            definition,
            band_half_width=config.local_mass_band_half_width,
            boundary_exclusion=config.boundary_exclusion_width,
            spatial_folds=config.spatial_folds,
        )
        if not predictions.empty and "prediction" in predictions:
            prediction_frames.append(predictions)
        metric_rows.append(metrics)

    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(metric_rows)
    predictions_df.to_csv(
        config.output_dir / "step11M_11_spatial_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df.to_csv(
        config.output_dir / "step11M_12_spatial_holdout_mass_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    junction_path = (
        config.output_dir / "step11M_03_right_junction_strict_grid.csv"
    )
    junction_grid = (
        pd.read_csv(junction_path, low_memory=False)
        if junction_path.is_file() else pd.DataFrame()
    )
    edge_cert_path = (
        config.output_dir / "step11M_06_junction_edge_certificates.csv"
    )
    edge_certificates = (
        pd.read_csv(edge_cert_path, low_memory=False)
        if edge_cert_path.is_file() else pd.DataFrame()
    )

    plot_fixed_slice_phase_map(
        formal_grid,
        points,
        fits,
        junction_grid,
        config.output_dir / "figures" / "step11M_fixed_slice_mechanism_map",
    )
    plot_mass_coordinate_validation(
        predictions_df,
        config.output_dir / "figures" / "step11M_local_mass_coordinate_validation",
    )

    background_json = {
        name: float(value) for name, value in background.items()
    }
    (config.output_dir / "step11M_00_fixed_background.json").write_text(
        json.dumps(background_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bridge_new = points[
        points["point_id"].astype(str).str.startswith("left_bridge_target_")
    ]
    certificate = {
        "code_version": CODE_VERSION,
        "research_design": "Lieb_aligned_fixed_reduced_plane",
        "fixed_background_parameters": background_json,
        "free_plane_parameters": ["r3", "r4"],
        "n_step10_input_points": int(len(tables["points"])),
        "n_new_certified_bridge_points": int(len(bridge_new)),
        "n_completed_certified_points": int(len(points)),
        "n_local_signed_mass_definitions": int(len(definitions_df)),
        "all_background_parameters_remain_fixed": True,
        "mass_coordinate_validation": metrics_df.to_dict("records"),
        "junction_grid_completed": bool(len(junction_grid) > 0),
        "n_junction_reliable_points": int(
            junction_grid["paper_phase_code"].isin([-2, 0, 2]).sum()
        ) if len(junction_grid) else 0,
        "n_junction_edge_certificates": int(len(edge_certificates)),
        "n_junction_edges_certified": int(
            edge_certificates.get(
                "edge_multiclosure_certificate_pass",
                pd.Series(dtype=int),
            ).fillna(0).astype(int).sum()
        ) if len(edge_certificates) else 0,
        "interpretation": {
            "methodological_scope": (
                "This is a representative fixed-background r3-r4 mechanism plane, "
                "analogous to the fixed-parameter planes used in the Lieb analysis."
            ),
            "valid_claim": (
                "The certified local signed masses organize the -2/+2 boundaries "
                "within this reduced plane."
            ),
            "not_claimed": (
                "No universal seven-dimensional phase boundary is claimed or required."
            ),
        },
    }
    (config.output_dir / "step11M_13_fixed_slice_certificate.json").write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    formula_lines = [
        "# TTS Step11M local signed masses",
        "",
        "Fixed background:",
        *[
            f"- `{name} = {background[name]:+.10g}`"
            for name in FIXED5
        ],
        "",
    ]
    for definition in definitions:
        formula_lines.extend([
            f"## {definition['branch_hint']}",
            "",
            f"`M = {definition['formula']}`",
            "",
            (
                f"Validated only for "
                f"{definition['independent_parameter']} in "
                f"[{definition['x_min']:+.6f}, {definition['x_max']:+.6f}]."
            ),
            "",
        ])
    (config.output_dir / "step11M_14_local_signed_mass_formulas.md").write_text(
        "\n".join(formula_lines),
        encoding="utf-8",
    )
    return certificate


# -----------------------------------------------------------------------------
# Complete public workflow
# -----------------------------------------------------------------------------


def run_step11m(
    *,
    tts_archive: str | Path,
    formal_scan_source: str | Path,
    step10_source: str | Path,
    output_dir: str | Path,
    config: Step11Config | None = None,
    run_bridge: bool = True,
    run_junction_grid: bool = True,
    run_junction_edges: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step11Config(output_dir=Path(output_dir))
    config.output_dir = Path(output_dir)
    config = config.normalized()

    if run_bridge:
        run_left_bridge_completion(
            tts_archive=tts_archive,
            step10_source=step10_source,
            output_dir=output_dir,
            config=config,
        )

    junction_grid = pd.DataFrame()
    if run_junction_grid:
        junction_grid = run_junction_strict_grid(
            tts_archive=tts_archive,
            step10_source=step10_source,
            output_dir=output_dir,
            config=config,
        )
    else:
        path = config.output_dir / "step11M_03_right_junction_strict_grid.csv"
        if path.is_file():
            junction_grid = pd.read_csv(path, low_memory=False)

    if run_junction_edges and len(junction_grid):
        run_junction_edge_certificates(
            tts_archive=tts_archive,
            step10_source=step10_source,
            junction_grid=junction_grid,
            output_dir=output_dir,
            config=config,
        )

    (config.output_dir / "step11M_run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return aggregate_step11(
        formal_scan_source=formal_scan_source,
        step10_source=step10_source,
        output_dir=output_dir,
        config=config,
    )
