from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CODE_VERSION = "TTS_STEP17M_RANDOM_MASS_COORDINATE_VALIDATION_V1_20260727"

PHASE_COLORS = {
    -2: "#4F81BD",
     0: "#D9D9D9",
     2: "#CF5A47",
    99: "#666666",
}


@dataclass
class Step17Config:
    output_dir: Path = Path("outputs_tts_step17M_random_mass_coordinate_validation")
    random_seed: int = 20260727

    # Random interior validation counts.
    n_left_minus2: int = 15
    n_left_plus2: int = 15
    n_right_minus2: int = 15
    n_right_plus2: int = 15
    n_right_zero: int = 15

    # Boundary stress-test points. These are not included in interior accuracy.
    n_left_boundary: int = 6
    n_right_four_valley_boundary: int = 6
    n_right_sigma_boundary: int = 6

    # Safety margins in normalized mass coordinates. Interior points are sampled
    # away from the zero-mass boundaries; boundary points are sampled inside a
    # narrow strip around zero.
    interior_mass_margin: float = 0.18
    boundary_half_width: float = 0.055

    # Sampling domains in normalized mass coordinates.
    left_abs_mass_max: float = 1.05
    left_s_min: float = -0.85
    left_s_max: float = 0.90

    right_abs_m4_max: float = 0.95
    right_msigma_min: float = -1.05
    right_msigma_max: float = 0.90

    # Valid raw-parameter windows. The left r3 interval is restricted to the
    # certified local-mass range rather than the entire plotting window.
    left_r3_min: float = -0.643364
    left_r3_max: float = -0.542527
    left_r4_min: float = -0.180
    left_r4_max: float = -0.135

    right_r3_min: float = 0.056
    right_r3_max: float = 0.064
    right_r4_min: float = 0.128
    right_r4_max: float = 0.134

    # Reject samples too close to an existing regular-grid coordinate. This
    # ensures that the validation points are genuinely new parameter points.
    min_distance_from_existing_raw: float = 2.0e-5

    # Rejection sampler controls.
    maximum_draws_per_region: int = 200000

    # Physics settings. quick=True is only for pipeline testing and must not be
    # used for paper conclusions.
    quick: bool = False
    force_recalculate: bool = False

    # Plot settings.
    point_size: float = 62.0
    figure_size_combined: tuple[float, float] = (14.8, 6.8)

    def normalized(self) -> "Step17Config":
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ["checkpoints", "wilson_traces", "figures", "_physics_source"]:
            (self.output_dir / name).mkdir(exist_ok=True)
        return self


# =============================================================================
# ZIP-stream readers: no full extraction of result archives.
# =============================================================================


def _find_member_by_basename(zf: zipfile.ZipFile, filename: str) -> str:
    matches = [name for name in zf.namelist() if Path(name).name == filename]
    if not matches:
        raise FileNotFoundError(f"{filename} was not found inside {zf.filename}")
    matches.sort(key=lambda name: (len(Path(name).parts), len(name), name))
    return matches[0]


def read_csv_any(source: str | Path, filename: str) -> pd.DataFrame:
    source = Path(source)
    if source.is_dir():
        matches = sorted(source.rglob(filename), key=lambda p: (len(p.parts), str(p)))
        if not matches:
            raise FileNotFoundError(f"{filename} was not found under {source}")
        return pd.read_csv(matches[0], low_memory=False)
    if source.is_file() and source.suffix.lower() == ".csv":
        return pd.read_csv(source, low_memory=False)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            member = _find_member_by_basename(zf, filename)
            return pd.read_csv(io.BytesIO(zf.read(member)), low_memory=False)
    raise FileNotFoundError(f"Unsupported or missing source: {source}")


def read_json_any(source: str | Path, filename: str) -> dict[str, Any]:
    source = Path(source)
    if source.is_dir():
        matches = sorted(source.rglob(filename), key=lambda p: (len(p.parts), str(p)))
        if not matches:
            raise FileNotFoundError(f"{filename} was not found under {source}")
        return json.loads(matches[0].read_text(encoding="utf-8"))
    if source.is_file() and source.suffix.lower() == ".json":
        return json.loads(source.read_text(encoding="utf-8"))
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            member = _find_member_by_basename(zf, filename)
            return json.loads(zf.read(member).decode("utf-8"))
    raise FileNotFoundError(f"Unsupported or missing source: {source}")


# =============================================================================
# Dynamic physics-module import and TTS source resolution.
# =============================================================================


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses under importlib.exec_module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_step15_module(source: str | Path) -> Path:
    source = Path(source).expanduser()
    if source.is_file() and source.suffix.lower() == ".py":
        return source.resolve()
    if source.is_dir():
        matches = sorted(source.rglob("TTS_step15M_reclassify_all_boundary_points.py"))
        if matches:
            return matches[0].resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        # Extract only the single Python module, not the whole ZIP.
        with zipfile.ZipFile(source) as zf:
            member = _find_member_by_basename(zf, "TTS_step15M_reclassify_all_boundary_points.py")
            target = Path.cwd() / "TTS_step15M_reclassify_all_boundary_points.py"
            if not target.is_file():
                target.write_bytes(zf.read(member))
            return target.resolve()
    raise FileNotFoundError(f"Could not resolve Step15M support module from {source}")


def _valid_tts_source(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".zip":
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return False
    return any(Path(n).name == "TTS_step01_model_and_label_audit_v2.py" for n in names) or any(
        Path(n).name.lower() in {"tts(1).zip", "tts.zip"} for n in names
    )


def resolve_tts_archive(requested: str | Path) -> Path:
    requested = Path(requested).expanduser()
    candidates: list[Path] = []
    if requested.is_file():
        candidates.append(requested.resolve())
    if not requested.is_absolute() and (Path.cwd() / requested).is_file():
        candidates.append((Path.cwd() / requested).resolve())
    for name in ["tts(1).zip", "tts.zip", "TTS_Step08M_Mechanism_Aware_ML.zip"]:
        candidate = Path.cwd() / name
        if candidate.is_file() and candidate.resolve() not in candidates:
            candidates.append(candidate.resolve())
    for candidate in candidates:
        if _valid_tts_source(candidate):
            print(f"[input] frozen TTS model source: {candidate}")
            return candidate
    raise FileNotFoundError(
        "No valid TTS model source was found. Put tts(1).zip or "
        "TTS_Step08M_Mechanism_Aware_ML.zip beside this script."
    )


# =============================================================================
# Mass-coordinate transformations and inverse maps.
# =============================================================================


def left_boundary_slope(r3: np.ndarray, a: float, b: float) -> np.ndarray:
    return -(2.0 * a * r3 + b)


def build_left_arc_map(r3_lo: float, r3_hi: float, ref_lo: float, ref_hi: float,
                       a: float, b: float, n: int = 30001) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(min(r3_lo, ref_lo), max(r3_hi, ref_hi), n)
    slope = left_boundary_slope(grid, a, b)
    integrand = np.sqrt(1.0 + slope**2)
    dg = np.diff(grid)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dg)])
    ref_r3 = 0.5 * (ref_lo + ref_hi)
    ref_s = float(np.interp(ref_r3, grid, cumulative))
    return grid, cumulative - ref_s


def inverse_left_mass_coordinates(m_tilde: float, s_tilde: float, *, mass_scale: float,
                                  tangent_scale: float, a: float, b: float, c: float,
                                  arc_r3: np.ndarray, arc_s: np.ndarray) -> tuple[float, float]:
    target_s = float(s_tilde * tangent_scale)
    r3 = float(np.interp(target_s, arc_s, arc_r3))
    slope = float(left_boundary_slope(np.array([r3]), a, b)[0])
    normal_mass = float(m_tilde * mass_scale)
    raw_mass = normal_mass * math.sqrt(1.0 + slope**2)
    r4 = raw_mass - a * r3**2 - b * r3 - c
    return r3, float(r4)


def forward_left_mass_coordinates(r3: float, r4: float, *, mass_scale: float,
                                  tangent_scale: float, a: float, b: float, c: float,
                                  arc_r3: np.ndarray, arc_s: np.ndarray) -> tuple[float, float]:
    slope = -(2.0 * a * r3 + b)
    raw_mass = r4 + a * r3**2 + b * r3 + c
    normal_mass = raw_mass / math.sqrt(1.0 + slope**2)
    s_arc = float(np.interp(r3, arc_r3, arc_s))
    return float(normal_mass / mass_scale), float(s_arc / tangent_scale)


def inverse_right_mass_coordinates(m4_tilde: float, ms_tilde: float, *,
                                   m4_scale: float, ms_scale: float,
                                   alpha: float, beta: float,
                                   anchor_r3: float, anchor_r4: float) -> tuple[float, float]:
    m4_raw = float(m4_tilde * m4_scale * math.sqrt(1.0 + alpha**2))
    ms_raw = float(ms_tilde * ms_scale * math.sqrt(1.0 + beta**2))
    constant = anchor_r4 - beta * anchor_r3
    denominator = 1.0 - alpha * beta
    if abs(denominator) < 1.0e-8:
        raise RuntimeError("The two right-junction mass coordinates are nearly linearly dependent")
    r3 = (m4_raw + alpha * (ms_raw + constant)) / denominator
    r4 = ms_raw + beta * r3 + constant
    return float(r3), float(r4)


def forward_right_mass_coordinates(r3: float, r4: float, *,
                                   m4_scale: float, ms_scale: float,
                                   alpha: float, beta: float,
                                   anchor_r3: float, anchor_r4: float) -> tuple[float, float]:
    m4_raw = r3 - alpha * r4
    ms_raw = r4 - (anchor_r4 + beta * (r3 - anchor_r3))
    m4_normal = m4_raw / math.sqrt(1.0 + alpha**2)
    ms_normal = ms_raw / math.sqrt(1.0 + beta**2)
    return float(m4_normal / m4_scale), float(ms_normal / ms_scale)


# =============================================================================
# Random design generation.
# =============================================================================


def _far_from_existing(r3: float, r4: float, existing: np.ndarray, minimum_distance: float) -> bool:
    if existing.size == 0:
        return True
    distance = np.sqrt((existing[:, 0] - r3) ** 2 + (existing[:, 1] - r4) ** 2)
    return bool(np.min(distance) >= minimum_distance)


def _accept_unique(r3: float, r4: float, accepted: list[tuple[float, float]], threshold: float = 1.0e-7) -> bool:
    return all(math.hypot(r3 - x, r4 - y) >= threshold for x, y in accepted)


def generate_random_design(cfg: Step17Config, step16_report: dict[str, Any],
                           existing_points: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    conf = step16_report["configuration"]
    left_meta = step16_report["left_mass_coordinates"]
    right_meta = step16_report["right_double_mass_coordinates"]

    a = float(conf["left_a"]); b = float(conf["left_b"]); c = float(conf["left_c"])
    alpha = float(conf["alpha_4v"])
    beta = float(right_meta["sigma_boundary_fit"]["beta"])
    anchor_r3 = float(conf["closure_B_r3"]); anchor_r4 = float(conf["closure_B_r4"])
    left_mass_scale = float(left_meta["mass_scale"])
    left_tangent_scale = float(left_meta["tangent_scale"])
    m4_scale = float(right_meta["M_4v_scale"])
    ms_scale = float(right_meta["M_Sigma_scale"])

    arc_r3, arc_s = build_left_arc_map(
        cfg.left_r3_min, cfg.left_r3_max,
        float(conf["left_r3_min"]), float(conf["left_r3_max"]), a, b,
    )
    # Limit the sampled S range to the certified r3 interval after conversion.
    s_cert_lo = float(np.interp(cfg.left_r3_min, arc_r3, arc_s) / left_tangent_scale)
    s_cert_hi = float(np.interp(cfg.left_r3_max, arc_r3, arc_s) / left_tangent_scale)
    left_s_lo = max(cfg.left_s_min, min(s_cert_lo, s_cert_hi))
    left_s_hi = min(cfg.left_s_max, max(s_cert_lo, s_cert_hi))
    if left_s_lo >= left_s_hi:
        raise RuntimeError("No overlap between the requested left S range and the certified r3 range")

    existing = existing_points[["r3", "r4"]].dropna().to_numpy(dtype=float)
    accepted_raw: list[tuple[float, float]] = []
    rows: list[dict[str, Any]] = []

    def add_left(region: str, count: int, m_lo: float, m_hi: float, expected: int | None):
        found = 0
        for _ in range(cfg.maximum_draws_per_region):
            if found >= count:
                break
            mt = float(rng.uniform(m_lo, m_hi))
            st = float(rng.uniform(left_s_lo, left_s_hi))
            r3, r4 = inverse_left_mass_coordinates(
                mt, st, mass_scale=left_mass_scale, tangent_scale=left_tangent_scale,
                a=a, b=b, c=c, arc_r3=arc_r3, arc_s=arc_s,
            )
            if not (cfg.left_r3_min <= r3 <= cfg.left_r3_max and cfg.left_r4_min <= r4 <= cfg.left_r4_max):
                continue
            if not _far_from_existing(r3, r4, existing, cfg.min_distance_from_existing_raw):
                continue
            if not _accept_unique(r3, r4, accepted_raw):
                continue
            mt_check, st_check = forward_left_mass_coordinates(
                r3, r4, mass_scale=left_mass_scale, tangent_scale=left_tangent_scale,
                a=a, b=b, c=c, arc_r3=arc_r3, arc_s=arc_s,
            )
            sample_id = f"L_{region}_{found+1:03d}"
            rows.append({
                "sample_id": sample_id, "mass_map_region": "left_local",
                "validation_region": region, "expected_phase_code": expected,
                "M_L_tilde_target": mt, "S_L_tilde_target": st,
                "M_L_tilde": mt_check, "S_L_tilde": st_check,
                "M_4v_tilde": np.nan, "M_Sigma_tilde": np.nan,
                "r3": r3, "r4": r4,
                "source_type": "step17_random_inverse_mass_sample",
                "source_region": "left_local",
                "is_boundary_stress": int(expected is None),
            })
            accepted_raw.append((r3, r4)); found += 1
        if found != count:
            raise RuntimeError(f"Generated only {found}/{count} samples for {region}")

    def add_right(region: str, count: int,
                  m4_lo: float, m4_hi: float, ms_lo: float, ms_hi: float,
                  expected: int | None):
        found = 0
        for _ in range(cfg.maximum_draws_per_region):
            if found >= count:
                break
            m4t = float(rng.uniform(m4_lo, m4_hi))
            mst = float(rng.uniform(ms_lo, ms_hi))
            r3, r4 = inverse_right_mass_coordinates(
                m4t, mst, m4_scale=m4_scale, ms_scale=ms_scale,
                alpha=alpha, beta=beta, anchor_r3=anchor_r3, anchor_r4=anchor_r4,
            )
            if not (cfg.right_r3_min <= r3 <= cfg.right_r3_max and cfg.right_r4_min <= r4 <= cfg.right_r4_max):
                continue
            if not _far_from_existing(r3, r4, existing, cfg.min_distance_from_existing_raw):
                continue
            if not _accept_unique(r3, r4, accepted_raw):
                continue
            m4_check, ms_check = forward_right_mass_coordinates(
                r3, r4, m4_scale=m4_scale, ms_scale=ms_scale,
                alpha=alpha, beta=beta, anchor_r3=anchor_r3, anchor_r4=anchor_r4,
            )
            sample_id = f"R_{region}_{found+1:03d}"
            rows.append({
                "sample_id": sample_id, "mass_map_region": "right_junction",
                "validation_region": region, "expected_phase_code": expected,
                "M_L_tilde_target": np.nan, "S_L_tilde_target": np.nan,
                "M_L_tilde": np.nan, "S_L_tilde": np.nan,
                "M_4v_tilde": m4_check, "M_Sigma_tilde": ms_check,
                "r3": r3, "r4": r4,
                "source_type": "step17_random_inverse_mass_sample",
                "source_region": "right_junction",
                "is_boundary_stress": int(expected is None),
            })
            accepted_raw.append((r3, r4)); found += 1
        if found != count:
            raise RuntimeError(f"Generated only {found}/{count} samples for {region}")

    margin = cfg.interior_mass_margin
    bw = cfg.boundary_half_width

    add_left("minus2_interior", cfg.n_left_minus2, -cfg.left_abs_mass_max, -margin, -2)
    add_left("plus2_interior", cfg.n_left_plus2, margin, cfg.left_abs_mass_max, 2)
    add_left("mass_boundary_stress", cfg.n_left_boundary, -bw, bw, None)

    add_right("minus2_interior", cfg.n_right_minus2,
              margin, cfg.right_abs_m4_max, cfg.right_msigma_min, -margin, -2)
    add_right("plus2_interior", cfg.n_right_plus2,
              -cfg.right_abs_m4_max, -margin, cfg.right_msigma_min, -margin, 2)
    add_right("zero_interior", cfg.n_right_zero,
              -cfg.right_abs_m4_max, cfg.right_abs_m4_max, margin, cfg.right_msigma_max, 0)
    add_right("four_valley_boundary_stress", cfg.n_right_four_valley_boundary,
              -bw, bw, cfg.right_msigma_min, -margin, None)
    add_right("sigma_boundary_stress", cfg.n_right_sigma_boundary,
              -cfg.right_abs_m4_max, cfg.right_abs_m4_max, -bw, bw, None)

    design = pd.DataFrame(rows)
    design["draw_order"] = np.arange(1, len(design) + 1)
    # Randomize execution order so that checkpoints accumulate across regions.
    design = design.sample(frac=1.0, random_state=cfg.random_seed).reset_index(drop=True)
    return design


# =============================================================================
# Physics evaluation, metrics, and plotting.
# =============================================================================


def configure_step15_physics(step15, cfg: Step17Config):
    p = step15.Step15Config(output_dir=cfg.output_dir / "strict_support")
    p.quick = cfg.quick
    p.force_recalculate = cfg.force_recalculate
    # Paper validation uses a slightly stricter Fukui hierarchy than the original
    # broad Step15M scan while retaining Wilson fallback for difficult points.
    if not cfg.quick:
        p.gap_grid_n = 121
        p.gap_local_starts = 18
        p.fukui_grids = (61, 81, 101, 121)
        p.fukui_shifts = ((0.00, 0.00), (0.50, 0.50), (0.25, 0.73))
        p.wilson_nkx = (401, 601, 801)
        p.wilson_initial_nky = (81, 101, 121)
        p.wilson_shifts = ((0.00, 0.00), (0.50, 0.37), (0.25, 0.73))
    return p.normalized()


def evaluate_design(design: pd.DataFrame, *, step15, core, background: dict[str, float],
                    orientation_factor: int, physics_cfg, cfg: Step17Config):
    results: list[dict[str, Any]] = []
    fukui_all: list[dict[str, Any]] = []
    wilson_all: list[dict[str, Any]] = []
    start = time.time()

    for index, row in design.iterrows():
        sample_id = str(row["sample_id"])
        checkpoint = cfg.output_dir / "checkpoints" / f"{sample_id}.json"
        if checkpoint.is_file() and not cfg.force_recalculate:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            results.append(payload["result"])
            fukui_all.extend(payload.get("fukui_attempts", []))
            wilson_all.extend(payload.get("wilson_attempts", []))
            print(f"[{index+1}/{len(design)}] {sample_id}: checkpoint")
            continue

        print(
            f"[{index+1}/{len(design)}] {sample_id}  "
            f"region={row['validation_region']}  r3={row['r3']:.9f} r4={row['r4']:.9f}"
        )
        result, fukui_rows, wilson_rows, traces = step15.classify_one(
            core=core, background=background, row=row,
            orientation_factor=orientation_factor, config=physics_cfg,
        )
        # Preserve validation design metadata in the final result.
        for key in [
            "mass_map_region", "validation_region", "expected_phase_code",
            "M_L_tilde", "S_L_tilde", "M_4v_tilde", "M_Sigma_tilde",
            "is_boundary_stress",
        ]:
            result[key] = row.get(key, np.nan)
        payload = {
            "result": result,
            "fukui_attempts": fukui_rows,
            "wilson_attempts": wilson_rows,
        }
        step15.atomic_json(payload, checkpoint)
        for trace in traces:
            sid = str(trace.iloc[0]["sample_id"])
            spin = str(trace.iloc[0]["spin"])
            attempt = int(trace.iloc[0]["attempt_index"])
            step15.atomic_csv(
                trace,
                cfg.output_dir / "wilson_traces" / f"{sid}__{spin}__attempt{attempt:02d}.csv",
            )
        results.append(result)
        fukui_all.extend(fukui_rows)
        wilson_all.extend(wilson_rows)
        print(
            f"    -> {result['final_status']}  C_up={result.get('final_chern_up')}  "
            f"method={result.get('topology_method')}"
        )

    return pd.DataFrame(results), pd.DataFrame(fukui_all), pd.DataFrame(wilson_all), time.time() - start


def validation_metrics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    results = results.copy()
    interior = results[results["is_boundary_stress"].astype(int).eq(0)].copy()
    interior["actual_phase_code"] = pd.to_numeric(interior["strict_phase_code"], errors="coerce")
    interior["prediction_match"] = (
        interior["actual_phase_code"].eq(pd.to_numeric(interior["expected_phase_code"], errors="coerce"))
    ).astype(int)

    labels = [-2, 0, 2]
    matrix = pd.crosstab(
        pd.Categorical(interior["expected_phase_code"], categories=labels),
        pd.Categorical(interior["actual_phase_code"], categories=labels),
        dropna=False,
    )
    matrix.index.name = "expected_phase_code"
    matrix.columns.name = "actual_phase_code"
    matrix_long = matrix.stack(future_stack=True).rename("count").reset_index()

    region_rows = []
    for region, group in interior.groupby("validation_region", sort=False):
        n = len(group)
        region_rows.append({
            "validation_region": region,
            "expected_phase_code": group["expected_phase_code"].iloc[0],
            "n_points": n,
            "n_match": int(group["prediction_match"].sum()),
            "accuracy": float(group["prediction_match"].mean()) if n else np.nan,
            "n_numeric_unresolved": int(group["final_status"].eq("numeric_unresolved").sum()),
            "n_noninsulating_or_boundary": int((~group["strict_phase_code"].isin([-2, 0, 2])).sum()),
        })
    region_summary = pd.DataFrame(region_rows)

    boundary = results[results["is_boundary_stress"].astype(int).eq(1)].copy()
    if boundary.empty:
        boundary_summary = pd.DataFrame()
    else:
        boundary_summary = (
            boundary.groupby(["validation_region", "final_status"], dropna=False)
            .size().rename("count").reset_index()
        )

    recalls = []
    for label in labels:
        group = interior[pd.to_numeric(interior["expected_phase_code"], errors="coerce").eq(label)]
        if len(group):
            recalls.append(float(group["prediction_match"].mean()))
    certificate = {
        "n_random_points_total": int(len(results)),
        "n_interior_validation_points": int(len(interior)),
        "n_boundary_stress_points": int(len(boundary)),
        "n_interior_matches": int(interior["prediction_match"].sum()),
        "interior_accuracy": float(interior["prediction_match"].mean()) if len(interior) else None,
        "interior_balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "n_interior_numeric_unresolved": int(interior["final_status"].eq("numeric_unresolved").sum()),
        "n_interior_noninsulating_or_boundary": int((~interior["strict_phase_code"].isin([-2, 0, 2])).sum()),
        "n_boundary_gap_closing_or_noninsulating": int((~boundary["strict_phase_code"].isin([-2, 0, 2])).sum()),
        "all_interior_random_points_match_mass_sector": bool(
            len(interior) > 0 and interior["prediction_match"].eq(1).all()
        ),
    }
    return interior, matrix_long, region_summary, boundary_summary, certificate


def _plot_points(ax, points: pd.DataFrame, x: str, y: str):
    for code in [-2, 0, 2]:
        subset = points[pd.to_numeric(points["strict_phase_code"], errors="coerce").eq(code)]
        if subset.empty:
            continue
        ax.scatter(
            subset[x], subset[y], s=64,
            facecolors="white", edgecolors=PHASE_COLORS[code], linewidths=1.7,
            zorder=5,
        )
    unresolved = points[~pd.to_numeric(points["strict_phase_code"], errors="coerce").isin([-2, 0, 2])]
    if not unresolved.empty:
        ax.scatter(
            unresolved[x], unresolved[y], s=70, marker="x",
            color=PHASE_COLORS[99], linewidths=1.8, zorder=6,
        )
    mismatch = points[
        points["is_boundary_stress"].astype(int).eq(0)
        & pd.to_numeric(points["expected_phase_code"], errors="coerce").ne(
            pd.to_numeric(points["strict_phase_code"], errors="coerce")
        )
    ]
    if not mismatch.empty:
        ax.scatter(
            mismatch[x], mismatch[y], s=115, facecolors="none",
            edgecolors="black", linewidths=2.0, zorder=7,
        )


def configure_publication_fonts():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        # Keep accents such as the tilde over mass coordinates at regular
        # weight; all other upright math glyphs are explicitly bolded below.
        "mathtext.rm": "Times New Roman:style=normal:weight=normal",
        "mathtext.it": "Times New Roman:italic:bold",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman:bold",
        "mathtext.tt": "Times New Roman:bold",
        "mathtext.fallback": None,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_validation_maps(results: pd.DataFrame, cfg: Step17Config):
    configure_publication_fonts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.figure_size_combined)

    # Left ideal sector background.
    ax1.axvspan(-1.4, 0, color=PHASE_COLORS[-2], alpha=0.28, zorder=0)
    ax1.axvspan(0, 1.4, color=PHASE_COLORS[2], alpha=0.28, zorder=0)
    ax1.axvline(0, color="black", linewidth=1.8)
    left = results[results["mass_map_region"].eq("left_local")]
    _plot_points(ax1, left, "M_L_tilde", "S_L_tilde")
    ax1.set_xlim(-1.35, 1.35); ax1.set_ylim(-1.15, 1.15)
    ax1.set_xlabel(r"$\overset{\text{~}}{M}_{\mathbf{L}}$", fontsize=25, fontweight="normal")
    ax1.set_ylabel(r"$\overset{\text{~}}{S}_{\mathbf{L}}$", fontsize=25, fontweight="normal")
    ax1.set_title("(a) Random validation in the left local-mass plane", fontsize=20, pad=12)

    # Right ideal sector background.
    ax2.axhspan(0, 1.4, color=PHASE_COLORS[0], alpha=0.45, zorder=0)
    ax2.fill_between([-1.4, 0], -1.4, 0, color=PHASE_COLORS[2], alpha=0.35, zorder=0)
    ax2.fill_between([0, 1.4], -1.4, 0, color=PHASE_COLORS[-2], alpha=0.35, zorder=0)
    ax2.axvline(0, color="black", linewidth=1.8)
    ax2.axhline(0, color="black", linewidth=1.8)
    right = results[results["mass_map_region"].eq("right_junction")]
    _plot_points(ax2, right, "M_4v_tilde", "M_Sigma_tilde")
    ax2.set_xlim(-1.35, 1.35); ax2.set_ylim(-1.35, 1.35)
    ax2.set_xlabel(r"$\overset{\text{~}}{M}_{\mathbf{4v}}$", fontsize=25, fontweight="normal")
    ax2.set_ylabel(r"$\overset{\text{~}}{M}_{\mathbf{\Sigma^\prime}}$", fontsize=25, fontweight="normal")
    ax2.set_title("(b) Random validation in the right double-mass plane", fontsize=20, pad=12)

    for ax in [ax1, ax2]:
        ax.set_box_aspect(1)
        ax.tick_params(direction="in", length=7, width=1.7, top=True, right=True, labelsize=15)
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)

    handles = [
        Patch(facecolor=PHASE_COLORS[-2], edgecolor="black", label=r"$C_{\mathbf{\uparrow}}\mathbf{=-2}$"),
        Patch(facecolor=PHASE_COLORS[0], edgecolor="black", label=r"$C_{\mathbf{\uparrow}}\mathbf{=0}$"),
        Patch(facecolor=PHASE_COLORS[2], edgecolor="black", label=r"$C_{\mathbf{\uparrow}}\mathbf{=2}$"),
        Line2D([0], [0], marker="x", linestyle="None", color=PHASE_COLORS[99],
               markersize=9, markeredgewidth=1.8, label="gapless / boundary / unresolved"),
        Line2D([0], [0], marker="o", linestyle="None", color="black",
               markerfacecolor="none", markersize=10, markeredgewidth=1.8,
               label="interior mismatch"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, 1.02), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    stem = cfg.output_dir / "figures" / "step17M_random_mass_coordinate_validation"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix_long: pd.DataFrame, cfg: Step17Config):
    configure_publication_fonts()
    labels = [-2, 0, 2]
    matrix = matrix_long.pivot(index="expected_phase_code", columns="actual_phase_code", values="count")
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0).fillna(0).to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=15, fontweight="bold")
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set_xlabel(r"Calculated $C_{\mathbf{\uparrow}}$", fontsize=17, fontweight="bold")
    ax.set_ylabel(r"Mass-sector prediction", fontsize=17, fontweight="bold")
    ax.set_title("Random out-of-grid validation", fontsize=18, fontweight="bold")
    ax.tick_params(direction="in", top=True, right=True, labelsize=14)
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    stem = cfg.output_dir / "figures" / "step17M_random_validation_confusion_matrix"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main workflow.
# =============================================================================


def run_step17M(
    *,
    tts_archive: str | Path = "tts(1).zip",
    step11_source: str | Path = "outputs_tts_step11M_lieb_aligned_fixed_slice.zip",
    step13_source: str | Path = "outputs_tts_step13M_adaptive_wilson_intermediate_chern.zip",
    step16_source: str | Path = "outputs_tts_step16M_mass_coordinate_phase_maps",
    step15_module_source: str | Path = "TTS_step15M_reclassify_all_boundary_points.py",
    output_dir: str | Path = "outputs_tts_step17M_random_mass_coordinate_validation",
    config: Step17Config | None = None,
    run_physics: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step17Config(output_dir=Path(output_dir))
    else:
        config.output_dir = Path(output_dir)
    config = config.normalized()

    print("Code version:", CODE_VERSION)
    step16_report = read_json_any(step16_source, "step16M_02_mass_coordinate_report.json")
    existing_points = read_csv_any(step16_source, "step16M_00_transformed_mass_coordinate_points.csv")

    design = generate_random_design(config, step16_report, existing_points)
    design_path = config.output_dir / "step17M_01_random_inverse_mass_samples.csv"
    design.to_csv(design_path, index=False)
    audit = {
        "code_version": CODE_VERSION,
        "sampling_design": "uniform_random_sampling_in_local_mass sectors followed by exact inverse mapping to r3-r4",
        "config": asdict(config),
        "n_samples": int(len(design)),
        "counts_by_region": design["validation_region"].value_counts().to_dict(),
        "step16_mass_report_code_version": step16_report.get("code_version"),
        "warnings": [
            "The mass coordinates are local and must not be extrapolated beyond the certified windows.",
            "Boundary-stress points are excluded from interior classification accuracy.",
            "quick=True is for pipeline testing only and is not paper evidence.",
        ],
    }
    (config.output_dir / "step17M_00_sampling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    if not run_physics:
        summary = {
            "code_version": CODE_VERSION,
            "sampling_only": True,
            "n_samples_generated": int(len(design)),
            "design_csv": str(design_path),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    module_path = resolve_step15_module(step15_module_source)
    step15 = import_module_from_path("tts_step15_support_for_step17", module_path)
    physics_cfg = configure_step15_physics(step15, config)
    model_archive = resolve_tts_archive(tts_archive)
    core = step15.load_core(model_archive, physics_cfg)
    background = step15.read_json_token(step11_source, "step11M_13_fixed_slice_certificate.json")[
        "fixed_background_parameters"
    ]
    orientation = step15.read_json_token(step13_source, "step13M_04b_orientation_calibration.json")
    if int(orientation.get("orientation_calibrated", 0)) != 1:
        raise RuntimeError("Step13 Wilson orientation was not calibrated")
    orientation_factor = int(orientation["orientation_factor"])

    results, fukui, wilson, elapsed = evaluate_design(
        design, step15=step15, core=core, background=background,
        orientation_factor=orientation_factor, physics_cfg=physics_cfg, cfg=config,
    )
    results.to_csv(config.output_dir / "step17M_02_random_validation_results.csv", index=False)
    fukui.to_csv(config.output_dir / "step17M_03_fukui_attempts.csv", index=False)
    wilson.to_csv(config.output_dir / "step17M_04_wilson_attempts.csv", index=False)

    interior, matrix_long, region_summary, boundary_summary, certificate = validation_metrics(results)
    interior.to_csv(config.output_dir / "step17M_05_interior_validation_points.csv", index=False)
    matrix_long.to_csv(config.output_dir / "step17M_06_confusion_matrix.csv", index=False)
    region_summary.to_csv(config.output_dir / "step17M_07_region_accuracy.csv", index=False)
    boundary_summary.to_csv(config.output_dir / "step17M_08_boundary_stress_summary.csv", index=False)

    mismatches = interior[interior["prediction_match"].eq(0)].copy()
    mismatches.to_csv(config.output_dir / "step17M_09_interior_mismatches.csv", index=False)

    certificate.update({
        "code_version": CODE_VERSION,
        "elapsed_seconds": float(elapsed),
        "fixed_background_parameters": background,
        "wilson_orientation_factor": orientation_factor,
        "physics_quick_mode": bool(config.quick),
        "validation_claim": (
            "Random out-of-grid parameter points were sampled in local mass-coordinate sectors, "
            "inverse-mapped to the original Hamiltonian parameters, and independently recalculated."
        ),
        "evidence_scope": {
            "left": "Only the certified left local four-valley window is tested.",
            "right": "Only the refined right-junction window is tested.",
            "global": "This is not a validation of a universal seven-dimensional mass rule.",
        },
    })
    (config.output_dir / "step17M_10_final_random_validation_certificate.json").write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    plot_validation_maps(results, config)
    plot_confusion_matrix(matrix_long, config)

    print(json.dumps(certificate, ensure_ascii=False, indent=2, default=str))
    return certificate


if __name__ == "__main__":
    run_step17M()
