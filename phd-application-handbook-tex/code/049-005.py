#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 05 — 自旋分辨临界谷、低能 k·p 与 valley 拓扑电荷

本步骤承接 TTS Step 04，但不再使用总八带半填充直接带隙来判定
spin-Chern 跳变的位置。真正的 C_up 跳变要求 spin-up 4×4 块的
第 2/3 条能带闭隙；C_down 同理。因此本脚本执行：

1. 从 Step 04 的严格 Chern 跳变区间重新搜索 spin-up 内部闭隙；
2. 在 Γ/M/X/Y 和 Σ/Σ' 线上做确定性多起点搜索，必要时全 BZ 回退；
3. 枚举 D4 对称轨道并判断每个 valley 属于 up 还是 down 自旋块；
4. 将临界两带投影为 H_eff = d0 I + d·σ，计算三参数 Jacobian；
5. 在 (kx, ky, λ) 小球上用非阿贝尔 determinant-link Berry flux
   直接计算每个 valley 的整数拓扑电荷；
6. 验证 valley 电荷之和是否严格等于 Step 04 观测的 ΔC_up；
7. 输出高对称质量梯度与局域半解析 k·p 系数，为后续解析公式准备数据。

模型：tts 3.3.4.3.4，P4'/mbm'，BNS 127.391，自旋守恒八带模型。
全部计算串行，适合 Windows/Jupyter，不使用 ProcessPoolExecutor。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
import argparse
import json
import math
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, differential_evolution, minimize, minimize_scalar
from scipy.spatial import ConvexHull

import TTS_step01_model_and_label_audit_v2 as core
import TTS_step04_chern_sector_boundary_valley_tracking as step4

CODE_VERSION = "TTS_STEP05_V2_20260714"
REDUCED7 = list(core.REDUCED7)
SPINS = ("up", "down")


@dataclass
class Step05Config:
    output_dir: Path = Path("outputs_tts_step05_kp_valley_topological_charge")
    step4_input: Path = Path("outputs_tts_step04_chern_sector_boundary_valley_tracking.zip")

    # Spin-resolved critical-point search.
    coarse_lambda_points: int = 9
    coarse_diagonal_k_points: int = 51
    local_starts_per_manifold: int = 6
    powell_maxiter: int = 500
    full_bz_fallback: bool = True
    full_bz_de_popsize: int = 10
    full_bz_de_maxiter: int = 160
    spin_gap_accept_tol: float = 2.0e-7
    candidate_lambda_dedup_tol: float = 2.0e-4
    candidate_k_dedup_tol: float = 2.0e-3
    active_spin_gap_tol: float = 2.0e-6

    # Finite-difference low-energy k·p.
    fd_k: float = 1.0e-5
    fd_lambda: float = 1.0e-5
    fd_parameter: float = 1.0e-6
    jacobian_singular_tol: float = 1.0e-8

    # Berry sphere: several radii must give the same integer charge.
    sphere_subdivision: int = 2
    sphere_k_radii: tuple[float, ...] = (0.02, 0.04, 0.08)
    sphere_lambda_radius_min: float = 8.0e-4
    sphere_lambda_radius_max: float = 1.5e-2
    sphere_integer_tol: float = 0.08
    sphere_min_link_tol: float = 1.0e-6

    # Optional Chern verification immediately on both sides of each transition.
    side_chern_grids: tuple[int, ...] = (31,)
    side_chern_shifts: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    side_lambda_fraction: float = 0.18
    chern_integer_tol: float = 0.08

    force_recalculate: bool = False
    random_seed: int = 20260714

    def normalized(self) -> "Step05Config":
        self.output_dir = Path(self.output_dir)
        self.step4_input = Path(self.step4_input)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        if self.coarse_lambda_points < 5:
            raise ValueError("coarse_lambda_points must be >= 5")
        if self.coarse_diagonal_k_points < 21:
            raise ValueError("coarse_diagonal_k_points must be >= 21")
        return self


# =============================================================================
# I/O
# =============================================================================


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _find_default_step4_input() -> Path:
    candidates = [
        Path("outputs_tts_step04_chern_sector_boundary_valley_tracking.zip"),
        Path("outputs_tts_step04_chern_sector_boundary_valley_tracking"),
    ]
    for p in candidates:
        if p.exists():
            return p
    zips = sorted(Path.cwd().glob("*step04*boundary*valley*.zip"))
    if zips:
        return zips[0]
    raise FileNotFoundError(
        "Cannot find Step 04 result ZIP/directory. Place it beside this script "
        "or pass --step4-input."
    )


def _read_csv_from_zip(zip_path: Path, basename: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if Path(n).name == basename]
        if len(names) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {basename!r} in {zip_path}; found {names}"
            )
        with zf.open(names[0], "r") as fh:
            return pd.read_csv(fh)


def _read_csv_from_directory(root: Path, basename: str) -> pd.DataFrame:
    names = list(root.rglob(basename))
    if len(names) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {basename!r} under {root}; found {names}"
        )
    return pd.read_csv(names[0])


def load_step4_tables(config: Step05Config) -> dict[str, pd.DataFrame]:
    source = config.step4_input
    if not source.exists():
        source = _find_default_step4_input()
        config.step4_input = source
    files = {
        "partners": "step04_01_nearest_trivial_path_partners.csv",
        "path_chern": "step04_04_path_adaptive_chern_labels.csv",
        "brackets": "step04_05_transition_brackets.csv",
        "step4_critical": "step04_05_refined_critical_valleys.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, basename in files.items():
        if source.is_file() and source.suffix.lower() == ".zip":
            out[key] = _read_csv_from_zip(source, basename)
        else:
            out[key] = _read_csv_from_directory(source, basename)
    return out


# =============================================================================
# Path and Hamiltonian helpers
# =============================================================================


def pair_vectors(pair: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([float(pair[f"anchor_{name}"]) for name in REDUCED7], dtype=float)
    b = np.array([float(pair[f"trivial_{name}"]) for name in REDUCED7], dtype=float)
    return a, b


def reduced_on_path(pair: pd.Series, lam: float) -> dict[str, float]:
    a, b = pair_vectors(pair)
    return step4.vector_to_reduced(step4.slerp(a, b, float(lam)))


def raw_on_path(pair: pd.Series, lam: float) -> dict[str, float]:
    return core.raw8_from_reduced7(reduced_on_path(pair, lam), e0=0.0)


def spin_block(kx: float, ky: float, pair: pd.Series, lam: float, spin: str) -> np.ndarray:
    return core.h_spin_block_periodic(float(kx), float(ky), raw_on_path(pair, lam), spin)


def spin_middle_gap(kx: float, ky: float, pair: pd.Series, lam: float, spin: str) -> float:
    eig = np.linalg.eigvalsh(spin_block(kx, ky, pair, lam, spin))
    return float(max(0.0, eig[2] - eig[1]))


def rounded_integer(value: float, tol: float) -> int | None:
    if not np.isfinite(value):
        return None
    nearest = int(np.rint(value))
    return nearest if abs(float(value) - nearest) <= tol else None


# =============================================================================
# Spin-resolved gap-closing search
# =============================================================================


HIGH_SYMMETRY_POINTS: dict[str, tuple[float, float]] = {
    "Gamma": (0.0, 0.0),
    "X": (math.pi, 0.0),
    "Y": (0.0, math.pi),
    "M": (math.pi, math.pi),
}


def _bounded_powell_2d(
    objective,
    start: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    maxiter: int,
) -> OptimizeResult:
    """Robust bounded local minimization for the two diagonal coordinates.

    Older/newer SciPy combinations can make bounded Powell construct a zero
    search direction and raise ``ValueError: zero-size array to reduction``.
    The physics does not require the Powell algorithm itself, so this wrapper
    uses L-BFGS-B first and bounded Nelder-Mead as a derivative-free fallback.
    Any local failure is converted into a normal unsuccessful OptimizeResult;
    the deterministic differential-evolution search below still runs.
    """
    x0 = np.asarray(start, dtype=float).reshape(-1)
    bnd = [(float(lo), float(hi)) for lo, hi in bounds]
    if x0.size != len(bnd):
        raise ValueError(f"start has {x0.size} variables but {len(bnd)} bounds were supplied")

    lower = np.asarray([lo for lo, _ in bnd], dtype=float)
    upper = np.asarray([hi for _, hi in bnd], dtype=float)
    if (not np.all(np.isfinite(x0)) or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))):
        return OptimizeResult(
            x=np.clip(np.nan_to_num(x0, nan=0.0), lower, upper),
            fun=np.inf, success=False, nfev=0,
            message="invalid non-finite local-search start or bounds",
            method="none",
        )
    if np.any(upper < lower):
        return OptimizeResult(
            x=np.clip(x0, np.minimum(lower, upper), np.maximum(lower, upper)),
            fun=np.inf, success=False, nfev=0,
            message="invalid reversed local-search bounds",
            method="none",
        )

    # Keep the starting point strictly inside nonzero-width intervals.
    widths = upper - lower
    margin = np.maximum(1.0e-12, 1.0e-10 * np.maximum(widths, 1.0))
    interior_lo = np.where(widths > 2.0 * margin, lower + margin, lower)
    interior_hi = np.where(widths > 2.0 * margin, upper - margin, upper)
    x0 = np.minimum(np.maximum(x0, interior_lo), interior_hi)

    attempts: list[OptimizeResult] = []
    errors: list[str] = []

    methods = [
        (
            "L-BFGS-B",
            {
                "maxiter": int(maxiter),
                "ftol": 1.0e-15,
                "gtol": 1.0e-10,
                "maxls": 80,
            },
        ),
        (
            "Nelder-Mead",
            {
                "maxiter": int(maxiter),
                "xatol": 1.0e-11,
                "fatol": 1.0e-22,
                "adaptive": True,
            },
        ),
    ]

    for method, options in methods:
        try:
            result = minimize(
                objective,
                x0=x0,
                method=method,
                bounds=bnd,
                options=options,
            )
            result.method = method
            if np.all(np.isfinite(np.asarray(result.x, dtype=float))) and np.isfinite(float(result.fun)):
                attempts.append(result)
        except Exception as exc:  # Local search must never abort the full workflow.
            errors.append(f"{method}: {type(exc).__name__}: {exc}")

    if attempts:
        best = min(attempts, key=lambda r: float(r.fun))
        if errors:
            best.message = f"{best.message}; recovered after {' | '.join(errors)}"
        return best

    try:
        fallback_fun = float(objective(x0))
    except Exception as exc:
        fallback_fun = np.inf
        errors.append(f"start evaluation: {type(exc).__name__}: {exc}")
    return OptimizeResult(
        x=x0, fun=fallback_fun, success=False, nfev=1,
        message="all bounded local optimizers failed: " + " | ".join(errors),
        method="start_point_fallback",
    )


def search_high_symmetry_candidates(
    pair: pd.Series,
    bracket: pd.Series,
    config: Step05Config,
) -> list[dict[str, Any]]:
    lo, hi = sorted([float(bracket["lambda_left"]), float(bracket["lambda_right"])])
    rows: list[dict[str, Any]] = []
    for region, (kx, ky) in HIGH_SYMMETRY_POINTS.items():
        result = minimize_scalar(
            lambda lam: spin_middle_gap(kx, ky, pair, float(lam), "up") ** 2,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1.0e-13, "maxiter": 1000},
        )
        gap = spin_middle_gap(kx, ky, pair, float(result.x), "up")
        rows.append({
            "search_manifold": region,
            "critical_lambda": float(result.x),
            "critical_kx": float(step4.wrap_k(kx)),
            "critical_ky": float(step4.wrap_k(ky)),
            "spin_up_gap": gap,
            "optimizer_success": int(bool(result.success)),
            "optimizer_message": str(result.message),
            "optimizer_nfev": int(result.nfev),
        })
    return rows


def diagonal_coordinates(kind: str, kappa: float) -> tuple[float, float]:
    if kind == "Sigma":
        return float(kappa), float(kappa)
    if kind == "SigmaPrime":
        return float(kappa), float(-kappa)
    raise ValueError(kind)


def search_one_diagonal(
    pair: pd.Series,
    bracket: pd.Series,
    kind: str,
    config: Step05Config,
    seed_offset: int,
) -> list[dict[str, Any]]:
    lo, hi = sorted([float(bracket["lambda_left"]), float(bracket["lambda_right"])])
    lambdas = np.linspace(lo, hi, int(config.coarse_lambda_points))
    kappas = np.linspace(-math.pi, math.pi, int(config.coarse_diagonal_k_points), endpoint=False)

    coarse: list[tuple[float, float, float]] = []
    for lam in lambdas:
        for kappa in kappas:
            kx, ky = diagonal_coordinates(kind, float(kappa))
            coarse.append((spin_middle_gap(kx, ky, pair, float(lam), "up"), float(lam), float(kappa)))
    coarse.sort(key=lambda x: x[0])

    starts: list[np.ndarray] = []
    for _, lam, kap in coarse:
        candidate = np.array([lam, kap], dtype=float)
        if all(
            abs(candidate[0] - old[0]) > 0.01 * max(hi - lo, 1.0e-5)
            or abs(step4.torus_delta(candidate[1], old[1])) > 0.08
            for old in starts
        ):
            starts.append(candidate)
        if len(starts) >= int(config.local_starts_per_manifold):
            break

    def objective(x: np.ndarray) -> float:
        lam, kappa = float(x[0]), float(x[1])
        kx, ky = diagonal_coordinates(kind, kappa)
        gap = spin_middle_gap(kx, ky, pair, lam, "up")
        return gap * gap

    results: list[dict[str, Any]] = []
    for start in starts:
        opt = _bounded_powell_2d(
            objective,
            start,
            [(lo, hi), (-math.pi, math.pi)],
            config.powell_maxiter,
        )
        lam, kap = float(opt.x[0]), float(opt.x[1])
        kx, ky = diagonal_coordinates(kind, kap)
        gap = spin_middle_gap(kx, ky, pair, lam, "up")
        results.append({
            "search_manifold": kind,
            "critical_lambda": lam,
            "critical_kx": float(step4.wrap_k(kx)),
            "critical_ky": float(step4.wrap_k(ky)),
            "spin_up_gap": gap,
            "optimizer_success": int(bool(opt.success)),
            "optimizer_message": str(opt.message),
            "optimizer_nfev": int(getattr(opt, "nfev", 0)),
            "optimizer_method": str(getattr(opt, "method", "local_hybrid")),
        })

    # Differential evolution is a deterministic fallback against narrow valleys.
    de = differential_evolution(
        objective,
        bounds=[(lo, hi), (-math.pi, math.pi)],
        seed=int(config.random_seed + seed_offset),
        popsize=max(8, int(config.full_bz_de_popsize // 2)),
        maxiter=max(120, int(config.full_bz_de_maxiter // 2)),
        tol=1.0e-11,
        polish=True,
        workers=1,
        updating="immediate",
    )
    lam, kap = float(de.x[0]), float(de.x[1])
    kx, ky = diagonal_coordinates(kind, kap)
    results.append({
        "search_manifold": kind + "_DE",
        "critical_lambda": lam,
        "critical_kx": float(step4.wrap_k(kx)),
        "critical_ky": float(step4.wrap_k(ky)),
        "spin_up_gap": spin_middle_gap(kx, ky, pair, lam, "up"),
        "optimizer_success": int(bool(de.success)),
        "optimizer_message": str(de.message),
        "optimizer_nfev": int(de.nfev),
    })
    return results


def search_full_bz_fallback(
    pair: pd.Series,
    bracket: pd.Series,
    config: Step05Config,
    seed_offset: int,
) -> dict[str, Any]:
    lo, hi = sorted([float(bracket["lambda_left"]), float(bracket["lambda_right"])])

    def objective(x: np.ndarray) -> float:
        gap = spin_middle_gap(float(x[1]), float(x[2]), pair, float(x[0]), "up")
        return gap * gap

    result = differential_evolution(
        objective,
        bounds=[(lo, hi), (-math.pi, math.pi), (-math.pi, math.pi)],
        seed=int(config.random_seed + 1000 + seed_offset),
        popsize=int(config.full_bz_de_popsize),
        maxiter=int(config.full_bz_de_maxiter),
        tol=1.0e-11,
        polish=True,
        workers=1,
        updating="immediate",
    )
    lam, kx, ky = [float(v) for v in result.x]
    return {
        "search_manifold": "full_BZ_DE",
        "critical_lambda": lam,
        "critical_kx": float(step4.wrap_k(kx)),
        "critical_ky": float(step4.wrap_k(ky)),
        "spin_up_gap": spin_middle_gap(kx, ky, pair, lam, "up"),
        "optimizer_success": int(bool(result.success)),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
    }


def _snap_boundary_k(value: float, tol: float = 5.0e-4) -> float:
    wrapped = step4.wrap_k(float(value))
    # +pi and -pi are the same torus point. Numerical optimizers often return
    # values a few 1e-7 away from opposite sides of the seam.
    if abs(abs(wrapped) - math.pi) <= tol:
        return -math.pi
    if abs(wrapped) <= tol:
        return 0.0
    return wrapped


def orbit_signature(kx: float, ky: float, digits: int = 4) -> tuple[tuple[float, float], ...]:
    orbit = step4.d4_orbit(_snap_boundary_k(kx), _snap_boundary_k(ky))
    points = []
    for x, y in orbit:
        points.append((round(_snap_boundary_k(x), digits), round(_snap_boundary_k(y), digits)))
    return tuple(sorted(set(points)))


def deduplicate_closure_candidates(
    candidates: list[dict[str, Any]],
    config: Step05Config,
) -> list[dict[str, Any]]:
    accepted = [r for r in candidates if float(r["spin_up_gap"]) <= config.spin_gap_accept_tol]
    accepted.sort(key=lambda r: float(r["spin_up_gap"]))
    unique: list[dict[str, Any]] = []
    for row in accepted:
        sig = orbit_signature(float(row["critical_kx"]), float(row["critical_ky"]))
        duplicate = False
        for old in unique:
            old_sig = orbit_signature(float(old["critical_kx"]), float(old["critical_ky"]))
            if (
                sig == old_sig
                and abs(float(row["critical_lambda"]) - float(old["critical_lambda"]))
                <= config.candidate_lambda_dedup_tol
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(row)
    return unique


def search_transition_closures(
    bracket: pd.Series,
    pair: pd.Series,
    config: Step05Config,
    transition_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = search_high_symmetry_candidates(pair, bracket, config)
    candidates.extend(search_one_diagonal(pair, bracket, "Sigma", config, 10 * transition_index + 1))
    candidates.extend(search_one_diagonal(pair, bracket, "SigmaPrime", config, 10 * transition_index + 2))

    unique = deduplicate_closure_candidates(candidates, config)
    if not unique and config.full_bz_fallback:
        candidates.append(search_full_bz_fallback(pair, bracket, config, transition_index))
        unique = deduplicate_closure_candidates(candidates, config)

    # If a full-BZ fallback finds a lower zero not represented by the symmetry searches,
    # retain it as an additional orbit.
    if config.full_bz_fallback:
        best_gap = min((float(r["spin_up_gap"]) for r in candidates), default=np.inf)
        if best_gap > config.spin_gap_accept_tol:
            candidates.append(search_full_bz_fallback(pair, bracket, config, transition_index))
            unique = deduplicate_closure_candidates(candidates, config)
    return candidates, unique


# =============================================================================
# D4 orbit and spin assignment
# =============================================================================


def enumerate_spin_valleys(
    closure: dict[str, Any],
    pair: pd.Series,
    config: Step05Config,
) -> pd.DataFrame:
    lam = float(closure["critical_lambda"])
    orbit = step4.d4_orbit(float(closure["critical_kx"]), float(closure["critical_ky"]))
    rows: list[dict[str, Any]] = []
    all_gaps: list[float] = []
    temporary: list[dict[str, Any]] = []
    for orbit_index, (kx, ky) in enumerate(orbit):
        for spin in SPINS:
            gap = spin_middle_gap(kx, ky, pair, lam, spin)
            all_gaps.append(gap)
            temporary.append({
                "orbit_index": int(orbit_index),
                "spin": spin,
                "kx": float(kx),
                "ky": float(ky),
                "spin_gap": float(gap),
            })
    min_gap = min(all_gaps)
    threshold = max(config.active_spin_gap_tol, 100.0 * min_gap)
    for row in temporary:
        row["is_active_valley"] = int(float(row["spin_gap"]) <= threshold)
        row.update(step4.classify_k_region(float(row["kx"]), float(row["ky"]), 0.08))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Two-band k·p projection
# =============================================================================


PAULI_LABELS = ("x", "y", "z")


def pauli_decompose(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    m = np.asarray(matrix, dtype=np.complex128)
    d0 = 0.5 * float((m[0, 0] + m[1, 1]).real)
    d = np.array([
        float(m[0, 1].real),
        float(-m[0, 1].imag),
        0.5 * float((m[0, 0] - m[1, 1]).real),
    ])
    return d0, d


def central_derivative(func, x0: float, step: float) -> np.ndarray:
    return (func(x0 + step) - func(x0 - step)) / (2.0 * step)


def crossing_basis(
    pair: pd.Series,
    lam: float,
    kx: float,
    ky: float,
    spin: str,
    config: Step05Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h0 = spin_block(kx, ky, pair, lam, spin)
    eig, vec = np.linalg.eigh(h0)
    u = vec[:, 1:3]

    dhl = central_derivative(
        lambda ll: spin_block(kx, ky, pair, ll, spin),
        lam,
        config.fd_lambda,
    )
    projected = u.conj().T @ dhl @ u
    _, mass_vec = pauli_decompose(projected)
    # Diagonalize the λ derivative to define a reproducible local mass axis.
    _, rotation = np.linalg.eigh(projected)
    u = u @ rotation
    return eig, u, mass_vec


def projected_derivative(
    u: np.ndarray,
    derivative: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    p = u.conj().T @ derivative @ u
    d0, d = pauli_decompose(p)
    return d0, d, p


def parameter_derivative_matrix(
    pair: pd.Series,
    lam: float,
    kx: float,
    ky: float,
    spin: str,
    parameter: str,
    step: float,
) -> np.ndarray:
    reduced = reduced_on_path(pair, lam)
    plus = dict(reduced)
    minus = dict(reduced)
    plus[parameter] += step
    minus[parameter] -= step
    hp = core.h_spin_block_periodic(
        kx, ky, core.raw8_from_reduced7(plus, e0=0.0), spin
    )
    hm = core.h_spin_block_periodic(
        kx, ky, core.raw8_from_reduced7(minus, e0=0.0), spin
    )
    return (hp - hm) / (2.0 * step)


def kp_analysis_one_valley(
    transition_id: str,
    closure_id: str,
    valley_id: str,
    pair: pd.Series,
    lam: float,
    kx: float,
    ky: float,
    spin: str,
    config: Step05Config,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    eig, u, _ = crossing_basis(pair, lam, kx, ky, spin, config)

    hk = config.fd_k
    hl = config.fd_lambda
    dh_kx = central_derivative(lambda xx: spin_block(xx, ky, pair, lam, spin), kx, hk)
    dh_ky = central_derivative(lambda yy: spin_block(kx, yy, pair, lam, spin), ky, hk)
    dh_lam = central_derivative(lambda ll: spin_block(kx, ky, pair, ll, spin), lam, hl)

    derivative_rows: list[dict[str, Any]] = []
    vectors: dict[str, np.ndarray] = {}
    d0s: dict[str, float] = {}
    for coordinate, derivative in (("kx", dh_kx), ("ky", dh_ky), ("lambda", dh_lam)):
        d0, d, projected = projected_derivative(u, derivative)
        vectors[coordinate] = d
        d0s[coordinate] = d0
        derivative_rows.append({
            "transition_id": transition_id,
            "closure_id": closure_id,
            "valley_id": valley_id,
            "spin": spin,
            "coordinate": coordinate,
            "d0": d0,
            "dx": float(d[0]),
            "dy": float(d[1]),
            "dz": float(d[2]),
            "projected_00_real": float(projected[0, 0].real),
            "projected_11_real": float(projected[1, 1].real),
            "projected_01_real": float(projected[0, 1].real),
            "projected_01_imag": float(projected[0, 1].imag),
        })

    jacobian = np.column_stack([vectors["kx"], vectors["ky"], vectors["lambda"]])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    det = float(np.linalg.det(jacobian))
    linear_charge = int(-np.sign(det)) if min(singular) > config.jacobian_singular_tol else 0

    # Diagonal coordinates are useful for Σ/Σ' valleys.
    d_parallel_sigma = (vectors["kx"] + vectors["ky"]) / math.sqrt(2.0)
    d_perp_sigma = (vectors["kx"] - vectors["ky"]) / math.sqrt(2.0)
    d_parallel_sigmap = (vectors["kx"] - vectors["ky"]) / math.sqrt(2.0)
    d_perp_sigmap = (vectors["kx"] + vectors["ky"]) / math.sqrt(2.0)

    gradient_rows: list[dict[str, Any]] = []
    mass_gradient = []
    for parameter in REDUCED7:
        derivative = parameter_derivative_matrix(
            pair, lam, kx, ky, spin, parameter, config.fd_parameter
        )
        d0, d, _ = projected_derivative(u, derivative)
        # The basis diagonalizes dH/dλ, so dz is the local mass-direction component.
        mass_gradient.append(float(d[2]))
        gradient_rows.append({
            "transition_id": transition_id,
            "closure_id": closure_id,
            "valley_id": valley_id,
            "spin": spin,
            "parameter": parameter,
            "mass_gradient": float(d[2]),
            "d0_gradient": float(d0),
            "dx_gradient": float(d[0]),
            "dy_gradient": float(d[1]),
            "dz_gradient": float(d[2]),
        })
    grad_norm = float(np.linalg.norm(mass_gradient))
    if grad_norm > 0:
        for row in gradient_rows:
            row["normalized_mass_gradient"] = float(row["mass_gradient"] / grad_norm)
    else:
        for row in gradient_rows:
            row["normalized_mass_gradient"] = np.nan

    summary = {
        "transition_id": transition_id,
        "closure_id": closure_id,
        "valley_id": valley_id,
        "spin": spin,
        "critical_lambda": float(lam),
        "critical_kx": float(kx),
        "critical_ky": float(ky),
        "crossing_energy_lower": float(eig[1]),
        "crossing_energy_upper": float(eig[2]),
        "residual_spin_gap": float(eig[2] - eig[1]),
        "jacobian_det_kx_ky_lambda": det,
        "jacobian_singular_1": float(singular[0]),
        "jacobian_singular_2": float(singular[1]),
        "jacobian_singular_3": float(singular[2]),
        "jacobian_rank": int(np.sum(singular > config.jacobian_singular_tol)),
        "linear_kp_predicted_charge": linear_charge,
        "norm_d_kx": float(np.linalg.norm(vectors["kx"])),
        "norm_d_ky": float(np.linalg.norm(vectors["ky"])),
        "norm_d_lambda": float(np.linalg.norm(vectors["lambda"])),
        "norm_d_parallel_Sigma": float(np.linalg.norm(d_parallel_sigma)),
        "norm_d_perp_Sigma": float(np.linalg.norm(d_perp_sigma)),
        "norm_d_parallel_SigmaPrime": float(np.linalg.norm(d_parallel_sigmap)),
        "norm_d_perp_SigmaPrime": float(np.linalg.norm(d_perp_sigmap)),
        **step4.classify_k_region(kx, ky, 0.08),
    }
    return summary, derivative_rows, gradient_rows


# =============================================================================
# Non-Abelian Berry flux on an icosphere
# =============================================================================


def make_icosphere(subdivision: int) -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = ConvexHull(vertices).simplices.tolist()

    def orient(face: Sequence[int], verts: np.ndarray) -> list[int]:
        i, j, k = [int(x) for x in face]
        a, b, c = verts[[i, j, k]]
        if float(np.dot(np.cross(b - a, c - a), (a + b + c) / 3.0)) < 0:
            return [i, k, j]
        return [i, j, k]

    faces = [orient(face, vertices) for face in faces]
    verts_list = [v.copy() for v in vertices]
    for _ in range(int(subdivision)):
        cache: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []

        def midpoint(i: int, j: int) -> int:
            key = tuple(sorted((int(i), int(j))))
            if key in cache:
                return cache[key]
            p = verts_list[i] + verts_list[j]
            p = p / np.linalg.norm(p)
            idx = len(verts_list)
            verts_list.append(p)
            cache[key] = idx
            return idx

        for i, j, k in faces:
            a = midpoint(i, j)
            b = midpoint(j, k)
            c = midpoint(k, i)
            new_faces.extend([[i, a, c], [a, j, b], [c, b, k], [a, b, c]])
        vertices_now = np.array(verts_list, dtype=float)
        faces = [orient(face, vertices_now) for face in new_faces]
    return np.array(verts_list, dtype=float), np.array(faces, dtype=int)


def occupied_spin_subspace(
    pair: pd.Series,
    lam: float,
    kx: float,
    ky: float,
    spin: str,
) -> np.ndarray:
    h = spin_block(kx, ky, pair, lam, spin)
    _, vec = np.linalg.eigh(h)
    return vec[:, :2]


def determinant_link(u: np.ndarray, v: np.ndarray) -> tuple[complex, float]:
    value = np.linalg.det(u.conj().T @ v)
    amp = float(abs(value))
    if amp <= 1.0e-15:
        raise FloatingPointError("Near-singular determinant link on Berry sphere")
    return value / amp, amp


def berry_sphere_charge(
    pair: pd.Series,
    lam: float,
    kx: float,
    ky: float,
    spin: str,
    k_radius: float,
    lambda_radius: float,
    subdivision: int,
) -> dict[str, float]:
    vertices, faces = make_icosphere(subdivision)
    subspaces: list[np.ndarray] = []
    for nx, ny, nz in vertices:
        ll = float(lam + lambda_radius * nz)
        # Do not wrap k around ±π. The periodic-gauge Hamiltonian is 2π periodic,
        # and an unwrapped local chart avoids an artificial seam at M.
        subspaces.append(occupied_spin_subspace(
            pair,
            ll,
            float(kx + k_radius * nx),
            float(ky + k_radius * ny),
            spin,
        ))

    total_phase = 0.0
    min_link = np.inf
    max_triangle_phase = 0.0
    for i, j, k in faces:
        lij, aij = determinant_link(subspaces[i], subspaces[j])
        ljk, ajk = determinant_link(subspaces[j], subspaces[k])
        lki, aki = determinant_link(subspaces[k], subspaces[i])
        phase = float(np.angle(lij * ljk * lki))
        total_phase += phase
        min_link = min(min_link, aij, ajk, aki)
        max_triangle_phase = max(max_triangle_phase, abs(phase))
    return {
        "berry_sphere_charge": float(total_phase / (2.0 * math.pi)),
        "sphere_min_link": float(min_link),
        "sphere_max_triangle_phase": float(max_triangle_phase),
        "sphere_vertex_count": int(len(vertices)),
        "sphere_face_count": int(len(faces)),
    }


def automatic_lambda_radius(kp_row: dict[str, Any], k_radius: float, lam: float, config: Step05Config) -> float:
    dk = max(float(kp_row["norm_d_kx"]), float(kp_row["norm_d_ky"]), 1.0e-12)
    dl = max(float(kp_row["norm_d_lambda"]), 1.0e-12)
    radius = float(k_radius * dk / dl)
    boundary_cap = max(1.0e-5, 0.35 * min(lam, 1.0 - lam))
    return float(np.clip(
        radius,
        config.sphere_lambda_radius_min,
        min(config.sphere_lambda_radius_max, boundary_cap),
    ))


def berry_charge_attempts_one_valley(
    kp_row: dict[str, Any],
    pair: pd.Series,
    config: Step05Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    lam = float(kp_row["critical_lambda"])
    for k_radius in config.sphere_k_radii:
        lambda_radius = automatic_lambda_radius(kp_row, float(k_radius), lam, config)
        result = berry_sphere_charge(
            pair,
            lam,
            float(kp_row["critical_kx"]),
            float(kp_row["critical_ky"]),
            str(kp_row["spin"]),
            float(k_radius),
            lambda_radius,
            int(config.sphere_subdivision),
        )
        integer = rounded_integer(result["berry_sphere_charge"], config.sphere_integer_tol)
        attempts.append({
            "transition_id": kp_row["transition_id"],
            "closure_id": kp_row["closure_id"],
            "valley_id": kp_row["valley_id"],
            "spin": kp_row["spin"],
            "k_radius": float(k_radius),
            "lambda_radius": lambda_radius,
            **result,
            "berry_charge_int": np.nan if integer is None else int(integer),
            "attempt_reliable": int(
                integer is not None and result["sphere_min_link"] > config.sphere_min_link_tol
            ),
        })

    reliable = [r for r in attempts if int(r["attempt_reliable"]) == 1]
    integers = [int(r["berry_charge_int"]) for r in reliable]
    consensus = bool(len(reliable) == len(attempts) and len(set(integers)) == 1)
    summary = {
        "transition_id": kp_row["transition_id"],
        "closure_id": kp_row["closure_id"],
        "valley_id": kp_row["valley_id"],
        "spin": kp_row["spin"],
        "berry_charge_consensus": int(consensus),
        "berry_charge_int": int(integers[0]) if consensus else np.nan,
        "berry_charge_attempt_count": int(len(attempts)),
        "berry_charge_reliable_count": int(len(reliable)),
        "berry_charge_unique_integer_count": int(len(set(integers))),
        "berry_charge_max_abs_residual": float(max(
            abs(float(r["berry_sphere_charge"]) - float(r["berry_charge_int"]))
            for r in reliable
        )) if reliable else np.nan,
        "berry_sphere_min_link_over_attempts": float(min(
            r["sphere_min_link"] for r in attempts
        )),
    }
    return attempts, summary


# =============================================================================
# Chern checks on both sides of the corrected transition
# =============================================================================


def side_chern_checks(
    transition_id: str,
    pair: pd.Series,
    bracket: pd.Series,
    critical_lambda: float,
    config: Step05Config,
) -> list[dict[str, Any]]:
    lo, hi = sorted([float(bracket["lambda_left"]), float(bracket["lambda_right"])])
    # The Step 04 bracket endpoints are themselves strict, reliable Chern points.
    # Using them avoids a coarse Chern mesh being asked to resolve an extremely
    # small gap immediately next to the critical lambda.
    lambdas = {"left": lo, "right": hi}
    rows: list[dict[str, Any]] = []
    for side, lam in lambdas.items():
        raw = raw_on_path(pair, lam)
        for nk in config.side_chern_grids:
            for shift in config.side_chern_shifts:
                base = {
                    "transition_id": transition_id,
                    "side": side,
                    "lambda": float(lam),
                    "chern_nk": int(nk),
                    "shift_x": float(shift[0]),
                    "shift_y": float(shift[1]),
                    "attempt_ok": 0,
                    "error": "",
                }
                try:
                    up = core.fukui_chern_subspace(
                        lambda kx, ky: core.h_spin_block_periodic(kx, ky, raw, "up"),
                        core.N_OCC_SPIN, int(nk), shift,
                    )
                    down = core.fukui_chern_subspace(
                        lambda kx, ky: core.h_spin_block_periodic(kx, ky, raw, "down"),
                        core.N_OCC_SPIN, int(nk), shift,
                    )
                    cu = float(up["chern"])
                    cd = float(down["chern"])
                    base.update({
                        "attempt_ok": 1,
                        "chern_up": cu,
                        "chern_down": cd,
                        "chern_up_int": rounded_integer(cu, config.chern_integer_tol),
                        "chern_down_int": rounded_integer(cd, config.chern_integer_tol),
                        "min_det_up": float(up["min_det_amp"]),
                        "min_det_down": float(down["min_det_amp"]),
                    })
                except Exception as exc:
                    base["error"] = repr(exc)
                rows.append(base)
    return rows


# =============================================================================
# Summaries and plots
# =============================================================================


def normalized_mass_formula(group: pd.DataFrame) -> str:
    g = group.sort_values("parameter")
    terms = []
    for _, row in g.iterrows():
        coefficient = float(row["normalized_mass_gradient"])
        if abs(coefficient) >= 0.08:
            terms.append(f"{coefficient:+.3f}*d{row['parameter']}")
    return " ".join(terms) if terms else "all coefficients < 0.08"


def mass_gradient_similarity(gradients: pd.DataFrame) -> pd.DataFrame:
    if gradients.empty:
        return pd.DataFrame()
    pivot = gradients.pivot_table(
        index=["transition_id", "closure_id", "valley_id", "spin"],
        columns="parameter",
        values="normalized_mass_gradient",
        aggfunc="first",
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for i in range(len(pivot)):
        for j in range(i + 1, len(pivot)):
            a = pivot.iloc[i]
            b = pivot.iloc[j]
            va = np.array([a.get(p, np.nan) for p in REDUCED7], dtype=float)
            vb = np.array([b.get(p, np.nan) for p in REDUCED7], dtype=float)
            if not np.isfinite(va).all() or not np.isfinite(vb).all():
                continue
            rows.append({
                "transition_id_a": a["transition_id"],
                "valley_id_a": a["valley_id"],
                "spin_a": a["spin"],
                "transition_id_b": b["transition_id"],
                "valley_id_b": b["valley_id"],
                "spin_b": b["spin"],
                "cosine_similarity": float(np.dot(va, vb)),
                "absolute_cosine_similarity": float(abs(np.dot(va, vb))),
            })
    return pd.DataFrame(rows)


def plot_corrected_valleys(valleys: pd.DataFrame, path: Path) -> None:
    if valleys.empty:
        return
    active = valleys[valleys["is_active_valley"].astype(int) == 1]
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for spin, group in active.groupby("spin"):
        ax.scatter(group["kx"], group["ky"], s=55, label=spin, alpha=0.8)
    ax.axline((-math.pi, -math.pi), (math.pi, math.pi), linewidth=0.8, linestyle="--")
    ax.axline((-math.pi, math.pi), (math.pi, -math.pi), linewidth=0.8, linestyle="--")
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title("Spin-resolved critical valleys")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_charge_certificate(certificate: pd.DataFrame, path: Path) -> None:
    if certificate.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    x = certificate["observed_delta_chern_up"].to_numpy(float)
    y = certificate["berry_charge_sum_up"].to_numpy(float)
    ax.scatter(x, y, s=65)
    low = min(-2.5, float(np.nanmin([x, y])))
    high = max(2.5, float(np.nanmax([x, y])))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.0)
    for _, row in certificate.iterrows():
        ax.annotate(str(row["transition_id"]).split("__transition")[0],
                    (row["observed_delta_chern_up"], row["berry_charge_sum_up"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"Observed $\Delta C_\uparrow$")
    ax.set_ylabel("Sum of spin-up valley charges")
    ax.set_title("Valley-charge certificate")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mass_gradients(gradients: pd.DataFrame, path: Path) -> None:
    if gradients.empty:
        return
    up = gradients[gradients["spin"] == "up"].copy()
    labels = up[["transition_id", "valley_id"]].drop_duplicates()
    keys = [(r.transition_id, r.valley_id) for r in labels.itertuples(index=False)]
    matrix = []
    for transition_id, valley_id in keys:
        group = up[(up["transition_id"] == transition_id) & (up["valley_id"] == valley_id)]
        mapping = dict(zip(group["parameter"], group["normalized_mass_gradient"]))
        matrix.append([mapping.get(p, np.nan) for p in REDUCED7])
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.35 * len(keys) + 1.8)))
    image = ax.imshow(arr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(REDUCED7)), REDUCED7, rotation=45, ha="right")
    ax.set_yticks(range(len(keys)), [f"{a.split('__transition')[0]} | {b}" for a, b in keys], fontsize=7)
    ax.set_title("Normalized local mass gradients")
    fig.colorbar(image, ax=ax, label="normalized coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================


def run_step05(config: Step05Config | None = None) -> Dict[str, Any]:
    if config is None:
        config = Step05Config()
    config = config.normalized()
    started = time.time()

    tables = load_step4_tables(config)
    partners = tables["partners"]
    brackets = tables["brackets"].copy()
    step4_critical = tables["step4_critical"].copy()
    partner_map = {str(row["path_id"]): row for _, row in partners.iterrows()}

    print("[1/8] Reload Step 04 strict transition brackets")
    print(f"      transitions = {len(brackets)}")

    candidate_checkpoint = config.output_dir / "step05_01_spin_gap_search_candidates_checkpoint.csv"
    closure_checkpoint = config.output_dir / "step05_01_corrected_spin_critical_points_checkpoint.csv"
    progress_checkpoint = config.output_dir / "step05_01_search_progress_checkpoint.json"

    candidate_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    completed_transitions: set[str] = set()

    if not config.force_recalculate and progress_checkpoint.exists():
        try:
            progress = json.loads(progress_checkpoint.read_text(encoding="utf-8"))
            if str(progress.get("code_version")) == CODE_VERSION:
                completed_transitions = {str(x) for x in progress.get("completed_transition_ids", [])}
                if candidate_checkpoint.exists():
                    candidate_rows = pd.read_csv(candidate_checkpoint).to_dict(orient="records")
                if closure_checkpoint.exists():
                    closure_rows = pd.read_csv(closure_checkpoint).to_dict(orient="records")
                print(
                    "      resume checkpoint: "
                    f"{len(completed_transitions)}/{len(brackets)} transitions already completed"
                )
        except Exception as exc:
            print(f"      warning: ignored unreadable Step 05 search checkpoint: {exc}")
            candidate_rows = []
            closure_rows = []
            completed_transitions = set()

    print("[2/8] Search spin-up internal gap closings")
    for index, (_, bracket) in enumerate(brackets.iterrows(), start=1):
        transition_id = str(bracket["transition_id"])
        if transition_id in completed_transitions:
            n_existing = sum(str(r.get("transition_id")) == transition_id for r in closure_rows)
            print(
                f"      {index}/{len(brackets)} {transition_id}: "
                f"resume skip, accepted spin closures = {n_existing}"
            )
            continue

        path_id = str(bracket["path_id"])
        pair = partner_map[path_id]
        candidates, closures = search_transition_closures(bracket, pair, config, index)
        step4_match = step4_critical[step4_critical["transition_id"].astype(str) == str(bracket["transition_id"])]
        old_region = str(step4_match.iloc[0]["critical_k_region"]) if not step4_match.empty else "missing"
        old_gap = float(step4_match.iloc[0]["critical_direct_gap"]) if not step4_match.empty else np.nan

        for row in candidates:
            candidate_rows.append({
                **bracket.to_dict(),
                "step4_critical_region": old_region,
                "step4_full_direct_gap": old_gap,
                **row,
            })
        for closure_index, row in enumerate(closures):
            closure_id = f"{bracket['transition_id']}__spinclosure{closure_index:02d}"
            classified = step4.classify_k_region(
                float(row["critical_kx"]), float(row["critical_ky"]), 0.08
            )
            closure = {
                **bracket.to_dict(),
                "closure_id": closure_id,
                "step4_critical_region": old_region,
                "step4_full_direct_gap": old_gap,
                **row,
                **classified,
                "step4_region_matches_spin_closure": int(old_region == classified["critical_k_region"]),
                **reduced_on_path(pair, float(row["critical_lambda"])),
            }
            closure_rows.append(closure)
        completed_transitions.add(transition_id)
        atomic_write_csv(pd.DataFrame(candidate_rows), candidate_checkpoint)
        atomic_write_csv(pd.DataFrame(closure_rows), closure_checkpoint)
        atomic_write_json(
            {
                "code_version": CODE_VERSION,
                "completed_transition_ids": sorted(completed_transitions),
                "updated_at_unix": float(time.time()),
            },
            progress_checkpoint,
        )
        print(
            f"      {index}/{len(brackets)} {bracket['transition_id']}: "
            f"accepted spin closures = {len(closures)}"
        )

    candidate_df = pd.DataFrame(candidate_rows)
    closure_df = pd.DataFrame(closure_rows)

    # Reconstruct all closure objects, including rows restored from a checkpoint.
    bracket_lookup = {str(r["transition_id"]): r for _, r in brackets.iterrows()}
    closure_objects: list[tuple[dict[str, Any], pd.Series, pd.Series]] = []
    for closure in closure_rows:
        transition_id = str(closure["transition_id"])
        bracket = bracket_lookup[transition_id]
        pair = partner_map[str(bracket["path_id"])]
        closure_objects.append((closure, bracket, pair))
    atomic_write_csv(candidate_df, config.output_dir / "step05_01_spin_gap_search_candidates.csv")
    atomic_write_csv(closure_df, config.output_dir / "step05_01_corrected_spin_critical_points.csv")
    if closure_df.empty:
        raise RuntimeError("No spin-resolved critical points were found.")

    print("[3/8] Assign D4-related valleys to spin-up and spin-down blocks")
    valley_frames: list[pd.DataFrame] = []
    closure_lookup = {c[0]["closure_id"]: c for c in closure_objects}
    for closure, _, pair in closure_objects:
        df = enumerate_spin_valleys(closure, pair, config)
        df.insert(0, "closure_id", closure["closure_id"])
        df.insert(0, "transition_id", closure["transition_id"])
        df.insert(2, "observed_delta_chern_up", closure["delta_chern_up"])
        valley_frames.append(df)
    valleys = pd.concat(valley_frames, ignore_index=True)
    valleys["valley_id"] = [
        f"{row.closure_id}__{row.spin}_v{int(row.orbit_index):02d}"
        for row in valleys.itertuples(index=False)
    ]
    atomic_write_csv(valleys, config.output_dir / "step05_02_spin_valley_assignment.csv")

    print("[4/8] Build projected two-band k.p Hamiltonians and Jacobians")
    kp_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    active = valleys[valleys["is_active_valley"].astype(int) == 1].copy()
    for count, (_, valley) in enumerate(active.iterrows(), start=1):
        closure, _, pair = closure_lookup[str(valley["closure_id"])]
        summary, derivatives, gradients = kp_analysis_one_valley(
            str(valley["transition_id"]),
            str(valley["closure_id"]),
            str(valley["valley_id"]),
            pair,
            float(closure["critical_lambda"]),
            float(valley["kx"]),
            float(valley["ky"]),
            str(valley["spin"]),
            config,
        )
        summary["observed_delta_chern_up"] = int(closure["delta_chern_up"])
        kp_rows.append(summary)
        derivative_rows.extend(derivatives)
        gradient_rows.extend(gradients)
        print(f"      k.p {count}/{len(active)}")
    kp_df = pd.DataFrame(kp_rows)
    derivatives_df = pd.DataFrame(derivative_rows)
    gradients_df = pd.DataFrame(gradient_rows)
    atomic_write_csv(kp_df, config.output_dir / "step05_03_kp_summary.csv")
    atomic_write_csv(derivatives_df, config.output_dir / "step05_03_kp_derivative_coefficients.csv")
    atomic_write_csv(gradients_df, config.output_dir / "step05_03_local_mass_gradients.csv")

    print("[5/8] Compute non-Abelian Berry charge on small spheres")
    sphere_attempt_rows: list[dict[str, Any]] = []
    charge_rows: list[dict[str, Any]] = []
    kp_lookup = {str(row["valley_id"]): row for row in kp_rows}
    for count, row in enumerate(kp_rows, start=1):
        closure, _, pair = closure_lookup[str(row["closure_id"])]
        attempts, charge = berry_charge_attempts_one_valley(row, pair, config)
        sphere_attempt_rows.extend(attempts)
        charge["linear_kp_predicted_charge"] = row["linear_kp_predicted_charge"]
        charge["kp_and_sphere_charge_match"] = int(
            int(charge["berry_charge_consensus"]) == 1
            and int(charge["berry_charge_int"]) == int(row["linear_kp_predicted_charge"])
        )
        charge_rows.append(charge)
        print(f"      Berry sphere {count}/{len(kp_rows)}")
    sphere_attempts = pd.DataFrame(sphere_attempt_rows)
    charges = pd.DataFrame(charge_rows)
    atomic_write_csv(sphere_attempts, config.output_dir / "step05_04_berry_sphere_attempts.csv")
    atomic_write_csv(charges, config.output_dir / "step05_04_valley_topological_charges.csv")

    print("[6/8] Verify side Chern numbers and valley-charge sum rule")
    side_rows: list[dict[str, Any]] = []
    # One corrected closure orbit is expected per Step 04 bracket in the present data.
    # The code still supports multiple closure orbits and sums all charges.
    for closure, bracket, pair in closure_objects:
        side_rows.extend(side_chern_checks(
            str(closure["transition_id"]),
            pair,
            bracket,
            float(closure["critical_lambda"]),
            config,
        ))
    side_df = pd.DataFrame(side_rows)
    atomic_write_csv(side_df, config.output_dir / "step05_05_side_chern_checks.csv")

    charge_with_spin = charges.merge(
        kp_df[["valley_id", "spin", "critical_k_region"]],
        on=["valley_id", "spin"],
        how="left",
    )
    certificate_rows: list[dict[str, Any]] = []
    for transition_id, group in charge_with_spin.groupby("transition_id", sort=False):
        bracket = brackets[brackets["transition_id"].astype(str) == str(transition_id)].iloc[0]
        up = group[group["spin"] == "up"]
        down = group[group["spin"] == "down"]
        up_valid = up[up["berry_charge_consensus"].astype(int) == 1]
        down_valid = down[down["berry_charge_consensus"].astype(int) == 1]
        sum_up = float(up_valid["berry_charge_int"].sum())
        sum_down = float(down_valid["berry_charge_int"].sum())
        expected = int(bracket["delta_chern_up"])
        certificate_rows.append({
            "transition_id": transition_id,
            "chern_left": int(bracket["chern_left"]),
            "chern_right": int(bracket["chern_right"]),
            "observed_delta_chern_up": expected,
            "n_active_spin_up_valleys": int(len(up)),
            "n_active_spin_down_valleys": int(len(down)),
            "n_reliable_spin_up_charges": int(len(up_valid)),
            "n_reliable_spin_down_charges": int(len(down_valid)),
            "berry_charge_sum_up": sum_up,
            "berry_charge_sum_down": sum_down,
            "up_charge_matches_delta_chern": int(abs(sum_up - expected) < 0.1),
            "down_charge_matches_opposite_delta": int(abs(sum_down + expected) < 0.1),
            "spin_charge_sum_zero": int(abs(sum_up + sum_down) < 0.1),
            "all_valley_charges_integer_consensus": int(
                len(up_valid) == len(up) and len(down_valid) == len(down)
            ),
            "all_kp_jacobians_match_sphere": int(
                (group["kp_and_sphere_charge_match"].astype(int) == 1).all()
            ),
            "critical_regions_up": ";".join(sorted(set(up["critical_k_region"].astype(str)))),
            "mechanism_certificate_pass": int(
                abs(sum_up - expected) < 0.1
                and abs(sum_down + expected) < 0.1
                and len(up_valid) == len(up)
                and len(down_valid) == len(down)
            ),
        })
    certificate = pd.DataFrame(certificate_rows)
    atomic_write_csv(certificate, config.output_dir / "step05_05_valley_charge_certificate.csv")

    print("[7/8] Summarize local mass formulas")
    formula_rows: list[dict[str, Any]] = []
    for keys, group in gradients_df.groupby(
        ["transition_id", "closure_id", "valley_id", "spin"], sort=False
    ):
        formula_rows.append({
            "transition_id": keys[0],
            "closure_id": keys[1],
            "valley_id": keys[2],
            "spin": keys[3],
            "normalized_local_mass_formula": normalized_mass_formula(group),
        })
    formulas = pd.DataFrame(formula_rows)
    similarities = mass_gradient_similarity(gradients_df)
    atomic_write_csv(formulas, config.output_dir / "step05_06_local_mass_formulas.csv")
    atomic_write_csv(similarities, config.output_dir / "step05_06_mass_gradient_similarity.csv")

    plot_corrected_valleys(
        valleys,
        config.output_dir / "figures" / "step05_spin_resolved_critical_valleys.png",
    )
    plot_charge_certificate(
        certificate,
        config.output_dir / "figures" / "step05_valley_charge_certificate.png",
    )
    plot_mass_gradients(
        gradients_df,
        config.output_dir / "figures" / "step05_local_mass_gradients.png",
    )

    print("[8/8] Save run summary")
    summary = {
        "code_version": CODE_VERSION,
        "step01_version": core.CODE_VERSION,
        "step04_version": step4.CODE_VERSION,
        "step4_input": str(config.step4_input),
        "n_transition_brackets": int(len(brackets)),
        "n_corrected_spin_closure_orbits": int(len(closure_df)),
        "n_active_spin_up_valleys": int(((valleys["spin"] == "up") & (valleys["is_active_valley"] == 1)).sum()),
        "n_active_spin_down_valleys": int(((valleys["spin"] == "down") & (valleys["is_active_valley"] == 1)).sum()),
        "n_kp_valleys": int(len(kp_df)),
        "n_integer_berry_charge_valleys": int(charges["berry_charge_consensus"].astype(int).sum()),
        "n_mechanism_certificates_pass": int(certificate["mechanism_certificate_pass"].astype(int).sum()),
        "n_mechanism_certificates_total": int(len(certificate)),
        "step4_region_corrections": int((closure_df["step4_region_matches_spin_closure"].astype(int) == 0).sum()),
        "critical_region_counts_spin_up": (
            kp_df[kp_df["spin"] == "up"]["critical_k_region"].value_counts().to_dict()
        ),
        "elapsed_seconds": float(time.time() - started),
        "configuration": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "step4_input": str(config.step4_input),
        },
    }
    atomic_write_json(summary, config.output_dir / "step05_07_run_summary.json")
    return {
        "summary": summary,
        "closures": closure_df,
        "valleys": valleys,
        "kp": kp_df,
        "charges": charges,
        "certificate": certificate,
        "formulas": formulas,
    }


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step4-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Reduce search and sphere settings for a smoke test")
    parser.add_argument("--no-full-bz-fallback", action="store_true")
    parser.add_argument("--force-recalculate", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Step05Config:
    config = Step05Config()
    if args.step4_input is not None:
        config.step4_input = args.step4_input
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    config.full_bz_fallback = not bool(args.no_full_bz_fallback)
    config.force_recalculate = bool(args.force_recalculate)
    if args.quick:
        config.output_dir = Path(str(config.output_dir) + "_quick")
        config.coarse_lambda_points = 9
        config.coarse_diagonal_k_points = 51
        config.local_starts_per_manifold = 6
        config.powell_maxiter = 500
        config.full_bz_de_popsize = 10
        config.full_bz_de_maxiter = 160
        config.sphere_subdivision = 1
        config.sphere_k_radii = (0.03, 0.06)
        config.side_chern_grids = (21,)
        config.side_chern_shifts = ((0.0, 0.0),)
    return config.normalized()


def main() -> int:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    results = run_step05(config)
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
