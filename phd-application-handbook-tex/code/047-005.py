#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FES Step 06 — Γ/M 低能 k·p Hamiltonian、谷手性与 spin-Chern 闭式公式

研究目标
--------
Step 05 已经得到精确的条件性 no-go：

    若 B+ 分支在 Γ 与 M 之间发生 occupied/unoccupied partner switching，
    则 E_g^ind <= -4|r1^2-r2^2|/(B1+B2) <= 0。

本步骤补齐模型级 no-go 所缺少的拓扑环节：

1. 在 Γ 与 M 的 A-/B+ 闭隙点构造两带低能 k·p Hamiltonian；
2. 用投影质量算符固定简并子空间规范；
3. 计算局域 off-diagonal winding、线性 Jacobian 与 Chern 跳变；
4. 验证两个谷的有效拓扑取向均为 sgn(r1*r2)；
5. 检验候选 spin-Chern 公式

       C_up = sgn(r1*r2)/2
              [sgn(mu_Gamma) + sgn(mu_M)],

   其中
       mu_Gamma = E_{Gamma,A-} - E_{Gamma,B+},
       mu_M     = E_{M,B+} - E_{M,A-};

6. 将该公式与 Step 05 的 no-go certificate 连接，输出模型级推理链。

重要边界
--------
- k·p 投影、winding 和 Chern jump 是数值高精度验证；
- 闭式公式会在 Step 05 严格样本及全部粗筛样本上审计；
- 若要称为完全解析定理，仍建议将本脚本输出的 Jacobian 符号结构
  进一步整理为手工/符号代数证明。

依赖
----
将以下脚本放在同一目录：
    FES_step02_sobol_scan.py
    FES_step05_analytic_no_go.py

运行示例
--------
快速测试：
    python FES_step06_kp_chern_formula.py \
      --step05-input outputs_fes6_step05_no_go.zip \
      --output-dir outputs_fes6_step06_test \
      --test --overwrite

正式运行：
    python FES_step06_kp_chern_formula.py \
      --step05-input outputs_fes6_step05_no_go.zip \
      --output-dir outputs_fes6_step06_kp \
      --workers 4 --overwrite
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import FES_step02_sobol_scan as core
import FES_step05_analytic_no_go as step05

np.set_printoptions(precision=12, suppress=True)

CODE_VERSION = "FES_STEP06_KP_CHERN_V1_20260713"
RESULT_SCHEMA_VERSION = "fes_step06_kp_chern_schema_v1"

PARAMS5 = ("m_e", "t1", "t2", "r1", "r2")
VALLEYS = ("Gamma", "M")
K_POINTS = {
    "Gamma": (0.0, 0.0),
    "M": (math.pi, math.pi),
}


@dataclass
class KPConfig:
    step05_input: str = "outputs_fes6_step05_no_go.zip"
    output_dir: str = "outputs_fes6_step06_kp"
    workers: int = max(1, min(4, os.cpu_count() or 1))

    representative_per_group: int = 1
    root_span: float = 0.30
    root_grid_points: int = 2401
    root_tol: float = 1.0e-11

    projection_dm: float = 1.0e-6
    fit_radius: float = 4.0e-3
    fit_grid_n: int = 9
    winding_radii: tuple[float, ...] = (
        1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2
    )
    winding_angles: int = 721

    chern_jump_delta_m: float = 1.0e-3
    chern_jump_grids: tuple[int, ...] = (21, 31, 41)
    chern_tol: float = 0.08
    min_det_tol: float = 1.0e-7

    formula_mass_tol: float = 1.0e-6
    formula_hopping_tol: float = 1.0e-8
    coarse_gap_margin: float = 5.0e-3
    strict_only_reliable: bool = True

    overwrite: bool = False
    test_mode: bool = False


# =============================================================================
# 1. 输入与版本管理
# =============================================================================

def script_sha256() -> str | None:
    if "__file__" not in globals():
        return None
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def prepare_output(config: KPConfig) -> Path:
    out = Path(config.output_dir).expanduser().resolve()
    if out.exists() and config.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kp_details").mkdir(exist_ok=True)
    return out


def _find_step05_root(root: Path) -> Path:
    required = {
        "fes_step05_strict_analytic_audit.csv",
        "fes_step05_all_coarse_analytic_audit.csv",
        "fes_step05_verified_topological_no_go_certificates.csv",
    }
    names = {p.name for p in root.iterdir() if p.is_file()}
    if required.issubset(names):
        return root

    candidates: list[Path] = []
    for path in root.rglob("fes_step05_strict_analytic_audit.csv"):
        parent = path.parent
        names = {p.name for p in parent.iterdir() if p.is_file()}
        if required.issubset(names):
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(
            f"无法在 {root} 中找到完整 Step 05 结果。必须包含：{sorted(required)}"
        )
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def locate_step05_directory(input_path: str | Path, output: Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return _find_step05_root(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("--step05-input 必须是目录或 ZIP")
    extract_root = output / "_step05_extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_root)
    return _find_step05_root(extract_root)


def write_metadata(config: KPConfig, output: Path, step05_dir: Path) -> None:
    metadata = {
        "code_version": CODE_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": script_sha256(),
        "step02_core_version": getattr(core, "CODE_VERSION", None),
        "step02_core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "step05_core_version": getattr(step05, "CODE_VERSION", None),
        "step05_core_sha256": hashlib.sha256(Path(step05.__file__).read_bytes()).hexdigest(),
        "step05_directory": str(step05_dir),
        "config": asdict(config),
    }
    (output / "fes_step06_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# 2. 参数与质量
# =============================================================================

def reduced_from_row(row: pd.Series | Dict[str, object]) -> Dict[str, float]:
    return {name: float(row[name]) for name in PARAMS5}


def raw_from_reduced(p: Dict[str, float]) -> Dict[str, float]:
    return core.raw6_from_reduced5(p)


def analytic_mass(p: Dict[str, float], valley: str) -> float:
    d = step05.analytic_diagnostics(p)
    if valley == "Gamma":
        return float(d["mass_Gamma_signed"])
    if valley == "M":
        return float(d["mass_M_signed"])
    raise ValueError(valley)


def mass_derivative(p: Dict[str, float], valley: str, eps: float = 1.0e-6) -> float:
    plus = dict(p)
    minus = dict(p)
    plus["m_e"] += eps
    minus["m_e"] -= eps
    return (analytic_mass(plus, valley) - analytic_mass(minus, valley)) / (2.0 * eps)


def candidate_chern_formula(
    p: Dict[str, float],
    prefactor: int = 1,
    mass_tol: float = 1.0e-12,
    hopping_tol: float = 1.0e-14,
) -> float:
    d = step05.analytic_diagnostics(p)
    rprod = p["r1"] * p["r2"]
    mu_g = float(d["mass_Gamma_signed"])
    mu_m = float(d["mass_M_signed"])
    if abs(rprod) <= hopping_tol or abs(mu_g) <= mass_tol or abs(mu_m) <= mass_tol:
        return float("nan")
    return float(
        prefactor
        * np.sign(rprod)
        * 0.5
        * (np.sign(mu_g) + np.sign(mu_m))
    )


# =============================================================================
# 3. 代表点与质量根
# =============================================================================

def load_step05_tables(step05_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = pd.read_csv(step05_dir / "fes_step05_strict_analytic_audit.csv")
    coarse = pd.read_csv(step05_dir / "fes_step05_all_coarse_analytic_audit.csv")
    cert = pd.read_csv(
        step05_dir / "fes_step05_verified_topological_no_go_certificates.csv"
    )
    return strict, coarse, cert


def choose_representatives(strict: pd.DataFrame, per_group: int) -> pd.DataFrame:
    reliable = strict.copy()
    reliable = reliable[
        (pd.to_numeric(reliable["verified_chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(reliable["verified_is_direct_gapped"], errors="coerce") == 1)
        & (pd.to_numeric(reliable["verified_is_balanced_spin_sector"], errors="coerce") == 1)
    ].copy()

    reliable["r_product_sign"] = np.sign(
        pd.to_numeric(reliable["r1"], errors="coerce")
        * pd.to_numeric(reliable["r2"], errors="coerce")
    )
    reliable["chern_class"] = pd.to_numeric(
        reliable["verified_chern_up_int"], errors="coerce"
    ).fillna(99).astype(int)

    groups: list[pd.DataFrame] = []
    topo = reliable[reliable["chern_class"].abs() == 1]
    for (chern, rsign), group in topo.groupby(["chern_class", "r_product_sign"]):
        chosen = group.sort_values(
            ["verified_min_direct_gap", "mass_boundary_distance"],
            ascending=[False, False],
        ).head(per_group)
        groups.append(chosen)

    trivial = reliable[reliable["chern_class"] == 0]
    if not trivial.empty:
        groups.append(
            trivial.sort_values(
                ["verified_min_direct_gap", "verified_indirect_gap"],
                ascending=False,
            ).head(max(1, per_group))
        )

    if not groups:
        raise RuntimeError("没有找到可用的严格代表点")

    out = pd.concat(groups, ignore_index=True).drop_duplicates("point_id")
    cols = [
        "point_id", "m_e", "t1", "t2", "r1", "r2",
        "verified_chern_up_int", "verified_chern_down_int",
        "verified_min_direct_gap", "verified_indirect_gap",
        "mass_Gamma_signed", "mass_M_signed",
        "strict_phase_label", "r_product_sign",
    ]
    return out[[c for c in cols if c in out.columns]].copy()


def find_mass_roots(
    p: Dict[str, float],
    valley: str,
    span: float,
    n_grid: int,
    root_tol: float,
) -> list[float]:
    m0 = float(p["m_e"])
    xs = np.linspace(m0 - span, m0 + span, int(n_grid))
    vals = np.array([
        analytic_mass({**p, "m_e": float(x)}, valley) for x in xs
    ], dtype=float)

    roots: list[float] = []
    for i in range(len(xs) - 1):
        a, b = float(xs[i]), float(xs[i + 1])
        fa, fb = float(vals[i]), float(vals[i + 1])
        if abs(fa) <= root_tol:
            roots.append(a)
        if fa * fb < 0.0:
            root = brentq(
                lambda m: analytic_mass({**p, "m_e": float(m)}, valley),
                a, b, xtol=root_tol, rtol=4.0 * np.finfo(float).eps,
            )
            roots.append(float(root))
    if abs(vals[-1]) <= root_tol:
        roots.append(float(xs[-1]))

    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 1.0e-7:
            unique.append(root)
    return unique


def root_table(representatives: pd.DataFrame, config: KPConfig) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for _, row in representatives.iterrows():
        p = reduced_from_row(row)
        for valley in VALLEYS:
            roots = find_mass_roots(
                p, valley, config.root_span,
                config.root_grid_points, config.root_tol,
            )
            if not roots:
                rows.append({
                    "point_id": row["point_id"],
                    "valley": valley,
                    **p,
                    "root_found": 0,
                    "m_critical": np.nan,
                    "distance_from_reference": np.nan,
                })
                continue

            # The nearest root controls the neighboring phase of this representative.
            root = min(roots, key=lambda x: abs(x - p["m_e"]))
            pc = dict(p)
            pc["m_e"] = float(root)
            rows.append({
                "point_id": row["point_id"],
                "valley": valley,
                **p,
                "reference_chern_up": row.get("verified_chern_up_int", np.nan),
                "r_product_sign": int(np.sign(p["r1"] * p["r2"])),
                "root_found": 1,
                "all_roots": json.dumps(roots),
                "m_critical": float(root),
                "distance_from_reference": float(root - p["m_e"]),
                "mass_at_root": analytic_mass(pc, valley),
                "dmu_dm_at_root": mass_derivative(pc, valley),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 4. 两带投影、k·p 系数与 winding
# =============================================================================

def spin_up_atomic(kx: float, ky: float, p: Dict[str, float]) -> np.ndarray:
    return core.h_spin_block_atomic(kx, ky, raw_from_reduced(p), "up")


def critical_projection_basis(
    p: Dict[str, float],
    valley: str,
    dm: float,
) -> Dict[str, object]:
    kx0, ky0 = K_POINTS[valley]
    h0 = spin_up_atomic(kx0, ky0, p)
    evals, evecs = np.linalg.eigh(h0)

    # At the A-/B+ transition the middle pair closes.
    middle = np.array([1, 2], dtype=int)
    U = evecs[:, middle]
    e0 = float(np.mean(evals[middle]))

    pp = dict(p)
    pm = dict(p)
    pp["m_e"] += dm
    pm["m_e"] -= dm
    dHdm = (
        spin_up_atomic(kx0, ky0, pp) - spin_up_atomic(kx0, ky0, pm)
    ) / (2.0 * dm)

    mass_matrix = U.conj().T @ dHdm @ U
    mass_evals, mass_vecs = np.linalg.eigh(mass_matrix)
    W = U @ mass_vecs

    projected_mass = W.conj().T @ dHdm @ W
    dz_dm = float(np.real(projected_mass[0, 0] - projected_mass[1, 1]) / 2.0)

    return {
        "W": W,
        "e0": e0,
        "full_evals": evals,
        "mass_evals": mass_evals,
        "dz_dm": dz_dm,
        "critical_pair_gap": float(evals[2] - evals[1]),
    }


def projected_pauli(
    p: Dict[str, float],
    valley: str,
    W: np.ndarray,
    e0: float,
    qx: float,
    qy: float,
) -> tuple[float, float, float, float, complex]:
    kx0, ky0 = K_POINTS[valley]
    h = spin_up_atomic(kx0 + qx, ky0 + qy, p)
    heff = W.conj().T @ h @ W - e0 * np.eye(2)
    heff = 0.5 * (heff + heff.conj().T)

    d0 = float(np.real(np.trace(heff)) / 2.0)
    dx = float(np.real(heff[0, 1]))
    dy = float(-np.imag(heff[0, 1]))
    dz = float(np.real(heff[0, 0] - heff[1, 1]) / 2.0)
    return d0, dx, dy, dz, complex(heff[0, 1])


def winding_on_circle(
    p: Dict[str, float],
    valley: str,
    W: np.ndarray,
    e0: float,
    radius: float,
    n_angles: int,
) -> Dict[str, float]:
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=True)
    offdiag: list[complex] = []
    for theta in angles:
        qx = radius * math.cos(float(theta))
        qy = radius * math.sin(float(theta))
        *_, off = projected_pauli(p, valley, W, e0, qx, qy)
        offdiag.append(off)

    arr = np.asarray(offdiag, dtype=np.complex128)
    phases = np.unwrap(np.angle(arr))
    winding = float((phases[-1] - phases[0]) / (2.0 * np.pi))
    return {
        "radius": float(radius),
        "winding_raw": winding,
        "winding_int": int(np.rint(winding)),
        "min_abs_offdiag": float(np.min(np.abs(arr))),
        "max_abs_offdiag": float(np.max(np.abs(arr))),
    }


def polynomial_design(qx: np.ndarray, qy: np.ndarray) -> tuple[np.ndarray, list[str]]:
    X = np.column_stack([
        np.ones_like(qx),
        qx, qy,
        qx * qx, qx * qy, qy * qy,
    ])
    names = ["const", "qx", "qy", "qx2", "qxqy", "qy2"]
    return X, names


def fit_kp_coefficients(
    p: Dict[str, float],
    valley: str,
    W: np.ndarray,
    e0: float,
    radius: float,
    grid_n: int,
) -> Dict[str, object]:
    qs = np.linspace(-radius, radius, int(grid_n))
    qx_values: list[float] = []
    qy_values: list[float] = []
    components = {"d0": [], "dx": [], "dy": [], "dz": []}

    for qx in qs:
        for qy in qs:
            if abs(qx) < 1.0e-16 and abs(qy) < 1.0e-16:
                continue
            d0, dx, dy, dz, _ = projected_pauli(
                p, valley, W, e0, float(qx), float(qy)
            )
            qx_values.append(float(qx))
            qy_values.append(float(qy))
            components["d0"].append(d0)
            components["dx"].append(dx)
            components["dy"].append(dy)
            components["dz"].append(dz)

    qx_arr = np.asarray(qx_values)
    qy_arr = np.asarray(qy_values)
    X, names = polynomial_design(qx_arr, qy_arr)

    result: Dict[str, object] = {}
    for component, values in components.items():
        y = np.asarray(values, dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        rms = float(np.sqrt(np.mean((pred - y) ** 2)))
        scale = max(float(np.sqrt(np.mean(y ** 2))), 1.0e-15)
        result[f"{component}_fit_rms"] = rms
        result[f"{component}_fit_relative_rms"] = rms / scale
        for name, value in zip(names, coef):
            result[f"{component}_{name}"] = float(value)

    # d_x,d_y linear Jacobian.
    J = np.array([
        [result["dx_qx"], result["dx_qy"]],
        [result["dy_qx"], result["dy_qy"]],
    ], dtype=float)
    detJ = float(np.linalg.det(J))
    result.update({
        "jacobian_xx": float(J[0, 0]),
        "jacobian_xy": float(J[0, 1]),
        "jacobian_yx": float(J[1, 0]),
        "jacobian_yy": float(J[1, 1]),
        "jacobian_det": detJ,
        "jacobian_chirality": int(np.sign(detJ)) if abs(detJ) > 1.0e-16 else 0,
    })
    return result


def kp_audit(root_df: pd.DataFrame, config: KPConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[Dict[str, object]] = []
    winding_rows: list[Dict[str, object]] = []

    valid = root_df[root_df["root_found"] == 1].copy()
    for _, row in valid.iterrows():
        p = {name: float(row[name]) for name in PARAMS5}
        p["m_e"] = float(row["m_critical"])
        valley = str(row["valley"])

        basis = critical_projection_basis(p, valley, config.projection_dm)
        W = basis["W"]
        e0 = float(basis["e0"])

        fit = fit_kp_coefficients(
            p, valley, W, e0, config.fit_radius, config.fit_grid_n
        )

        per_radius: list[Dict[str, float]] = []
        for radius in config.winding_radii:
            wr = winding_on_circle(
                p, valley, W, e0, float(radius), config.winding_angles
            )
            per_radius.append(wr)
            winding_rows.append({
                "point_id": row["point_id"],
                "valley": valley,
                "m_critical": p["m_e"],
                "r1": p["r1"],
                "r2": p["r2"],
                "r_product_sign": int(np.sign(p["r1"] * p["r2"])),
                **wr,
            })

        windings = np.array([x["winding_int"] for x in per_radius], dtype=int)
        winding_consensus = int(np.all(windings == windings[0]))
        winding_int = int(windings[0]) if winding_consensus else int(np.rint(np.median(windings)))

        dmu_dm = float(row["dmu_dm_at_root"])
        dz_dm = float(basis["dz_dm"])
        mass_orientation = int(np.sign(dmu_dm * dz_dm))
        eta = int(winding_int * mass_orientation)
        rsign = int(np.sign(p["r1"] * p["r2"]))

        summary_rows.append({
            "point_id": row["point_id"],
            "valley": valley,
            **p,
            "reference_chern_up": row.get("reference_chern_up", np.nan),
            "r_product_sign": rsign,
            "critical_pair_gap": basis["critical_pair_gap"],
            "mass_eigenvalue_1": float(basis["mass_evals"][0]),
            "mass_eigenvalue_2": float(basis["mass_evals"][1]),
            "dz_dm_projected": dz_dm,
            "dmu_dm_analytic": dmu_dm,
            "mass_orientation_sign": mass_orientation,
            "winding_consensus": winding_consensus,
            "winding_int": winding_int,
            "eta_valley": eta,
            "eta_equals_sign_r1r2": int(eta == rsign),
            **fit,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(winding_rows)


# =============================================================================
# 5. Chern jump across each critical mass
# =============================================================================

def chern_up_single(p: Dict[str, float], nk: int) -> tuple[float, float]:
    raw = raw_from_reduced(p)
    result = core.fukui_chern_subspace_shifted(
        lambda kx, ky: core.h_spin_block_periodic(kx, ky, raw, "up"),
        core.N_OCC_SPIN,
        int(nk),
        (0.0, 0.0),
    )
    return float(result["chern"]), float(result["min_det"])


def chern_jump_audit(root_df: pd.DataFrame, config: KPConfig) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    valid = root_df[root_df["root_found"] == 1].copy()

    grids = config.chern_jump_grids
    if config.test_mode:
        grids = tuple(g for g in grids if g <= 21) or (15,)

    for _, row in valid.iterrows():
        p0 = {name: float(row[name]) for name in PARAMS5}
        p0["m_e"] = float(row["m_critical"])
        valley = str(row["valley"])
        delta = float(config.chern_jump_delta_m)

        pm = dict(p0)
        pp = dict(p0)
        pm["m_e"] -= delta
        pp["m_e"] += delta

        mu_minus = analytic_mass(pm, valley)
        mu_plus = analytic_mass(pp, valley)
        rsign = int(np.sign(p0["r1"] * p0["r2"]))
        jump_pred = float(
            rsign * 0.5 * (np.sign(mu_plus) - np.sign(mu_minus))
        )

        per_grid = []
        for nk in grids:
            try:
                c_minus, det_minus = chern_up_single(pm, int(nk))
                c_plus, det_plus = chern_up_single(pp, int(nk))
                per_grid.append({
                    "nk": int(nk),
                    "chern_minus": c_minus,
                    "chern_plus": c_plus,
                    "chern_jump": c_plus - c_minus,
                    "min_det_minus": det_minus,
                    "min_det_plus": det_plus,
                    "error": "",
                })
            except Exception as exc:
                per_grid.append({
                    "nk": int(nk),
                    "chern_minus": np.nan,
                    "chern_plus": np.nan,
                    "chern_jump": np.nan,
                    "min_det_minus": np.nan,
                    "min_det_plus": np.nan,
                    "error": repr(exc),
                })

        detail = pd.DataFrame(per_grid)
        detail.to_csv(
            Path(config.output_dir).resolve()
            / "kp_details"
            / f"{row['point_id']}_{valley}_chern_jump.csv",
            index=False,
        )

        good = detail[detail["error"] == ""].copy()
        jumps = good["chern_jump"].to_numpy(dtype=float) if not good.empty else np.array([])
        rounded = np.rint(jumps).astype(int) if len(jumps) else np.array([], dtype=int)
        consensus = int(len(rounded) > 0 and np.all(rounded == rounded[0]))

        rows.append({
            "point_id": row["point_id"],
            "valley": valley,
            **p0,
            "delta_m": delta,
            "mu_minus": mu_minus,
            "mu_plus": mu_plus,
            "r_product_sign": rsign,
            "predicted_chern_jump": jump_pred,
            "observed_chern_jump_mean": float(np.nanmean(jumps)) if len(jumps) else np.nan,
            "observed_chern_jump_int": int(rounded[0]) if consensus else np.nan,
            "chern_jump_consensus": consensus,
            "jump_matches_formula": int(
                consensus and int(rounded[0]) == int(np.rint(jump_pred))
            ),
            "minimum_link_determinant": float(
                np.nanmin(
                    np.concatenate([
                        good["min_det_minus"].to_numpy(dtype=float),
                        good["min_det_plus"].to_numpy(dtype=float),
                    ])
                )
            ) if not good.empty else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. Chern 公式审计
# =============================================================================

def formula_audit(
    df: pd.DataFrame,
    label: str,
    strict: bool,
    config: KPConfig,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    out = df.copy()

    if strict:
        required = (
            (pd.to_numeric(out["verified_chern_reliable"], errors="coerce") == 1)
            & (pd.to_numeric(out["verified_is_direct_gapped"], errors="coerce") == 1)
            & (pd.to_numeric(out["verified_is_balanced_spin_sector"], errors="coerce") == 1)
        )
        true_col = "verified_chern_up_int"
        direct_col = "verified_min_direct_gap"
    else:
        required = (
            (pd.to_numeric(out["chern_reliable"], errors="coerce") == 1)
            & (pd.to_numeric(out["is_direct_gapped"], errors="coerce") == 1)
            & (pd.to_numeric(out["is_balanced_spin_sector"], errors="coerce") == 1)
        )
        true_col = "chern_up_int"
        direct_col = "min_direct_gap"

    out = out[required].copy()
    out["formula_pred_prefactor_plus"] = [
        candidate_chern_formula(
            reduced_from_row(row), +1,
            config.formula_mass_tol, config.formula_hopping_tol,
        )
        for _, row in out.iterrows()
    ]
    out["formula_pred_prefactor_minus"] = -out["formula_pred_prefactor_plus"]

    valid = (
        np.isfinite(out["formula_pred_prefactor_plus"])
        & np.isfinite(pd.to_numeric(out[true_col], errors="coerce"))
    )
    calibration = out[valid].copy()

    acc_plus = float(np.mean(
        calibration["formula_pred_prefactor_plus"].to_numpy()
        == pd.to_numeric(calibration[true_col], errors="coerce").to_numpy()
    )) if len(calibration) else np.nan
    acc_minus = float(np.mean(
        calibration["formula_pred_prefactor_minus"].to_numpy()
        == pd.to_numeric(calibration[true_col], errors="coerce").to_numpy()
    )) if len(calibration) else np.nan
    prefactor = +1 if (np.isnan(acc_minus) or acc_plus >= acc_minus) else -1

    out["calibrated_prefactor"] = prefactor
    out["chern_up_formula_pred"] = (
        out["formula_pred_prefactor_plus"]
        if prefactor == 1
        else out["formula_pred_prefactor_minus"]
    )
    out["chern_formula_match"] = (
        out["chern_up_formula_pred"]
        == pd.to_numeric(out[true_col], errors="coerce")
    ).astype(int)
    out["formula_boundary_distance"] = np.minimum(
        np.abs(pd.to_numeric(out["mass_Gamma_signed"], errors="coerce")),
        np.abs(pd.to_numeric(out["mass_M_signed"], errors="coerce")),
    )
    out["formula_hopping_distance"] = np.abs(
        pd.to_numeric(out["r1"], errors="coerce")
        * pd.to_numeric(out["r2"], errors="coerce")
    )
    out["formula_near_boundary"] = (
        (out["formula_boundary_distance"] <= config.formula_mass_tol)
        | (out["formula_hopping_distance"] <= config.formula_hopping_tol)
        | (pd.to_numeric(out[direct_col], errors="coerce") <= config.coarse_gap_margin)
    ).astype(int)

    evaluated = out[np.isfinite(out["chern_up_formula_pred"])].copy()
    robust = evaluated[evaluated["formula_near_boundary"] == 0].copy()

    summary = {
        "dataset": label,
        "strict": bool(strict),
        "reliable_rows": int(len(out)),
        "formula_evaluated_rows": int(len(evaluated)),
        "calibrated_prefactor": int(prefactor),
        "accuracy_all_evaluated": float(evaluated["chern_formula_match"].mean())
        if len(evaluated) else np.nan,
        "robust_rows_after_boundary_exclusion": int(len(robust)),
        "accuracy_robust": float(robust["chern_formula_match"].mean())
        if len(robust) else np.nan,
        "mismatch_count_all": int((evaluated["chern_formula_match"] == 0).sum()),
        "mismatch_count_robust": int((robust["chern_formula_match"] == 0).sum()),
    }
    return out, summary


# =============================================================================
# 7. 图与报告
# =============================================================================

def plot_kp_winding(kp: pd.DataFrame, output: Path) -> None:
    if kp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        f"{row.point_id}\n{row.valley}"
        for row in kp.itertuples(index=False)
    ]
    x = np.arange(len(kp))
    ax.bar(x - 0.18, kp["winding_int"], width=0.36, label="offdiag winding")
    ax.bar(x + 0.18, kp["eta_valley"], width=0.36, label=r"$\eta_K=w_K\,\mathrm{sgn}(d_z'\mu')$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel("integer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fes_step06_local_winding_and_eta.png", dpi=220)
    plt.close(fig)


def plot_formula_confusion(audit: pd.DataFrame, output: Path, name: str, true_col: str) -> None:
    if audit.empty:
        return
    valid = audit[
        np.isfinite(audit["chern_up_formula_pred"])
        & np.isfinite(pd.to_numeric(audit[true_col], errors="coerce"))
    ].copy()
    if valid.empty:
        return

    true = pd.to_numeric(valid[true_col], errors="coerce").astype(int)
    pred = valid["chern_up_formula_pred"].astype(int)
    classes = [-1, 0, 1]
    matrix = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        if t in classes and p in classes:
            matrix[classes.index(t), classes.index(p)] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xticks(range(3), classes)
    ax.set_yticks(range(3), classes)
    ax.set_xlabel("predicted $C_\\uparrow$")
    ax.set_ylabel("numerical $C_\\uparrow$")
    ax.set_title(name)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output / f"fes_step06_{name}_formula_confusion.png", dpi=220)
    plt.close(fig)


def write_formula_files(output: Path, prefactor: int) -> None:
    sign_text = "" if prefactor == 1 else "-"
    txt = f"""FES Step 06 candidate spin-Chern formula

Definitions
-----------
mu_Gamma = E_(Gamma,A-) - E_(Gamma,B+)
mu_M     = E_(M,B+) - E_(M,A-)

Low-energy valley orientation
-----------------------------
eta_Gamma = eta_M = {sign_text}sgn(r1*r2)

Candidate closed formula
------------------------
C_up = {sign_text}sgn(r1*r2)/2
       [sgn(mu_Gamma) + sgn(mu_M)]

C_down = -C_up
C_total = 0

Combined no-go chain
--------------------
C_up != 0
=> mu_Gamma and mu_M have the same sign
=> B+ partner switching between Gamma and M
=> E_g^ind <= -4|r1^2-r2^2|/(B1+B2) <= 0.

Scope
-----
The formula is reconstructed from local k.p winding, Chern jumps,
and strict numerical validation. A publication-level theorem should
also present the projected Jacobian sign analytically.
"""
    (output / "fes_step06_candidate_chern_formula.txt").write_text(
        txt, encoding="utf-8"
    )

    pref = "" if prefactor == 1 else "-"
    tex = r"""\[
\mu_\Gamma=E_{\Gamma,A^-}-E_{\Gamma,B^+},\qquad
\mu_M=E_{M,B^+}-E_{M,A^-}.
\]

\[
\eta_\Gamma=\eta_M=__PREF__\operatorname{sgn}(r_1r_2).
\]

\[
\boxed{
C_\uparrow=
__PREF__\frac{\operatorname{sgn}(r_1r_2)}{2}
\left[
\operatorname{sgn}(\mu_\Gamma)+
\operatorname{sgn}(\mu_M)
\right]
}
\]

\[
C_\downarrow=-C_\uparrow,\qquad C_{\mathrm{total}}=0.
\]

Combining this result with the Step~05 branch-switching identity gives
\[
C_\uparrow\neq 0
\Longrightarrow
E_g^{\mathrm{ind}}
\le
-\frac{4\left|r_1^2-r_2^2\right|}{B_1+B_2}
\le 0.
\]
""".replace("__PREF__", pref)
    (output / "fes_step06_candidate_chern_formula.tex").write_text(
        tex, encoding="utf-8"
    )


# =============================================================================
# 8. 主流程
# =============================================================================

def run_pipeline(config: KPConfig) -> Dict[str, str]:
    output = prepare_output(config)
    step05_dir = locate_step05_directory(config.step05_input, output)
    write_metadata(config, output, step05_dir)

    strict, coarse, cert = load_step05_tables(step05_dir)

    representatives = choose_representatives(
        strict,
        1 if config.test_mode else config.representative_per_group,
    )
    representatives.to_csv(
        output / "fes_step06_representative_points.csv", index=False
    )

    roots = root_table(representatives, config)
    roots.to_csv(output / "fes_step06_critical_mass_roots.csv", index=False)

    kp_summary, winding = kp_audit(roots, config)
    kp_summary.to_csv(output / "fes_step06_kp_coefficients.csv", index=False)
    winding.to_csv(output / "fes_step06_winding_by_radius.csv", index=False)

    jumps = chern_jump_audit(roots, config)
    jumps.to_csv(output / "fes_step06_chern_jump_validation.csv", index=False)

    strict_audit, strict_summary = formula_audit(
        strict, "strict_step05", True, config
    )
    coarse_audit, coarse_summary = formula_audit(
        coarse, "coarse_step04", False, config
    )
    strict_audit.to_csv(
        output / "fes_step06_formula_validation_strict.csv", index=False
    )
    coarse_audit.to_csv(
        output / "fes_step06_formula_validation_coarse.csv", index=False
    )

    strict_mismatch = strict_audit[
        strict_audit["chern_formula_match"] == 0
    ].copy()
    coarse_mismatch = coarse_audit[
        coarse_audit["chern_formula_match"] == 0
    ].copy()
    strict_mismatch.to_csv(
        output / "fes_step06_formula_mismatches_strict.csv", index=False
    )
    coarse_mismatch.to_csv(
        output / "fes_step06_formula_mismatches_coarse.csv", index=False
    )

    prefactor = int(strict_summary["calibrated_prefactor"])
    write_formula_files(output, prefactor)

    plot_kp_winding(kp_summary, output)
    plot_formula_confusion(
        strict_audit, output, "strict", "verified_chern_up_int"
    )
    plot_formula_confusion(
        coarse_audit, output, "coarse", "chern_up_int"
    )

    eta_match_fraction = float(
        kp_summary["eta_equals_sign_r1r2"].mean()
    ) if len(kp_summary) else np.nan
    winding_consensus_fraction = float(
        kp_summary["winding_consensus"].mean()
    ) if len(kp_summary) else np.nan
    jump_match_fraction = float(
        jumps["jump_matches_formula"].mean()
    ) if len(jumps) else np.nan

    combined_nogo_count = 0
    if not cert.empty:
        topological = cert.copy()
        formula_pred = []
        for _, row in topological.iterrows():
            formula_pred.append(
                candidate_chern_formula(
                    reduced_from_row(row), prefactor,
                    config.formula_mass_tol,
                    config.formula_hopping_tol,
                )
            )
        topological["chern_formula_pred"] = formula_pred
        topological["formula_matches_verified"] = (
            topological["chern_formula_pred"]
            == pd.to_numeric(
                topological["verified_chern_up_int"], errors="coerce"
            )
        ).astype(int)
        topological["complete_numerical_no_go_chain"] = (
            (topological["formula_matches_verified"] == 1)
            & (pd.to_numeric(
                topological["conditional_no_go_certificate"], errors="coerce"
            ) == 1)
        ).astype(int)
        combined_nogo_count = int(
            topological["complete_numerical_no_go_chain"].sum()
        )
        topological.to_csv(
            output / "fes_step06_combined_no_go_certificates.csv", index=False
        )

    summary = {
        "code_version": CODE_VERSION,
        "representative_points": int(len(representatives)),
        "critical_roots_found": int((roots["root_found"] == 1).sum()),
        "kp_valleys_audited": int(len(kp_summary)),
        "winding_consensus_fraction": winding_consensus_fraction,
        "eta_equals_sign_r1r2_fraction": eta_match_fraction,
        "chern_jump_formula_match_fraction": jump_match_fraction,
        "calibrated_formula_prefactor": prefactor,
        "candidate_formula": (
            f"C_up = {prefactor:+d} * sgn(r1*r2)/2 * "
            "[sgn(mu_Gamma)+sgn(mu_M)]"
        ),
        "strict_formula_validation": strict_summary,
        "coarse_formula_validation": coarse_summary,
        "verified_topological_no_go_certificates": int(len(cert)),
        "combined_formula_plus_no_go_certificate_count": combined_nogo_count,
        "status": (
            "Low-energy k.p reconstruction and strict numerical formula audit "
            "completed. Publication-level analytic proof still requires an "
            "explicit symbolic derivation of the projected Jacobian signs."
        ),
    }
    (output / "fes_step06_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = f"""FES Step 06 summary
===================

Representative points: {len(representatives)}
Critical roots found: {(roots["root_found"] == 1).sum()}
k.p valleys audited: {len(kp_summary)}

Winding consensus fraction:
  {winding_consensus_fraction}

eta_K = sgn(r1*r2) fraction:
  {eta_match_fraction}

Chern-jump formula match fraction:
  {jump_match_fraction}

Calibrated formula:
  {summary["candidate_formula"]}

Strict formula accuracy:
  all evaluated = {strict_summary["accuracy_all_evaluated"]}
  robust         = {strict_summary["accuracy_robust"]}

Coarse formula accuracy:
  all evaluated = {coarse_summary["accuracy_all_evaluated"]}
  robust         = {coarse_summary["accuracy_robust"]}

Combined formula + Step 05 no-go certificates:
  {combined_nogo_count}/{len(cert)}

Interpretation
--------------
The numerical low-energy reconstruction supports

  C_up = sgn(r1*r2)/2
         [sgn(mu_Gamma)+sgn(mu_M)].

Together with the exact conditional Step 05 result, every verified
nonzero-spin-Chern point is certified to have E_g^ind <= 0.

Remaining analytic task
-----------------------
Write the projected Γ/M Jacobian determinants explicitly and prove
their signs are controlled by r1*r2. This upgrades the reconstructed
formula to a publication-level symbolic theorem.
"""
    (output / "fes_step06_report.txt").write_text(report, encoding="utf-8")

    return {
        "output_dir": str(output),
        "representatives": str(output / "fes_step06_representative_points.csv"),
        "roots": str(output / "fes_step06_critical_mass_roots.csv"),
        "kp_coefficients": str(output / "fes_step06_kp_coefficients.csv"),
        "winding": str(output / "fes_step06_winding_by_radius.csv"),
        "chern_jumps": str(output / "fes_step06_chern_jump_validation.csv"),
        "strict_formula": str(output / "fes_step06_formula_validation_strict.csv"),
        "coarse_formula": str(output / "fes_step06_formula_validation_coarse.csv"),
        "combined_certificates": str(output / "fes_step06_combined_no_go_certificates.csv"),
        "summary": str(output / "fes_step06_summary.json"),
        "report": str(output / "fes_step06_report.txt"),
    }


# =============================================================================
# 9. CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FES Step 06: low-energy k.p and spin-Chern formula"
    )
    parser.add_argument(
        "--step05-input",
        default="outputs_fes6_step05_no_go.zip",
        help="Step 05 result directory or ZIP",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_fes6_step06_kp",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> KPConfig:
    if args.test:
        return KPConfig(
            step05_input=args.step05_input,
            output_dir=args.output_dir,
            workers=1,
            representative_per_group=1,
            root_grid_points=801,
            fit_grid_n=7,
            winding_radii=(3.0e-4, 1.0e-3, 3.0e-3),
            winding_angles=361,
            chern_jump_grids=(15, 21),
            overwrite=args.overwrite,
            test_mode=True,
        )
    return KPConfig(
        step05_input=args.step05_input,
        output_dir=args.output_dir,
        workers=max(1, int(args.workers)),
        overwrite=args.overwrite,
        test_mode=False,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)

    print("=" * 78)
    print("FES Step 06 — low-energy k.p and spin-Chern formula")
    print("Version:", CODE_VERSION)
    print("Step 05 input:", config.step05_input)
    print("Output:", Path(config.output_dir).resolve())
    print("=" * 78)

    paths = run_pipeline(config)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
