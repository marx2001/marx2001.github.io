#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 02 — fes net 五维 Sobol 全局扫描与拓扑存在性筛选

任务目标
--------
1. 在去除整体能量平移后的五维参数空间
       (m_e, t1, t2, r1, r2)
   中进行可重复的 Sobol 低差异采样；
2. 对每个参数点完成完整二维 BZ 的直接带隙、间接带隙和
   2-up + 2-down 半填充占据检查；
3. 仅对直接开隙且自旋占据平衡的样本计算非阿贝尔 spin Chern；
4. 找出 spin_chern_TI_candidate、spin_chern_band_metal 和相边界样本；
5. 对所有拓扑/可疑样本执行多网格、多 shift 的严格复核；
6. 输出可直接供 Step 03 机器学习使用的结构化数据集。

本步骤不把非零 spin Chern 自动命名为 confirmed type-II QSH。
confirmed type-II QSH 仍需后续验证动量分离带反转、边缘态和输运平台。

默认计算规模
------------
Sobol 点数：2^12 = 4096
粗筛：25x25 BZ gap，17x17 Chern
严格复核：51/71 BZ gap，多 shift；21/31/41 Chern，多 shift

运行示例
--------
快速自检：
    python FES_step02_sobol_scan.py --test

正式扫描：
    python FES_step02_sobol_scan.py --sobol-power 12 --workers 8 --overwrite

断点续算：
    python FES_step02_sobol_scan.py --sobol-power 12 --workers 8 --resume
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Sequence
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt

np.set_printoptions(precision=10, suppress=True)

CODE_VERSION = "FES_STEP02_SOBOL_V1_20260712"
RESULT_SCHEMA_VERSION = "fes_step02_schema_v1"

# -----------------------------------------------------------------------------
# fes 模型常数
# -----------------------------------------------------------------------------
RAW6 = ["e1", "e2", "t1", "t2", "r1", "r2"]
REDUCED5 = ["m_e", "t1", "t2", "r1", "r2"]
N_OCC_TOTAL = 4
N_OCC_SPIN = 2

X_FES = 0.207
ALPHA = 2.0 * X_FES
BETA = 0.5 - X_FES

ORBITAL_POSITIONS = np.array(
    [
        [ X_FES,  0.5], [ X_FES,  0.5],
        [-X_FES,  0.5], [-X_FES,  0.5],
        [ 0.5,   -X_FES], [ 0.5,   -X_FES],
        [ 0.5,    X_FES], [ 0.5,    X_FES],
    ],
    dtype=float,
)

SPIN_UP_INDICES = [0, 2, 4, 6]
SPIN_DOWN_INDICES = [1, 3, 5, 7]

PAPER_PARAMS = {
    "e1": 0.7,
    "e2": -0.4,
    "t1": 0.3,
    "t2": 0.2,
    "r1": 0.3,
    "r2": -0.1,
}

DEFAULT_BOUNDS = {
    "m_e": (-1.0, 1.0),
    "t1": (-1.0, 1.0),
    "t2": (-1.0, 1.0),
    "r1": (-1.0, 1.0),
    "r2": (-1.0, 1.0),
}


@dataclass
class ScanConfig:
    output_dir: str = "outputs_fes6_step02_sobol"
    sobol_power: int = 12
    sobol_seed: int = 20260712
    scramble: bool = True
    workers: int = max(1, min(8, os.cpu_count() or 1))
    checkpoint_every: int = 128

    bounds: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BOUNDS)
    )

    coarse_gap_nk: int = 25
    coarse_chern_nk: int = 17
    gap_tol: float = 1.0e-3
    chern_tol: float = 0.08
    min_det_tol: float = 1.0e-7

    refine_gap_grids: tuple[int, ...] = (51, 71)
    refine_gap_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    )
    refine_chern_grids: tuple[int, ...] = (21, 31, 41)
    refine_chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )

    refine_gap_window: float = 0.05
    refine_det_window: float = 0.20
    max_boundary_refine: int = 512

    resume: bool = False
    overwrite: bool = False
    test_mode: bool = False

    @property
    def n_samples(self) -> int:
        return 2 ** int(self.sobol_power)


# =============================================================================
# 1. 参数与 Hamiltonian
# =============================================================================

def validate_params(params: Dict[str, float]) -> Dict[str, float]:
    missing = [name for name in RAW6 if name not in params]
    extra = [name for name in params if name not in RAW6]
    if missing:
        raise ValueError(f"缺少参数: {missing}")
    if extra:
        raise ValueError(f"出现未定义参数: {extra}")
    clean = {name: float(params[name]) for name in RAW6}
    if not all(np.isfinite(v) for v in clean.values()):
        raise ValueError("参数含 NaN 或无穷大")
    return clean


def raw6_from_reduced5(reduced: Dict[str, float], e0: float = 0.0) -> Dict[str, float]:
    missing = [name for name in REDUCED5 if name not in reduced]
    if missing:
        raise ValueError(f"缺少五维参数: {missing}")
    m_e = float(reduced["m_e"])
    return {
        "e1": float(e0 + m_e),
        "e2": float(e0 - m_e),
        "t1": float(reduced["t1"]),
        "t2": float(reduced["t2"]),
        "r1": float(reduced["r1"]),
        "r2": float(reduced["r2"]),
    }


def derived_features(params: Dict[str, float]) -> Dict[str, float]:
    p = validate_params(params)
    root2 = math.sqrt(2.0)
    values = np.array(
        [
            0.5 * (p["e1"] - p["e2"]),
            p["t1"], p["t2"], p["r1"], p["r2"],
        ],
        dtype=float,
    )
    return {
        "e0": 0.5 * (p["e1"] + p["e2"]),
        "m_e": 0.5 * (p["e1"] - p["e2"]),
        "t_sum": p["t1"] + p["t2"],
        "t_diff": p["t1"] - p["t2"],
        "t_s": (p["t1"] + p["t2"]) / root2,
        "t_d": (p["t1"] - p["t2"]) / root2,
        "r_sum": p["r1"] + p["r2"],
        "r_diff": p["r1"] - p["r2"],
        "r_s": (p["r1"] + p["r2"]) / root2,
        "r_d": (p["r1"] - p["r2"]) / root2,
        "abs_r_mismatch": abs(abs(p["r1"]) - abs(p["r2"])),
        "t_product": p["t1"] * p["t2"],
        "r_product": p["r1"] * p["r2"],
        "parameter_l2": float(np.linalg.norm(values)),
        "parameter_linf": float(np.max(np.abs(values))),
    }


def h_fes_atomic(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    p = validate_params(params)
    e1, e2, t1, t2, r1, r2 = [p[name] for name in RAW6]

    e11 = np.zeros((4, 4), dtype=np.complex128)
    np.fill_diagonal(e11, [e1, e2, e1, e2])
    e11[0, 2] = t1 * np.exp(-1j * ALPHA * kx)
    e11[2, 0] = np.conj(e11[0, 2])
    e11[1, 3] = t2 * np.exp(-1j * ALPHA * kx)
    e11[3, 1] = np.conj(e11[1, 3])

    e22 = np.zeros((4, 4), dtype=np.complex128)
    np.fill_diagonal(e22, [e2, e1, e2, e1])
    e22[0, 2] = t2 * np.exp(+1j * ALPHA * ky)
    e22[2, 0] = np.conj(e22[0, 2])
    e22[1, 3] = t1 * np.exp(+1j * ALPHA * ky)
    e22[3, 1] = np.conj(e22[1, 3])

    r_plus = 1j * r1 + r2
    r_minus = -1j * r1 + r2
    phase_pp = np.exp(+1j * BETA * (kx + ky))
    phase_pm = np.exp(+1j * BETA * (kx - ky))
    phase_mp = np.exp(-1j * BETA * (kx + ky))
    phase_mm = np.exp(-1j * BETA * (kx - ky))

    r12 = np.array(
        [
            [phase_pp * r_plus, 0.0, phase_pm * r_minus, 0.0],
            [0.0, phase_pp * r_minus, 0.0, phase_pm * r_plus],
            [phase_mm * r_minus, 0.0, phase_mp * r_plus, 0.0],
            [0.0, phase_mm * r_plus, 0.0, phase_mp * r_minus],
        ],
        dtype=np.complex128,
    )

    return np.block([[e11, r12], [r12.conj().T, e22]])


def periodic_gauge_matrix(kx: float, ky: float) -> np.ndarray:
    phases = np.exp(
        1j * (
            ORBITAL_POSITIONS[:, 0] * float(kx)
            + ORBITAL_POSITIONS[:, 1] * float(ky)
        )
    )
    return np.diag(phases).astype(np.complex128)


def h_fes_periodic(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    u = periodic_gauge_matrix(kx, ky)
    h = h_fes_atomic(kx, ky, params)
    return u @ h @ u.conj().T


def spin_indices(spin: str) -> list[int]:
    key = spin.lower()
    if key == "up":
        return SPIN_UP_INDICES.copy()
    if key == "down":
        return SPIN_DOWN_INDICES.copy()
    raise ValueError("spin 必须为 'up' 或 'down'")


def h_spin_block_atomic(
    kx: float, ky: float, params: Dict[str, float], spin: str
) -> np.ndarray:
    idx = spin_indices(spin)
    h = h_fes_atomic(kx, ky, params)
    return h[np.ix_(idx, idx)]


def h_spin_block_periodic(
    kx: float, ky: float, params: Dict[str, float], spin: str
) -> np.ndarray:
    idx = spin_indices(spin)
    h = h_fes_periodic(kx, ky, params)
    return h[np.ix_(idx, idx)]


# =============================================================================
# 2. BZ 带隙审计
# =============================================================================

def shifted_k_grid(nk: int, shift: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = shift
    kx = -np.pi + 2.0 * np.pi * (np.arange(nk) + float(sx)) / nk
    ky = -np.pi + 2.0 * np.pi * (np.arange(nk) + float(sy)) / nk
    return kx, ky


def scan_band_gaps_shifted(
    params: Dict[str, float],
    nk: int,
    shift: tuple[float, float] = (0.0, 0.0),
    gap_tol: float = 1.0e-3,
) -> Dict[str, object]:
    """
    使用两个 4x4 自旋块完成半填充 gap 审计，比重复对角化 8x8 更高效。
    """
    kxs, kys = shifted_k_grid(nk, shift)

    min_direct = np.inf
    direct_kx = np.nan
    direct_ky = np.nan
    max_valence = -np.inf
    vbm_kx = np.nan
    vbm_ky = np.nan
    min_conduction = np.inf
    cbm_kx = np.nan
    cbm_ky = np.nan
    min_spin_gap_up = np.inf
    min_spin_gap_down = np.inf
    min_balanced_sector_gap = np.inf
    balanced_gap_kx = np.nan
    balanced_gap_ky = np.nan
    mismatch_count = 0
    min_n_up = N_OCC_TOTAL
    max_n_up = 0

    for kx in kxs:
        for ky in kys:
            kx_f = float(kx)
            ky_f = float(ky)
            up_e = np.linalg.eigvalsh(h_spin_block_atomic(kx_f, ky_f, params, "up"))
            down_e = np.linalg.eigvalsh(h_spin_block_atomic(kx_f, ky_f, params, "down"))

            tagged = [(float(e), 1) for e in up_e]
            tagged += [(float(e), 0) for e in down_e]
            tagged.sort(key=lambda item: item[0])
            energies = np.array([e for e, _ in tagged], dtype=float)

            direct_gap = float(energies[N_OCC_TOTAL] - energies[N_OCC_TOTAL - 1])
            if direct_gap < min_direct:
                min_direct = direct_gap
                direct_kx = kx_f
                direct_ky = ky_f

            valence = float(energies[N_OCC_TOTAL - 1])
            conduction = float(energies[N_OCC_TOTAL])
            if valence > max_valence:
                max_valence = valence
                vbm_kx = kx_f
                vbm_ky = ky_f
            if conduction < min_conduction:
                min_conduction = conduction
                cbm_kx = kx_f
                cbm_ky = ky_f

            spin_gap_up = float(up_e[N_OCC_SPIN] - up_e[N_OCC_SPIN - 1])
            spin_gap_down = float(down_e[N_OCC_SPIN] - down_e[N_OCC_SPIN - 1])
            min_spin_gap_up = min(min_spin_gap_up, spin_gap_up)
            min_spin_gap_down = min(min_spin_gap_down, spin_gap_down)

            balanced_gap = float(
                min(up_e[N_OCC_SPIN], down_e[N_OCC_SPIN])
                - max(up_e[N_OCC_SPIN - 1], down_e[N_OCC_SPIN - 1])
            )
            if balanced_gap < min_balanced_sector_gap:
                min_balanced_sector_gap = balanced_gap
                balanced_gap_kx = kx_f
                balanced_gap_ky = ky_f

            n_up_occ = int(sum(tag for _, tag in tagged[:N_OCC_TOTAL]))
            min_n_up = min(min_n_up, n_up_occ)
            max_n_up = max(max_n_up, n_up_occ)
            if n_up_occ != N_OCC_SPIN:
                mismatch_count += 1

    indirect_gap = float(min_conduction - max_valence)
    balanced = min_balanced_sector_gap > gap_tol and mismatch_count == 0

    return {
        "gap_nk": int(nk),
        "gap_shift_x": float(shift[0]),
        "gap_shift_y": float(shift[1]),
        "min_direct_gap": float(min_direct),
        "indirect_gap": indirect_gap,
        "direct_gap_kx": direct_kx,
        "direct_gap_ky": direct_ky,
        "vbm": float(max_valence),
        "vbm_kx": vbm_kx,
        "vbm_ky": vbm_ky,
        "cbm": float(min_conduction),
        "cbm_kx": cbm_kx,
        "cbm_ky": cbm_ky,
        "min_spin_gap_up": float(min_spin_gap_up),
        "min_spin_gap_down": float(min_spin_gap_down),
        "min_balanced_sector_gap": float(min_balanced_sector_gap),
        "balanced_gap_kx": balanced_gap_kx,
        "balanced_gap_ky": balanced_gap_ky,
        "spin_occupancy_mismatch_count": int(mismatch_count),
        "min_n_up_in_lowest4": int(min_n_up),
        "max_n_up_in_lowest4": int(max_n_up),
        "is_direct_gapped": int(min_direct > gap_tol),
        "is_physical_insulator": int(indirect_gap > gap_tol),
        "is_balanced_spin_sector": int(balanced),
    }


def consensus_gap_audit(
    params: Dict[str, float],
    grids: Sequence[int],
    shifts: Sequence[tuple[float, float]],
    gap_tol: float,
) -> tuple[Dict[str, object], pd.DataFrame]:
    rows: list[Dict[str, object]] = []
    for nk in grids:
        for shift in shifts:
            row = scan_band_gaps_shifted(params, int(nk), shift, gap_tol)
            rows.append(row)
    df = pd.DataFrame(rows)

    # 保守汇总：跨所有网格/shift 取最小直接带隙，跨所有采样取全局 VBM/CBM。
    min_direct_idx = int(df["min_direct_gap"].idxmin())
    vbm_idx = int(df["vbm"].idxmax())
    cbm_idx = int(df["cbm"].idxmin())
    max_vbm = float(df.loc[vbm_idx, "vbm"])
    min_cbm = float(df.loc[cbm_idx, "cbm"])

    summary = {
        "verified_min_direct_gap": float(df["min_direct_gap"].min()),
        "verified_indirect_gap": float(min_cbm - max_vbm),
        "verified_direct_gap_kx": float(df.loc[min_direct_idx, "direct_gap_kx"]),
        "verified_direct_gap_ky": float(df.loc[min_direct_idx, "direct_gap_ky"]),
        "verified_vbm": max_vbm,
        "verified_vbm_kx": float(df.loc[vbm_idx, "vbm_kx"]),
        "verified_vbm_ky": float(df.loc[vbm_idx, "vbm_ky"]),
        "verified_cbm": min_cbm,
        "verified_cbm_kx": float(df.loc[cbm_idx, "cbm_kx"]),
        "verified_cbm_ky": float(df.loc[cbm_idx, "cbm_ky"]),
        "verified_min_spin_gap_up": float(df["min_spin_gap_up"].min()),
        "verified_min_spin_gap_down": float(df["min_spin_gap_down"].min()),
        "verified_min_balanced_sector_gap": float(
            df["min_balanced_sector_gap"].min()
        ),
        "verified_total_mismatch_count": int(
            df["spin_occupancy_mismatch_count"].sum()
        ),
        "verified_is_direct_gapped": int(
            float(df["min_direct_gap"].min()) > gap_tol
        ),
        "verified_is_physical_insulator": int(
            float(min_cbm - max_vbm) > gap_tol
        ),
        "verified_is_balanced_spin_sector": int(
            float(df["min_balanced_sector_gap"].min()) > gap_tol
            and int(df["spin_occupancy_mismatch_count"].sum()) == 0
        ),
        "verified_gap_runs": int(len(df)),
    }
    return summary, df


# =============================================================================
# 3. 非阿贝尔 Fukui Chern
# =============================================================================

def _normalize_link(value: complex, eps: float = 1.0e-14) -> complex:
    amplitude = abs(value)
    if amplitude < eps:
        raise FloatingPointError(f"Link determinant too small: {amplitude:.3e}")
    return value / amplitude


def fukui_chern_subspace_shifted(
    h_func: Callable[[float, float], np.ndarray],
    n_occ: int,
    nk: int,
    shift: tuple[float, float] = (0.0, 0.0),
) -> Dict[str, object]:
    sample_h = h_func(0.0, 0.0)
    dim = sample_h.shape[0]
    if n_occ <= 0 or n_occ >= dim:
        raise ValueError(f"n_occ={n_occ} 与 Hamiltonian 维数 {dim} 不兼容")

    sx, sy = shift
    kxs = 2.0 * np.pi * (np.arange(nk) + float(sx)) / nk
    kys = 2.0 * np.pi * (np.arange(nk) + float(sy)) / nk
    occupied = np.empty((nk, nk, dim, n_occ), dtype=np.complex128)

    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            _, vecs = np.linalg.eigh(h_func(float(kx), float(ky)))
            occupied[ix, iy] = vecs[:, :n_occ]

    total_phase = 0.0
    min_det = 1.0
    for ix in range(nk):
        for iy in range(nk):
            v = occupied[ix, iy]
            vx = occupied[(ix + 1) % nk, iy]
            vy = occupied[ix, (iy + 1) % nk]
            vxy = occupied[(ix + 1) % nk, (iy + 1) % nk]

            lx = np.linalg.det(v.conj().T @ vx)
            ly = np.linalg.det(v.conj().T @ vy)
            lx_y = np.linalg.det(vy.conj().T @ vxy)
            ly_x = np.linalg.det(vx.conj().T @ vxy)
            min_det = min(min_det, abs(lx), abs(ly), abs(lx_y), abs(ly_x))

            ux = _normalize_link(lx)
            uy = _normalize_link(ly)
            ux_y = _normalize_link(lx_y)
            uy_x = _normalize_link(ly_x)
            total_phase += float(np.angle(ux * uy_x / (ux_y * uy)))

    return {
        "chern": float(total_phase / (2.0 * np.pi)),
        "min_det": float(min_det),
        "nk": int(nk),
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
    }


def rounded_integer_if_close(value: float, tolerance: float) -> int | None:
    nearest = int(np.rint(value))
    return nearest if abs(value - nearest) <= tolerance else None


def calculate_coarse_spin_chern(
    params: Dict[str, float],
    nk: int,
    chern_tol: float,
    min_det_tol: float,
) -> Dict[str, object]:
    up = fukui_chern_subspace_shifted(
        lambda kx, ky: h_spin_block_periodic(kx, ky, params, "up"),
        N_OCC_SPIN,
        nk,
        (0.0, 0.0),
    )
    down = fukui_chern_subspace_shifted(
        lambda kx, ky: h_spin_block_periodic(kx, ky, params, "down"),
        N_OCC_SPIN,
        nk,
        (0.0, 0.0),
    )

    c_up = float(up["chern"])
    c_down = float(down["chern"])
    c_total = c_up + c_down
    c_up_int = rounded_integer_if_close(c_up, chern_tol)
    c_down_int = rounded_integer_if_close(c_down, chern_tol)
    c_total_int = rounded_integer_if_close(c_total, chern_tol)
    reliable = (
        c_up_int is not None
        and c_down_int is not None
        and c_total_int is not None
        and float(up["min_det"]) > min_det_tol
        and float(down["min_det"]) > min_det_tol
    )

    return {
        "chern_nk": int(nk),
        "chern_up": c_up,
        "chern_down": c_down,
        "chern_total_inferred": c_total,
        "spin_chern": 0.5 * (c_up - c_down),
        "chern_up_int": c_up_int,
        "chern_down_int": c_down_int,
        "chern_total_int": c_total_int,
        "min_det_up": float(up["min_det"]),
        "min_det_down": float(down["min_det"]),
        "chern_reliable": int(reliable),
    }


def consensus_chern_audit(
    params: Dict[str, float],
    grids: Sequence[int],
    shifts: Sequence[tuple[float, float]],
    chern_tol: float,
    min_det_tol: float,
    include_total: bool = True,
) -> tuple[Dict[str, object], pd.DataFrame]:
    rows: list[Dict[str, object]] = []
    for nk in grids:
        for shift in shifts:
            try:
                up = fukui_chern_subspace_shifted(
                    lambda kx, ky: h_spin_block_periodic(kx, ky, params, "up"),
                    N_OCC_SPIN,
                    int(nk),
                    shift,
                )
                down = fukui_chern_subspace_shifted(
                    lambda kx, ky: h_spin_block_periodic(kx, ky, params, "down"),
                    N_OCC_SPIN,
                    int(nk),
                    shift,
                )
                row: Dict[str, object] = {
                    "nk": int(nk),
                    "shift_x": float(shift[0]),
                    "shift_y": float(shift[1]),
                    "chern_up": float(up["chern"]),
                    "chern_down": float(down["chern"]),
                    "min_det_up": float(up["min_det"]),
                    "min_det_down": float(down["min_det"]),
                    "error": "",
                }
                if include_total:
                    total = fukui_chern_subspace_shifted(
                        lambda kx, ky: h_fes_periodic(kx, ky, params),
                        N_OCC_TOTAL,
                        int(nk),
                        shift,
                    )
                    row["chern_total"] = float(total["chern"])
                    row["min_det_total"] = float(total["min_det"])
                else:
                    row["chern_total"] = float(up["chern"] + down["chern"])
                    row["min_det_total"] = np.nan
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "nk": int(nk),
                        "shift_x": float(shift[0]),
                        "shift_y": float(shift[1]),
                        "error": repr(exc),
                    }
                )

    df = pd.DataFrame(rows)
    valid = df[df["error"].fillna("") == ""].copy()
    if valid.empty:
        return {
            "verified_chern_reliable": 0,
            "verified_chern_error": "all_chern_runs_failed",
            "verified_chern_runs": 0,
        }, df

    int_rows = []
    all_integer = True
    for _, row in valid.iterrows():
        cu = rounded_integer_if_close(float(row["chern_up"]), chern_tol)
        cd = rounded_integer_if_close(float(row["chern_down"]), chern_tol)
        ct = rounded_integer_if_close(float(row["chern_total"]), chern_tol)
        int_rows.append((cu, cd, ct))
        if cu is None or cd is None or ct is None:
            all_integer = False

    consensus = len(set(int_rows)) == 1 if all_integer else False
    min_det_up = float(valid["min_det_up"].min())
    min_det_down = float(valid["min_det_down"].min())
    if include_total:
        min_det_total = float(valid["min_det_total"].min())
    else:
        min_det_total = np.nan

    det_ok = min_det_up > min_det_tol and min_det_down > min_det_tol
    if include_total:
        det_ok = det_ok and min_det_total > min_det_tol

    if consensus:
        cu_int, cd_int, ct_int = int_rows[0]
    else:
        cu_int = cd_int = ct_int = None

    sum_rule_error = float(
        np.max(
            np.abs(
                valid["chern_total"].to_numpy(float)
                - valid["chern_up"].to_numpy(float)
                - valid["chern_down"].to_numpy(float)
            )
        )
    )

    reliable = (
        len(valid) == len(rows)
        and consensus
        and det_ok
        and sum_rule_error < chern_tol
    )

    summary = {
        "verified_chern_up_mean": float(valid["chern_up"].mean()),
        "verified_chern_down_mean": float(valid["chern_down"].mean()),
        "verified_chern_total_mean": float(valid["chern_total"].mean()),
        "verified_chern_up_int": cu_int,
        "verified_chern_down_int": cd_int,
        "verified_chern_total_int": ct_int,
        "verified_min_det_up": min_det_up,
        "verified_min_det_down": min_det_down,
        "verified_min_det_total": min_det_total,
        "verified_chern_sum_rule_error": sum_rule_error,
        "verified_chern_consensus": int(consensus),
        "verified_chern_reliable": int(reliable),
        "verified_chern_runs": int(len(valid)),
        "verified_chern_error": "" if len(valid) == len(rows) else "some_runs_failed",
    }
    return summary, df


# =============================================================================
# 4. 粗筛与严格复核标签
# =============================================================================

def empty_chern_fields() -> Dict[str, object]:
    return {
        "chern_nk": np.nan,
        "chern_up": np.nan,
        "chern_down": np.nan,
        "chern_total_inferred": np.nan,
        "spin_chern": np.nan,
        "chern_up_int": None,
        "chern_down_int": None,
        "chern_total_int": None,
        "min_det_up": np.nan,
        "min_det_down": np.nan,
        "chern_reliable": 0,
    }


def classify_from_chern(
    gap: Dict[str, object],
    chern: Dict[str, object],
) -> tuple[str, int]:
    direct = bool(int(gap["is_direct_gapped"]))
    insulator = bool(int(gap["is_physical_insulator"]))
    balanced = bool(int(gap["is_balanced_spin_sector"]))
    reliable = bool(int(chern.get("chern_reliable", 0)))

    if not direct:
        return "noninsulating_or_gap_closing", 0
    if not balanced:
        return "spin_sector_filling_mismatch", 0
    if not reliable:
        return "chern_unreliable", 0

    cu = chern.get("chern_up_int")
    cd = chern.get("chern_down_int")
    ct = chern.get("chern_total_int")
    spin_topological = (
        cu is not None
        and cd is not None
        and ct == 0
        and cu == -cd
        and cu != 0
    )

    if not insulator:
        return (
            "spin_chern_band_metal" if spin_topological else "indirect_overlap",
            0,
        )
    if spin_topological:
        return "spin_chern_TI_candidate", 1
    if cu == 0 and cd == 0 and ct == 0:
        return "trivial_insulator", 0
    if ct not in (None, 0):
        return "Chern_insulator", 0
    return "other_gapped_phase", 0


def coarse_label_sample(task: tuple) -> Dict[str, object]:
    (
        sample_id,
        sobol_index,
        params,
        gap_nk,
        chern_nk,
        gap_tol,
        chern_tol,
        min_det_tol,
    ) = task

    p = validate_params(params)
    row: Dict[str, object] = {
        "sample_id": str(sample_id),
        "sobol_index": int(sobol_index),
        **p,
        **derived_features(p),
        **empty_chern_fields(),
        "is_spin_chern_TI_candidate": 0,
        "is_typeII_QSH_confirmed": 0,
        "phase_label": "numeric_error",
        "error": "",
    }

    try:
        gap = scan_band_gaps_shifted(p, int(gap_nk), (0.0, 0.0), float(gap_tol))
        row.update(gap)
    except Exception as exc:
        row["error"] = f"gap_scan_failed: {repr(exc)}"
        return row

    if not bool(int(gap["is_direct_gapped"])):
        row["phase_label"] = "noninsulating_or_gap_closing"
        return row
    if not bool(int(gap["is_balanced_spin_sector"])):
        row["phase_label"] = "spin_sector_filling_mismatch"
        return row

    try:
        chern = calculate_coarse_spin_chern(
            p,
            int(chern_nk),
            float(chern_tol),
            float(min_det_tol),
        )
        row.update(chern)
    except Exception as exc:
        row["phase_label"] = "chern_unreliable"
        row["error"] = f"chern_failed: {repr(exc)}"
        return row

    label, candidate = classify_from_chern(gap, chern)
    row["phase_label"] = label
    row["is_spin_chern_TI_candidate"] = int(candidate)
    return row


def verified_phase_label(row: Dict[str, object]) -> tuple[str, int]:
    if not bool(int(row.get("verified_is_direct_gapped", 0))):
        return "verified_noninsulating_or_gap_closing", 0
    if not bool(int(row.get("verified_is_balanced_spin_sector", 0))):
        return "verified_spin_sector_filling_mismatch", 0
    if not bool(int(row.get("verified_chern_reliable", 0))):
        return "verified_chern_unreliable", 0

    cu = row.get("verified_chern_up_int")
    cd = row.get("verified_chern_down_int")
    ct = row.get("verified_chern_total_int")
    insulator = bool(int(row.get("verified_is_physical_insulator", 0)))
    spin_topological = (
        cu is not None
        and cd is not None
        and ct == 0
        and cu == -cd
        and cu != 0
    )

    if not insulator:
        return (
            "verified_spin_chern_band_metal"
            if spin_topological
            else "verified_indirect_overlap",
            0,
        )
    if spin_topological:
        return "verified_spin_chern_TI_candidate", 1
    if cu == 0 and cd == 0 and ct == 0:
        return "verified_trivial_insulator", 0
    if ct not in (None, 0):
        return "verified_Chern_insulator", 0
    return "verified_other_gapped_phase", 0


def refine_sample(
    coarse_row: Dict[str, object],
    config: ScanConfig,
    detail_dir: Path,
) -> Dict[str, object]:
    params = {name: float(coarse_row[name]) for name in RAW6}
    sample_id = str(coarse_row["sample_id"])
    result: Dict[str, object] = dict(coarse_row)
    result["refined"] = 1
    result["refine_error"] = ""

    try:
        gap_summary, gap_table = consensus_gap_audit(
            params,
            config.refine_gap_grids,
            config.refine_gap_shifts,
            config.gap_tol,
        )
        result.update(gap_summary)
        gap_table.insert(0, "sample_id", sample_id)
        gap_table.to_csv(
            detail_dir / f"{sample_id}_gap_consensus.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        result["verified_phase_label"] = "verified_numeric_error"
        result["verified_is_spin_chern_TI_candidate"] = 0
        result["refine_error"] = f"verified_gap_failed: {repr(exc)}"
        return result

    if not bool(int(result["verified_is_direct_gapped"])):
        label, candidate = verified_phase_label(result)
        result["verified_phase_label"] = label
        result["verified_is_spin_chern_TI_candidate"] = candidate
        return result

    if not bool(int(result["verified_is_balanced_spin_sector"])):
        label, candidate = verified_phase_label(result)
        result["verified_phase_label"] = label
        result["verified_is_spin_chern_TI_candidate"] = candidate
        return result

    try:
        chern_summary, chern_table = consensus_chern_audit(
            params,
            config.refine_chern_grids,
            config.refine_chern_shifts,
            config.chern_tol,
            config.min_det_tol,
            include_total=True,
        )
        result.update(chern_summary)
        chern_table.insert(0, "sample_id", sample_id)
        chern_table.to_csv(
            detail_dir / f"{sample_id}_chern_consensus.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        result["verified_chern_reliable"] = 0
        result["verified_phase_label"] = "verified_chern_unreliable"
        result["verified_is_spin_chern_TI_candidate"] = 0
        result["refine_error"] = f"verified_chern_failed: {repr(exc)}"
        return result

    label, candidate = verified_phase_label(result)
    result["verified_phase_label"] = label
    result["verified_is_spin_chern_TI_candidate"] = candidate
    result["is_typeII_QSH_confirmed"] = 0
    return result


# =============================================================================
# 5. Sobol 采样、断点续算与输出
# =============================================================================

def validate_bounds(bounds: Dict[str, tuple[float, float]]) -> None:
    missing = [name for name in REDUCED5 if name not in bounds]
    extra = [name for name in bounds if name not in REDUCED5]
    if missing or extra:
        raise ValueError(f"bounds 不匹配: missing={missing}, extra={extra}")
    for name in REDUCED5:
        lo, hi = bounds[name]
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"非法范围 {name}: {(lo, hi)}")


def generate_sobol_parameters(config: ScanConfig) -> pd.DataFrame:
    validate_bounds(config.bounds)
    sampler = qmc.Sobol(
        d=len(REDUCED5),
        scramble=bool(config.scramble),
        seed=int(config.sobol_seed),
    )
    unit = sampler.random_base2(m=int(config.sobol_power))
    lower = np.array([config.bounds[name][0] for name in REDUCED5], dtype=float)
    upper = np.array([config.bounds[name][1] for name in REDUCED5], dtype=float)
    scaled = qmc.scale(unit, lower, upper)

    rows = []
    for index, values in enumerate(scaled):
        reduced = {name: float(values[i]) for i, name in enumerate(REDUCED5)}
        raw = raw6_from_reduced5(reduced, e0=0.0)
        rows.append(
            {
                "sample_id": f"sobol_{index:07d}",
                "sobol_index": int(index),
                **raw,
            }
        )
    return pd.DataFrame(rows)


def config_to_jsonable(config: ScanConfig) -> Dict[str, object]:
    data = asdict(config)
    data["n_samples"] = config.n_samples
    return data


def script_sha256() -> str | None:
    if "__file__" not in globals():
        return None
    path = Path(__file__).resolve()
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_output(config: ScanConfig) -> Path:
    output = Path(config.output_dir)
    if output.exists() and config.overwrite and not config.resume:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "refine_details").mkdir(exist_ok=True)
    return output


def write_metadata(config: ScanConfig, output: Path) -> None:
    metadata = {
        "code_version": CODE_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": script_sha256(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "config": config_to_jsonable(config),
        "scientific_scope": (
            "Five-dimensional fes global scan; nonzero spin Chern is a candidate, "
            "not confirmed type-II QSH."
        ),
    }
    (output / "fes_step02_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_checkpoint(rows: list[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("sobol_index").reset_index(drop=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def run_coarse_scan(
    parameter_df: pd.DataFrame,
    config: ScanConfig,
    output: Path,
) -> pd.DataFrame:
    result_path = output / "fes_step02_coarse_results.csv"
    existing_rows: list[Dict[str, object]] = []
    completed_ids: set[str] = set()

    if config.resume and result_path.exists():
        existing = pd.read_csv(result_path)
        existing_rows = existing.to_dict("records")
        completed_ids = set(existing["sample_id"].astype(str))
        print(f"Resume: loaded {len(existing_rows)} completed samples")

    pending = parameter_df[~parameter_df["sample_id"].isin(completed_ids)].copy()
    tasks = []
    for _, row in pending.iterrows():
        params = {name: float(row[name]) for name in RAW6}
        tasks.append(
            (
                str(row["sample_id"]),
                int(row["sobol_index"]),
                params,
                int(config.coarse_gap_nk),
                int(config.coarse_chern_nk),
                float(config.gap_tol),
                float(config.chern_tol),
                float(config.min_det_tol),
            )
        )

    rows = list(existing_rows)
    total = len(tasks)
    if total == 0:
        return pd.DataFrame(rows).sort_values("sobol_index").reset_index(drop=True)

    print(
        f"Coarse scan: {total} pending / {len(parameter_df)} total, "
        f"workers={config.workers}"
    )
    start = time.time()

    if int(config.workers) <= 1:
        iterator: Iterable[Dict[str, object]] = map(coarse_label_sample, tasks)
        for count, result in enumerate(iterator, start=1):
            rows.append(result)
            if count % config.checkpoint_every == 0 or count == total:
                save_checkpoint(rows, result_path)
                elapsed = time.time() - start
                print(f"Coarse {count}/{total}; elapsed={elapsed:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=int(config.workers)) as executor:
            iterator = executor.map(coarse_label_sample, tasks, chunksize=4)
            for count, result in enumerate(iterator, start=1):
                rows.append(result)
                if count % config.checkpoint_every == 0 or count == total:
                    save_checkpoint(rows, result_path)
                    elapsed = time.time() - start
                    print(f"Coarse {count}/{total}; elapsed={elapsed:.1f}s")

    result_df = pd.DataFrame(rows).sort_values("sobol_index").reset_index(drop=True)
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    return result_df


def select_refine_queue(coarse_df: pd.DataFrame, config: ScanConfig) -> pd.DataFrame:
    mandatory_labels = {
        "spin_chern_TI_candidate",
        "spin_chern_band_metal",
        "chern_unreliable",
        "other_gapped_phase",
        "Chern_insulator",
    }
    mandatory = coarse_df[coarse_df["phase_label"].isin(mandatory_labels)].copy()

    direct = pd.to_numeric(coarse_df["min_direct_gap"], errors="coerce")
    min_det_up = pd.to_numeric(coarse_df["min_det_up"], errors="coerce")
    min_det_down = pd.to_numeric(coarse_df["min_det_down"], errors="coerce")
    balanced = pd.to_numeric(
        coarse_df["is_balanced_spin_sector"], errors="coerce"
    ).fillna(0).astype(int)

    boundary_mask = (
        (balanced == 1)
        & (
            (direct.abs() <= float(config.refine_gap_window))
            | (min_det_up <= float(config.refine_det_window))
            | (min_det_down <= float(config.refine_det_window))
        )
    )
    boundary = coarse_df[boundary_mask].copy()
    if len(boundary) > int(config.max_boundary_refine):
        boundary["_priority_gap"] = pd.to_numeric(
            boundary["min_direct_gap"], errors="coerce"
        ).abs()
        boundary = boundary.sort_values("_priority_gap").head(
            int(config.max_boundary_refine)
        )
        boundary = boundary.drop(columns=["_priority_gap"])

    queue = pd.concat([mandatory, boundary], ignore_index=True)
    queue = queue.drop_duplicates("sample_id").sort_values("sobol_index")
    queue["refine_reason"] = queue["phase_label"].astype(str)
    return queue.reset_index(drop=True)


def run_refinement(
    queue_df: pd.DataFrame,
    config: ScanConfig,
    output: Path,
) -> pd.DataFrame:
    result_path = output / "fes_step02_verified_results.csv"
    detail_dir = output / "refine_details"
    existing_rows: list[Dict[str, object]] = []
    completed_ids: set[str] = set()

    if config.resume and result_path.exists():
        existing = pd.read_csv(result_path)
        existing_rows = existing.to_dict("records")
        completed_ids = set(existing["sample_id"].astype(str))

    pending = queue_df[~queue_df["sample_id"].isin(completed_ids)]
    rows = list(existing_rows)
    total = len(pending)
    print(f"Refinement queue: {len(queue_df)} total, {total} pending")

    for count, (_, coarse_row) in enumerate(pending.iterrows(), start=1):
        result = refine_sample(coarse_row.to_dict(), config, detail_dir)
        rows.append(result)
        if count % 10 == 0 or count == total:
            save_checkpoint(rows, result_path)
            print(f"Refined {count}/{total}")

    if rows:
        verified_df = pd.DataFrame(rows).sort_values("sobol_index").reset_index(drop=True)
    else:
        verified_df = pd.DataFrame()
    verified_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    return verified_df


def build_ml_dataset(
    coarse_df: pd.DataFrame,
    verified_df: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    data = coarse_df.copy()
    data["final_phase_label"] = data["phase_label"].astype(str)
    data["final_label_source"] = "coarse"
    data["final_is_spin_chern_TI_candidate"] = pd.to_numeric(
        data["is_spin_chern_TI_candidate"], errors="coerce"
    ).fillna(0).astype(int)

    if not verified_df.empty:
        verified_map = verified_df.set_index("sample_id")
        for idx, sample_id in data["sample_id"].items():
            if sample_id not in verified_map.index:
                continue
            v = verified_map.loc[sample_id]
            data.at[idx, "final_phase_label"] = v.get(
                "verified_phase_label", data.at[idx, "phase_label"]
            )
            data.at[idx, "final_label_source"] = "verified"
            data.at[idx, "final_is_spin_chern_TI_candidate"] = int(
                v.get("verified_is_spin_chern_TI_candidate", 0)
            )

    data["ml_primary_class"] = data["final_phase_label"].replace(
        {
            "verified_noninsulating_or_gap_closing": "noninsulating_or_gap_closing",
            "verified_spin_sector_filling_mismatch": "spin_sector_filling_mismatch",
            "verified_chern_unreliable": "boundary_or_unreliable",
            "verified_indirect_overlap": "indirect_overlap",
            "verified_spin_chern_band_metal": "spin_chern_band_metal",
            "verified_trivial_insulator": "trivial_insulator",
            "verified_spin_chern_TI_candidate": "spin_chern_TI_candidate",
            "verified_Chern_insulator": "Chern_insulator",
            "verified_other_gapped_phase": "other_gapped_phase",
        }
    )

    data.to_csv(
        output / "fes_step02_ml_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return data


def write_summaries(
    coarse_df: pd.DataFrame,
    verified_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    output: Path,
) -> None:
    coarse_counts = (
        coarse_df["phase_label"].value_counts(dropna=False).rename_axis("phase_label")
        .reset_index(name="count")
    )
    coarse_counts["fraction"] = coarse_counts["count"] / len(coarse_df)
    coarse_counts.to_csv(
        output / "fes_step02_coarse_phase_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_counts = (
        ml_df["ml_primary_class"].value_counts(dropna=False).rename_axis("phase_label")
        .reset_index(name="count")
    )
    final_counts["fraction"] = final_counts["count"] / len(ml_df)
    final_counts.to_csv(
        output / "fes_step02_final_phase_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not verified_df.empty and "verified_phase_label" in verified_df.columns:
        verified_counts = (
            verified_df["verified_phase_label"].value_counts(dropna=False)
            .rename_axis("verified_phase_label")
            .reset_index(name="count")
        )
        verified_counts.to_csv(
            output / "fes_step02_verified_phase_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )

    topo = ml_df[
        ml_df["ml_primary_class"].isin(
            ["spin_chern_TI_candidate", "spin_chern_band_metal"]
        )
    ].copy()
    topo.to_csv(
        output / "fes_step02_topological_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 简单诊断图：每张图单独输出，不设置固定颜色。
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.bar(final_counts["phase_label"].astype(str), final_counts["count"])
    ax.set_ylabel("Count")
    ax.set_title("fes Step 02 final phase counts")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output / "fes_step02_final_phase_counts.png", dpi=220)
    plt.close(fig)

    summary = {
        "n_total": int(len(ml_df)),
        "n_refined": int(len(verified_df)),
        "n_final_spin_chern_TI_candidate": int(
            (ml_df["ml_primary_class"] == "spin_chern_TI_candidate").sum()
        ),
        "n_final_spin_chern_band_metal": int(
            (ml_df["ml_primary_class"] == "spin_chern_band_metal").sum()
        ),
        "coarse_phase_counts": {
            str(row["phase_label"]): int(row["count"])
            for _, row in coarse_counts.iterrows()
        },
        "final_phase_counts": {
            str(row["phase_label"]): int(row["count"])
            for _, row in final_counts.iterrows()
        },
    }
    (output / "fes_step02_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# 6. 主流程
# =============================================================================

def audit_paper_point(config: ScanConfig, output: Path) -> None:
    gap = scan_band_gaps_shifted(
        PAPER_PARAMS,
        max(31, config.coarse_gap_nk),
        (0.0, 0.0),
        config.gap_tol,
    )
    chern = calculate_coarse_spin_chern(
        PAPER_PARAMS,
        max(21, config.coarse_chern_nk),
        config.chern_tol,
        config.min_det_tol,
    )
    label, candidate = classify_from_chern(gap, chern)
    row = {
        "sample_id": "paper_reference",
        **PAPER_PARAMS,
        **derived_features(PAPER_PARAMS),
        **gap,
        **chern,
        "phase_label": label,
        "is_spin_chern_TI_candidate": candidate,
    }
    pd.DataFrame([row]).to_csv(
        output / "fes_step02_paper_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if label != "trivial_insulator":
        raise AssertionError(
            f"文献参数基准标签异常: expected trivial_insulator, got {label}"
        )


def run_pipeline(config: ScanConfig) -> Dict[str, Path]:
    if config.test_mode:
        config.sobol_power = 5
        config.workers = 1
        config.coarse_gap_nk = 13
        config.coarse_chern_nk = 9
        config.refine_gap_grids = (17,)
        config.refine_gap_shifts = ((0.0, 0.0), (0.5, 0.5))
        config.refine_chern_grids = (11, 15)
        config.refine_chern_shifts = ((0.0, 0.0),)
        config.max_boundary_refine = 8
        config.output_dir = str(Path(config.output_dir).with_name(
            Path(config.output_dir).name + "_test"
        ))

    output = prepare_output(config)
    write_metadata(config, output)

    print("=" * 80)
    print("fes Step 02 Sobol global scan")
    print("Version :", CODE_VERSION)
    print("Output  :", output.resolve())
    print("Samples :", config.n_samples)
    print("Workers :", config.workers)
    print("Bounds  :", config.bounds)
    print("=" * 80)

    audit_paper_point(config, output)

    parameter_path = output / "fes_step02_sobol_parameters.csv"
    if config.resume and parameter_path.exists():
        parameter_df = pd.read_csv(parameter_path)
        if len(parameter_df) != config.n_samples:
            raise ValueError(
                "断点续算时参数文件样本数与当前 sobol_power 不一致"
            )
    else:
        parameter_df = generate_sobol_parameters(config)
        parameter_df.to_csv(
            parameter_path,
            index=False,
            encoding="utf-8-sig",
        )

    coarse_df = run_coarse_scan(parameter_df, config, output)

    queue_df = select_refine_queue(coarse_df, config)
    queue_df.to_csv(
        output / "fes_step02_refine_queue.csv",
        index=False,
        encoding="utf-8-sig",
    )

    verified_df = run_refinement(queue_df, config, output)
    ml_df = build_ml_dataset(coarse_df, verified_df, output)
    write_summaries(coarse_df, verified_df, ml_df, output)

    print("\nFinal phase counts:")
    print(ml_df["ml_primary_class"].value_counts(dropna=False).to_string())
    print("\nStep 02 completed.")

    return {
        "output_dir": output,
        "parameters": parameter_path,
        "coarse_results": output / "fes_step02_coarse_results.csv",
        "verified_results": output / "fes_step02_verified_results.csv",
        "ml_dataset": output / "fes_step02_ml_dataset.csv",
        "topological_candidates": output / "fes_step02_topological_candidates.csv",
        "summary": output / "fes_step02_summary.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fes Step 02: five-dimensional Sobol topology scan"
    )
    parser.add_argument("--output", default="outputs_fes6_step02_sobol")
    parser.add_argument("--sobol-power", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用")
    config = ScanConfig(
        output_dir=args.output,
        sobol_power=args.sobol_power,
        sobol_seed=args.seed,
        workers=args.workers,
        resume=args.resume,
        overwrite=args.overwrite,
        test_mode=args.test,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
