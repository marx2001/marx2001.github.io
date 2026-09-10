#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01 — fes net 六参数八带交错磁模型：最小可验证闭环

任务目标
--------
1. 按用户 MagneticTB 导出的 fes Hamiltonian 实现 8x8 模型；
2. 检查 Hermiticity、自旋块解耦、周期规范及对称性简并；
3. 复现文献 Fig. 4(c) 的高对称路径能带；
4. 在完整二维 Brillouin 区审计直接带隙与间接带隙；
5. 在半填充下计算 spin-up、spin-down 和总占据子空间的 Chern 数；
6. 建立可靠的一级相标签，为后续 Sobol 采样和机器学习提供统一接口；
7. 修正闭隙样本标签逻辑，并检查半填充是否始终为 2-up + 2-down；
8. 可选地生成少量调试样本，或与 wannier90_hr.dat 做能谱对照。

模型信息
--------
磁空间群：P4'/mm'm，BNS #123.342
网络：fes net，Wyckoff 4o
原始参数：(e1, e2, t1, t2, r1, r2)
去除整体能量平移后的独立参数：(m_e, t1, t2, r1, r2)
矩阵维数：8
半填充：总占据 4 带；每个守恒自旋块占据 2 带

重要
----
本脚本只完成模型与数值标签器审计，不预设 fes 一定存在 type-II QSH 相。
非零且相反的 spin Chern 只标记为 spin_chern_TI_candidate；是否属于
confirmed type-II QSH，需要后续带反转、动量分离边缘态和输运验证。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=10, suppress=True)

# =============================================================================
# 0. 用户配置
# =============================================================================

CODE_VERSION = "FES_STEP01_V2_20260712"
RESULT_SCHEMA_VERSION = "fes_step01_schema_v2_59cols"

# 默认写入全新的版本化目录，避免旧 CSV 混入。
# 也可通过环境变量覆盖，例如：
# FES_STEP01_OUTPUT_DIR=my_results python FES_step01_v2.py
OUTPUT_DIR = Path(os.getenv("FES_STEP01_OUTPUT_DIR", "outputs_fes6_step01_v2"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 默认执行 24 个调试样本，以便直接生成修正后的 59 列结果。
# 临时只做文献参数审计时：
# FES_RUN_SMALL_BATCH=0 python FES_step01_v2.py
RUN_SMALL_BATCH = os.getenv("FES_RUN_SMALL_BATCH", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
N_BATCH = 24
BATCH_RANDOM_SEED = 20260712

# 8 带半填充
N_OCC_TOTAL = 4
N_OCC_SPIN = 2

# 调试网格。正式大数据标签阶段应增加网格、shift 共识和闭隙局部精修。
GAP_NK_DEBUG = 61
CHERN_NK_DEBUG = 25

GAP_TOL = 1.0e-3
CHERN_TOL = 0.08
MIN_DET_TOL = 1.0e-7

RAW6 = ["e1", "e2", "t1", "t2", "r1", "r2"]
REDUCED5 = ["m_e", "t1", "t2", "r1", "r2"]

# 用户当前 MagneticTB 输入采用 x=0.207，因此默认严格匹配导出的 207/500 与 293/1000 相位。
# 若以后使用 Mathematica 精确值重新导出，可改为：
# X_FES = 0.5 * (np.sqrt(2.0) - 1.0)
X_FES = 0.207
ALPHA = 2.0 * X_FES       # 0.414
BETA = 0.5 - X_FES        # 0.293

BASIS_LABELS = [
    "A_up", "A_down",
    "B_up", "B_down",
    "C_up", "C_down",
    "D_up", "D_down",
]

# 与用户 fes_ham.txt 的相位约定一致：
# A=(x,1/2), B=(-x,1/2), C=(1/2,-x), D=(1/2,x)
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

# 文献主文 Fig. 4(c) 参数
PAPER_PARAMS: Dict[str, float] = {
    "e1": 0.7,
    "e2": -0.4,
    "t1": 0.3,
    "t2": 0.2,
    "r1": 0.3,
    "r2": -0.1,
}

# 仅用于 Step 01 的小批量调试，不代表最终科学采样范围。
# 采用去除 e0 后的五维空间，避免把无物理意义的整体能量平移送入 ML。
DEBUG_REDUCED_BOUNDS = {
    "m_e": (-1.0, 1.0),
    "t1": (-0.8, 0.8),
    "t2": (-0.8, 0.8),
    "r1": (-0.8, 0.8),
    "r2": (-0.8, 0.8),
}


# =============================================================================
# 1. 参数处理与 fes 解析 Hamiltonian
# =============================================================================

def validate_params(params: Dict[str, float]) -> Dict[str, float]:
    """验证并标准化六参数字典。"""
    missing = [name for name in RAW6 if name not in params]
    extra = [name for name in params if name not in RAW6]
    if missing:
        raise ValueError(f"缺少参数: {missing}")
    if extra:
        raise ValueError(f"出现未定义参数: {extra}")

    clean = {name: float(params[name]) for name in RAW6}
    if not all(np.isfinite(value) for value in clean.values()):
        raise ValueError("参数中存在 NaN 或无穷大")
    return clean


def raw6_from_reduced5(reduced: Dict[str, float], e0: float = 0.0) -> Dict[str, float]:
    """由五维独立参数恢复六参数；e0 只平移全部能带。"""
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
    """输出供后续 ML 使用的低维候选特征；此处不预设其为拓扑判据。"""
    p = validate_params(params)
    root2 = np.sqrt(2.0)
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
    }


def h_fes_atomic(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    """
    用户 MagneticTB 导出 fes_ham.txt 对应的 8x8 原子规范 Hamiltonian。

    基底：A_up, A_down, B_up, B_down, C_up, C_down, D_up, D_down。
    """
    p = validate_params(params)
    e1, e2, t1, t2, r1, r2 = [p[name] for name in RAW6]

    # E11
    e11 = np.zeros((4, 4), dtype=np.complex128)
    np.fill_diagonal(e11, [e1, e2, e1, e2])
    e11[0, 2] = t1 * np.exp(-1j * ALPHA * kx)
    e11[2, 0] = np.conj(e11[0, 2])
    e11[1, 3] = t2 * np.exp(-1j * ALPHA * kx)
    e11[3, 1] = np.conj(e11[1, 3])

    # E22
    e22 = np.zeros((4, 4), dtype=np.complex128)
    np.fill_diagonal(e22, [e2, e1, e2, e1])
    e22[0, 2] = t2 * np.exp(+1j * ALPHA * ky)
    e22[2, 0] = np.conj(e22[0, 2])
    e22[1, 3] = t1 * np.exp(+1j * ALPHA * ky)
    e22[3, 1] = np.conj(e22[1, 3])

    # R12
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

    h = np.block([[e11, r12], [r12.conj().T, e22]])
    return h.astype(np.complex128)


def eigvals_full(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    return np.linalg.eigvalsh(h_fes_atomic(kx, ky, params))


def spin_indices(spin: str) -> list[int]:
    spin_key = spin.lower()
    if spin_key == "up":
        return SPIN_UP_INDICES.copy()
    if spin_key == "down":
        return SPIN_DOWN_INDICES.copy()
    raise ValueError("spin 必须为 'up' 或 'down'")


def h_spin_block_atomic(
    kx: float,
    ky: float,
    params: Dict[str, float],
    spin: str,
) -> np.ndarray:
    idx = spin_indices(spin)
    h = h_fes_atomic(kx, ky, params)
    return h[np.ix_(idx, idx)]


# =============================================================================
# 2. 周期规范
# =============================================================================

def periodic_gauge_matrix(kx: float, ky: float) -> np.ndarray:
    phases = np.exp(
        1j * (
            ORBITAL_POSITIONS[:, 0] * float(kx)
            + ORBITAL_POSITIONS[:, 1] * float(ky)
        )
    )
    return np.diag(phases).astype(np.complex128)


def h_fes_periodic(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    # 对当前 fes_ham.txt 相位约定，H_per = U H_atomic U^†。
    u = periodic_gauge_matrix(kx, ky)
    h = h_fes_atomic(kx, ky, params)
    return u @ h @ u.conj().T


def h_spin_block_periodic(
    kx: float,
    ky: float,
    params: Dict[str, float],
    spin: str,
) -> np.ndarray:
    idx = spin_indices(spin)
    h = h_fes_periodic(kx, ky, params)
    return h[np.ix_(idx, idx)]


# =============================================================================
# 3. 模型硬审计与对称性测试
# =============================================================================

def hermiticity_error(
    params: Dict[str, float],
    n_test: int = 40,
    seed: int = 1234,
) -> float:
    rng = np.random.default_rng(seed)
    max_error = 0.0
    for _ in range(n_test):
        kx, ky = rng.uniform(-np.pi, np.pi, size=2)
        h = h_fes_atomic(float(kx), float(ky), params)
        max_error = max(max_error, float(np.max(np.abs(h - h.conj().T))))
    return max_error


def spin_mixing_error(
    params: Dict[str, float],
    n_test: int = 40,
    seed: int = 2026,
) -> float:
    rng = np.random.default_rng(seed)
    max_error = 0.0
    for _ in range(n_test):
        kx, ky = rng.uniform(-np.pi, np.pi, size=2)
        h = h_fes_atomic(float(kx), float(ky), params)
        mix_ud = h[np.ix_(SPIN_UP_INDICES, SPIN_DOWN_INDICES)]
        mix_du = h[np.ix_(SPIN_DOWN_INDICES, SPIN_UP_INDICES)]
        max_error = max(
            max_error,
            float(np.max(np.abs(mix_ud))),
            float(np.max(np.abs(mix_du))),
        )
    return max_error


def periodicity_errors(
    params: Dict[str, float],
    n_test: int = 40,
    seed: int = 99,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    max_x = 0.0
    max_y = 0.0
    for _ in range(n_test):
        kx, ky = rng.uniform(-np.pi, np.pi, size=2)
        h0 = h_fes_periodic(float(kx), float(ky), params)
        hx = h_fes_periodic(float(kx + 2.0 * np.pi), float(ky), params)
        hy = h_fes_periodic(float(kx), float(ky + 2.0 * np.pi), params)
        max_x = max(max_x, float(np.max(np.abs(hx - h0))))
        max_y = max(max_y, float(np.max(np.abs(hy - h0))))
    return {"periodicity_error_x": max_x, "periodicity_error_y": max_y}


def gamma_pair_degeneracy_error(params: Dict[str, float]) -> float:
    """Γ 点最多四个能级且均为二重简并。"""
    e = np.sort(eigvals_full(0.0, 0.0, params))
    pair_errors = [abs(e[2 * i + 1] - e[2 * i]) for i in range(4)]
    return float(max(pair_errors))


def spin_spectrum_difference(
    kx: float,
    ky: float,
    params: Dict[str, float],
) -> float:
    up = np.linalg.eigvalsh(h_spin_block_atomic(kx, ky, params, "up"))
    down = np.linalg.eigvalsh(h_spin_block_atomic(kx, ky, params, "down"))
    return float(np.max(np.abs(np.sort(up) - np.sort(down))))


def sigma_degeneracy_errors(
    params: Dict[str, float],
    n_points: int = 41,
) -> Dict[str, float]:
    """检查 Σ:kx=ky 与 Σ':-kx=ky 路径上的自旋简并。"""
    qs = np.linspace(-np.pi, np.pi, n_points)
    sigma_error = 0.0
    sigma_prime_error = 0.0
    for q in qs:
        sigma_error = max(
            sigma_error,
            spin_spectrum_difference(float(q), float(q), params),
        )
        sigma_prime_error = max(
            sigma_prime_error,
            spin_spectrum_difference(float(-q), float(q), params),
        )
    return {
        "sigma_kx_eq_ky_error": sigma_error,
        "sigma_prime_minus_kx_eq_ky_error": sigma_prime_error,
    }


def run_model_tests(params: Dict[str, float]) -> Dict[str, float]:
    results = {
        "hermiticity_error": hermiticity_error(params),
        "spin_mixing_error": spin_mixing_error(params),
        "gamma_pair_degeneracy_error": gamma_pair_degeneracy_error(params),
    }
    results.update(periodicity_errors(params))
    results.update(sigma_degeneracy_errors(params))

    hard_tolerance = 1.0e-10
    failed = {key: value for key, value in results.items() if value > hard_tolerance}
    if failed:
        raise AssertionError(f"fes 模型硬审计失败: {failed}")
    return results


# =============================================================================
# 4. 高对称路径能带
# =============================================================================

HIGH_SYM_POINTS = [
    ("Γ", np.array([0.0, 0.0])),
    ("X", np.array([np.pi, 0.0])),
    ("M", np.array([np.pi, np.pi])),
    ("Γ", np.array([0.0, 0.0])),
    ("M′", np.array([-np.pi, np.pi])),
    ("X′", np.array([0.0, np.pi])),
    ("Γ", np.array([0.0, 0.0])),
]


def make_k_path(
    points: Sequence[tuple[str, np.ndarray]] = HIGH_SYM_POINTS,
    n_per_segment: int = 100,
):
    k_list: list[np.ndarray] = []
    x_list: list[float] = []
    tick_positions = [0.0]
    tick_labels = [points[0][0]]
    distance = 0.0

    for segment in range(len(points) - 1):
        _, ka = points[segment]
        label_b, kb = points[segment + 1]
        ts = np.linspace(0.0, 1.0, n_per_segment, endpoint=False)
        for t in ts:
            k = (1.0 - t) * ka + t * kb
            if k_list:
                distance += float(np.linalg.norm(k - k_list[-1]))
            k_list.append(k)
            x_list.append(distance)

        distance += float(np.linalg.norm(kb - k_list[-1]))
        k_list.append(kb.copy())
        x_list.append(distance)
        tick_positions.append(distance)
        tick_labels.append(label_b)

    return (
        np.asarray(k_list),
        np.asarray(x_list),
        tick_positions,
        tick_labels,
    )


def calculate_path_bands(
    params: Dict[str, float],
    n_per_segment: int = 100,
):
    k_list, x_axis, ticks, labels = make_k_path(n_per_segment=n_per_segment)
    bands = np.array(
        [eigvals_full(float(k[0]), float(k[1]), params) for k in k_list]
    )
    return k_list, x_axis, bands, ticks, labels


def plot_path_bands(
    params: Dict[str, float],
    title: str,
    n_per_segment: int = 100,
    save_name: str | None = None,
) -> None:
    _, x_axis, bands, ticks, labels = calculate_path_bands(
        params,
        n_per_segment=n_per_segment,
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for band_index in range(bands.shape[1]):
        ax.plot(x_axis, bands[:, band_index], linewidth=1.25)
    for position in ticks:
        ax.axvline(position, linewidth=0.6, alpha=0.45)
    ax.axhline(0.0, linewidth=0.7, alpha=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.set_xlim(x_axis[0], x_axis[-1])
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(OUTPUT_DIR / save_name, dpi=240, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. 完整 BZ 带隙审计
# =============================================================================

def scan_band_gaps(
    params: Dict[str, float],
    nk: int = GAP_NK_DEBUG,
    n_occ_total: int = N_OCC_TOTAL,
) -> Dict[str, float | int]:
    """在 [-pi,pi)^2 网格上审计半填充直接带隙与全局间接带隙。"""
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)

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

    # 对 fes 八带模型，半填充的目标 spin-Chern 子空间必须在每个 k 点
    # 都对应 2 条 spin-up 占据带 + 2 条 spin-down 占据带。
    # balanced_sector_gap > 0 等价于：两自旋块的第 2 条带全部低于第 3 条带。
    min_balanced_sector_gap = np.inf
    balanced_gap_kx = np.nan
    balanced_gap_ky = np.nan
    spin_occupancy_mismatch_count = 0
    min_n_up_in_lowest4 = N_OCC_TOTAL
    max_n_up_in_lowest4 = 0

    for kx in ks:
        for ky in ks:
            kx_f = float(kx)
            ky_f = float(ky)

            energies = eigvals_full(kx_f, ky_f, params)
            direct_gap = float(energies[n_occ_total] - energies[n_occ_total - 1])
            if direct_gap < min_direct:
                min_direct = direct_gap
                direct_kx = kx_f
                direct_ky = ky_f

            valence = float(energies[n_occ_total - 1])
            conduction = float(energies[n_occ_total])
            if valence > max_valence:
                max_valence = valence
                vbm_kx = kx_f
                vbm_ky = ky_f
            if conduction < min_conduction:
                min_conduction = conduction
                cbm_kx = kx_f
                cbm_ky = ky_f

            up_e = np.linalg.eigvalsh(
                h_spin_block_atomic(kx_f, ky_f, params, "up")
            )
            down_e = np.linalg.eigvalsh(
                h_spin_block_atomic(kx_f, ky_f, params, "down")
            )

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

            tagged = [(float(value), 1) for value in up_e]
            tagged += [(float(value), 0) for value in down_e]
            tagged.sort(key=lambda item: item[0])
            n_up_occ = int(sum(tag for _, tag in tagged[:n_occ_total]))
            min_n_up_in_lowest4 = min(min_n_up_in_lowest4, n_up_occ)
            max_n_up_in_lowest4 = max(max_n_up_in_lowest4, n_up_occ)
            if n_up_occ != N_OCC_SPIN:
                spin_occupancy_mismatch_count += 1

    indirect_gap = float(min_conduction - max_valence)
    balanced_sector_valid = (
        min_balanced_sector_gap > GAP_TOL
        and spin_occupancy_mismatch_count == 0
    )

    return {
        "grid_nk": int(nk),
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
        "spin_occupancy_mismatch_count": int(spin_occupancy_mismatch_count),
        "min_n_up_in_lowest4": int(min_n_up_in_lowest4),
        "max_n_up_in_lowest4": int(max_n_up_in_lowest4),
        "is_direct_gapped": int(min_direct > GAP_TOL),
        "is_physical_insulator": int(indirect_gap > GAP_TOL),
        "is_balanced_spin_sector": int(balanced_sector_valid),
    }


# =============================================================================
# 6. 非阿贝尔 Fukui Chern
# =============================================================================

def _normalize_link(value: complex, eps: float = 1.0e-14) -> complex:
    amplitude = abs(value)
    if amplitude < eps:
        raise FloatingPointError(f"Link determinant too small: {amplitude:.3e}")
    return value / amplitude


def fukui_chern_subspace(
    h_func: Callable[[float, float], np.ndarray],
    n_occ: int,
    nk: int = CHERN_NK_DEBUG,
) -> Dict[str, float | int]:
    """对多占据带子空间使用 determinant-link 非阿贝尔 Fukui 方法。"""
    sample_h = h_func(0.0, 0.0)
    dimension = sample_h.shape[0]
    if n_occ <= 0 or n_occ >= dimension:
        raise ValueError(f"n_occ={n_occ} 与 Hamiltonian 维数 {dimension} 不兼容")

    ks = 2.0 * np.pi * np.arange(nk) / nk
    occupied = np.empty((nk, nk, dimension, n_occ), dtype=np.complex128)

    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            _, eigenvectors = np.linalg.eigh(h_func(float(kx), float(ky)))
            occupied[ix, iy] = eigenvectors[:, :n_occ]

    total_phase = 0.0
    min_det_amplitude = 1.0

    for ix in range(nk):
        for iy in range(nk):
            v = occupied[ix, iy]
            vx = occupied[(ix + 1) % nk, iy]
            vy = occupied[ix, (iy + 1) % nk]
            vxy = occupied[(ix + 1) % nk, (iy + 1) % nk]

            link_x = np.linalg.det(v.conj().T @ vx)
            link_y = np.linalg.det(v.conj().T @ vy)
            link_x_at_y = np.linalg.det(vy.conj().T @ vxy)
            link_y_at_x = np.linalg.det(vx.conj().T @ vxy)

            min_det_amplitude = min(
                min_det_amplitude,
                abs(link_x),
                abs(link_y),
                abs(link_x_at_y),
                abs(link_y_at_x),
            )

            ux = _normalize_link(link_x)
            uy = _normalize_link(link_y)
            ux_at_y = _normalize_link(link_x_at_y)
            uy_at_x = _normalize_link(link_y_at_x)

            plaquette = ux * uy_at_x / (ux_at_y * uy)
            total_phase += float(np.angle(plaquette))

    return {
        "chern": float(total_phase / (2.0 * np.pi)),
        "min_det_amp": float(min_det_amplitude),
        "nk": int(nk),
    }


def calculate_spin_cherns(
    params: Dict[str, float],
    nk: int = CHERN_NK_DEBUG,
) -> Dict[str, float | int]:
    up = fukui_chern_subspace(
        lambda kx, ky: h_spin_block_periodic(kx, ky, params, "up"),
        n_occ=N_OCC_SPIN,
        nk=nk,
    )
    down = fukui_chern_subspace(
        lambda kx, ky: h_spin_block_periodic(kx, ky, params, "down"),
        n_occ=N_OCC_SPIN,
        nk=nk,
    )
    total = fukui_chern_subspace(
        lambda kx, ky: h_fes_periodic(kx, ky, params),
        n_occ=N_OCC_TOTAL,
        nk=nk,
    )

    c_up = float(up["chern"])
    c_down = float(down["chern"])
    c_total = float(total["chern"])
    return {
        "chern_nk": int(nk),
        "chern_up": c_up,
        "chern_down": c_down,
        "chern_total": c_total,
        "spin_chern": 0.5 * (c_up - c_down),
        "sum_rule_error": abs(c_total - (c_up + c_down)),
        "min_det_up": float(up["min_det_amp"]),
        "min_det_down": float(down["min_det_amp"]),
        "min_det_total": float(total["min_det_amp"]),
    }


def chern_convergence_table(
    params: Dict[str, float],
    grids: Iterable[int] = (15, 21, 31, 41),
) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for nk in grids:
        try:
            rows.append(calculate_spin_cherns(params, nk=int(nk)))
        except Exception as exc:  # 保留失败信息，避免静默丢样本
            rows.append({"chern_nk": int(nk), "error": repr(exc)})
    return pd.DataFrame(rows)


# =============================================================================
# 7. 单样本标签
# =============================================================================

def rounded_integer_if_close(value: float, tolerance: float = CHERN_TOL):
    nearest = int(np.rint(value))
    return nearest if abs(value - nearest) <= tolerance else None


def _empty_chern_fields() -> Dict[str, object]:
    """为未执行或失败的 Chern 计算补齐统一字段。"""
    return {
        "chern_nk": np.nan,
        "chern_up": np.nan,
        "chern_down": np.nan,
        "chern_total": np.nan,
        "spin_chern": np.nan,
        "sum_rule_error": np.nan,
        "min_det_up": np.nan,
        "min_det_down": np.nan,
        "min_det_total": np.nan,
        "chern_up_int": None,
        "chern_down_int": None,
        "chern_total_int": None,
        "chern_reliable": 0,
    }


def calculate_sample(
    sample_id: str,
    params: Dict[str, float],
    gap_nk: int = GAP_NK_DEBUG,
    chern_nk: int = CHERN_NK_DEBUG,
) -> Dict[str, object]:
    """
    对单个参数点执行严格标签。

    逻辑顺序：
    1. 先完成完整 BZ 带隙审计；
    2. 若总直接带隙闭合，立即标记为 noninsulating_or_gap_closing，
       不再计算没有定义的总占据子空间 Chern；
    3. 若半填充不是全 BZ 的 2-up + 2-down 平衡占据，单独标记；
    4. 只有满足以上条件时才计算非阿贝尔 spin-Chern；
    5. Chern 数值失败保留已获得的 gap 信息，并标记 chern_unreliable，
       不再误标为 numeric_error。
    """
    p = validate_params(params)
    row: Dict[str, object] = {
        "sample_id": sample_id,
        **p,
        **derived_features(p),
        **_empty_chern_fields(),
        "is_spin_chern_TI_candidate": 0,
        "is_typeII_QSH_confirmed": 0,
        "phase_label": "numeric_error",
        "error": "",
    }

    # ------------------------------------------------------------------
    # A. 完整 BZ 带隙审计。这里若失败才属于真正的 numeric_error。
    # ------------------------------------------------------------------
    try:
        gap = scan_band_gaps(p, nk=gap_nk)
        row.update(gap)
    except Exception as exc:
        row["phase_label"] = "numeric_error"
        row["error"] = f"gap_scan_failed: {repr(exc)}"
        return row

    direct_gapped = float(gap["min_direct_gap"]) > GAP_TOL
    physical_insulator = float(gap["indirect_gap"]) > GAP_TOL
    balanced_spin_sector = bool(int(gap["is_balanced_spin_sector"]))

    # 先判断直接带隙。闭隙时 Chern 不定义，禁止继续调用 Fukui。
    if not direct_gapped:
        row["phase_label"] = "noninsulating_or_gap_closing"
        return row

    # 总占据流形虽有直接带隙，但最低四带并非处处为 2-up + 2-down。
    # 这种样本不属于当前固定 N_OCC_SPIN=2 的 spin-Chern 分类问题。
    if not balanced_spin_sector:
        row["phase_label"] = "spin_sector_filling_mismatch"
        return row

    # ------------------------------------------------------------------
    # B. 仅对有定义的 2+2 占据子空间计算 Chern。
    # ------------------------------------------------------------------
    try:
        chern = calculate_spin_cherns(p, nk=chern_nk)
        row.update(chern)
    except Exception as exc:
        row["phase_label"] = "chern_unreliable"
        row["error"] = f"chern_failed: {repr(exc)}"
        return row

    c_up_int = rounded_integer_if_close(float(chern["chern_up"]))
    c_down_int = rounded_integer_if_close(float(chern["chern_down"]))
    c_total_int = rounded_integer_if_close(float(chern["chern_total"]))

    reliable = (
        c_up_int is not None
        and c_down_int is not None
        and c_total_int is not None
        and float(chern["min_det_up"]) > MIN_DET_TOL
        and float(chern["min_det_down"]) > MIN_DET_TOL
        and float(chern["min_det_total"]) > MIN_DET_TOL
        and float(chern["sum_rule_error"]) < CHERN_TOL
    )

    row["chern_up_int"] = c_up_int
    row["chern_down_int"] = c_down_int
    row["chern_total_int"] = c_total_int
    row["chern_reliable"] = int(reliable)

    if not reliable:
        row["phase_label"] = "chern_unreliable"
        return row

    spin_chern_nonzero = (
        c_up_int is not None
        and c_down_int is not None
        and c_total_int == 0
        and c_up_int == -c_down_int
        and c_up_int != 0
    )

    spin_chern_candidate = physical_insulator and spin_chern_nonzero
    row["is_spin_chern_TI_candidate"] = int(spin_chern_candidate)

    # 候选不直接等同于 confirmed type-II QSH。
    row["is_typeII_QSH_confirmed"] = 0

    if not physical_insulator:
        if spin_chern_nonzero:
            label = "spin_chern_band_metal"
        else:
            label = "indirect_overlap"
    elif spin_chern_candidate:
        label = "spin_chern_TI_candidate"
    elif c_up_int == 0 and c_down_int == 0 and c_total_int == 0:
        label = "trivial_insulator"
    elif c_total_int not in (None, 0):
        label = "Chern_insulator"
    else:
        label = "other_gapped_phase"

    row["phase_label"] = label
    return row


# =============================================================================
# 8. 小批量调试采样
# =============================================================================

def sample_reduced_debug_parameters(
    n: int,
    bounds: Dict[str, tuple[float, float]],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        reduced = {
            name: float(rng.uniform(bounds[name][0], bounds[name][1]))
            for name in REDUCED5
        }
        raw = raw6_from_reduced5(reduced, e0=0.0)
        rows.append({"sample_id": f"debug_{index:06d}", **raw})
    return pd.DataFrame(rows)


def run_small_batch() -> pd.DataFrame:
    parameter_df = sample_reduced_debug_parameters(
        n=N_BATCH,
        bounds=DEBUG_REDUCED_BOUNDS,
        seed=BATCH_RANDOM_SEED,
    )
    parameter_df.to_csv(
        OUTPUT_DIR / "fes6_debug_parameter_batch.csv",
        index=False,
        encoding="utf-8-sig",
    )

    results = []
    for index, row in parameter_df.iterrows():
        params = {name: float(row[name]) for name in RAW6}
        results.append(
            calculate_sample(
                sample_id=str(row["sample_id"]),
                params=params,
                gap_nk=41,
                chern_nk=21,
            )
        )
        if (index + 1) % 4 == 0 or (index + 1) == len(parameter_df):
            print(f"Finished debug batch: {index + 1}/{len(parameter_df)}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        OUTPUT_DIR / "fes6_debug_batch_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result_df


# =============================================================================
# 9. 可选：wannier90_hr.dat 对照
# =============================================================================

def read_wannier90_hr(path: str | Path) -> Dict[str, object]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"文件过短: {path}")

    num_wann = int(lines[1].strip())
    nrpts = int(lines[2].strip())

    degeneracies: list[int] = []
    line_index = 3
    while len(degeneracies) < nrpts:
        degeneracies.extend(int(value) for value in lines[line_index].split())
        line_index += 1
    degeneracies = degeneracies[:nrpts]

    records = []
    for line in lines[line_index:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        rx, ry, rz, m, n = map(int, parts[:5])
        real_part, imag_part = map(float, parts[5:7])
        records.append((rx, ry, rz, m - 1, n - 1, real_part + 1j * imag_part))

    expected = nrpts * num_wann * num_wann
    if len(records) != expected:
        raise ValueError(f"hr 数据行数不匹配: got={len(records)}, expected={expected}")

    hr: Dict[tuple[int, int, int], np.ndarray] = {}
    cursor = 0
    for ir in range(nrpts):
        r_vector = records[cursor][:3]
        matrix = np.zeros((num_wann, num_wann), dtype=np.complex128)
        for _ in range(num_wann * num_wann):
            rx, ry, rz, m, n, value = records[cursor]
            if (rx, ry, rz) != r_vector:
                raise ValueError("hr.dat 中 R 分组不连续或格式异常")
            matrix[m, n] = value
            cursor += 1
        hr[r_vector] = matrix / float(degeneracies[ir])

    return {"path": path, "num_wann": num_wann, "nrpts": nrpts, "hr": hr}


def hk_from_hr(
    hr_data: Dict[str, object],
    kx: float,
    ky: float,
    kz: float = 0.0,
) -> np.ndarray:
    num_wann = int(hr_data["num_wann"])
    h = np.zeros((num_wann, num_wann), dtype=np.complex128)
    hr = hr_data["hr"]
    if not isinstance(hr, dict):
        raise TypeError("hr_data['hr'] 不是字典")

    for (rx, ry, rz), matrix in hr.items():
        phase = np.exp(1j * (kx * rx + ky * ry + kz * rz))
        h += matrix * phase
    return 0.5 * (h + h.conj().T)


def compare_hr_spectra(
    hr_path: str | Path,
    params: Dict[str, float],
    n_test: int = 20,
    seed: int = 20260712,
) -> pd.DataFrame:
    hr_data = read_wannier90_hr(hr_path)
    if int(hr_data["num_wann"]) != 8:
        raise ValueError(
            f"fes 六参数模型应有 8 个 Wannier 轨道，当前为 {hr_data['num_wann']}"
        )

    rng = np.random.default_rng(seed)
    rows = []
    for test_index in range(n_test):
        kx, ky = rng.uniform(-np.pi, np.pi, size=2)
        analytic = np.linalg.eigvalsh(h_fes_periodic(float(kx), float(ky), params))
        from_hr = np.linalg.eigvalsh(hk_from_hr(hr_data, float(kx), float(ky)))
        rows.append(
            {
                "test": test_index,
                "kx": float(kx),
                "ky": float(ky),
                "max_eigenvalue_error": float(np.max(np.abs(analytic - from_hr))),
            }
        )
    return pd.DataFrame(rows)



def write_run_metadata() -> Dict[str, object]:
    """写入版本、脚本哈希和关键数值设置，便于确认结果来源。"""
    source_name = "Jupyter notebook"
    source_sha256 = None

    if "__file__" in globals():
        candidate = Path(__file__).resolve()
        source_name = candidate.name
        if candidate.exists():
            source_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()

    metadata: Dict[str, object] = {
        "code_version": CODE_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "source_sha256": source_sha256,
        "output_directory": str(OUTPUT_DIR.resolve()),
        "run_small_batch": bool(RUN_SMALL_BATCH),
        "n_batch": int(N_BATCH),
        "batch_random_seed": int(BATCH_RANDOM_SEED),
        "gap_nk_debug": int(GAP_NK_DEBUG),
        "chern_nk_debug": int(CHERN_NK_DEBUG),
        "gap_tolerance": float(GAP_TOL),
        "chern_tolerance": float(CHERN_TOL),
        "min_det_tolerance": float(MIN_DET_TOL),
        "expected_debug_result_columns": 59,
    }

    path = OUTPUT_DIR / "fes_step01_run_metadata.json"
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


# =============================================================================
# 10. 主程序
# =============================================================================

def main() -> None:
    metadata = write_run_metadata()

    print("=" * 78)
    print("Step 01: fes net six-parameter eight-band model audit")
    print("Version:", CODE_VERSION)
    print("Schema :", RESULT_SCHEMA_VERSION)
    print("Output :", OUTPUT_DIR.resolve())
    print("Batch  :", RUN_SMALL_BATCH)
    print("Basis  :", BASIS_LABELS)
    print("x      :", X_FES)
    print("SHA256 :", metadata.get("source_sha256"))
    print("=" * 78)

    # 1) 模型硬审计
    tests = run_model_tests(PAPER_PARAMS)
    test_df = pd.DataFrame([tests])
    print("\n[Model tests]")
    print(test_df.to_string(index=False))
    test_df.to_csv(
        OUTPUT_DIR / "fes6_model_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2) 文献参数能带
    plot_path_bands(
        PAPER_PARAMS,
        title="fes eight-band model: literature parameter set",
        save_name="fes_paper_parameter_bands.png",
    )
    print("\nBand plot written:", OUTPUT_DIR / "fes_paper_parameter_bands.png")

    # 3) 完整 BZ gap
    paper_gap = scan_band_gaps(PAPER_PARAMS, nk=GAP_NK_DEBUG)
    gap_df = pd.DataFrame([{"case": "paper", **paper_gap}])
    print("\n[Full-BZ gap audit]")
    print(gap_df.to_string(index=False))
    gap_df.to_csv(
        OUTPUT_DIR / "fes_paper_gap_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4) Chern 网格收敛
    chern_df = chern_convergence_table(PAPER_PARAMS, grids=(15, 21, 31, 41))
    print("\n[Chern convergence: paper parameters]")
    print(chern_df.to_string(index=False))
    chern_df.to_csv(
        OUTPUT_DIR / "fes_paper_chern_convergence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 5) 单样本统一标签
    paper_result = calculate_sample(
        sample_id="paper_fes",
        params=PAPER_PARAMS,
        gap_nk=GAP_NK_DEBUG,
        chern_nk=CHERN_NK_DEBUG,
    )
    print("\n[Paper sample label]")
    print(json.dumps(paper_result, ensure_ascii=False, indent=2))
    pd.DataFrame([paper_result]).to_csv(
        OUTPUT_DIR / "fes_paper_sample_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 6) 可选小批量
    if RUN_SMALL_BATCH:
        batch_df = run_small_batch()
        print("\n[Debug batch labels]")
        print(batch_df["phase_label"].value_counts(dropna=False).to_string())
    else:
        print("\nRUN_SMALL_BATCH=False：跳过小批量。")

    # 7) 可选 hr.dat 对照
    hr_candidates = [
        Path("wannier90_hr.dat"),
        Path("wannier90_test.dat"),
        Path("fes_wannier90_hr.dat"),
    ]
    existing_hr = next((path for path in hr_candidates if path.exists()), None)
    if existing_hr is not None:
        comparison = compare_hr_spectra(existing_hr, PAPER_PARAMS)
        print("\n[hr.dat spectrum comparison]")
        print(comparison.to_string(index=False))
        comparison.to_csv(
            OUTPUT_DIR / "fes_hr_spectrum_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        print("\n未发现 wannier90_hr.dat，跳过 MagneticTB/Wannier90 对照。")

    print("\nStep 01 finished.")


if __name__ == "__main__":
    main()
