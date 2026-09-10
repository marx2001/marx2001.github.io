#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 01 v2 — 八带交错磁 Hamiltonian 与严格拓扑标签器调试

本脚本参考用户提供的 01_Lieb8_AMTI_pipeline_debug.ipynb 的组织方式，
将同一套“模型定义 → 单元测试 → 能带 → 全 BZ 带隙 → 周期规范 →
spin-Chern → 小批量标签”流程改写为 tts 网络版本。

核心模型
--------
- tts tessellation: 3.3.4.3.4
- magnetic space group: P4'/mbm' (BNS 127.391)
- magnetic Wyckoff position: 4h
- representative coordinate: (x, x+1/2, 1/2), x = 0.183
- raw parameters: (e1,e2,t1,t2,r1,r2,r3,r4)
- reduced parameters after removing e0: (m_e,t1,t2,r1,r2,r3,r4)
- eight bands, half filling: four occupied bands

重要实现说明
------------
1. 全部计算均为串行，不使用 ProcessPoolExecutor，因此可直接在 Windows/Jupyter 运行。
2. Chern 数使用周期规范 Hamiltonian；不能直接在含分数坐标相位的 atomic gauge 上跨 BZ 边界。
3. 只有在完整 BZ 中最低四带始终为 2↑+2↓，且直接带隙为正时，才计算物理 spin-Chern 标签。
4. “spin_chern_TI_candidate”仅表示模型级候选；边缘态、DOS 与 SHC 留给后续步骤验证。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence
import argparse
import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=10, suppress=True)

# =============================================================================
# Step 0. Constants and configuration
# =============================================================================

CODE_VERSION = "TTS_STEP01_V2_20260713"
RAW8 = ["e1", "e2", "t1", "t2", "r1", "r2", "r3", "r4"]
REDUCED7 = ["m_e", "t1", "t2", "r1", "r2", "r3", "r4"]

N_OCC_TOTAL = 4
N_OCC_SPIN = 2

X_TTS = 0.183
ALPHA = 2.0 * X_TTS          # 0.366 = 183/500
BETA = 0.5 - 2.0 * X_TTS     # 0.134 = 67/500

SPIN_UP_INDICES = [0, 2, 4, 6]
SPIN_DOWN_INDICES = [1, 3, 5, 7]

# MagneticTB basis positions in the cached tts.nb Hamiltonian.
ORBITAL_POSITIONS = np.array(
    [
        [X_TTS, X_TTS + 0.5], [X_TTS, X_TTS + 0.5],
        [X_TTS + 0.5, -X_TTS], [X_TTS + 0.5, -X_TTS],
        [0.5 - X_TTS, X_TTS], [0.5 - X_TTS, X_TTS],
        [-X_TTS, 0.5 - X_TTS], [-X_TTS, 0.5 - X_TTS],
    ],
    dtype=float,
)

PAPER_PARAMS: Dict[str, float] = {
    "e1": 0.5,
    "e2": -0.4,
    "t1": 0.2,
    "t2": 0.1,
    "r1": 0.3,
    "r2": -0.1,
    "r3": 0.1,
    "r4": -0.4,
}

# Step 01 小批量中发现的正全局带隙 spin-Chern 候选。
TOPO_PARAMS: Dict[str, float] = {
    "e1": 0.003886630742,
    "e2": -0.003886630742,
    "t1": 0.330302133158,
    "t2": 0.691254202294,
    "r1": 0.793309071205,
    "r2": -0.491195093495,
    "r3": -0.542657657699,
    "r4": -0.578737306339,
}

DEBUG_BOUNDS = {
    "m_e": (-1.0, 1.0),
    "t1": (-0.8, 0.8),
    "t2": (-0.8, 0.8),
    "r1": (-0.8, 0.8),
    "r2": (-0.8, 0.8),
    "r3": (-0.8, 0.8),
    "r4": (-0.8, 0.8),
}


@dataclass
class Step01Config:
    output_dir: Path = Path("outputs_tts_step01_v2")
    run_small_batch: bool = True
    n_batch: int = 24
    batch_seed: int = 20260713
    debug_gap_nk: int = 41
    debug_chern_nk: int = 21
    paper_gap_nk: int = 81
    paper_chern_grids: tuple[int, ...] = (21, 31, 41)
    strict_gap_grids: tuple[int, ...] = (51, 71, 101)
    strict_chern_grids: tuple[int, ...] = (21, 31, 41)
    gap_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)
    )
    chern_shifts: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.5))
    gap_tol: float = 1.0e-3
    chern_tol: float = 0.08
    min_det_tol: float = 1.0e-7

    def normalized(self) -> "Step01Config":
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.n_batch < 0:
            raise ValueError("n_batch must be >= 0")
        for name in ("debug_gap_nk", "debug_chern_nk", "paper_gap_nk"):
            if int(getattr(self, name)) < 5:
                raise ValueError(f"{name} must be >= 5")
        return self


# =============================================================================
# Step 1. Parameters and tts Hamiltonian
# =============================================================================

def validate_params(params: Dict[str, float]) -> Dict[str, float]:
    missing = [name for name in RAW8 if name not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}")
    clean = {name: float(params[name]) for name in RAW8}
    if not all(np.isfinite(v) for v in clean.values()):
        raise ValueError("Parameter vector contains NaN or infinity")
    return clean


def raw8_from_reduced7(reduced: Dict[str, float], e0: float = 0.0) -> Dict[str, float]:
    missing = [name for name in REDUCED7 if name not in reduced]
    if missing:
        raise ValueError(f"Missing reduced parameters: {missing}")
    m_e = float(reduced["m_e"])
    return {
        "e1": float(e0) + m_e,
        "e2": float(e0) - m_e,
        "t1": float(reduced["t1"]),
        "t2": float(reduced["t2"]),
        "r1": float(reduced["r1"]),
        "r2": float(reduced["r2"]),
        "r3": float(reduced["r3"]),
        "r4": float(reduced["r4"]),
    }


def reduced7_from_raw8(params: Dict[str, float]) -> Dict[str, float]:
    p = validate_params(params)
    return {
        "m_e": 0.5 * (p["e1"] - p["e2"]),
        "t1": p["t1"], "t2": p["t2"],
        "r1": p["r1"], "r2": p["r2"],
        "r3": p["r3"], "r4": p["r4"],
    }


def derived_features(params: Dict[str, float]) -> Dict[str, float]:
    p = validate_params(params)
    root2 = math.sqrt(2.0)
    return {
        "e0": 0.5 * (p["e1"] + p["e2"]),
        "m_e": 0.5 * (p["e1"] - p["e2"]),
        "t_sum": p["t1"] + p["t2"],
        "t_diff": p["t1"] - p["t2"],
        "t_product": p["t1"] * p["t2"],
        "t_s": (p["t1"] + p["t2"]) / root2,
        "t_d": (p["t1"] - p["t2"]) / root2,
        "r13_sum": p["r1"] + p["r3"],
        "r13_diff": p["r1"] - p["r3"],
        "r24_sum": p["r2"] + p["r4"],
        "r24_diff": p["r2"] - p["r4"],
        "r_all_sum": p["r1"] + p["r2"] + p["r3"] + p["r4"],
    }


def h_tts_atomic(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    """Supplemental Material Eqs. S24-S25 in the atomic Bloch gauge."""
    p = validate_params(params)
    e1, e2, t1, t2, r1, r2, r3, r4 = [p[name] for name in RAW8]

    r = 1j * r1 + r3
    rp = 1j * r2 + r4

    xi = np.exp(1j * (0.5 * kx + BETA * ky))
    xip = np.exp(1j * (-0.5 * kx + BETA * ky))
    eta = np.exp(1j * (BETA * kx + 0.5 * ky))
    etap = np.exp(1j * (-BETA * kx + 0.5 * ky))
    phi = np.exp(1j * (ALPHA * kx + ALPHA * ky))
    phip = np.exp(1j * (-ALPHA * kx + ALPHA * ky))

    h = np.zeros((8, 8), dtype=np.complex128)
    np.fill_diagonal(h, [e1, e2, e2, e1, e2, e1, e1, e2])

    h[0, 2] = xi * r + xip * np.conj(rp)
    h[0, 4] = eta * np.conj(r) + np.conj(etap) * rp
    h[0, 6] = np.conj(phi) * t1

    h[1, 3] = xip * np.conj(r) + xi * rp
    h[1, 5] = np.conj(etap) * r + eta * np.conj(rp)
    h[1, 7] = np.conj(phi) * t2

    h[2, 4] = phip * t2
    h[2, 6] = eta * r + np.conj(etap) * np.conj(rp)

    h[3, 5] = phip * t1
    h[3, 7] = np.conj(etap) * np.conj(r) + eta * rp

    h[4, 6] = xi * np.conj(r) + xip * rp
    h[5, 7] = xip * r + xi * np.conj(rp)

    h = h + np.triu(h, 1).conj().T
    return h


def periodic_gauge_matrix(kx: float, ky: float) -> np.ndarray:
    phases = np.exp(
        1j * (
            ORBITAL_POSITIONS[:, 0] * float(kx)
            + ORBITAL_POSITIONS[:, 1] * float(ky)
        )
    )
    return np.diag(phases).astype(np.complex128)


def h_tts_periodic(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    u = periodic_gauge_matrix(kx, ky)
    h = h_tts_atomic(kx, ky, params)
    return u @ h @ u.conj().T


def spin_indices(spin: str) -> list[int]:
    key = spin.lower()
    if key == "up":
        return SPIN_UP_INDICES.copy()
    if key == "down":
        return SPIN_DOWN_INDICES.copy()
    raise ValueError("spin must be 'up' or 'down'")


def h_spin_block_atomic(kx: float, ky: float, params: Dict[str, float], spin: str) -> np.ndarray:
    idx = spin_indices(spin)
    h = h_tts_atomic(kx, ky, params)
    return h[np.ix_(idx, idx)]


def h_spin_block_periodic(kx: float, ky: float, params: Dict[str, float], spin: str) -> np.ndarray:
    idx = spin_indices(spin)
    h = h_tts_periodic(kx, ky, params)
    return h[np.ix_(idx, idx)]


def eigvals_full(kx: float, ky: float, params: Dict[str, float]) -> np.ndarray:
    return np.linalg.eigvalsh(h_tts_atomic(kx, ky, params))


# =============================================================================
# Step 2. Hard audits
# =============================================================================

def spin_spectrum_difference(kx: float, ky: float, params: Dict[str, float]) -> float:
    eu = np.linalg.eigvalsh(h_spin_block_atomic(kx, ky, params, "up"))
    ed = np.linalg.eigvalsh(h_spin_block_atomic(kx, ky, params, "down"))
    return float(np.max(np.abs(np.sort(eu) - np.sort(ed))))


def model_tests(params: Dict[str, float], n_random: int = 40, tol: float = 1.0e-10) -> Dict[str, float]:
    rng = np.random.default_rng(127391)
    herm = spin_mixing = per_x = per_y = 0.0

    for _ in range(n_random):
        kx, ky = rng.uniform(-np.pi, np.pi, size=2)
        h = h_tts_atomic(float(kx), float(ky), params)
        herm = max(herm, float(np.max(np.abs(h - h.conj().T))))
        spin_mixing = max(
            spin_mixing,
            float(np.max(np.abs(h[np.ix_(SPIN_UP_INDICES, SPIN_DOWN_INDICES)]))),
            float(np.max(np.abs(h[np.ix_(SPIN_DOWN_INDICES, SPIN_UP_INDICES)]))),
        )
        hp = h_tts_periodic(float(kx), float(ky), params)
        per_x = max(
            per_x,
            float(np.max(np.abs(h_tts_periodic(kx + 2*np.pi, ky, params) - hp))),
        )
        per_y = max(
            per_y,
            float(np.max(np.abs(h_tts_periodic(kx, ky + 2*np.pi, params) - hp))),
        )

    def pair_error(kx: float, ky: float) -> float:
        e = np.sort(eigvals_full(kx, ky, params))
        return float(max(abs(e[2*i + 1] - e[2*i]) for i in range(4)))

    qs = np.linspace(-np.pi, np.pi, 101)
    results = {
        "hermiticity_error": herm,
        "spin_mixing_error": spin_mixing,
        "periodicity_error_x": per_x,
        "periodicity_error_y": per_y,
        "gamma_pair_degeneracy_error": pair_error(0.0, 0.0),
        "M_pair_degeneracy_error": pair_error(np.pi, np.pi),
        "Mprime_pair_degeneracy_error": pair_error(-np.pi, np.pi),
        "delta_ky0_spin_degeneracy_error": max(
            spin_spectrum_difference(float(q), 0.0, params) for q in qs
        ),
        "delta_prime_kx0_spin_degeneracy_error": max(
            spin_spectrum_difference(0.0, float(q), params) for q in qs
        ),
        "Z_kxpi_spin_degeneracy_error": max(
            spin_spectrum_difference(np.pi, float(q), params) for q in qs
        ),
        "Z_prime_kypi_spin_degeneracy_error": max(
            spin_spectrum_difference(float(q), np.pi, params) for q in qs
        ),
        "sigma_kx_eq_ky_max_spin_splitting": max(
            spin_spectrum_difference(float(q), float(q), params) for q in qs
        ),
        "sigma_prime_minus_kx_eq_ky_max_spin_splitting": max(
            spin_spectrum_difference(float(-q), float(q), params) for q in qs
        ),
    }

    hard_keys = [
        "hermiticity_error", "spin_mixing_error",
        "periodicity_error_x", "periodicity_error_y",
        "gamma_pair_degeneracy_error", "M_pair_degeneracy_error",
        "Mprime_pair_degeneracy_error",
        "delta_ky0_spin_degeneracy_error",
        "delta_prime_kx0_spin_degeneracy_error",
        "Z_kxpi_spin_degeneracy_error",
        "Z_prime_kypi_spin_degeneracy_error",
    ]
    failed = {key: results[key] for key in hard_keys if results[key] > tol}
    if failed:
        raise AssertionError(f"tts Hamiltonian hard audit failed: {failed}")
    if results["sigma_kx_eq_ky_max_spin_splitting"] < 1.0e-6:
        raise AssertionError("Expected Sigma spin splitting was not found")
    if results["sigma_prime_minus_kx_eq_ky_max_spin_splitting"] < 1.0e-6:
        raise AssertionError("Expected Sigma' spin splitting was not found")
    return results


# =============================================================================
# Step 3. High-symmetry path and bands
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

    for seg in range(len(points) - 1):
        _, ka = points[seg]
        label_b, kb = points[seg + 1]
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

    return np.asarray(k_list), np.asarray(x_list), tick_positions, tick_labels


def path_band_data(params: Dict[str, float], n_per_segment: int = 100):
    k_list, x_axis, ticks, labels = make_k_path(n_per_segment=n_per_segment)
    up = np.array([
        np.linalg.eigvalsh(h_spin_block_atomic(float(k[0]), float(k[1]), params, "up"))
        for k in k_list
    ])
    down = np.array([
        np.linalg.eigvalsh(h_spin_block_atomic(float(k[0]), float(k[1]), params, "down"))
        for k in k_list
    ])
    return k_list, x_axis, ticks, labels, up, down


def plot_path_bands(
    params: Dict[str, float],
    save_path: Path,
    title: str,
    n_per_segment: int = 100,
) -> pd.DataFrame:
    k_list, x_axis, ticks, labels, up, down = path_band_data(params, n_per_segment)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for ib in range(up.shape[1]):
        ax.plot(x_axis, up[:, ib], linewidth=1.25, label="spin up" if ib == 0 else None)
        ax.plot(
            x_axis, down[:, ib], linewidth=1.10, linestyle="--",
            label="spin down" if ib == 0 else None,
        )
    for tick in ticks:
        ax.axvline(tick, linewidth=0.6, alpha=0.45)
    ax.axhline(0.0, linewidth=0.7, alpha=0.5)
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    data = {
        "path_x": x_axis,
        "kx": k_list[:, 0],
        "ky": k_list[:, 1],
    }
    for ib in range(4):
        data[f"up_{ib+1}"] = up[:, ib]
        data[f"down_{ib+1}"] = down[:, ib]
    return pd.DataFrame(data)


# =============================================================================
# Step 4. Full-BZ gap and spin-filling audit
# =============================================================================

def bz_grid(nk: int, shift: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    # Periodic grid in [-pi, pi), shifted by fractions of one mesh spacing.
    kx = -np.pi + 2.0 * np.pi * (np.arange(nk) + float(shift[0])) / nk
    ky = -np.pi + 2.0 * np.pi * (np.arange(nk) + float(shift[1])) / nk
    return kx, ky


def scan_band_gaps(
    params: Dict[str, float],
    nk: int = 81,
    shift: tuple[float, float] = (0.0, 0.0),
    gap_tol: float = 1.0e-3,
) -> Dict[str, float | int]:
    ks_x, ks_y = bz_grid(int(nk), shift)

    min_direct = np.inf
    max_valence = -np.inf
    min_conduction = np.inf
    min_balanced = np.inf
    min_spin_up = np.inf
    min_spin_down = np.inf
    mismatch_count = 0
    min_n_up = N_OCC_TOTAL
    max_n_up = 0

    direct_k = (np.nan, np.nan)
    vbm_k = (np.nan, np.nan)
    cbm_k = (np.nan, np.nan)
    balanced_k = (np.nan, np.nan)

    for kx in ks_x:
        for ky in ks_y:
            kx_f, ky_f = float(kx), float(ky)
            e = eigvals_full(kx_f, ky_f, params)
            direct = float(e[N_OCC_TOTAL] - e[N_OCC_TOTAL - 1])
            if direct < min_direct:
                min_direct = direct
                direct_k = (kx_f, ky_f)
            if float(e[N_OCC_TOTAL - 1]) > max_valence:
                max_valence = float(e[N_OCC_TOTAL - 1])
                vbm_k = (kx_f, ky_f)
            if float(e[N_OCC_TOTAL]) < min_conduction:
                min_conduction = float(e[N_OCC_TOTAL])
                cbm_k = (kx_f, ky_f)

            eu = np.linalg.eigvalsh(h_spin_block_atomic(kx_f, ky_f, params, "up"))
            ed = np.linalg.eigvalsh(h_spin_block_atomic(kx_f, ky_f, params, "down"))
            min_spin_up = min(min_spin_up, float(eu[2] - eu[1]))
            min_spin_down = min(min_spin_down, float(ed[2] - ed[1]))

            balanced = float(min(eu[2], ed[2]) - max(eu[1], ed[1]))
            if balanced < min_balanced:
                min_balanced = balanced
                balanced_k = (kx_f, ky_f)

            tagged = [(float(x), 1) for x in eu] + [(float(x), 0) for x in ed]
            tagged.sort(key=lambda item: item[0])
            n_up = int(sum(tag for _, tag in tagged[:N_OCC_TOTAL]))
            min_n_up = min(min_n_up, n_up)
            max_n_up = max(max_n_up, n_up)
            mismatch_count += int(n_up != N_OCC_SPIN)

    indirect = float(min_conduction - max_valence)
    return {
        "grid_nk": int(nk),
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
        "min_direct_gap": float(min_direct),
        "indirect_gap": indirect,
        "direct_gap_kx": direct_k[0], "direct_gap_ky": direct_k[1],
        "vbm": float(max_valence), "vbm_kx": vbm_k[0], "vbm_ky": vbm_k[1],
        "cbm": float(min_conduction), "cbm_kx": cbm_k[0], "cbm_ky": cbm_k[1],
        "min_spin_gap_up": float(min_spin_up),
        "min_spin_gap_down": float(min_spin_down),
        "min_balanced_sector_gap": float(min_balanced),
        "balanced_gap_kx": balanced_k[0], "balanced_gap_ky": balanced_k[1],
        "spin_occupancy_mismatch_count": int(mismatch_count),
        "min_n_up_in_lowest4": int(min_n_up),
        "max_n_up_in_lowest4": int(max_n_up),
        "is_direct_gapped": int(min_direct > gap_tol),
        "is_physical_insulator": int(indirect > gap_tol),
        "is_balanced_spin_sector": int(min_balanced > gap_tol and mismatch_count == 0),
    }


# =============================================================================
# Step 5. Non-Abelian Fukui spin-Chern
# =============================================================================

def _normalize_link(value: complex, eps: float = 1.0e-14) -> complex:
    amp = abs(value)
    if amp < eps:
        raise FloatingPointError(f"Link determinant too small: {amp:.3e}")
    return value / amp


def fukui_chern_subspace(
    h_func: Callable[[float, float], np.ndarray],
    n_occ: int,
    nk: int,
    shift: tuple[float, float] = (0.0, 0.0),
) -> Dict[str, float | int]:
    dim = h_func(0.0, 0.0).shape[0]
    ks_x = 2.0 * np.pi * (np.arange(nk) + float(shift[0])) / nk
    ks_y = 2.0 * np.pi * (np.arange(nk) + float(shift[1])) / nk
    occ = np.empty((nk, nk, dim, n_occ), dtype=np.complex128)

    for ix, kx in enumerate(ks_x):
        for iy, ky in enumerate(ks_y):
            _, vec = np.linalg.eigh(h_func(float(kx), float(ky)))
            occ[ix, iy] = vec[:, :n_occ]

    total_phase = 0.0
    min_det = 1.0
    for ix in range(nk):
        for iy in range(nk):
            v = occ[ix, iy]
            vx = occ[(ix + 1) % nk, iy]
            vy = occ[ix, (iy + 1) % nk]
            vxy = occ[(ix + 1) % nk, (iy + 1) % nk]

            lx = np.linalg.det(v.conj().T @ vx)
            ly = np.linalg.det(v.conj().T @ vy)
            lx_y = np.linalg.det(vy.conj().T @ vxy)
            ly_x = np.linalg.det(vx.conj().T @ vxy)
            min_det = min(min_det, abs(lx), abs(ly), abs(lx_y), abs(ly_x))

            plaquette = (
                _normalize_link(lx) * _normalize_link(ly_x)
                / (_normalize_link(lx_y) * _normalize_link(ly))
            )
            total_phase += float(np.angle(plaquette))

    return {
        "chern": total_phase / (2.0 * np.pi),
        "min_det_amp": float(min_det),
        "nk": int(nk),
    }


def calculate_spin_cherns(
    params: Dict[str, float],
    nk: int = 31,
    shift: tuple[float, float] = (0.0, 0.0),
) -> Dict[str, float | int]:
    up = fukui_chern_subspace(
        lambda kx, ky: h_spin_block_periodic(kx, ky, params, "up"),
        N_OCC_SPIN, int(nk), shift,
    )
    down = fukui_chern_subspace(
        lambda kx, ky: h_spin_block_periodic(kx, ky, params, "down"),
        N_OCC_SPIN, int(nk), shift,
    )
    total = fukui_chern_subspace(
        lambda kx, ky: h_tts_periodic(kx, ky, params),
        N_OCC_TOTAL, int(nk), shift,
    )

    c_up = float(up["chern"])
    c_down = float(down["chern"])
    c_total = float(total["chern"])
    return {
        "chern_nk": int(nk),
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
        "chern_up": c_up,
        "chern_down": c_down,
        "chern_total": c_total,
        "spin_chern": 0.5 * (c_up - c_down),
        "sum_rule_error": abs(c_total - c_up - c_down),
        "min_det_up": float(up["min_det_amp"]),
        "min_det_down": float(down["min_det_amp"]),
        "min_det_total": float(total["min_det_amp"]),
    }


def rounded_integer_if_close(value: float, tolerance: float = 0.08) -> int | None:
    nearest = int(np.rint(value))
    return nearest if abs(value - nearest) <= tolerance else None


def chern_convergence_table(
    params: Dict[str, float],
    grids: Iterable[int] = (21, 31, 41),
    shifts: Iterable[tuple[float, float]] = ((0.0, 0.0), (0.5, 0.5)),
) -> pd.DataFrame:
    rows: list[dict] = []
    for nk in grids:
        for shift in shifts:
            try:
                rows.append(calculate_spin_cherns(params, int(nk), shift))
            except Exception as exc:
                rows.append({
                    "chern_nk": int(nk),
                    "shift_x": float(shift[0]),
                    "shift_y": float(shift[1]),
                    "error": repr(exc),
                })
    return pd.DataFrame(rows)


# =============================================================================
# Step 6. Unified phase label
# =============================================================================

def calculate_sample(
    sample_id: str,
    params: Dict[str, float],
    gap_nk: int = 41,
    chern_nk: int = 21,
    gap_tol: float = 1.0e-3,
    chern_tol: float = 0.08,
    min_det_tol: float = 1.0e-7,
) -> Dict[str, object]:
    p = validate_params(params)
    row: Dict[str, object] = {
        "sample_id": sample_id,
        **p,
        **derived_features(p),
        "phase_label": "numeric_error",
        "error": "",
        "chern_reliable": 0,
        "is_spin_chern_topological": 0,
        "is_spin_chern_TI_candidate": 0,
        "is_typeII_QSH_confirmed": 0,
    }

    try:
        row.update(scan_band_gaps(p, nk=gap_nk, gap_tol=gap_tol))
    except Exception as exc:
        row["error"] = f"gap_scan_failed: {exc!r}"
        return row

    if float(row["min_direct_gap"]) <= gap_tol:
        row["phase_label"] = "noninsulating_or_gap_closing"
        return row
    if int(row["is_balanced_spin_sector"]) != 1:
        row["phase_label"] = "spin_sector_filling_mismatch"
        return row

    try:
        row.update(calculate_spin_cherns(p, nk=chern_nk))
    except Exception as exc:
        row["phase_label"] = "chern_unreliable"
        row["error"] = f"chern_failed: {exc!r}"
        return row

    cu = rounded_integer_if_close(float(row["chern_up"]), chern_tol)
    cd = rounded_integer_if_close(float(row["chern_down"]), chern_tol)
    ct = rounded_integer_if_close(float(row["chern_total"]), chern_tol)
    row.update({"chern_up_int": cu, "chern_down_int": cd, "chern_total_int": ct})

    reliable = (
        cu is not None and cd is not None and ct is not None
        and float(row["min_det_up"]) > min_det_tol
        and float(row["min_det_down"]) > min_det_tol
        and float(row["min_det_total"]) > min_det_tol
        and float(row["sum_rule_error"]) <= 0.15
    )
    row["chern_reliable"] = int(reliable)
    if not reliable:
        row["phase_label"] = "chern_unreliable"
        return row

    topological = bool(cu == -cd and abs(cu) >= 1 and ct == 0)
    insulator = bool(float(row["indirect_gap"]) > gap_tol)
    row["is_spin_chern_topological"] = int(topological)

    if topological and insulator:
        row["phase_label"] = "spin_chern_TI_candidate"
        row["is_spin_chern_TI_candidate"] = 1
    elif topological:
        row["phase_label"] = "spin_chern_band_metal"
    elif insulator and cu == 0 and cd == 0:
        row["phase_label"] = "trivial_insulator"
    elif cu == 0 and cd == 0:
        row["phase_label"] = "indirect_overlap"
    else:
        row["phase_label"] = "other_spin_chern_sector"
    return row


# =============================================================================
# Step 7. Strict multi-grid verification
# =============================================================================

def strict_verify_sample(
    sample_id: str,
    params: Dict[str, float],
    config: Step01Config,
) -> Dict[str, object]:
    config = config.normalized()
    gap_rows: list[dict] = []
    chern_rows: list[dict] = []

    for nk in config.strict_gap_grids:
        for shift in config.gap_shifts:
            row = scan_band_gaps(
                params,
                nk=int(nk),
                shift=shift,
                gap_tol=config.gap_tol,
            )
            row["sample_id"] = sample_id
            gap_rows.append(row)

    for nk in config.strict_chern_grids:
        for shift in config.chern_shifts:
            row = calculate_spin_cherns(params, int(nk), shift)
            row["sample_id"] = sample_id
            chern_rows.append(row)

    gap_df = pd.DataFrame(gap_rows)
    chern_df = pd.DataFrame(chern_rows)
    gap_df.to_csv(config.output_dir / f"{sample_id}_strict_gap_checks.csv", index=False)
    chern_df.to_csv(config.output_dir / f"{sample_id}_strict_chern_checks.csv", index=False)

    cu = [rounded_integer_if_close(v, config.chern_tol) for v in chern_df["chern_up"]]
    cd = [rounded_integer_if_close(v, config.chern_tol) for v in chern_df["chern_down"]]
    ct = [rounded_integer_if_close(v, config.chern_tol) for v in chern_df["chern_total"]]
    tuples = list(zip(cu, cd, ct))

    gaps_ok = bool(
        (gap_df["min_direct_gap"] > config.gap_tol).all()
        and (gap_df["indirect_gap"] > config.gap_tol).all()
        and (gap_df["spin_occupancy_mismatch_count"] == 0).all()
        and (gap_df["is_balanced_spin_sector"] == 1).all()
    )
    chern_ok = bool(
        all(x is not None for triple in tuples for x in triple)
        and len(set(tuples)) == 1
        and tuples[0][0] == -tuples[0][1]
        and abs(int(tuples[0][0])) >= 1
        and tuples[0][2] == 0
        and float(chern_df[["min_det_up", "min_det_down", "min_det_total"]].min().min())
            > config.min_det_tol
    )

    summary = {
        "sample_id": sample_id,
        **validate_params(params),
        **derived_features(params),
        "strict_min_direct_gap": float(gap_df["min_direct_gap"].min()),
        "strict_min_indirect_gap": float(gap_df["indirect_gap"].min()),
        "strict_max_spin_occupancy_mismatch_count": int(
            gap_df["spin_occupancy_mismatch_count"].max()
        ),
        "strict_min_link_determinant": float(
            chern_df[["min_det_up", "min_det_down", "min_det_total"]].min().min()
        ),
        "strict_chern_tuple": str(tuples[0]) if tuples else "None",
        "strict_gap_consensus": int(gaps_ok),
        "strict_chern_consensus": int(chern_ok),
        "strict_phase_label": (
            "spin_chern_TI_candidate" if gaps_ok and chern_ok else "boundary_or_unreliable"
        ),
    }
    pd.DataFrame([summary]).to_csv(
        config.output_dir / f"{sample_id}_strict_summary.csv", index=False
    )
    return summary


# =============================================================================
# Step 8. Small serial debug batch
# =============================================================================

def sample_debug_parameters(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_samples)):
        reduced = {
            name: float(rng.uniform(*DEBUG_BOUNDS[name]))
            for name in REDUCED7
        }
        rows.append({"sample_id": f"debug_{i:04d}", **reduced})
    return pd.DataFrame(rows)


def run_small_batch(config: Step01Config) -> pd.DataFrame:
    config = config.normalized()
    parameter_df = sample_debug_parameters(config.n_batch, config.batch_seed)
    parameter_df.to_csv(config.output_dir / "tts_debug_parameters.csv", index=False)

    rows: list[dict] = []
    total = len(parameter_df)
    for i, item in parameter_df.iterrows():
        reduced = {name: float(item[name]) for name in REDUCED7}
        params = raw8_from_reduced7(reduced)
        rows.append(
            calculate_sample(
                str(item["sample_id"]),
                params,
                gap_nk=config.debug_gap_nk,
                chern_nk=config.debug_chern_nk,
                gap_tol=config.gap_tol,
                chern_tol=config.chern_tol,
                min_det_tol=config.min_det_tol,
            )
        )
        print(f"  serial debug sample {i + 1:>3d}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(config.output_dir / "tts_debug_batch_results.csv", index=False)
    counts = (
        df["phase_label"].value_counts(dropna=False)
        .rename_axis("phase_label").reset_index(name="count")
    )
    counts.to_csv(config.output_dir / "tts_debug_phase_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(counts["phase_label"].astype(str), counts["count"])
    ax.set_ylabel("Count")
    ax.set_title("TTS Step 01 serial debug phase counts")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(config.output_dir / "tts_debug_phase_counts.png", dpi=220)
    plt.close(fig)
    return df


# =============================================================================
# Step 9. Complete Step 01 workflow
# =============================================================================

def run_step01(config: Step01Config | None = None) -> Dict[str, object]:
    config = (config or Step01Config()).normalized()
    started = time.time()

    metadata = {
        "code_version": CODE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "tts eight-band MagneticTB minimal Hamiltonian",
        "msg_bns": "127.391",
        "wyckoff": "4h",
        "x_tts": X_TTS,
        "raw_parameters": RAW8,
        "reduced_parameters": REDUCED7,
        "serial_execution_only": True,
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
    }
    (config.output_dir / "tts_step01_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[1/6] Hamiltonian hard audit")
    tests = model_tests(PAPER_PARAMS)
    pd.DataFrame([tests]).to_csv(config.output_dir / "tts_model_tests.csv", index=False)
    print(pd.DataFrame([tests]).to_string(index=False))

    print("\n[2/6] Literature parameter bands and label")
    paper_bands = plot_path_bands(
        PAPER_PARAMS,
        config.output_dir / "tts_paper_parameter_bands.png",
        "tts eight-band model: literature parameter set",
    )
    paper_bands.to_csv(config.output_dir / "tts_paper_path_bands.csv", index=False)
    paper_gap = scan_band_gaps(
        PAPER_PARAMS, nk=config.paper_gap_nk, gap_tol=config.gap_tol
    )
    pd.DataFrame([paper_gap]).to_csv(config.output_dir / "tts_paper_gap_audit.csv", index=False)
    paper_chern = chern_convergence_table(
        PAPER_PARAMS, config.paper_chern_grids, config.chern_shifts
    )
    paper_chern.to_csv(config.output_dir / "tts_paper_chern_convergence.csv", index=False)
    paper_label = calculate_sample(
        "paper_tts",
        PAPER_PARAMS,
        gap_nk=config.paper_gap_nk,
        chern_nk=max(config.paper_chern_grids),
        gap_tol=config.gap_tol,
        chern_tol=config.chern_tol,
        min_det_tol=config.min_det_tol,
    )
    pd.DataFrame([paper_label]).to_csv(
        config.output_dir / "tts_paper_sample_label.csv", index=False
    )
    print(
        "  paper:", paper_label["phase_label"],
        "Edir=", f"{float(paper_label['min_direct_gap']):.8f}",
        "Eind=", f"{float(paper_label['indirect_gap']):.8f}",
    )

    print("\n[3/6] Known topological anchor bands and basic label")
    topo_bands = plot_path_bands(
        TOPO_PARAMS,
        config.output_dir / "tts_topological_anchor_bands.png",
        "tts eight-band model: Step 01 topological anchor",
    )
    topo_bands.to_csv(config.output_dir / "tts_topological_anchor_path_bands.csv", index=False)
    topo_label = calculate_sample(
        "tts_topological_anchor",
        TOPO_PARAMS,
        gap_nk=config.paper_gap_nk,
        chern_nk=max(config.paper_chern_grids),
        gap_tol=config.gap_tol,
        chern_tol=config.chern_tol,
        min_det_tol=config.min_det_tol,
    )
    pd.DataFrame([topo_label]).to_csv(
        config.output_dir / "tts_topological_anchor_label.csv", index=False
    )
    print(
        "  anchor:", topo_label["phase_label"],
        "Edir=", f"{float(topo_label['min_direct_gap']):.8f}",
        "Eind=", f"{float(topo_label['indirect_gap']):.8f}",
        "C=", (topo_label.get("chern_up_int"), topo_label.get("chern_down_int")),
    )

    print("\n[4/6] Strict multi-grid/multi-shift anchor verification")
    strict_topo = strict_verify_sample("tts_topological_anchor", TOPO_PARAMS, config)
    print(pd.DataFrame([strict_topo]).to_string(index=False))

    print("\n[5/6] Small serial debug batch")
    if config.run_small_batch and config.n_batch > 0:
        batch = run_small_batch(config)
        print(batch["phase_label"].value_counts(dropna=False).to_string())
    else:
        batch = pd.DataFrame()
        print("  skipped")

    print("\n[6/6] Summary")
    summary = {
        "code_version": CODE_VERSION,
        "paper_phase_label": paper_label["phase_label"],
        "paper_min_direct_gap": float(paper_label["min_direct_gap"]),
        "paper_indirect_gap": float(paper_label["indirect_gap"]),
        "paper_chern_up": float(paper_label.get("chern_up", np.nan)),
        "paper_chern_down": float(paper_label.get("chern_down", np.nan)),
        "topological_anchor_phase_label": topo_label["phase_label"],
        "topological_anchor_strict_phase_label": strict_topo["strict_phase_label"],
        "topological_anchor_strict_min_direct_gap": strict_topo["strict_min_direct_gap"],
        "topological_anchor_strict_min_indirect_gap": strict_topo["strict_min_indirect_gap"],
        "topological_anchor_strict_chern_tuple": strict_topo["strict_chern_tuple"],
        "small_batch_size": int(len(batch)),
        "elapsed_seconds": float(time.time() - started),
        "next_step": "TTS Step 02: seven-dimensional Sobol global scan",
    }
    (config.output_dir / "tts_step01_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nStep 01 finished:", config.output_dir.resolve())
    return {
        "summary": summary,
        "model_tests": pd.DataFrame([tests]),
        "paper_label": pd.DataFrame([paper_label]),
        "topological_anchor_label": pd.DataFrame([topo_label]),
        "strict_topological_anchor": pd.DataFrame([strict_topo]),
        "debug_batch": batch,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs_tts_step01_v2")
    parser.add_argument("--n-batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced meshes and four debug samples for an environment test.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Step01Config:
    if args.quick:
        return Step01Config(
            output_dir=Path(args.output_dir),
            run_small_batch=not args.skip_batch,
            n_batch=min(max(args.n_batch, 0), 4),
            batch_seed=args.seed,
            debug_gap_nk=21,
            debug_chern_nk=15,
            paper_gap_nk=31,
            paper_chern_grids=(15, 21),
            strict_gap_grids=(31, 41),
            strict_chern_grids=(15, 21),
        )
    return Step01Config(
        output_dir=Path(args.output_dir),
        run_small_batch=not args.skip_batch,
        n_batch=args.n_batch,
        batch_seed=args.seed,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_step01(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
