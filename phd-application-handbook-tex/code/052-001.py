#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 18: three-system WannierTools-style edge spectra and Hall examples.

The workflow selects exactly three states for each of the Lieb, TTS and FES
models:

1. a comparatively large-gap topological representative;
2. the refined critical Hamiltonian on the same transition path;
3. a gapped Hamiltonian on the Chern-changed side of that boundary.

It then calculates:

- square-lattice bulk bands;
- a finite ribbon spectrum with left/right edge weights and spin character;
- a semi-infinite surface spectral function using the iterative Sancho method;
- energy-resolved spin-up, spin-down, charge and spin Hall responses.

The numerical definitions follow the existing WannierTools-style project
scripts.  Plot colors are taken exactly from the corresponding phase maps.
No existing result directory is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
import argparse
import importlib.util
import json
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

_THREADPOOL_LIMITER = threadpool_limits(limits=1)


CODE_VERSION = "TTS_STEP18_THREE_SYSTEM_WT_STYLE_V1_20260730"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LIEB_ROOT = PROJECT_ROOT / "lieb"
TTS_ROOT = PROJECT_ROOT / "tts"
FES_ROOT = PROJECT_ROOT / "fes"

DEFAULT_OUTPUT = (
    TTS_ROOT / "outputs_three_system_wanniertools_style_edge_ahc_examples"
)

STATE_ORDER = ("topological", "critical", "chern_changed")
STATE_LABELS = {
    "topological": "Topological",
    "critical": "Critical",
    "chern_changed": "Chern-changed",
}
SYSTEM_LABELS = {"lieb": "Lieb", "tts": "TTS", "fes": "FES"}

# Exact strong colors used by the phase maps.  Region backgrounds are derived
# by blending these colors toward white.
LIEB_FES_COLORS = {
    +1: "#A92425",
    0: "#D3D3D3",
    -1: "#3F63AD",
}
TTS_COLORS = {
    +2: "#F28E8B",
    +1: "#7FCBA1",
    0: "#B2B2B2",
    -1: "#8AA3CD",
    -2: "#3F63AD",
}
CRITICAL_COLOR = "#4D4D4D"
BODY_COLOR = "#D7D7D7"
BODY_DARK = "#AFAFAF"
BLACK = "#111111"


@dataclass
class ModelSpec:
    key: str
    label: str
    h_periodic: Callable[[float, float, dict[str, float]], np.ndarray]
    h_atomic: Callable[[float, float, dict[str, float]], np.ndarray]
    spin_up_indices: tuple[int, ...]
    spin_down_indices: tuple[int, ...]
    n_occ_total: int
    n_occ_spin: int
    palette: dict[int, str]
    parameter_names: tuple[str, ...]
    ribbon_width: int
    energy_reference: str = "global_midgap"


@dataclass
class SelectedState:
    system: str
    state: str
    source_id: str
    path_id: str
    params: dict[str, float]
    expected_chern_up: int | None
    expected_chern_down: int | None
    critical_kx: float
    critical_ky: float
    selection_note: str


@dataclass
class RunSettings:
    gap_nk: int
    chern_nk: int
    band_points_per_segment: int
    ribbon_k_points: int
    surface_k_points: int
    surface_energy_points: int
    response_nk: int
    response_energy_points: int
    fourier_n: int
    dpi: int
    quick: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.8,
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.35,
        length=4.5,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color(BLACK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> list[str]:
    paths: list[str] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        paths.append(str(path))
    plt.close(fig)
    return paths


def lighten(color: str, factor: float = 0.78) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * (1.0 - factor) + factor, 0.0, 1.0))


def phase_color(spec: ModelSpec, chern: int | None) -> str:
    if chern is None:
        return CRITICAL_COLOR
    return spec.palette[int(chern)]


def build_model_specs() -> tuple[dict[str, ModelSpec], dict[str, Any]]:
    lieb = load_module(
        "step18_lieb_source",
        LIEB_ROOT / "08_Lieb8_Analytic_Mechanism_and_WannierTools_Edge_v5.py",
    )
    fes = load_module("step18_fes_source", FES_ROOT / "FES_step01_v2.py")
    tts = load_module(
        "step18_tts_source",
        TTS_ROOT / "TTS_step01_model_and_label_audit_v2.py",
    )

    specs = {
        "lieb": ModelSpec(
            key="lieb",
            label="Lieb",
            h_periodic=lieb.h_periodic,
            h_atomic=lieb.h_lieb_atomic,
            spin_up_indices=tuple(int(x) for x in lieb.SPIN_INDICES["up"]),
            spin_down_indices=tuple(int(x) for x in lieb.SPIN_INDICES["down"]),
            n_occ_total=2,
            n_occ_spin=1,
            palette=LIEB_FES_COLORS,
            parameter_names=tuple(lieb.RAW8),
            ribbon_width=60,
        ),
        "tts": ModelSpec(
            key="tts",
            label="TTS",
            h_periodic=tts.h_tts_periodic,
            h_atomic=tts.h_tts_atomic,
            spin_up_indices=tuple(int(x) for x in tts.SPIN_UP_INDICES),
            spin_down_indices=tuple(int(x) for x in tts.SPIN_DOWN_INDICES),
            n_occ_total=int(tts.N_OCC_TOTAL),
            n_occ_spin=int(tts.N_OCC_SPIN),
            palette=TTS_COLORS,
            parameter_names=tuple(tts.RAW8),
            ribbon_width=36,
        ),
        "fes": ModelSpec(
            key="fes",
            label="FES",
            h_periodic=fes.h_fes_periodic,
            h_atomic=fes.h_fes_atomic,
            spin_up_indices=tuple(int(x) for x in fes.SPIN_UP_INDICES),
            spin_down_indices=tuple(int(x) for x in fes.SPIN_DOWN_INDICES),
            n_occ_total=int(fes.N_OCC_TOTAL),
            n_occ_spin=int(fes.N_OCC_SPIN),
            palette=LIEB_FES_COLORS,
            parameter_names=tuple(fes.RAW6),
            ribbon_width=44,
            energy_reference="critical_valley_midgap",
        ),
    }
    return specs, {"lieb": lieb, "tts": tts, "fes": fes}


def _clean_params(
    params: dict[str, float],
    names: Iterable[str],
) -> dict[str, float]:
    return {name: float(params[name]) for name in names}


def select_lieb_states(lieb: Any) -> list[SelectedState]:
    output = (
        LIEB_ROOT
        / "outputs_step08_analytic_boundary_wanniertools_edge_msg123342_lieb8_from_step06_n1024_pairs4_seed20260711"
    )
    roots = pd.read_csv(output / "step08_04_refined_analytic_transition_roots.csv")
    profiles = pd.read_csv(output / "step08_04_dense_transition_path_profiles.csv")

    direct_flip = roots[
        (roots["boundary"].astype(str) == "t2")
        & (pd.to_numeric(roots["before_chern_up_int"], errors="coerce")
           * pd.to_numeric(roots["after_chern_up_int"], errors="coerce") < 0)
    ].copy()
    if direct_flip.empty:
        raise RuntimeError("No direct Lieb Chern-sign flip was found")
    root = direct_flip.sort_values("verified_direct_gap").iloc[0]
    pair_id = int(root["pair_id"])
    tc = float(root["t_critical"])

    pair_roots = roots[roots["pair_id"].astype(int) == pair_id].sort_values("t_critical")
    later_roots = pair_roots[pair_roots["t_critical"] > tc + 1.0e-8]
    upper = float(later_roots["t_critical"].min()) if not later_roots.empty else 1.0

    profile = profiles[profiles["pair_id"].astype(int) == pair_id].copy()
    pre = profile[profile["t"] <= tc - 0.04].sort_values(
        ["indirect_gap", "direct_gap"], ascending=False
    ).iloc[0]
    post = profile[
        (profile["t"] >= tc + 0.04) & (profile["t"] <= upper - 0.04)
    ].sort_values(["indirect_gap", "direct_gap"], ascending=False).iloc[0]

    def raw_from_row(row: pd.Series) -> dict[str, float]:
        vector = [float(row[name]) for name in lieb.PHYS7]
        return _clean_params(lieb.phys7_to_raw8(vector), lieb.RAW8)

    critical_params = raw_from_row(root)
    pre_c = int(root["before_chern_up_int"])
    post_c = int(root["after_chern_up_int"])
    critical_kx = float(root["global_direct_kx"])
    critical_ky = float(root["global_direct_ky"])
    path_id = f"pair_{pair_id}_direct_t2_flip"
    return [
        SelectedState(
            system="lieb",
            state="topological",
            source_id=f"pair{pair_id}_t{float(pre['t']):.6f}",
            path_id=path_id,
            params=raw_from_row(pre),
            expected_chern_up=pre_c,
            expected_chern_down=-pre_c,
            critical_kx=critical_kx,
            critical_ky=critical_ky,
            selection_note=(
                "Largest indirect gap on the pre-critical Chern-sign segment "
                f"(t < {tc:.8f})."
            ),
        ),
        SelectedState(
            system="lieb",
            state="critical",
            source_id=f"pair{pair_id}_tcritical_{tc:.10f}",
            path_id=path_id,
            params=critical_params,
            expected_chern_up=None,
            expected_chern_down=None,
            critical_kx=critical_kx,
            critical_ky=critical_ky,
            selection_note="Refined exact t2=0 Chern-sign-flip root.",
        ),
        SelectedState(
            system="lieb",
            state="chern_changed",
            source_id=f"pair{pair_id}_t{float(post['t']):.6f}",
            path_id=path_id,
            params=raw_from_row(post),
            expected_chern_up=post_c,
            expected_chern_down=-post_c,
            critical_kx=critical_kx,
            critical_ky=critical_ky,
            selection_note=(
                "Largest indirect gap after the direct sign flip and before "
                "the next analytic boundary."
            ),
        ),
    ]


def select_tts_states(tts: Any) -> list[SelectedState]:
    output = TTS_ROOT / "outputs_tts_step04_chern_sector_boundary_valley_tracking"
    refined = pd.read_csv(output / "step04_05_refined_critical_valleys.csv")
    gaps = pd.read_csv(output / "step04_04_path_dense_gap_scan.csv")

    path_id = "Cup+1_anchor01__nearest"
    candidates = refined[
        (refined["path_id"].astype(str) == path_id)
        & (pd.to_numeric(refined["chern_left"], errors="coerce") == 1)
        & (pd.to_numeric(refined["chern_right"], errors="coerce") == 0)
    ]
    if candidates.empty:
        raise RuntimeError("The requested TTS C_up=+1 to 0 transition is missing")
    root = candidates.iloc[0]
    lc = float(root["critical_lambda"])
    path = gaps[gaps["path_id"].astype(str) == path_id].copy()
    pre = path[path["lambda"] <= lc - 0.05].sort_values(
        ["indirect_gap", "min_direct_gap"], ascending=False
    ).iloc[0]
    post = path[path["lambda"] >= lc + 0.05].sort_values(
        ["indirect_gap", "min_direct_gap"], ascending=False
    ).iloc[0]

    def raw_from_row(row: pd.Series) -> dict[str, float]:
        reduced = {name: float(row[name]) for name in tts.REDUCED7}
        return _clean_params(tts.raw8_from_reduced7(reduced), tts.RAW8)

    kx = float(root["critical_kx"])
    ky = float(root["critical_ky"])
    return [
        SelectedState(
            system="tts",
            state="topological",
            source_id=str(pre["path_point_id"]),
            path_id=path_id,
            params=raw_from_row(pre),
            expected_chern_up=1,
            expected_chern_down=-1,
            critical_kx=kx,
            critical_ky=ky,
            selection_note="Largest indirect gap on the C_up=+1 side of the path.",
        ),
        SelectedState(
            system="tts",
            state="critical",
            source_id=str(root["transition_id"]),
            path_id=path_id,
            params=raw_from_row(root),
            expected_chern_up=None,
            expected_chern_down=None,
            critical_kx=kx,
            critical_ky=ky,
            selection_note="Refined M-valley critical Hamiltonian.",
        ),
        SelectedState(
            system="tts",
            state="chern_changed",
            source_id=str(post["path_point_id"]),
            path_id=path_id,
            params=raw_from_row(post),
            expected_chern_up=0,
            expected_chern_down=0,
            critical_kx=kx,
            critical_ky=ky,
            selection_note="Largest indirect gap on the C_up=0 side of the same path.",
        ),
    ]


def select_fes_states(fes: Any) -> list[SelectedState]:
    boundary = FES_ROOT / "outputs_fes6_step04_boundary"
    kp = FES_ROOT / "outputs_fes6_step06_kp"
    strict = pd.read_csv(boundary / "fes_step04_strict_verified_results.csv")
    roots = pd.read_csv(kp / "fes_step06_critical_mass_roots.csv")

    reliable = strict[
        (pd.to_numeric(strict["verified_chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(strict["verified_chern_up_int"], errors="coerce").abs() == 1)
    ].copy()
    topo = reliable.sort_values(
        ["verified_min_direct_gap", "verified_indirect_gap"],
        ascending=False,
    ).iloc[0]
    point_id = str(topo["point_id"])
    root_candidates = roots[
        (roots["point_id"].astype(str) == point_id)
        & (roots["valley"].astype(str) == "Gamma")
        & (pd.to_numeric(roots["root_found"], errors="coerce") == 1)
    ]
    if root_candidates.empty:
        raise RuntimeError(f"No FES Gamma critical root for {point_id}")
    root = root_candidates.iloc[0]

    reduced_topo = {
        name: float(topo[name]) for name in ("m_e", "t1", "t2", "r1", "r2")
    }
    reduced_critical = dict(reduced_topo)
    reduced_critical["m_e"] = float(root["m_critical"])

    direction = float(np.sign(float(root["m_critical"]) - float(topo["m_e"])))
    if direction == 0.0:
        direction = 1.0
    reduced_changed = dict(reduced_critical)
    reduced_changed["m_e"] += direction * 0.03

    raw_topo = _clean_params(fes.raw6_from_reduced5(reduced_topo), fes.RAW6)
    raw_critical = _clean_params(
        fes.raw6_from_reduced5(reduced_critical), fes.RAW6
    )
    raw_changed = _clean_params(
        fes.raw6_from_reduced5(reduced_changed), fes.RAW6
    )
    expected_topo = int(topo["verified_chern_up_int"])
    path_id = f"{point_id}_Gamma_mass_crossing"
    return [
        SelectedState(
            system="fes",
            state="topological",
            source_id=point_id,
            path_id=path_id,
            params=raw_topo,
            expected_chern_up=expected_topo,
            expected_chern_down=-expected_topo,
            critical_kx=0.0,
            critical_ky=0.0,
            selection_note=(
                "Largest verified direct gap among reliable FES |C_up|=1 "
                "samples; the indirect gap remains negative by the FES no-go result."
            ),
        ),
        SelectedState(
            system="fes",
            state="critical",
            source_id=f"{point_id}_Gamma_mcrit",
            path_id=path_id,
            params=raw_critical,
            expected_chern_up=None,
            expected_chern_down=None,
            critical_kx=0.0,
            critical_ky=0.0,
            selection_note="Analytic Gamma-valley mass root from FES Step06.",
        ),
        SelectedState(
            system="fes",
            state="chern_changed",
            source_id=f"{point_id}_Gamma_post_0p03",
            path_id=path_id,
            params=raw_changed,
            expected_chern_up=0,
            expected_chern_down=0,
            critical_kx=0.0,
            critical_ky=0.0,
            selection_note=(
                "C_up=0 point 0.03 in m_e beyond the same Gamma boundary."
            ),
        ),
    ]


def select_all_states(modules: dict[str, Any]) -> list[SelectedState]:
    return (
        select_lieb_states(modules["lieb"])
        + select_tts_states(modules["tts"])
        + select_fes_states(modules["fes"])
    )


def extract_hoppings(
    spec: ModelSpec,
    params: dict[str, float],
    nfft: int,
    tolerance: float = 1.0e-10,
) -> dict[tuple[int, int], np.ndarray]:
    nfft = int(nfft)
    ks = 2.0 * np.pi * np.arange(nfft) / nfft
    dim = int(spec.h_periodic(0.0, 0.0, params).shape[0])
    hk = np.empty((nfft, nfft, dim, dim), dtype=np.complex128)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            hk[ix, iy] = spec.h_periodic(float(kx), float(ky), params)
    coeff = np.fft.fft2(hk, axes=(0, 1)) / float(nfft * nfft)
    hoppings: dict[tuple[int, int], np.ndarray] = {}
    for ix in range(nfft):
        rx = ix if ix <= nfft // 2 else ix - nfft
        for iy in range(nfft):
            ry = iy if iy <= nfft // 2 else iy - nfft
            mat = coeff[ix, iy]
            if float(np.max(np.abs(mat))) > tolerance:
                hoppings[(int(rx), int(ry))] = np.asarray(
                    mat, dtype=np.complex128
                )
    for kx, ky in ((0.173, -1.029), (2.037, 0.691), (-2.4, 2.1)):
        rec = np.zeros((dim, dim), dtype=np.complex128)
        for (rx, ry), mat in hoppings.items():
            rec += mat * np.exp(1j * (kx * rx + ky * ry))
        ref = spec.h_periodic(kx, ky, params)
        error = float(np.max(np.abs(rec - ref)))
        if error > 2.0e-9:
            raise RuntimeError(
                f"{spec.key} Fourier reconstruction failed: {error:.3e}"
            )
    if any(abs(ry) > 1 for _, ry in hoppings):
        raise RuntimeError(
            f"{spec.key} axial principal layer is too thin for hoppings "
            f"{sorted(hoppings)}"
        )
    return hoppings


def bloch_from_hoppings(
    kx: float,
    ky: float,
    hoppings: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = next(iter(hoppings.values())).shape[0]
    h = np.zeros((dim, dim), dtype=np.complex128)
    vx = np.zeros_like(h)
    vy = np.zeros_like(h)
    for (rx, ry), mat in hoppings.items():
        phase = np.exp(1j * (float(kx) * rx + float(ky) * ry))
        h += mat * phase
        vx += 1j * rx * mat * phase
        vy += 1j * ry * mat * phase
    h = 0.5 * (h + h.conj().T)
    vx = 0.5 * (vx + vx.conj().T)
    vy = 0.5 * (vy + vy.conj().T)
    return h, vx, vy


def axial_blocks(
    k_parallel: float,
    hoppings: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    dim = next(iter(hoppings.values())).shape[0]
    blocks = {
        -1: np.zeros((dim, dim), dtype=np.complex128),
        0: np.zeros((dim, dim), dtype=np.complex128),
        1: np.zeros((dim, dim), dtype=np.complex128),
    }
    for (rx, ry), mat in hoppings.items():
        blocks[int(ry)] += mat * np.exp(1j * float(k_parallel) * rx)
    h0 = 0.5 * (blocks[0] + blocks[0].conj().T)
    hp = 0.5 * (blocks[1] + blocks[-1].conj().T)
    return h0, hp


def gap_audit(
    spec: ModelSpec,
    selected: SelectedState,
    nk: int,
) -> dict[str, float | int]:
    nk = int(nk)
    if nk % 2:
        nk += 1
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    min_direct = np.inf
    direct_kx = np.nan
    direct_ky = np.nan
    vbm = -np.inf
    cbm = np.inf
    vbm_kx = vbm_ky = cbm_kx = cbm_ky = np.nan
    for kx in ks:
        for ky in ks:
            energies = np.linalg.eigvalsh(
                spec.h_periodic(float(kx), float(ky), selected.params)
            )
            dg = float(
                energies[spec.n_occ_total] - energies[spec.n_occ_total - 1]
            )
            if dg < min_direct:
                min_direct = dg
                direct_kx, direct_ky = float(kx), float(ky)
            valence = float(energies[spec.n_occ_total - 1])
            conduction = float(energies[spec.n_occ_total])
            if valence > vbm:
                vbm, vbm_kx, vbm_ky = valence, float(kx), float(ky)
            if conduction < cbm:
                cbm, cbm_kx, cbm_ky = conduction, float(kx), float(ky)

    kc_e = np.linalg.eigvalsh(
        spec.h_periodic(
            float(selected.critical_kx),
            float(selected.critical_ky),
            selected.params,
        )
    )
    critical_valence = float(kc_e[spec.n_occ_total - 1])
    critical_conduction = float(kc_e[spec.n_occ_total])
    critical_gap = critical_conduction - critical_valence
    # The Lieb sign-flip closes at an off-grid axis momentum.  Include the
    # explicitly refined transition momentum in the global extrema audit so
    # the reported indirect gap cannot spuriously exceed the exact direct gap.
    if critical_valence > vbm:
        vbm = critical_valence
        vbm_kx = float(selected.critical_kx)
        vbm_ky = float(selected.critical_ky)
    if critical_conduction < cbm:
        cbm = critical_conduction
        cbm_kx = float(selected.critical_kx)
        cbm_ky = float(selected.critical_ky)
    if critical_gap < min_direct:
        min_direct = critical_gap
        direct_kx = float(selected.critical_kx)
        direct_ky = float(selected.critical_ky)

    global_midgap = 0.5 * (vbm + cbm)
    local_midgap = 0.5 * (critical_valence + critical_conduction)
    if selected.state == "critical":
        ef = local_midgap
        reference_kind = "critical_closing_energy"
    elif spec.energy_reference == "critical_valley_midgap":
        ef = local_midgap
        reference_kind = "critical_valley_local_midgap"
    else:
        ef = global_midgap
        reference_kind = "global_midgap"

    return {
        "gap_nk": int(nk),
        "min_direct_gap": float(min_direct),
        "indirect_gap": float(cbm - vbm),
        "direct_gap_kx": float(direct_kx),
        "direct_gap_ky": float(direct_ky),
        "vbm": float(vbm),
        "vbm_kx": float(vbm_kx),
        "vbm_ky": float(vbm_ky),
        "cbm": float(cbm),
        "cbm_kx": float(cbm_kx),
        "cbm_ky": float(cbm_ky),
        "critical_valley_gap": float(critical_gap),
        "critical_valley_midgap": float(local_midgap),
        "energy_reference": float(ef),
        "energy_reference_kind": reference_kind,
        "is_direct_gapped": int(min_direct > 1.0e-4),
        "is_global_insulator": int((cbm - vbm) > 1.0e-4),
    }


def _normalized_link(value: complex, floor: float = 1.0e-14) -> complex:
    amplitude = abs(value)
    if amplitude < floor:
        raise FloatingPointError(f"Link determinant too small: {amplitude:.3e}")
    return value / amplitude


def fukui_chern(
    h_func: Callable[[float, float], np.ndarray],
    n_occ: int,
    nk: int,
) -> tuple[float, float]:
    sample = h_func(0.0, 0.0)
    dim = sample.shape[0]
    ks = 2.0 * np.pi * np.arange(int(nk)) / int(nk)
    occupied = np.empty((nk, nk, dim, n_occ), dtype=np.complex128)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            _, vectors = np.linalg.eigh(h_func(float(kx), float(ky)))
            occupied[ix, iy] = vectors[:, :n_occ]
    total = 0.0
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
            plaquette = (
                _normalized_link(lx)
                * _normalized_link(ly_x)
                / (_normalized_link(lx_y) * _normalized_link(ly))
            )
            total += float(np.angle(plaquette))
    return float(total / (2.0 * np.pi)), float(min_det)


def chern_audit(
    spec: ModelSpec,
    selected: SelectedState,
    nk: int,
) -> dict[str, float | int]:
    if selected.state == "critical":
        return {
            "chern_nk": int(nk),
            "chern_up": np.nan,
            "chern_down": np.nan,
            "chern_up_int": np.nan,
            "chern_down_int": np.nan,
            "min_link_up": np.nan,
            "min_link_down": np.nan,
            "chern_defined": 0,
        }
    up_idx = np.asarray(spec.spin_up_indices, dtype=int)
    down_idx = np.asarray(spec.spin_down_indices, dtype=int)

    def block(indices: np.ndarray, kx: float, ky: float) -> np.ndarray:
        h = spec.h_periodic(kx, ky, selected.params)
        return h[np.ix_(indices, indices)]

    cup, det_up = fukui_chern(
        lambda kx, ky: block(up_idx, kx, ky),
        spec.n_occ_spin,
        int(nk),
    )
    cdown, det_down = fukui_chern(
        lambda kx, ky: block(down_idx, kx, ky),
        spec.n_occ_spin,
        int(nk),
    )
    return {
        "chern_nk": int(nk),
        "chern_up": float(cup),
        "chern_down": float(cdown),
        "chern_up_int": int(np.rint(cup)),
        "chern_down_int": int(np.rint(cdown)),
        "min_link_up": float(det_up),
        "min_link_down": float(det_down),
        "chern_defined": 1,
    }


def high_symmetry_path(
    points_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    named = [
        (r"$\Gamma$", (0.0, 0.0)),
        ("X", (np.pi, 0.0)),
        ("M", (np.pi, np.pi)),
        (r"$\Gamma$", (0.0, 0.0)),
    ]
    klist: list[np.ndarray] = []
    distance: list[float] = []
    ticks = [0.0]
    labels = [named[0][0]]
    current = 0.0
    previous: np.ndarray | None = None
    for iseg in range(len(named) - 1):
        p0 = np.asarray(named[iseg][1], dtype=float)
        p1 = np.asarray(named[iseg + 1][1], dtype=float)
        for j in range(points_per_segment + 1):
            if iseg > 0 and j == 0:
                continue
            t = j / float(points_per_segment)
            k = (1.0 - t) * p0 + t * p1
            if previous is not None:
                current += float(np.linalg.norm(k - previous))
            klist.append(k)
            distance.append(current)
            previous = k
        ticks.append(current)
        labels.append(named[iseg + 1][0])
    return np.asarray(klist), np.asarray(distance), ticks, labels


def calculate_bulk_bands(
    spec: ModelSpec,
    selected: SelectedState,
    ef: float,
    points_per_segment: int,
) -> tuple[pd.DataFrame, list[float], list[str]]:
    klist, distance, ticks, labels = high_symmetry_path(points_per_segment)
    spin_diag = np.zeros(
        spec.h_atomic(0.0, 0.0, selected.params).shape[0], dtype=float
    )
    spin_diag[list(spec.spin_up_indices)] = 1.0
    spin_diag[list(spec.spin_down_indices)] = -1.0
    rows: list[dict[str, float | int]] = []
    for ik, ((kx, ky), xpos) in enumerate(zip(klist, distance)):
        energies, vectors = np.linalg.eigh(
            spec.h_atomic(float(kx), float(ky), selected.params)
        )
        probability = np.abs(vectors) ** 2
        spin = np.sum(probability * spin_diag[:, None], axis=0)
        for ib, energy in enumerate(energies):
            rows.append(
                {
                    "k_index": int(ik),
                    "path_coordinate": float(xpos),
                    "kx": float(kx),
                    "ky": float(ky),
                    "band": int(ib + 1),
                    "energy": float(energy - ef),
                    "spin_z": float(spin[ib]),
                }
            )
    return pd.DataFrame(rows), ticks, labels


def ribbon_hamiltonian(
    k_parallel: float,
    width: int,
    hoppings: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, int]:
    h0, hp = axial_blocks(k_parallel, hoppings)
    dim = h0.shape[0]
    h = np.zeros((width * dim, width * dim), dtype=np.complex128)
    for layer in range(width):
        sl = slice(layer * dim, (layer + 1) * dim)
        h[sl, sl] = h0
        if layer + 1 < width:
            sr = slice((layer + 1) * dim, (layer + 2) * dim)
            h[sl, sr] = hp
            h[sr, sl] = hp.conj().T
    return 0.5 * (h + h.conj().T), dim


def calculate_ribbon(
    spec: ModelSpec,
    selected: SelectedState,
    hoppings: dict[tuple[int, int], np.ndarray],
    ef: float,
    k_points: int,
    width: int,
    edge_layers: int = 3,
) -> pd.DataFrame:
    kvalues = np.linspace(-np.pi, np.pi, int(k_points), endpoint=True)
    spin_primitive = np.zeros(
        next(iter(hoppings.values())).shape[0], dtype=float
    )
    spin_primitive[list(spec.spin_up_indices)] = 1.0
    spin_primitive[list(spec.spin_down_indices)] = -1.0
    rows: list[dict[str, float | int]] = []
    for ik, kp in enumerate(kvalues):
        h, dim = ribbon_hamiltonian(float(kp), int(width), hoppings)
        energies, vectors = eigh(h, overwrite_a=True, check_finite=False)
        probability = np.abs(vectors) ** 2
        nedge = min(edge_layers, max(1, width // 3))
        left_weight = np.sum(probability[: nedge * dim], axis=0)
        right_weight = np.sum(probability[(width - nedge) * dim :], axis=0)
        total_edge = left_weight + right_weight
        spin_diag = np.tile(spin_primitive, width)
        spin_z = np.real(np.sum(probability * spin_diag[:, None], axis=0))
        for ib, energy in enumerate(energies):
            rows.append(
                {
                    "k_index": int(ik),
                    "k_parallel": float(kp),
                    "band": int(ib + 1),
                    "energy": float(energy - ef),
                    "left_edge_weight": float(left_weight[ib]),
                    "right_edge_weight": float(right_weight[ib]),
                    "total_edge_weight": float(total_edge[ib]),
                    "spin_z": float(spin_z[ib]),
                }
            )
    return pd.DataFrame(rows)


def sancho_surface_green(
    energy: float,
    eta: float,
    h0: np.ndarray,
    hp: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    z = complex(float(energy), float(eta))
    identity = np.eye(h0.shape[0], dtype=np.complex128)
    eps = h0.copy()
    eps_surface = h0.copy()
    alpha = hp.copy()
    beta = hp.conj().T
    for _ in range(max_iter):
        g = np.linalg.inv(z * identity - eps)
        agb = alpha @ g @ beta
        bga = beta @ g @ alpha
        eps_surface += agb
        eps += agb + bga
        alpha_new = alpha @ g @ alpha
        beta_new = beta @ g @ beta
        alpha, beta = alpha_new, beta_new
        if max(
            float(np.linalg.norm(alpha, ord="fro")),
            float(np.linalg.norm(beta, ord="fro")),
        ) < tolerance:
            break
    return np.linalg.inv(z * identity - eps_surface)


def calculate_surface(
    spec: ModelSpec,
    hoppings: dict[tuple[int, int], np.ndarray],
    ef: float,
    energy_window: float,
    k_points: int,
    energy_points: int,
    eta: float,
) -> dict[str, np.ndarray | float]:
    kvalues = np.linspace(-np.pi, np.pi, int(k_points), endpoint=True)
    energies = np.linspace(-energy_window, energy_window, int(energy_points))
    dim = next(iter(hoppings.values())).shape[0]
    up_mask = np.zeros(dim, dtype=float)
    down_mask = np.zeros(dim, dtype=float)
    up_mask[list(spec.spin_up_indices)] = 1.0
    down_mask[list(spec.spin_down_indices)] = 1.0
    total = np.zeros((len(energies), len(kvalues)), dtype=float)
    up = np.zeros_like(total)
    down = np.zeros_like(total)
    for ik, kp in enumerate(kvalues):
        h0, hp = axial_blocks(float(kp), hoppings)
        for ie, energy_rel in enumerate(energies):
            g = sancho_surface_green(float(energy_rel + ef), eta, h0, hp)
            diagonal = np.maximum(
                0.0, -np.imag(np.diag(g)) / np.pi
            )
            total[ie, ik] = float(np.sum(diagonal))
            up[ie, ik] = float(np.sum(diagonal * up_mask))
            down[ie, ik] = float(np.sum(diagonal * down_mask))
    return {
        "k_parallel": kvalues,
        "energy": energies,
        "surface_dos": total,
        "surface_dos_up": up,
        "surface_dos_down": down,
        "eta": float(eta),
    }


def band_berry_curvature(
    h: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    denominator_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    energies, vectors = np.linalg.eigh(h)
    omega = np.zeros(len(energies), dtype=float)
    for ib in range(len(energies)):
        value = 0.0
        for jb in range(len(energies)):
            if ib == jb:
                continue
            de = float(energies[ib] - energies[jb])
            denom = max(de * de, denominator_floor)
            a = np.vdot(vectors[:, ib], vx @ vectors[:, jb])
            b = np.vdot(vectors[:, jb], vy @ vectors[:, ib])
            value += 2.0 * float(np.imag(a * b)) / denom
        omega[ib] = value
    return energies, omega


def calculate_hall(
    spec: ModelSpec,
    hoppings: dict[tuple[int, int], np.ndarray],
    ef: float,
    energy_window: float,
    nk: int,
    energy_points: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    records: dict[str, list[tuple[float, float]]] = {"up": [], "down": []}
    area_weight = (2.0 * np.pi / nk) ** 2 / (2.0 * np.pi)
    kvalues = -np.pi + 2.0 * np.pi * (np.arange(nk) + 0.5) / nk
    spin_blocks = (
        ("up", np.asarray(spec.spin_up_indices, dtype=int)),
        ("down", np.asarray(spec.spin_down_indices, dtype=int)),
    )
    for kx in kvalues:
        for ky in kvalues:
            h, vx, vy = bloch_from_hoppings(float(kx), float(ky), hoppings)
            for spin_name, indices in spin_blocks:
                es, omega = band_berry_curvature(
                    h[np.ix_(indices, indices)],
                    vx[np.ix_(indices, indices)],
                    vy[np.ix_(indices, indices)],
                )
                for energy, curvature in zip(es, omega):
                    records[spin_name].append(
                        (float(energy - ef), float(curvature * area_weight))
                    )

    energy = np.linspace(-energy_window, energy_window, int(energy_points))
    cumulative: dict[str, np.ndarray] = {}
    for spin_name, values in records.items():
        rec = np.asarray(values, dtype=float)
        order = np.argsort(rec[:, 0])
        e_sorted = rec[order, 0]
        w_sorted = rec[order, 1]
        cumulative_weight = np.cumsum(w_sorted)
        indices = np.searchsorted(e_sorted, energy, side="right") - 1
        curve = np.zeros_like(energy)
        valid = indices >= 0
        curve[valid] = cumulative_weight[indices[valid]]
        cumulative[spin_name] = curve
    charge = cumulative["up"] + cumulative["down"]
    spin = 0.5 * (cumulative["up"] - cumulative["down"])
    df = pd.DataFrame(
        {
            "energy": energy,
            "sigma_xy_up_e2_over_h": cumulative["up"],
            "sigma_xy_down_e2_over_h": cumulative["down"],
            "sigma_xy_charge_e2_over_h": charge,
            "sigma_xy_spin_e_over_2pi": spin,
        }
    )
    mid = int(np.argmin(np.abs(energy)))
    summary = {
        "response_nk": int(nk),
        "mid_reference_sigma_up": float(cumulative["up"][mid]),
        "mid_reference_sigma_down": float(cumulative["down"][mid]),
        "mid_reference_sigma_charge": float(charge[mid]),
        "mid_reference_sigma_spin": float(spin[mid]),
    }
    return df, summary


def choose_system_window(
    audits: list[dict[str, Any]],
    system: str,
) -> float:
    direct = [
        abs(float(row["min_direct_gap"]))
        for row in audits
        if row["system"] == system and row["state"] != "critical"
    ]
    base = max(direct, default=0.05)
    minimum = {"lieb": 0.24, "tts": 0.36, "fes": 0.10}[system]
    maximum = {"lieb": 0.55, "tts": 0.60, "fes": 0.24}[system]
    return float(np.clip(2.2 * base + 0.08, minimum, maximum))


def state_title(selected: SelectedState) -> str:
    if selected.expected_chern_up is None:
        return "Critical"
    return rf"{STATE_LABELS[selected.state]}  $C_\uparrow={selected.expected_chern_up:+d}$"


def spin_colors(spec: ModelSpec, selected: SelectedState) -> tuple[str, str]:
    if selected.expected_chern_up is None:
        return CRITICAL_COLOR, BODY_DARK
    return (
        phase_color(spec, selected.expected_chern_up),
        phase_color(spec, selected.expected_chern_down),
    )


def surface_rgb(
    spec: ModelSpec,
    selected: SelectedState,
    surface: dict[str, np.ndarray | float],
) -> np.ndarray:
    up = np.log1p(np.asarray(surface["surface_dos_up"], dtype=float))
    down = np.log1p(np.asarray(surface["surface_dos_down"], dtype=float))
    scale = max(float(np.max(up)), float(np.max(down)), 1.0e-14)
    up /= scale
    down /= scale
    intensity = np.maximum(up, down)
    total = np.maximum(up + down, 1.0e-14)
    color_up, color_down = spin_colors(spec, selected)
    cup = np.asarray(to_rgb(color_up), dtype=float)
    cdown = np.asarray(to_rgb(color_down), dtype=float)
    mixed = (
        up[..., None] * cup[None, None, :]
        + down[..., None] * cdown[None, None, :]
    ) / total[..., None]
    white = np.ones_like(mixed)
    rgb = white * (1.0 - intensity[..., None]) + mixed * intensity[..., None]
    return np.clip(rgb, 0.0, 1.0)


def plot_bulk_grid(
    results: list[dict[str, Any]],
    specs: dict[str, ModelSpec],
    systems: list[str],
    output_stem: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(
        len(systems),
        3,
        figsize=(10.7, 3.0 * len(systems)),
        squeeze=False,
        sharey="row",
    )
    for irow, system in enumerate(systems):
        spec = specs[system]
        for icol, state in enumerate(STATE_ORDER):
            result = next(
                x for x in results
                if x["selected"].system == system and x["selected"].state == state
            )
            selected = result["selected"]
            df = result["bulk"]
            ax = axes[irow, icol]
            color_up, color_down = spin_colors(spec, selected)
            for _, band in df.groupby("band"):
                mean_spin = float(band["spin_z"].mean())
                color = color_up if mean_spin >= 0.0 else color_down
                ax.plot(
                    band["path_coordinate"],
                    band["energy"],
                    color=color,
                    lw=1.35,
                    alpha=0.96,
                )
            for tick in result["bulk_ticks"]:
                ax.axvline(tick, color="#B8B8B8", lw=0.65, zorder=0)
            ax.axhline(0.0, color=BLACK, lw=1.0, ls="--")
            ax.set_xticks(result["bulk_ticks"])
            ax.set_xticklabels(result["bulk_labels"])
            ax.set_ylim(-result["window"], result["window"])
            ax.set_title(state_title(selected))
            if icol == 0:
                ax.set_ylabel(
                    f"{spec.label}\n" + r"$E-E_{\mathrm{ref}}$"
                )
            style_axis(ax)
    fig.supxlabel("Bulk momentum path", fontweight="bold", y=0.01)
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    return save_figure(fig, output_stem, dpi)


def plot_ribbon_grid(
    results: list[dict[str, Any]],
    specs: dict[str, ModelSpec],
    systems: list[str],
    output_stem: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(
        len(systems),
        3,
        figsize=(10.7, 3.0 * len(systems)),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for irow, system in enumerate(systems):
        spec = specs[system]
        for icol, state in enumerate(STATE_ORDER):
            result = next(
                x for x in results
                if x["selected"].system == system and x["selected"].state == state
            )
            selected = result["selected"]
            df = result["ribbon"]
            ax = axes[irow, icol]
            visible = df[np.abs(df["energy"]) <= result["window"]]
            ax.scatter(
                visible["k_parallel"],
                visible["energy"],
                s=2.0,
                color=BODY_COLOR,
                alpha=0.40,
                linewidths=0,
                rasterized=True,
            )
            edge = visible[visible["total_edge_weight"] >= 0.28].copy()
            up = edge[edge["spin_z"] >= 0.0]
            down = edge[edge["spin_z"] < 0.0]
            color_up, color_down = spin_colors(spec, selected)
            for frame, color, marker in (
                (up, color_up, "o"),
                (down, color_down, "o"),
            ):
                ax.scatter(
                    frame["k_parallel"],
                    frame["energy"],
                    s=4.0 + 9.0 * frame["total_edge_weight"],
                    color=color,
                    alpha=0.92,
                    marker=marker,
                    linewidths=0,
                    rasterized=True,
                )
            ax.axhline(0.0, color=BLACK, lw=1.0, ls="--")
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-result["window"], result["window"])
            ax.set_xticks([-np.pi, 0.0, np.pi])
            ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
            ax.set_title(state_title(selected))
            if icol == 0:
                ax.set_ylabel(
                    f"{spec.label}\n" + r"$E-E_{\mathrm{ref}}$"
                )
            style_axis(ax)
    fig.supxlabel(r"$k_{\parallel}$", fontweight="bold", y=0.01)
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    return save_figure(fig, output_stem, dpi)


def plot_surface_grid(
    results: list[dict[str, Any]],
    specs: dict[str, ModelSpec],
    systems: list[str],
    output_stem: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(
        len(systems),
        3,
        figsize=(10.7, 3.0 * len(systems)),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for irow, system in enumerate(systems):
        spec = specs[system]
        for icol, state in enumerate(STATE_ORDER):
            result = next(
                x for x in results
                if x["selected"].system == system and x["selected"].state == state
            )
            selected = result["selected"]
            surface = result["surface"]
            rgb = surface_rgb(spec, selected, surface)
            ax = axes[irow, icol]
            ax.imshow(
                rgb,
                origin="lower",
                extent=[
                    -np.pi,
                    np.pi,
                    -result["window"],
                    result["window"],
                ],
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            ax.axhline(0.0, color=BLACK, lw=1.0, ls="--")
            ax.set_xticks([-np.pi, 0.0, np.pi])
            ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
            ax.set_title(state_title(selected))
            if icol == 0:
                ax.set_ylabel(
                    f"{spec.label}\n" + r"$E-E_{\mathrm{ref}}$"
                )
            style_axis(ax)
    fig.supxlabel(r"$k_{\parallel}$", fontweight="bold", y=0.01)
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    return save_figure(fig, output_stem, dpi)


def plot_hall_grid(
    results: list[dict[str, Any]],
    specs: dict[str, ModelSpec],
    systems: list[str],
    output_stem: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(
        len(systems),
        3,
        figsize=(10.7, 3.0 * len(systems)),
        squeeze=False,
        sharex="row",
        sharey=True,
    )
    for irow, system in enumerate(systems):
        spec = specs[system]
        for icol, state in enumerate(STATE_ORDER):
            result = next(
                x for x in results
                if x["selected"].system == system and x["selected"].state == state
            )
            selected = result["selected"]
            df = result["hall"]
            ax = axes[irow, icol]
            color_up, color_down = spin_colors(spec, selected)
            ax.axvspan(
                -0.04 * result["window"],
                0.04 * result["window"],
                color=lighten(
                    phase_color(spec, selected.expected_chern_up), 0.76
                ),
                alpha=0.65,
                zorder=0,
            )
            ax.plot(
                df["energy"],
                df["sigma_xy_up_e2_over_h"],
                color=color_up,
                lw=2.0,
                label=r"$\sigma_{xy}^{\uparrow}$",
            )
            ax.plot(
                df["energy"],
                df["sigma_xy_down_e2_over_h"],
                color=color_down,
                lw=2.0,
                label=r"$\sigma_{xy}^{\downarrow}$",
            )
            ax.plot(
                df["energy"],
                df["sigma_xy_charge_e2_over_h"],
                color=BLACK,
                lw=1.3,
                ls="--",
                label=r"$\sigma_{xy}^{\mathrm{tot}}$",
            )
            ax.axhline(0.0, color="#777777", lw=0.8)
            ax.axvline(0.0, color=BLACK, lw=0.9, ls=":")
            ax.set_xlim(-result["window"], result["window"])
            ax.set_ylim(-2.35, 2.35)
            ax.set_title(state_title(selected))
            if icol == 0:
                ax.set_ylabel(
                    f"{spec.label}\n" + r"$\sigma_{xy}\ (e^2/h)$"
                )
            if irow == 0 and icol == 0:
                ax.legend(
                    loc="upper left",
                    frameon=False,
                    handlelength=2.0,
                )
            style_axis(ax)
    fig.supxlabel(r"$E-E_{\mathrm{ref}}$", fontweight="bold", y=0.01)
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 1.0))
    return save_figure(fig, output_stem, dpi)


def build_settings(quick: bool) -> RunSettings:
    if quick:
        return RunSettings(
            gap_nk=48,
            chern_nk=21,
            band_points_per_segment=28,
            ribbon_k_points=51,
            surface_k_points=51,
            surface_energy_points=71,
            response_nk=25,
            response_energy_points=101,
            fourier_n=8,
            dpi=150,
            quick=True,
        )
    return RunSettings(
        gap_nk=120,
        chern_nk=41,
        band_points_per_segment=70,
        ribbon_k_points=161,
        surface_k_points=141,
        surface_energy_points=181,
        response_nk=81,
        response_energy_points=241,
        fourier_n=8,
        dpi=320,
        quick=False,
    )


def hall_mesh_for_state(
    selected: SelectedState,
    settings: RunSettings,
) -> int:
    """Use denser Hall meshes only where the Berry curvature is concentrated."""
    if settings.quick:
        return int(settings.response_nk)
    if selected.system == "lieb":
        # The post-t2 sign-flip state has a small generic-valley gap and needs
        # a much denser Kubo mesh to recover the integer plateau.
        return 241 if selected.state == "chern_changed" else 121
    if selected.system == "fes":
        # The topological and critical FES examples have millielectronvolt-
        # scale direct gaps, although the topological point is an indirect
        # overlap metal and therefore has no quantized Fermi-level plateau.
        # At the exact Gamma critical point an odd midpoint mesh contains
        # k=(0,0), where the band-resolved Kubo curvature is singular.  The
        # neighboring even mesh is the symmetric principal-value convention.
        if selected.state == "critical":
            return 160
        return 161 if selected.state == "topological" else 81
    return 81


def refresh_hall_outputs(
    output_dir: Path,
    quick: bool = False,
) -> dict[str, Any]:
    """Refresh Hall data whose stored mesh differs from the current policy."""
    configure_plot_style()
    settings = build_settings(quick)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    audit_path = output_dir / "step18_01_selected_state_audit.csv"
    summary_path = output_dir / "step18_02_hall_summary.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    specs, modules = build_model_specs()
    selected_states = select_all_states(modules)
    audit_df = pd.read_csv(audit_path)
    old_summary = (
        pd.read_csv(summary_path)
        if summary_path.is_file()
        else pd.DataFrame()
    )
    old_lookup = {
        (str(row.system), str(row.state)): row
        for row in old_summary.itertuples(index=False)
    }
    audit_records = audit_df.to_dict(orient="records")
    windows = {
        system: choose_system_window(audit_records, system)
        for system in ("lieb", "tts", "fes")
    }
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    refreshed: list[str] = []
    for selected in selected_states:
        spec = specs[selected.system]
        desired_nk = hall_mesh_for_state(selected, settings)
        sample_dir = data_dir / f"{selected.system}_{selected.state}"
        hall_path = sample_dir / "wanniertools_style_hall.csv"
        old = old_lookup.get((selected.system, selected.state))
        stored_nk = int(old.response_nk) if old is not None else -1
        audit = audit_df[
            (audit_df["system"].astype(str) == selected.system)
            & (audit_df["state"].astype(str) == selected.state)
        ].iloc[0]
        ef = float(audit["energy_reference"])
        window = float(windows[selected.system])
        if hall_path.is_file() and stored_nk == desired_nk:
            hall = pd.read_csv(hall_path)
        else:
            hoppings = extract_hoppings(
                spec,
                selected.params,
                nfft=settings.fourier_n,
            )
            hall, _ = calculate_hall(
                spec,
                hoppings,
                ef,
                window,
                desired_nk,
                settings.response_energy_points,
            )
            hall.to_csv(hall_path, index=False)
            refreshed.append(f"{selected.system}:{selected.state}")
        mid = int(np.argmin(np.abs(hall["energy"].to_numpy(float))))
        summaries.append(
            {
                "system": selected.system,
                "state": selected.state,
                "source_id": selected.source_id,
                "expected_chern_up": selected.expected_chern_up,
                "response_nk": int(desired_nk),
                "mid_reference_sigma_up": float(
                    hall["sigma_xy_up_e2_over_h"].iloc[mid]
                ),
                "mid_reference_sigma_down": float(
                    hall["sigma_xy_down_e2_over_h"].iloc[mid]
                ),
                "mid_reference_sigma_charge": float(
                    hall["sigma_xy_charge_e2_over_h"].iloc[mid]
                ),
                "mid_reference_sigma_spin": float(
                    hall["sigma_xy_spin_e_over_2pi"].iloc[mid]
                ),
            }
        )
        results.append(
            {
                "selected": selected,
                "window": window,
                "hall": hall,
            }
        )
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    plot_hall_grid(
        results,
        specs,
        ["lieb", "tts", "fes"],
        figures_dir / "step18_three_system_hall_triptychs",
        settings.dpi,
    )
    for system in ("lieb", "tts", "fes"):
        plot_hall_grid(
            results,
            specs,
            [system],
            figures_dir / f"step18_{system}_hall_triptych",
            settings.dpi,
        )
    certificate_path = output_dir / "step18_04_final_certificate.json"
    if certificate_path.is_file():
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        certificate["hall_refresh_utc"] = utc_now()
        certificate["hall_refreshed_states"] = refreshed
        certificate["critical_hall_policy"] = (
            "The exact FES Gamma critical state uses an even 160x160 midpoint "
            "mesh, which excludes the singular closing momentum and implements "
            "the symmetric principal-value Kubo limit. Critical Hall values are "
            "not topological plateaus."
        )
        certificate_path.write_text(
            json.dumps(certificate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return {"refreshed": refreshed, "summary_rows": len(summaries)}


def run(
    output_dir: Path,
    quick: bool = False,
) -> dict[str, Any]:
    started = time.time()
    configure_plot_style()
    settings = build_settings(quick)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    specs, modules = build_model_specs()
    selected_states = select_all_states(modules)
    if len(selected_states) != 9:
        raise RuntimeError(f"Expected exactly 9 selected states, got {len(selected_states)}")

    print("[1/6] Gap and Chern audits for nine selected states", flush=True)
    audits: list[dict[str, Any]] = []
    for selected in selected_states:
        spec = specs[selected.system]
        gap = gap_audit(spec, selected, settings.gap_nk)
        chern = chern_audit(spec, selected, settings.chern_nk)
        audit = {
            "system": selected.system,
            "state": selected.state,
            "source_id": selected.source_id,
            "path_id": selected.path_id,
            "expected_chern_up": selected.expected_chern_up,
            "expected_chern_down": selected.expected_chern_down,
            "critical_kx": selected.critical_kx,
            "critical_ky": selected.critical_ky,
            "selection_note": selected.selection_note,
            **selected.params,
            **gap,
            **chern,
        }
        if selected.expected_chern_up is None:
            audit["chern_matches_selection"] = np.nan
            audit["critical_gap_pass"] = int(
                float(gap["critical_valley_gap"]) < 2.0e-5
            )
        else:
            audit["chern_matches_selection"] = int(
                int(chern["chern_up_int"]) == selected.expected_chern_up
                and int(chern["chern_down_int"]) == selected.expected_chern_down
            )
            audit["critical_gap_pass"] = np.nan
        audits.append(audit)
        print(
            f"  {selected.system:4s} {selected.state:13s} "
            f"gap={float(gap['min_direct_gap']):.4e} "
            f"indirect={float(gap['indirect_gap']):.4e} "
            f"Cup={chern['chern_up_int']}",
            flush=True,
        )
    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(output_dir / "step18_01_selected_state_audit.csv", index=False)

    windows = {
        system: choose_system_window(audits, system)
        for system in ("lieb", "tts", "fes")
    }

    print("[2/6] Wannier hopping reconstruction and bulk bands", flush=True)
    results: list[dict[str, Any]] = []
    hall_summaries: list[dict[str, Any]] = []
    for index, selected in enumerate(selected_states, start=1):
        spec = specs[selected.system]
        audit = next(
            row for row in audits
            if row["system"] == selected.system and row["state"] == selected.state
        )
        ef = float(audit["energy_reference"])
        window = float(windows[selected.system])
        sample_dir = data_dir / f"{selected.system}_{selected.state}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"  [{index}/9] {selected.system} {selected.state}: Fourier/bulk",
            flush=True,
        )
        hoppings = extract_hoppings(
            spec, selected.params, nfft=settings.fourier_n
        )
        hopping_rows = [
            {
                "rx": rx,
                "ry": ry,
                "max_abs_hopping": float(np.max(np.abs(mat))),
            }
            for (rx, ry), mat in sorted(hoppings.items())
        ]
        pd.DataFrame(hopping_rows).to_csv(
            sample_dir / "wannier_hopping_summary.csv", index=False
        )
        bulk, ticks, labels = calculate_bulk_bands(
            spec,
            selected,
            ef,
            settings.band_points_per_segment,
        )
        bulk.to_csv(sample_dir / "bulk_bands.csv", index=False)

        width = spec.ribbon_width
        if quick:
            width = min(width, 18 if selected.system == "lieb" else 16)
        print(
            f"  [{index}/9] {selected.system} {selected.state}: ribbon width={width}",
            flush=True,
        )
        ribbon = calculate_ribbon(
            spec,
            selected,
            hoppings,
            ef,
            settings.ribbon_k_points,
            width,
        )
        ribbon.to_csv(
            sample_dir / "wanniertools_style_ribbon.csv.gz",
            index=False,
            compression="gzip",
        )

        eta = max(
            5.0e-4 if selected.system == "fes" else 1.5e-3,
            0.006 * window,
        )
        print(
            f"  [{index}/9] {selected.system} {selected.state}: "
            f"semi-infinite surface eta={eta:.3e}",
            flush=True,
        )
        surface = calculate_surface(
            spec,
            hoppings,
            ef,
            window,
            settings.surface_k_points,
            settings.surface_energy_points,
            eta,
        )
        np.savez_compressed(
            sample_dir / "wanniertools_style_surface_spectrum.npz",
            **surface,
        )

        print(
            f"  [{index}/9] {selected.system} {selected.state}: Hall response",
            flush=True,
        )
        hall_nk = hall_mesh_for_state(selected, settings)
        hall, hall_summary = calculate_hall(
            spec,
            hoppings,
            ef,
            window,
            hall_nk,
            settings.response_energy_points,
        )
        hall.to_csv(sample_dir / "wanniertools_style_hall.csv", index=False)
        hall_summaries.append(
            {
                "system": selected.system,
                "state": selected.state,
                "source_id": selected.source_id,
                "expected_chern_up": selected.expected_chern_up,
                **hall_summary,
            }
        )
        results.append(
            {
                "selected": selected,
                "audit": audit,
                "window": window,
                "bulk": bulk,
                "bulk_ticks": ticks,
                "bulk_labels": labels,
                "ribbon": ribbon,
                "surface": surface,
                "hall": hall,
                "width": int(width),
            }
        )

    pd.DataFrame(hall_summaries).to_csv(
        output_dir / "step18_02_hall_summary.csv", index=False
    )

    print("[3/6] Three-system publication figures", flush=True)
    figure_manifest: list[dict[str, str]] = []
    plotters = (
        ("bulk", plot_bulk_grid),
        ("ribbon", plot_ribbon_grid),
        ("surface", plot_surface_grid),
        ("hall", plot_hall_grid),
    )
    for kind, plotter in plotters:
        paths = plotter(
            results,
            specs,
            ["lieb", "tts", "fes"],
            figures_dir / f"step18_three_system_{kind}_triptychs",
            settings.dpi,
        )
        for path in paths:
            figure_manifest.append(
                {"scope": "three_system", "kind": kind, "path": path}
            )

    print("[4/6] Individual-system figures", flush=True)
    for system in ("lieb", "tts", "fes"):
        for kind, plotter in plotters:
            paths = plotter(
                results,
                specs,
                [system],
                figures_dir / f"step18_{system}_{kind}_triptych",
                settings.dpi,
            )
            for path in paths:
                figure_manifest.append(
                    {"scope": system, "kind": kind, "path": path}
                )
    pd.DataFrame(figure_manifest).to_csv(
        output_dir / "step18_03_figure_manifest.csv", index=False
    )

    print("[5/6] Physics and color certificates", flush=True)
    noncritical = audit_df[audit_df["state"] != "critical"]
    critical = audit_df[audit_df["state"] == "critical"]
    certificate = {
        "code_version": CODE_VERSION,
        "created_utc": utc_now(),
        "quick": bool(quick),
        "selected_state_count": int(len(audit_df)),
        "exactly_three_states_per_system": bool(
            audit_df.groupby("system")["state"].nunique().eq(3).all()
        ),
        "noncritical_chern_match": bool(
            noncritical["chern_matches_selection"].fillna(0).astype(int).eq(1).all()
        ),
        "critical_gap_closure_pass": bool(
            critical["critical_gap_pass"].fillna(0).astype(int).eq(1).all()
        ),
        "fes_topological_global_insulator": bool(
            audit_df[
                (audit_df["system"] == "fes")
                & (audit_df["state"] == "topological")
            ]["is_global_insulator"].iloc[0]
        ),
        "fes_topological_classification": (
            "global_insulator"
            if bool(
                audit_df[
                    (audit_df["system"] == "fes")
                    & (audit_df["state"] == "topological")
                ]["is_global_insulator"].iloc[0]
            )
            else "spin_chern_band_metal"
        ),
        "palettes": {
            "lieb": LIEB_FES_COLORS,
            "fes": LIEB_FES_COLORS,
            "tts": TTS_COLORS,
            "critical": CRITICAL_COLOR,
            "bulk": BODY_COLOR,
        },
        "color_policy": (
            "Exact phase-map strong colors for spin-resolved edge and Hall data; "
            "critical spectra use neutral dark gray; bulk states use light gray."
        ),
        "energy_reference_policy": {
            "lieb": "global midpoint between VBM and CBM",
            "tts": "global midpoint between VBM and CBM",
            "fes": (
                "local midpoint at the selected Gamma transition valley because "
                "the topological representative is an indirect-overlap band metal"
            ),
            "critical": "critical closing energy",
        },
        "hall_units": {
            "spin_resolved_and_charge": "e^2/h",
            "spin_hall": "e/(2*pi)",
        },
        "run_settings": settings.__dict__,
        "elapsed_seconds": float(time.time() - started),
    }
    with (output_dir / "step18_04_final_certificate.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(certificate, handle, indent=2, ensure_ascii=False)

    print("[6/6] Complete", flush=True)
    print(json.dumps(certificate, indent=2, ensure_ascii=False), flush=True)
    return certificate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--refresh-hall-only",
        action="store_true",
        help="Reuse existing nine-state outputs and refresh Hall data/figures only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.refresh_hall_only:
        result = refresh_hall_outputs(
            Path(args.output_dir),
            quick=bool(args.quick),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        run(Path(args.output_dir), quick=bool(args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
