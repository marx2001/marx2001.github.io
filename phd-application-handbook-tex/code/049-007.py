#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 07 — final observable validation
==========================================

Final validation of the tts eight-band altermagnetic Hamiltonian through:

1. bulk bands on the square-lattice high-symmetry path;
2. finite ribbons for axial and diagonal terminations;
3. semi-infinite surface spectral functions with spin resolution;
4. bulk DOS around the global insulating gap;
5. energy-resolved intrinsic spin Hall conductivity (SHC);
6. an observable certificate for C_up = 0, ±1, ±2 representatives.

Plotting conventions follow the uploaded WannierTools ``src`` style:
- thick borders and large labels;
- ribbon edge weight: green -> yellow -> red;
- spin-resolved surface spectrum: blue -> white -> red;
- DOS and conductivity: clean thick-line panels;
- 1920x1680-like publication aspect ratios at high DPI.

The code is serial and Windows/Jupyter safe. It does not use ProcessPoolExecutor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
import argparse
import io
import json
import math
import os
import time
import zipfile

# Dense eigensolvers become extremely slow on some Windows/Jupyter and CI
# environments when BLAS starts many threads for small/medium matrices.
# Keep these calculations deterministic and avoid thread oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

# Apply the limit even when NumPy/SciPy was imported earlier by a notebook.
_THREADPOOL_LIMITER = threadpool_limits(limits=1)

import TTS_step01_model_and_label_audit_v2 as step1


CODE_VERSION = "TTS_STEP07_V1_20260714"
PARAM7 = tuple(step1.REDUCED7)
PARAM8 = tuple(step1.RAW8)

# Two boundary orientations. Columns are real-space supercell vectors A1, A2.
# A1 is periodic along the ribbon; A2 is finite / surface-normal.
EDGE_SPECS: Dict[str, Dict[str, Any]] = {
    "axial_x": {
        "label": r"axial edge $A_1=(1,0)$",
        "A": np.array([[1, 0], [0, 1]], dtype=int),
    },
    "diagonal_11": {
        "label": r"diagonal edge $A_1=(1,1)$",
        "A": np.array([[1, 1], [1, -1]], dtype=int),
    },
}

# WannierTools-like palettes inferred from uploaded src.
WT_EDGE_CMAP = LinearSegmentedColormap.from_list(
    "wt_edge_green_yellow_red", ["green", "yellow", "red"]
)
WT_SPIN_CMAP = LinearSegmentedColormap.from_list(
    "wt_spin_blue_white_red", ["#194eff", "white", "red"]
)


@dataclass
class Step07Config:
    step3_input: Path = Path("outputs_tts_step03_global_sobol_hierarchical_topology_ml.zip")
    output_dir: Path = Path("outputs_tts_step07_observable_validation")

    # Representative Chern sectors. The paper parameter set is added as C=0.
    chern_sectors: tuple[int, ...] = (-2, -1, 1, 2)

    # Bulk and response grids.
    gap_nk: int = 101
    band_points_per_segment: int = 70
    response_nk: int = 81
    response_energy_points: int = 321
    dos_sigma_fraction_of_gap: float = 0.025
    minimum_dos_sigma: float = 0.002

    # Ribbon calculations.
    ribbon_k_points: int = 161
    ribbon_width_axial: int = 36
    ribbon_width_diagonal: int = 24
    ribbon_edge_layers: int = 3
    ribbon_plot_margin_factor: float = 0.75
    ribbon_min_plot_margin: float = 0.35
    ribbon_edge_weight_threshold: float = 0.25

    # Surface Green functions.
    surface_k_points: int = 141
    surface_energy_points: int = 181
    surface_eta_fraction_of_gap: float = 0.012
    surface_min_eta: float = 0.002
    surface_max_iter: int = 80
    surface_tol: float = 1.0e-12
    surface_peak_prominence_fraction: float = 0.08

    # Fourier decomposition of periodic Hamiltonian.
    fourier_n: int = 8
    fourier_tolerance: float = 1.0e-10

    # Observable certificate tolerances.
    shc_plateau_tolerance: float = 0.08
    charge_hall_tolerance: float = 0.08
    normalized_gap_dos_tolerance: float = 0.06

    dpi: int = 240
    quick: bool = False
    force_recalculate: bool = False

    def normalized(self) -> "Step07Config":
        self.step3_input = Path(self.step3_input)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)

        if self.quick:
            self.chern_sectors = (-1, 2)
            self.gap_nk = min(self.gap_nk, 41)
            self.band_points_per_segment = min(self.band_points_per_segment, 24)
            self.response_nk = min(self.response_nk, 31)
            self.response_energy_points = min(self.response_energy_points, 101)
            self.ribbon_k_points = min(self.ribbon_k_points, 41)
            self.ribbon_width_axial = min(self.ribbon_width_axial, 14)
            self.ribbon_width_diagonal = min(self.ribbon_width_diagonal, 10)
            self.ribbon_edge_layers = min(self.ribbon_edge_layers, 2)
            self.surface_k_points = min(self.surface_k_points, 41)
            self.surface_energy_points = min(self.surface_energy_points, 61)
            self.surface_max_iter = min(self.surface_max_iter, 50)
            self.dpi = min(self.dpi, 150)

        for name in (
            "gap_nk", "band_points_per_segment", "response_nk",
            "response_energy_points", "ribbon_k_points", "surface_k_points",
            "surface_energy_points", "fourier_n",
        ):
            if int(getattr(self, name)) < 5:
                raise ValueError(f"{name} must be >= 5")
        return self


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def wrap_angle(x: float) -> float:
    return float((float(x) + np.pi) % (2.0 * np.pi) - np.pi)


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.linewidth": 2.0,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "legend.fontsize": 10,
            "savefig.bbox": "tight",
        }
    )


def resolve_existing_path(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    candidates = sorted(Path.cwd().glob(f"*{path.name}*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(path)


def read_matching_csv_from_zip(zip_path: Path, keyword: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if keyword in n and n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError(f"No CSV containing {keyword!r} in {zip_path}")
        if len(names) > 1:
            names.sort(key=len)
        return pd.read_csv(zf.open(names[0]))


def raw_params_from_row(row: pd.Series) -> Dict[str, float]:
    if all(name in row.index for name in PARAM8):
        return {name: float(row[name]) for name in PARAM8}
    reduced = {name: float(row[name]) for name in PARAM7}
    return step1.raw8_from_reduced7(reduced)


def select_representatives(step3_zip: Path, sectors: Sequence[int]) -> pd.DataFrame:
    """Select the largest-gap strict representative in each requested sector."""
    best = read_matching_csv_from_zip(step3_zip, "best_gap_candidate_by_chern_sector")
    rows: list[dict[str, Any]] = []
    for c in sectors:
        subset = best[np.rint(best["chern_up_int"]).astype(int) == int(c)]
        if subset.empty:
            strict = read_matching_csv_from_zip(step3_zip, "strict_spin_chern_TI_candidates")
            subset = strict[np.rint(strict["chern_up_int"]).astype(int) == int(c)]
        if subset.empty:
            raise RuntimeError(f"No strict representative found for C_up={c}")
        row = subset.sort_values("indirect_gap", ascending=False).iloc[0]
        rec = {key: row[key] for key in row.index}
        rec["representative_label"] = f"C_up_{int(c):+d}"
        rec["expected_chern_up"] = int(c)
        rec["expected_chern_down"] = -int(c)
        rec["selection_reason"] = "largest strict indirect gap in Step 03"
        rows.append(rec)

    # The exact literature parameter set is the clean trivial control.
    paper_gap = step1.scan_band_gaps(step1.PAPER_PARAMS, nk=81, shift=(0.0, 0.0))
    paper = {
        "sample_id": "paper_trivial_exact",
        "sample_source": "literature_parameter_control",
        "representative_label": "C_up_+0",
        "expected_chern_up": 0,
        "expected_chern_down": 0,
        "selection_reason": "exact tts literature parameter set",
        **step1.reduced7_from_raw8(step1.PAPER_PARAMS),
        **step1.PAPER_PARAMS,
        **paper_gap,
    }
    rows.append(paper)

    result = pd.DataFrame(rows)
    # Put trivial first, followed by increasing Chern.
    order = {0: 0, -1: 1, 1: 2, -2: 3, 2: 4}
    result["_order"] = result["expected_chern_up"].map(order).fillna(99)
    result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return result


# -----------------------------------------------------------------------------
# Finite Fourier representation H(k) = sum_R H_R exp(i k.R)
# -----------------------------------------------------------------------------

def extract_hopping_matrices(
    params: Dict[str, float],
    nfft: int = 8,
    tolerance: float = 1.0e-10,
) -> Dict[tuple[int, int], np.ndarray]:
    nfft = int(nfft)
    ks = 2.0 * np.pi * np.arange(nfft) / nfft
    hk = np.empty((nfft, nfft, 8, 8), dtype=np.complex128)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            hk[ix, iy] = step1.h_tts_periodic(float(kx), float(ky), params)

    coeff = np.fft.fft2(hk, axes=(0, 1)) / float(nfft * nfft)
    hoppings: Dict[tuple[int, int], np.ndarray] = {}
    for ix in range(nfft):
        rx = ix if ix <= nfft // 2 else ix - nfft
        for iy in range(nfft):
            ry = iy if iy <= nfft // 2 else iy - nfft
            mat = coeff[ix, iy]
            if float(np.max(np.abs(mat))) > float(tolerance):
                hoppings[(int(rx), int(ry))] = np.asarray(mat, dtype=np.complex128)

    # Hard reconstruction audit.
    for kx, ky in ((0.173, -1.029), (2.037, 0.691), (-2.4, 2.1)):
        rec = np.zeros((8, 8), dtype=np.complex128)
        for (rx, ry), mat in hoppings.items():
            rec += mat * np.exp(1j * (kx * rx + ky * ry))
        ref = step1.h_tts_periodic(kx, ky, params)
        err = float(np.max(np.abs(rec - ref)))
        if err > 1.0e-9:
            raise RuntimeError(f"Fourier reconstruction failed: {err:.3e}")
    return hoppings


def bloch_from_hoppings(
    kx: float,
    ky: float,
    hoppings: Dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.zeros((8, 8), dtype=np.complex128)
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


# -----------------------------------------------------------------------------
# Supercell / ribbon geometry
# -----------------------------------------------------------------------------

def integer_coset_representatives(A: np.ndarray) -> list[np.ndarray]:
    A = np.asarray(A, dtype=int)
    det = abs(int(round(np.linalg.det(A))))
    if det < 1:
        raise ValueError("Supercell matrix must be nonsingular")
    invA = np.linalg.inv(A.astype(float))

    candidates = [np.array([0, 0], dtype=int)]
    radius = 1
    while len(candidates) < 20 * det:
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                candidates.append(np.array([x, y], dtype=int))
        radius += 1
        if radius > 8:
            break

    reps: list[np.ndarray] = []
    for candidate in candidates:
        equivalent = False
        for rep in reps:
            delta = invA @ (candidate - rep)
            if np.allclose(delta, np.rint(delta), atol=1.0e-10):
                equivalent = True
                break
        if not equivalent:
            reps.append(candidate.copy())
        if len(reps) == det:
            break
    if len(reps) != det:
        raise RuntimeError("Failed to construct integer supercell cosets")
    return reps


def locate_coset(q: np.ndarray, A: np.ndarray, reps: Sequence[np.ndarray]) -> tuple[int, np.ndarray]:
    invA = np.linalg.inv(np.asarray(A, dtype=float))
    for i, rep in enumerate(reps):
        delta = invA @ (np.asarray(q, dtype=float) - np.asarray(rep, dtype=float))
        rounded = np.rint(delta)
        if np.allclose(delta, rounded, atol=1.0e-9):
            return int(i), rounded.astype(int)
    raise RuntimeError(f"Unable to map primitive cell {q} into supercell")


def transformed_principal_layer_blocks(
    k_parallel: float,
    hoppings: Dict[tuple[int, int], np.ndarray],
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return H0, H(+normal), H(-normal) for one principal layer."""
    A = np.asarray(A, dtype=int)
    reps = integer_coset_representatives(A)
    ncoset = len(reps)
    n = 8 * ncoset
    blocks = {-1: np.zeros((n, n), complex), 0: np.zeros((n, n), complex), 1: np.zeros((n, n), complex)}

    for ia, ca in enumerate(reps):
        for (rx, ry), mat in hoppings.items():
            q = ca + np.array([rx, ry], dtype=int)
            ib, delta = locate_coset(q, A, reps)
            dpar, dperp = int(delta[0]), int(delta[1])
            if dperp not in blocks:
                raise RuntimeError(
                    f"Principal layer too thin: normal displacement {dperp}; A={A.tolist()}"
                )
            rs = slice(8 * ia, 8 * (ia + 1))
            cs = slice(8 * ib, 8 * (ib + 1))
            blocks[dperp][rs, cs] += mat * np.exp(1j * float(k_parallel) * dpar)

    h0 = 0.5 * (blocks[0] + blocks[0].conj().T)
    hp = 0.5 * (blocks[1] + blocks[-1].conj().T)
    hm = hp.conj().T
    return h0, hp, hm, reps


def ribbon_hamiltonian(
    k_parallel: float,
    width: int,
    hoppings: Dict[tuple[int, int], np.ndarray],
    A: np.ndarray,
) -> tuple[np.ndarray, int]:
    h0, hp, hm, reps = transformed_principal_layer_blocks(k_parallel, hoppings, A)
    layer_dim = h0.shape[0]
    width = int(width)
    h = np.zeros((layer_dim * width, layer_dim * width), dtype=np.complex128)
    for layer in range(width):
        sl = slice(layer * layer_dim, (layer + 1) * layer_dim)
        h[sl, sl] = h0
        if layer + 1 < width:
            sr = slice((layer + 1) * layer_dim, (layer + 2) * layer_dim)
            h[sl, sr] = hp
            h[sr, sl] = hm
    h = 0.5 * (h + h.conj().T)
    return h, layer_dim


def spin_operator_for_layer(ncoset: int) -> np.ndarray:
    primitive = np.diag([1.0 if i in step1.SPIN_UP_INDICES else -1.0 for i in range(8)])
    return np.kron(np.eye(int(ncoset)), primitive)


def projected_bulk_bands(
    k_parallel_values: np.ndarray,
    hoppings: Dict[tuple[int, int], np.ndarray],
    A: np.ndarray,
    n_perp: int = 61,
) -> tuple[np.ndarray, np.ndarray]:
    invAT = np.linalg.inv(np.asarray(A, dtype=float).T)
    kperp_values = np.linspace(-np.pi, np.pi, int(n_perp), endpoint=False)
    records_k: list[float] = []
    records_e: list[float] = []
    for kp in k_parallel_values:
        for kn in kperp_values:
            kx, ky = invAT @ np.array([kp, kn], dtype=float)
            h, _, _ = bloch_from_hoppings(float(kx), float(ky), hoppings)
            energies = np.linalg.eigvalsh(h)
            records_k.extend([float(kp)] * len(energies))
            records_e.extend([float(x) for x in energies])
    return np.asarray(records_k), np.asarray(records_e)


# -----------------------------------------------------------------------------
# Bulk bands, ribbon spectra and edge crossing diagnostics
# -----------------------------------------------------------------------------

def high_symmetry_path(n_per_segment: int) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    points = [
        ("Γ", (0.0, 0.0)),
        ("X", (np.pi, 0.0)),
        ("M", (np.pi, np.pi)),
        ("Γ", (0.0, 0.0)),
        ("M′", (-np.pi, np.pi)),
        ("X′", (0.0, np.pi)),
        ("Γ", (0.0, 0.0)),
    ]
    klist: list[tuple[float, float]] = []
    distance: list[float] = []
    tick_positions: list[float] = [0.0]
    labels = [points[0][0]]
    current = 0.0
    previous: np.ndarray | None = None
    for iseg in range(len(points) - 1):
        p0 = np.asarray(points[iseg][1], float)
        p1 = np.asarray(points[iseg + 1][1], float)
        for j in range(int(n_per_segment)):
            if iseg > 0 and j == 0:
                continue
            t = j / float(n_per_segment)
            k = (1.0 - t) * p0 + t * p1
            if previous is not None:
                current += float(np.linalg.norm(k - previous))
            klist.append((float(k[0]), float(k[1])))
            distance.append(current)
            previous = k
        k = p1
        current += float(np.linalg.norm(k - previous)) if previous is not None else 0.0
        klist.append((float(k[0]), float(k[1])))
        distance.append(current)
        previous = k
        tick_positions.append(current)
        labels.append(points[iseg + 1][0])
    return np.asarray(klist), np.asarray(distance), tick_positions, labels


def calculate_bulk_band_data(
    params: Dict[str, float],
    ef: float,
    n_per_segment: int,
) -> pd.DataFrame:
    klist, distance, _, _ = high_symmetry_path(n_per_segment)
    rows: list[dict[str, Any]] = []
    for ik, ((kx, ky), xpos) in enumerate(zip(klist, distance)):
        h = step1.h_tts_atomic(float(kx), float(ky), params)
        evals, evecs = np.linalg.eigh(h)
        for band, energy in enumerate(evals):
            spin = float(np.sum(np.abs(evecs[step1.SPIN_UP_INDICES, band]) ** 2) - np.sum(np.abs(evecs[step1.SPIN_DOWN_INDICES, band]) ** 2))
            rows.append(
                {
                    "k_index": ik,
                    "path_coordinate": float(xpos),
                    "kx": float(kx),
                    "ky": float(ky),
                    "band": int(band + 1),
                    "energy": float(energy - ef),
                    "spin_z": spin,
                }
            )
    return pd.DataFrame(rows)


def calculate_ribbon_data(
    params: Dict[str, float],
    hoppings: Dict[tuple[int, int], np.ndarray],
    edge_name: str,
    width: int,
    k_points: int,
    edge_layers: int,
    ef: float,
    vbm_rel: float,
    cbm_rel: float,
    edge_weight_threshold: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    A = np.asarray(EDGE_SPECS[edge_name]["A"], int)
    reps = integer_coset_representatives(A)
    ncoset = len(reps)
    sz_layer = spin_operator_for_layer(ncoset)
    kvalues = np.linspace(-np.pi, np.pi, int(k_points), endpoint=True)
    rows: list[dict[str, Any]] = []

    for ik, kp in enumerate(kvalues):
        h, layer_dim = ribbon_hamiltonian(float(kp), int(width), hoppings, A)
        evals, evecs = eigh(h, overwrite_a=True, check_finite=False)
        edge_layers_eff = min(int(edge_layers), max(1, int(width) // 3))
        left_dim = edge_layers_eff * layer_dim
        right_start = (int(width) - edge_layers_eff) * layer_dim

        probability = np.abs(evecs) ** 2
        left_weight = np.sum(probability[:left_dim, :], axis=0)
        right_weight = np.sum(probability[right_start:, :], axis=0)
        total_edge = left_weight + right_weight

        spin_diag = np.tile(np.diag(sz_layer), int(width))
        spin_z = np.real(np.sum(probability * spin_diag[:, None], axis=0))

        for ib, energy in enumerate(evals):
            rows.append(
                {
                    "edge_name": edge_name,
                    "k_index": int(ik),
                    "k_parallel": float(kp),
                    "band": int(ib + 1),
                    "energy": float(energy - ef),
                    "left_edge_weight": float(left_weight[ib]),
                    "right_edge_weight": float(right_weight[ib]),
                    "total_edge_weight": float(total_edge[ib]),
                    "spin_z": float(spin_z[ib]),
                    "is_global_gap_state": int(vbm_rel < float(energy - ef) < cbm_rel),
                    "is_edge_localized": int(float(total_edge[ib]) >= float(edge_weight_threshold)),
                }
            )

    df = pd.DataFrame(rows)
    ingap = df[(df["is_global_gap_state"] == 1) & (df["is_edge_localized"] == 1)]
    summary = {
        "edge_name": edge_name,
        "width": int(width),
        "principal_layer_orbitals": int(8 * ncoset),
        "in_gap_edge_state_records": int(len(ingap)),
        "in_gap_spin_up_records": int(np.count_nonzero(ingap["spin_z"].to_numpy(float) > 0.5)),
        "in_gap_spin_down_records": int(np.count_nonzero(ingap["spin_z"].to_numpy(float) < -0.5)),
        "max_in_gap_edge_weight": float(ingap["total_edge_weight"].max()) if not ingap.empty else 0.0,
        "edge_states_detected": int(not ingap.empty),
    }
    return df, summary


# -----------------------------------------------------------------------------
# Semi-infinite surface Green function
# -----------------------------------------------------------------------------

def sancho_surface_green(
    energy: float,
    eta: float,
    h0: np.ndarray,
    hp: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    z = complex(float(energy), float(eta))
    n = h0.shape[0]
    identity = np.eye(n, dtype=np.complex128)

    eps = h0.copy()
    eps_surface = h0.copy()
    alpha = hp.copy()
    beta = hp.conj().T

    for _ in range(int(max_iter)):
        g = np.linalg.inv(z * identity - eps)
        agb = alpha @ g @ beta
        bga = beta @ g @ alpha
        eps_surface = eps_surface + agb
        eps = eps + agb + bga
        alpha_new = alpha @ g @ alpha
        beta_new = beta @ g @ beta
        alpha, beta = alpha_new, beta_new
        if max(float(np.linalg.norm(alpha, ord="fro")), float(np.linalg.norm(beta, ord="fro"))) < float(tol):
            break
    return np.linalg.inv(z * identity - eps_surface)


def calculate_surface_spectrum(
    hoppings: Dict[tuple[int, int], np.ndarray],
    edge_name: str,
    k_points: int,
    energies: np.ndarray,
    ef: float,
    eta: float,
    max_iter: int,
    tol: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    A = np.asarray(EDGE_SPECS[edge_name]["A"], int)
    reps = integer_coset_representatives(A)
    ncoset = len(reps)
    sz = spin_operator_for_layer(ncoset)
    up_mask = np.diag((np.diag(sz) > 0).astype(float))
    down_mask = np.diag((np.diag(sz) < 0).astype(float))

    kvalues = np.linspace(-np.pi, np.pi, int(k_points), endpoint=True)
    energies_abs = np.asarray(energies, float) + float(ef)
    total = np.zeros((len(energies), len(kvalues)), dtype=float)
    spin = np.zeros_like(total)
    up = np.zeros_like(total)
    down = np.zeros_like(total)

    for ik, kp in enumerate(kvalues):
        h0, hp, _, _ = transformed_principal_layer_blocks(float(kp), hoppings, A)
        for ie, energy in enumerate(energies_abs):
            g = sancho_surface_green(float(energy), eta, h0, hp, max_iter, tol)
            total[ie, ik] = max(0.0, float(-np.imag(np.trace(g)) / np.pi))
            up[ie, ik] = max(0.0, float(-np.imag(np.trace(up_mask @ g)) / np.pi))
            down[ie, ik] = max(0.0, float(-np.imag(np.trace(down_mask @ g)) / np.pi))
            spin[ie, ik] = float(up[ie, ik] - down[ie, ik])

    # Midgap peak count by spin. These peaks are a direct surface-state signature.
    imid = int(np.argmin(np.abs(energies)))
    peak_counts: Dict[str, int] = {}
    for name, curve in (("up", up[imid]), ("down", down[imid]), ("total", total[imid])):
        scale = max(float(np.max(curve)), 1.0e-14)
        prominence = max(1.0e-12, 0.08 * scale)
        peaks, _ = find_peaks(curve, prominence=prominence, distance=max(1, len(kvalues) // 50))
        peak_counts[name] = int(len(peaks))

    intensity = np.log1p(total)
    normalized_intensity = intensity / max(float(np.max(intensity)), 1.0e-14)
    polarization = spin / np.maximum(total, 1.0e-14)
    spin_field = np.clip(polarization, -1.0, 1.0) * normalized_intensity

    data = {
        "k_parallel": kvalues,
        "energy": np.asarray(energies, float),
        "surface_dos": total,
        "surface_dos_up": up,
        "surface_dos_down": down,
        "spin_polarization": polarization,
        "spin_weighted_log_intensity": spin_field,
    }
    summary = {
        "edge_name": edge_name,
        "surface_eta": float(eta),
        "midgap_peak_count_up": peak_counts["up"],
        "midgap_peak_count_down": peak_counts["down"],
        "midgap_peak_count_total": peak_counts["total"],
        "max_surface_dos": float(np.max(total)),
    }
    return data, summary


# -----------------------------------------------------------------------------
# Bulk DOS and energy-resolved Hall responses
# -----------------------------------------------------------------------------

def band_berry_curvature(
    h: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    denominator_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    energies, vectors = np.linalg.eigh(h)
    n = len(energies)
    omega = np.zeros(n, dtype=float)
    # Sign chosen to match the determinant-link Fukui convention used in Steps 01-06.
    for ib in range(n):
        value = 0.0
        for jb in range(n):
            if ib == jb:
                continue
            de = float(energies[ib] - energies[jb])
            denom = max(de * de, float(denominator_floor))
            a = np.vdot(vectors[:, ib], vx @ vectors[:, jb])
            b = np.vdot(vectors[:, jb], vy @ vectors[:, ib])
            value += 2.0 * float(np.imag(a * b)) / denom
        omega[ib] = value
    return energies, omega


def calculate_dos_and_hall(
    hoppings: Dict[tuple[int, int], np.ndarray],
    ef: float,
    energy_grid: np.ndarray,
    nk: int,
    dos_sigma: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    all_energies: list[float] = []
    spin_records: Dict[str, list[tuple[float, float]]] = {"up": [], "down": []}
    area_weight = (2.0 * np.pi / int(nk)) ** 2 / (2.0 * np.pi)
    kvalues = -np.pi + 2.0 * np.pi * (np.arange(int(nk)) + 0.5) / int(nk)

    for kx in kvalues:
        for ky in kvalues:
            h, vx, vy = bloch_from_hoppings(float(kx), float(ky), hoppings)
            evals = np.linalg.eigvalsh(h)
            all_energies.extend([float(e - ef) for e in evals])
            for spin_name, indices in (
                ("up", step1.SPIN_UP_INDICES),
                ("down", step1.SPIN_DOWN_INDICES),
            ):
                idx = np.asarray(indices, int)
                es, omega = band_berry_curvature(
                    h[np.ix_(idx, idx)],
                    vx[np.ix_(idx, idx)],
                    vy[np.ix_(idx, idx)],
                )
                for energy, curvature in zip(es, omega):
                    spin_records[spin_name].append((float(energy - ef), float(curvature * area_weight)))

    energy_grid = np.asarray(energy_grid, float)
    all_e = np.asarray(all_energies, float)
    sigma = max(float(dos_sigma), 1.0e-8)
    dos = np.zeros_like(energy_grid)
    prefactor = 1.0 / (math.sqrt(2.0 * math.pi) * sigma * int(nk) * int(nk))
    # Chunk the Gaussian evaluation to keep memory bounded.
    for start in range(0, len(all_e), 1024):
        chunk = all_e[start:start + 1024]
        x = (energy_grid[:, None] - chunk[None, :]) / sigma
        dos += prefactor * np.sum(np.exp(-0.5 * x * x), axis=1)

    cumulative: Dict[str, np.ndarray] = {}
    plateau_chern: Dict[str, float] = {}
    for spin_name, records in spin_records.items():
        rec = np.asarray(records, dtype=float)
        order = np.argsort(rec[:, 0])
        energies_sorted = rec[order, 0]
        weights_sorted = rec[order, 1]
        cumulative_weight = np.cumsum(weights_sorted)
        indices = np.searchsorted(energies_sorted, energy_grid, side="right") - 1
        values = np.zeros_like(energy_grid)
        valid = indices >= 0
        values[valid] = cumulative_weight[indices[valid]]
        cumulative[spin_name] = values
        mid = int(np.argmin(np.abs(energy_grid)))
        plateau_chern[spin_name] = float(values[mid])

    charge_hall = cumulative["up"] + cumulative["down"]
    # Dimensionless spin Hall conductivity in units e/(2*pi).
    spin_hall = 0.5 * (cumulative["up"] - cumulative["down"])
    df = pd.DataFrame(
        {
            "energy": energy_grid,
            "dos": dos,
            "chern_up_below_energy": cumulative["up"],
            "chern_down_below_energy": cumulative["down"],
            "charge_hall_e2_over_h": charge_hall,
            "spin_hall_e_over_2pi": spin_hall,
        }
    )
    summary = {
        "response_nk": int(nk),
        "dos_sigma": float(sigma),
        "midgap_chern_up_kubo": plateau_chern["up"],
        "midgap_chern_down_kubo": plateau_chern["down"],
        "midgap_charge_hall_e2_over_h": float(charge_hall[np.argmin(np.abs(energy_grid))]),
        "midgap_spin_hall_e_over_2pi": float(spin_hall[np.argmin(np.abs(energy_grid))]),
    }
    return df, summary


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def add_gap_shading(ax: plt.Axes, vbm: float, cbm: float) -> None:
    ax.axhspan(float(vbm), float(cbm), color="0.92", zorder=0)
    ax.axhline(0.0, color="0.25", lw=1.2, ls="--", zorder=1)


def plot_bulk_bands(
    df: pd.DataFrame,
    ticks: Sequence[float],
    labels: Sequence[str],
    vbm: float,
    cbm: float,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    add_gap_shading(ax, vbm, cbm)
    for band in sorted(df["band"].unique()):
        sub = df[df["band"] == band]
        ax.plot(sub["path_coordinate"], sub["energy"], color="black", lw=1.2)
    for x in ticks[1:-1]:
        ax.axvline(float(x), color="0.7", lw=0.8)
    ax.set_xticks(ticks, labels)
    ax.set_xlim(float(min(ticks)), float(max(ticks)))
    margin = max(0.35, 0.7 * float(cbm - vbm))
    ax.set_ylim(float(vbm - margin), float(cbm + margin))
    ax.set_ylabel("Energy relative to midgap")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_ribbon(
    df: pd.DataFrame,
    bulk_k: np.ndarray,
    bulk_e: np.ndarray,
    vbm: float,
    cbm: float,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    add_gap_shading(ax, vbm, cbm)
    margin = max(0.35, 0.75 * float(cbm - vbm))
    ymin, ymax = float(vbm - margin), float(cbm + margin)
    mask_bulk = (bulk_e >= ymin) & (bulk_e <= ymax)
    ax.scatter(bulk_k[mask_bulk], bulk_e[mask_bulk], s=1.0, color="0.83", alpha=0.35, rasterized=True)

    mask = (df["energy"] >= ymin) & (df["energy"] <= ymax)
    sub = df[mask]
    weight = np.clip(sub["total_edge_weight"].to_numpy(float), 0.0, 1.0)
    order = np.argsort(weight)
    scatter = ax.scatter(
        sub["k_parallel"].to_numpy(float)[order],
        sub["energy"].to_numpy(float)[order],
        c=weight[order],
        cmap=WT_EDGE_CMAP,
        norm=Normalize(0.0, 1.0),
        s=3.5 + 14.0 * weight[order],
        linewidths=0,
        rasterized=True,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Edge weight")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.set_xlabel(r"$k_{\parallel}$")
    ax.set_ylabel("Energy relative to midgap")
    ax.set_title(title)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_surface_spectrum(
    data: Dict[str, np.ndarray],
    vbm: float,
    cbm: float,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    k = data["k_parallel"]
    e = data["energy"]
    field = data["spin_weighted_log_intensity"]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    mesh = ax.pcolormesh(k, e, field, shading="auto", cmap=WT_SPIN_CMAP, vmin=-1.0, vmax=1.0, rasterized=True)
    ax.axhline(0.0, color="black", lw=1.1, ls="--")
    ax.axhline(vbm, color="0.25", lw=0.8, ls=":")
    ax.axhline(cbm, color="0.25", lw=0.8, ls=":")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(float(e.min()), float(e.max()))
    ax.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.set_xlabel(r"$k_{\parallel}$")
    ax.set_ylabel("Energy relative to midgap")
    ax.set_title(title + "\nred: spin up, blue: spin down")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Spin-weighted log spectral intensity")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_dos_shc(
    df: pd.DataFrame,
    vbm: float,
    cbm: float,
    expected_chern: int,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)
    energy = df["energy"].to_numpy(float)

    axes[0].axvspan(vbm, cbm, color="0.92")
    axes[0].plot(energy, df["dos"], color="black", lw=2.0)
    axes[0].axvline(0.0, color="0.3", lw=1.0, ls="--")
    axes[0].set_xlabel("Energy relative to midgap")
    axes[0].set_ylabel("DOS (states / model energy / cell)")
    axes[0].set_title("Bulk DOS")
    axes[0].set_ylim(bottom=0.0)

    axes[1].axvspan(vbm, cbm, color="0.92")
    axes[1].plot(energy, df["spin_hall_e_over_2pi"], color="red", lw=2.2, label=r"$\sigma^s_{xy}$")
    axes[1].plot(energy, df["charge_hall_e2_over_h"], color="#194eff", lw=1.6, label=r"$\sigma_{xy}$")
    axes[1].axhline(float(expected_chern), color="0.25", lw=1.0, ls=":", label=r"expected $C_\uparrow$")
    axes[1].axhline(0.0, color="0.3", lw=0.9)
    axes[1].axvline(0.0, color="0.3", lw=1.0, ls="--")
    axes[1].set_xlabel("Energy relative to midgap")
    axes[1].set_ylabel(r"Hall response: $\sigma^s$ in $e/2\pi$, $\sigma$ in $e^2/h$")
    axes[1].set_title("Energy-resolved Hall response")
    axes[1].legend(frameon=False, loc="best")

    fig.suptitle(title, y=1.02, fontsize=15)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_observable_certificate(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    ordered = summary.sort_values("expected_chern_up")
    labels = [f"C↑={int(x):+d}" for x in ordered["expected_chern_up"]]
    expected = ordered["expected_chern_up"].to_numpy(float)
    measured = ordered["shc_plateau_median"].to_numpy(float)
    dos_ratio = ordered["normalized_gap_dos_max"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(labels))
    axes[0].plot(x, expected, "o-", color="black", lw=1.8, label="Expected C↑")
    axes[0].plot(x, measured, "s--", color="red", lw=1.8, label=r"SHC plateau ($e/2\pi$)")
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Topological response")
    axes[0].set_title("Quantized spin Hall plateau")
    axes[0].legend(frameon=False)

    axes[1].bar(x, dos_ratio, color="0.45")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("max gap DOS / max DOS")
    axes[1].set_title("Clean bulk insulating window")
    axes[1].axhline(0.06, color="red", lw=1.2, ls="--")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def energy_window_from_gap(gap: Dict[str, Any], margin_factor: float, min_margin: float) -> tuple[float, float, float, float, float]:
    vbm = float(gap["vbm"])
    cbm = float(gap["cbm"])
    ef = 0.5 * (vbm + cbm)
    vbm_rel = vbm - ef
    cbm_rel = cbm - ef
    gap_value = cbm - vbm
    margin = max(float(min_margin), float(margin_factor) * gap_value)
    return ef, vbm_rel, cbm_rel, vbm_rel - margin, cbm_rel + margin


def plateau_metrics(
    response: pd.DataFrame,
    vbm: float,
    cbm: float,
    expected_chern: int,
) -> Dict[str, float]:
    lo = float(vbm + 0.2 * (cbm - vbm))
    hi = float(cbm - 0.2 * (cbm - vbm))
    central = response[(response["energy"] >= lo) & (response["energy"] <= hi)]
    if central.empty:
        central = response.iloc[[int(np.argmin(np.abs(response["energy"].to_numpy(float))))]]
    shc = central["spin_hall_e_over_2pi"].to_numpy(float)
    charge = central["charge_hall_e2_over_h"].to_numpy(float)
    dos = central["dos"].to_numpy(float)
    max_dos_all = max(float(response["dos"].max()), 1.0e-14)
    return {
        "plateau_energy_min": lo,
        "plateau_energy_max": hi,
        "shc_plateau_median": float(np.median(shc)),
        "shc_plateau_max_error": float(np.max(np.abs(shc - float(expected_chern)))),
        "charge_hall_plateau_max_abs": float(np.max(np.abs(charge))),
        "gap_dos_max": float(np.max(dos)),
        "normalized_gap_dos_max": float(np.max(dos) / max_dos_all),
    }


def run_step07(config: Step07Config | None = None) -> Dict[str, Any]:
    if config is None:
        config = Step07Config()
    config = config.normalized()
    configure_plot_style()
    started = time.time()

    step3_zip = resolve_existing_path(config.step3_input)
    print("[1/8] Select strict Chern-sector representatives")
    representatives = select_representatives(step3_zip, config.chern_sectors)
    atomic_write_csv(representatives, config.output_dir / "step07_01_selected_observable_representatives.csv")
    print(representatives[["sample_id", "expected_chern_up", "selection_reason"]].to_string(index=False))

    summary_rows: list[dict[str, Any]] = []
    ribbon_summaries: list[dict[str, Any]] = []
    surface_summaries: list[dict[str, Any]] = []
    bulk_gap_rows: list[dict[str, Any]] = []

    _, _, band_ticks, band_labels = high_symmetry_path(config.band_points_per_segment)

    print("[2/8] Bulk gap, band and Fourier audits")
    for irow, row in representatives.iterrows():
        sample_id = str(row["sample_id"])
        expected_c = int(row["expected_chern_up"])
        params = raw_params_from_row(row)
        prefix = f"{irow:02d}_{sample_id}_Cup{expected_c:+d}".replace("+", "p").replace("-", "m")
        print(f"      {irow+1}/{len(representatives)} {sample_id}: C_up={expected_c:+d}")

        gap = step1.scan_band_gaps(params, nk=config.gap_nk, shift=(0.0, 0.0))
        ef, vbm_rel, cbm_rel, plot_emin, plot_emax = energy_window_from_gap(
            gap, config.ribbon_plot_margin_factor, config.ribbon_min_plot_margin
        )
        gap_record = {
            "sample_id": sample_id,
            "expected_chern_up": expected_c,
            "fermi_midgap": ef,
            "vbm_relative": vbm_rel,
            "cbm_relative": cbm_rel,
            **gap,
        }
        bulk_gap_rows.append(gap_record)

        hoppings = extract_hopping_matrices(params, config.fourier_n, config.fourier_tolerance)
        hr_rows = []
        for (rx, ry), mat in sorted(hoppings.items()):
            hr_rows.append(
                {
                    "sample_id": sample_id,
                    "Rx": rx,
                    "Ry": ry,
                    "max_abs_matrix_element": float(np.max(np.abs(mat))),
                    "frobenius_norm": float(np.linalg.norm(mat, ord="fro")),
                }
            )
        atomic_write_csv(pd.DataFrame(hr_rows), config.output_dir / "data" / f"{prefix}_hopping_summary.csv")

        band_df = calculate_bulk_band_data(params, ef, config.band_points_per_segment)
        atomic_write_csv(band_df, config.output_dir / "data" / f"{prefix}_bulk_bands.csv")
        plot_bulk_bands(
            band_df, band_ticks, band_labels, vbm_rel, cbm_rel,
            f"tts bulk bands — {sample_id}, C↑={expected_c:+d}",
            config.output_dir / "figures" / f"{prefix}_bulk_bands.png",
            config.dpi,
        )

        print("[3/8] Ribbon spectra for axial and diagonal terminations")
        sample_ribbon_records: list[dict[str, Any]] = []
        for edge_name, edge_spec in EDGE_SPECS.items():
            width = config.ribbon_width_axial if edge_name == "axial_x" else config.ribbon_width_diagonal
            ribbon_df, ribbon_summary = calculate_ribbon_data(
                params, hoppings, edge_name, width, config.ribbon_k_points,
                config.ribbon_edge_layers, ef, vbm_rel, cbm_rel,
                config.ribbon_edge_weight_threshold,
            )
            ribbon_summary.update({"sample_id": sample_id, "expected_chern_up": expected_c})
            ribbon_summaries.append(ribbon_summary)
            sample_ribbon_records.append(ribbon_summary)
            atomic_write_csv(ribbon_df, config.output_dir / "data" / f"{prefix}_ribbon_{edge_name}.csv")

            A = np.asarray(edge_spec["A"], int)
            bulk_k, bulk_e_abs = projected_bulk_bands(
                np.linspace(-np.pi, np.pi, config.ribbon_k_points), hoppings, A,
                n_perp=31 if config.quick else 61,
            )
            bulk_e = bulk_e_abs - ef
            plot_ribbon(
                ribbon_df, bulk_k, bulk_e, vbm_rel, cbm_rel,
                f"tts ribbon — {edge_spec['label']}, C↑={expected_c:+d}",
                config.output_dir / "figures" / f"{prefix}_ribbon_{edge_name}.png",
                config.dpi,
            )

        print("[4/8] Semi-infinite spin-resolved surface spectral functions")
        surface_margin = max(config.ribbon_min_plot_margin, 0.7 * float(gap["indirect_gap"]))
        surface_energies = np.linspace(vbm_rel - surface_margin, cbm_rel + surface_margin, config.surface_energy_points)
        eta = max(config.surface_min_eta, config.surface_eta_fraction_of_gap * float(gap["indirect_gap"]))
        sample_surface_records: list[dict[str, Any]] = []
        for edge_name, edge_spec in EDGE_SPECS.items():
            surface_data, surface_summary = calculate_surface_spectrum(
                hoppings, edge_name, config.surface_k_points, surface_energies,
                ef, eta, config.surface_max_iter, config.surface_tol,
            )
            gap_lo = vbm_rel + 0.10 * (cbm_rel - vbm_rel)
            gap_hi = cbm_rel - 0.10 * (cbm_rel - vbm_rel)
            gap_mask = (surface_data["energy"] >= gap_lo) & (surface_data["energy"] <= gap_hi)
            up_global_max = max(float(np.max(surface_data["surface_dos_up"])), 1.0e-14)
            down_global_max = max(float(np.max(surface_data["surface_dos_down"])), 1.0e-14)
            if np.any(gap_mask):
                gap_up_max = float(np.max(surface_data["surface_dos_up"][gap_mask, :]))
                gap_down_max = float(np.max(surface_data["surface_dos_down"][gap_mask, :]))
            else:
                gap_up_max = 0.0
                gap_down_max = 0.0
            surface_summary.update({
                "sample_id": sample_id,
                "expected_chern_up": expected_c,
                "central_gap_surface_up_max": gap_up_max,
                "central_gap_surface_down_max": gap_down_max,
                "central_gap_surface_up_ratio": gap_up_max / up_global_max,
                "central_gap_surface_down_ratio": gap_down_max / down_global_max,
            })
            surface_summaries.append(surface_summary)
            sample_surface_records.append(surface_summary)
            np.savez_compressed(
                config.output_dir / "data" / f"{prefix}_surface_{edge_name}.npz",
                **surface_data,
            )
            plot_surface_spectrum(
                surface_data, vbm_rel, cbm_rel,
                f"tts semi-infinite surface — {edge_spec['label']}, C↑={expected_c:+d}",
                config.output_dir / "figures" / f"{prefix}_surface_{edge_name}.png",
                config.dpi,
            )

        print("[5/8] Bulk DOS and intrinsic spin Hall conductivity")
        # Cover the full near-gap ribbon plotting window, with modest extra response margin.
        response_energies = np.linspace(plot_emin, plot_emax, config.response_energy_points)
        dos_sigma = max(
            config.minimum_dos_sigma,
            config.dos_sigma_fraction_of_gap * float(gap["indirect_gap"]),
        )
        response_df, response_summary = calculate_dos_and_hall(
            hoppings, ef, response_energies, config.response_nk, dos_sigma
        )
        atomic_write_csv(response_df, config.output_dir / "data" / f"{prefix}_dos_shc.csv")
        plot_dos_shc(
            response_df, vbm_rel, cbm_rel, expected_c,
            f"tts observable response — {sample_id}, C↑={expected_c:+d}",
            config.output_dir / "figures" / f"{prefix}_dos_shc.png",
            config.dpi,
        )

        metrics = plateau_metrics(response_df, vbm_rel, cbm_rel, expected_c)
        axial_ribbon = next(x for x in sample_ribbon_records if x["edge_name"] == "axial_x")
        diagonal_ribbon = next(x for x in sample_ribbon_records if x["edge_name"] == "diagonal_11")
        axial_surface = next(x for x in sample_surface_records if x["edge_name"] == "axial_x")
        diagonal_surface = next(x for x in sample_surface_records if x["edge_name"] == "diagonal_11")

        is_topological = expected_c != 0
        shc_pass = int(metrics["shc_plateau_max_error"] <= config.shc_plateau_tolerance)
        charge_pass = int(metrics["charge_hall_plateau_max_abs"] <= config.charge_hall_tolerance)
        dos_pass = int(metrics["normalized_gap_dos_max"] <= config.normalized_gap_dos_tolerance)
        ribbon_pass = int(
            (not is_topological)
            or (
                axial_ribbon["edge_states_detected"] == 1
                and diagonal_ribbon["edge_states_detected"] == 1
            )
        )
        # A topological edge branch must traverse the gap, but it need not cross exactly
        # at the chosen midgap energy. Use central-gap spectral weight as the hard test;
        # retain the exact-midgap peak counts as a diagnostic.
        surface_ratio_threshold = 0.02
        surface_pass = int(
            (not is_topological)
            or (
                axial_surface["central_gap_surface_up_ratio"] >= surface_ratio_threshold
                and axial_surface["central_gap_surface_down_ratio"] >= surface_ratio_threshold
                and diagonal_surface["central_gap_surface_up_ratio"] >= surface_ratio_threshold
                and diagonal_surface["central_gap_surface_down_ratio"] >= surface_ratio_threshold
            )
        )

        summary_rows.append(
            {
                "sample_id": sample_id,
                "representative_label": str(row["representative_label"]),
                "expected_chern_up": expected_c,
                "expected_chern_down": -expected_c,
                "indirect_gap": float(gap["indirect_gap"]),
                "min_direct_gap": float(gap["min_direct_gap"]),
                "fermi_midgap": ef,
                "vbm_relative": vbm_rel,
                "cbm_relative": cbm_rel,
                **response_summary,
                **metrics,
                "axial_in_gap_edge_records": axial_ribbon["in_gap_edge_state_records"],
                "diagonal_in_gap_edge_records": diagonal_ribbon["in_gap_edge_state_records"],
                "axial_surface_midgap_peaks_up": axial_surface["midgap_peak_count_up"],
                "axial_surface_midgap_peaks_down": axial_surface["midgap_peak_count_down"],
                "diagonal_surface_midgap_peaks_up": diagonal_surface["midgap_peak_count_up"],
                "diagonal_surface_midgap_peaks_down": diagonal_surface["midgap_peak_count_down"],
                "axial_surface_gap_ratio_up": axial_surface["central_gap_surface_up_ratio"],
                "axial_surface_gap_ratio_down": axial_surface["central_gap_surface_down_ratio"],
                "diagonal_surface_gap_ratio_up": diagonal_surface["central_gap_surface_up_ratio"],
                "diagonal_surface_gap_ratio_down": diagonal_surface["central_gap_surface_down_ratio"],
                "shc_quantization_pass": shc_pass,
                "charge_hall_zero_pass": charge_pass,
                "bulk_dos_gap_pass": dos_pass,
                "ribbon_edge_state_pass": ribbon_pass,
                "surface_spectral_pass": surface_pass,
                "observable_certificate_pass": int(shc_pass and charge_pass and dos_pass and ribbon_pass and surface_pass),
            }
        )

    print("[6/8] Build representative observable certificate")
    summary_df = pd.DataFrame(summary_rows)
    ribbon_summary_df = pd.DataFrame(ribbon_summaries)
    surface_summary_df = pd.DataFrame(surface_summaries)
    gap_df = pd.DataFrame(bulk_gap_rows)
    atomic_write_csv(gap_df, config.output_dir / "step07_02_bulk_gap_audit.csv")
    atomic_write_csv(ribbon_summary_df, config.output_dir / "step07_03_ribbon_edge_state_summary.csv")
    atomic_write_csv(surface_summary_df, config.output_dir / "step07_04_surface_spectral_summary.csv")
    atomic_write_csv(summary_df, config.output_dir / "step07_05_observable_certificate.csv")
    plot_observable_certificate(
        summary_df,
        config.output_dir / "figures" / "step07_observable_certificate.png",
        config.dpi,
    )

    print("[7/8] Final interpretation checks")
    topological = summary_df[summary_df["expected_chern_up"] != 0]
    trivial = summary_df[summary_df["expected_chern_up"] == 0]
    all_topological_pass = int(not topological.empty and bool((topological["observable_certificate_pass"] == 1).all()))
    trivial_response_pass = int(
        not trivial.empty
        and bool((trivial["shc_quantization_pass"] == 1).all())
        and bool((trivial["charge_hall_zero_pass"] == 1).all())
        and bool((trivial["bulk_dos_gap_pass"] == 1).all())
    )
    high_chern_edge_enhancement = np.nan
    if not topological.empty:
        low = topological[np.abs(topological["expected_chern_up"]) == 1]
        high = topological[np.abs(topological["expected_chern_up"]) == 2]
        if not low.empty and not high.empty:
            high_chern_edge_enhancement = float(
                high[["axial_in_gap_edge_records", "diagonal_in_gap_edge_records"]].to_numpy(float).mean()
                / max(low[["axial_in_gap_edge_records", "diagonal_in_gap_edge_records"]].to_numpy(float).mean(), 1.0)
            )

    final_certificate = {
        "code_version": CODE_VERSION,
        "created_utc": utc_now(),
        "n_representatives": int(len(summary_df)),
        "chern_sectors": [int(x) for x in summary_df["expected_chern_up"].tolist()],
        "all_topological_observable_certificates_pass": all_topological_pass,
        "trivial_bulk_response_control_pass": trivial_response_pass,
        "high_chern_to_unit_chern_edge_record_ratio": high_chern_edge_enhancement,
        "spin_hall_unit": "e/(2*pi); plateau equals C_up for spin-conserved model",
        "charge_hall_unit": "e^2/h",
        "plot_style_reference": {
            "source": "uploaded WannierTools src archive",
            "ribbon_palette": "green-yellow-red edge weight",
            "surface_palette": "blue-white-red spin-resolved spectral intensity",
            "visual_style": "thick borders, large labels, high-resolution raster output",
        },
        "final_observable_validation_pass": int(all_topological_pass and trivial_response_pass),
        "elapsed_seconds": float(time.time() - started),
        "config": asdict(config) | {"step3_input": str(config.step3_input), "output_dir": str(config.output_dir)},
    }
    atomic_write_json(final_certificate, config.output_dir / "step07_06_final_observable_certificate.json")

    print("[8/8] Save summary")
    run_summary = {
        "code_version": CODE_VERSION,
        "created_utc": utc_now(),
        "step3_input": str(step3_zip),
        "output_dir": str(config.output_dir),
        "n_representatives": int(len(summary_df)),
        "observable_certificate_pass_count": int(summary_df["observable_certificate_pass"].sum()),
        "final_observable_validation_pass": final_certificate["final_observable_validation_pass"],
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_write_json(run_summary, config.output_dir / "step07_07_run_summary.json")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))

    return {
        "summary": run_summary,
        "representatives": representatives,
        "observable_certificate": summary_df,
        "final_certificate": final_certificate,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS Step 07 final observable validation")
    parser.add_argument("--step3-input", type=Path, default=Step07Config.step3_input)
    parser.add_argument("--output-dir", type=Path, default=Step07Config.output_dir)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-recalculate", action="store_true")
    parser.add_argument("--response-nk", type=int, default=Step07Config.response_nk)
    parser.add_argument("--ribbon-k-points", type=int, default=Step07Config.ribbon_k_points)
    parser.add_argument("--surface-k-points", type=int, default=Step07Config.surface_k_points)
    parser.add_argument("--surface-energy-points", type=int, default=Step07Config.surface_energy_points)
    return parser


def config_from_args(args: argparse.Namespace) -> Step07Config:
    return Step07Config(
        step3_input=args.step3_input,
        output_dir=args.output_dir,
        quick=bool(args.quick),
        force_recalculate=bool(args.force_recalculate),
        response_nk=int(args.response_nk),
        ribbon_k_points=int(args.ribbon_k_points),
        surface_k_points=int(args.surface_k_points),
        surface_energy_points=int(args.surface_energy_points),
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_step07(config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
