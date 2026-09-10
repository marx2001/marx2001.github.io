#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 09: Analytic Chern formula and three-sector phase map for the
spin-conserving MSG 123.342 Lieb Hamiltonian.

This script reads an EXISTING Step 08 output directory directly. It never reads
or extracts ZIP archives.

Main result audited by this script
----------------------------------
For nonzero t1, t2 and nonzero Dirac masses

    M_plus  =  m_e + d1 + d2,
    M_minus = -m_e + d1 + d2,

we test the model-specific analytic formula

    C_up   = -sgn(t1*t2)/2 * [sgn(M_plus) + sgn(M_minus)],
    C_down = -C_up,
    C_total = 0.

The formula follows from the two spin-block d-vectors and the massive-Dirac
contributions at X=(pi,0) and Y=(0,pi). The script validates the formula against
all reliable numerical Chern labels available in Step 08, separately for the
original Sobol set and the independent targeted-boundary set.

Outputs include:
- exact spin-block d-vector formulas (JSON and LaTeX),
- Dirac point masses, Jacobian chiralities and Chern contributions,
- analytic-vs-numerical Chern master table,
- formula metrics and mismatch samples,
- three-sector phase taxonomy,
- figures suitable for mechanism analysis,
- a concise research-significance manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1.0e-12
BOUNDARY_TOL = 1.0e-10
STEP08_PREFIX = "outputs_step08_analytic_boundary_wanniertools_edge_"
STEP09_PREFIX = "outputs_step09_analytic_chern_formula_"

REQUIRED_BASE_COLUMNS = [
    "sample_id", "m_e", "t1", "t2", "s1", "d1", "s2", "d2",
    "M_plus", "M_minus", "chern_up_int", "chern_down_int",
    "chern_total_int", "phase_label", "final_direct_gap",
    "final_indirect_gap", "chern_reliable",
]

REQUIRED_TARGET_COLUMNS = [
    "targeted_sample_id", "sampling_family", "m_e", "t1", "t2",
    "s1", "d1", "s2", "d2", "M_plus", "M_minus",
    "chern_up_int", "chern_down_int", "chern_total_int", "phase_label",
    "fine_direct_gap", "fine_indirect_gap", "gap_high_confidence",
    "chern_reliable",
]


@dataclass(frozen=True)
class Paths:
    step08_dir: Path
    output_dir: Path
    figures_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an existing Step 08 directory and derive/audit the exact "
            "spin-resolved Chern formula for the MSG 123.342 Lieb model."
        )
    )
    parser.add_argument(
        "--step08-dir",
        type=Path,
        default=None,
        help=(
            "Existing Step 08 output directory. If omitted, the newest directory "
            f"matching '{STEP08_PREFIX}*' is auto-detected beside this script or "
            "in the current working directory. ZIP files are rejected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional Step 09 output directory.",
    )
    parser.add_argument(
        "--boundary-tol",
        type=float,
        default=BOUNDARY_TOL,
        help="Absolute tolerance used to mark formula-boundary samples.",
    )
    parser.add_argument(
        "--validation-kpoints",
        type=int,
        default=64,
        help="Random k points per selected sample for d-vector reconstruction audit.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=40,
        help="Number of samples used in the Hamiltonian reconstruction audit.",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def reject_zip(path: Path) -> None:
    if path.suffix.lower() == ".zip" or path.is_file():
        raise ValueError(
            f"Step 09 requires an existing Step 08 DIRECTORY, not a file/ZIP: {path}"
        )


def candidate_roots() -> list[Path]:
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    unique: list[Path] = []
    for root in roots:
        root = root.resolve()
        if root not in unique:
            unique.append(root)
    return unique


def discover_step08_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        reject_zip(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Step 08 directory does not exist: {path}")
        return path

    candidates: list[Path] = []
    for root in candidate_roots():
        candidates.extend(p for p in root.glob(f"{STEP08_PREFIX}*") if p.is_dir())
    if not candidates:
        searched = ", ".join(str(p) for p in candidate_roots())
        raise FileNotFoundError(
            f"No Step 08 directory matching '{STEP08_PREFIX}*' was found in: {searched}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def infer_run_tag(step08_dir: Path) -> str:
    name = step08_dir.name
    if name.startswith(STEP08_PREFIX):
        return name[len(STEP08_PREFIX):]
    return name


def make_paths(args: argparse.Namespace) -> Paths:
    step08_dir = discover_step08_dir(args.step08_dir)
    run_tag = infer_run_tag(step08_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else step08_dir.parent / f"{STEP09_PREFIX}{run_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return Paths(step08_dir, output_dir, figures_dir)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 320), (".pdf", None), (".svg", None)):
        kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)


def find_required_file(directory: Path, exact_name: str) -> Path:
    path = directory / exact_name
    if path.exists():
        return path
    matches = list(directory.glob(exact_name.replace(".csv", "*.csv")))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Required Step 08 file not found: {path}")


def require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def load_step08_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_path = find_required_file(paths.step08_dir, "step08_01_analytic_feature_master.csv")
    target_path = find_required_file(
        paths.step08_dir, "step08_06_targeted_boundary_validation_samples.csv"
    )
    summary_path = paths.step08_dir / "step08_00_run_summary.json"

    base = pd.read_csv(base_path)
    target = pd.read_csv(target_path)
    require_columns(base, REQUIRED_BASE_COLUMNS, "Step 08 analytic master")
    require_columns(target, REQUIRED_TARGET_COLUMNS, "Step 08 targeted samples")

    base = base[
        (base["chern_reliable"] == 1)
        & base["phase_label"].isin(["typeII_QSH", "trivial_insulator"])
    ].copy()
    base["source_dataset"] = "sobol_global"
    base["source_sample_id"] = base["sample_id"].astype(str)
    base["direct_gap"] = pd.to_numeric(base["final_direct_gap"], errors="coerce")
    base["indirect_gap"] = pd.to_numeric(base["final_indirect_gap"], errors="coerce")

    target = target[
        (target["gap_high_confidence"] == 1)
        & (target["chern_reliable"] == 1)
        & target["phase_label"].isin(["typeII_QSH", "trivial_insulator"])
    ].copy()
    target["source_dataset"] = "targeted_boundary"
    target["source_sample_id"] = target["targeted_sample_id"].astype(str)
    target["direct_gap"] = pd.to_numeric(target["fine_direct_gap"], errors="coerce")
    target["indirect_gap"] = pd.to_numeric(target["fine_indirect_gap"], errors="coerce")

    common = [
        "source_dataset", "source_sample_id", "m_e", "t1", "t2", "s1", "d1",
        "s2", "d2", "M_plus", "M_minus", "chern_up_int", "chern_down_int",
        "chern_total_int", "phase_label", "direct_gap", "indirect_gap",
    ]
    if "sampling_family" not in base.columns:
        base["sampling_family"] = "global_sobol"
    common.append("sampling_family")

    combined = pd.concat([base[common], target[common]], ignore_index=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return base, target, {"combined": combined, "step08_summary": summary}


# -----------------------------------------------------------------------------
# Hamiltonian and d-vector formulas in the atomic gauge
# -----------------------------------------------------------------------------


def raw_from_phys(row: pd.Series | dict) -> dict[str, float]:
    me = float(row["m_e"])
    s1, d1, s2, d2 = [float(row[k]) for k in ("s1", "d1", "s2", "d2")]
    return {
        "e1": me,
        "e2": -me,
        "t1": float(row["t1"]),
        "t2": float(row["t2"]),
        "r1": 0.5 * (s1 + d1),
        "r3": 0.5 * (s1 - d1),
        "r2": 0.5 * (s2 + d2),
        "r4": 0.5 * (s2 - d2),
    }


def h_atomic(kx: float, ky: float, row: pd.Series | dict) -> np.ndarray:
    p = raw_from_phys(row)
    e1, e2 = p["e1"], p["e2"]
    t1, t2 = p["t1"], p["t2"]
    r1, r2, r3, r4 = p["r1"], p["r2"], p["r3"], p["r4"]
    a = e1 + 2 * r1 * np.cos(ky) + 2 * r3 * np.cos(kx)
    b = e2 + 2 * r2 * np.cos(ky) + 2 * r4 * np.cos(kx)
    c = e2 + 2 * r2 * np.cos(kx) + 2 * r4 * np.cos(ky)
    d = e1 + 2 * r1 * np.cos(kx) + 2 * r3 * np.cos(ky)
    cp = np.cos((kx + ky) / 2)
    cm = np.cos((kx - ky) / 2)
    u = 2 * (-1j * t1 + t2) * cp + 2 * (1j * t1 + t2) * cm
    v = 2 * (-1j * t1 + t2) * cm + 2 * (1j * t1 + t2) * cp
    return np.array(
        [[a, 0, u, 0], [0, b, 0, v], [np.conj(u), 0, c, 0], [0, np.conj(v), 0, d]],
        dtype=complex,
    )


def dvector(kx: float, ky: float, row: pd.Series | dict, spin: str) -> tuple[float, float, float, float]:
    me, t1, t2, s1, d1, s2, d2 = [
        float(row[k]) for k in ("m_e", "t1", "t2", "s1", "d1", "s2", "d2")
    ]
    sx, sy = np.sin(kx / 2), np.sin(ky / 2)
    cx, cy = np.cos(kx / 2), np.cos(ky / 2)
    dx = 4 * t2 * cx * cy
    S = s1 - s2
    D = d1 + d2

    if spin == "up":
        d0 = 0.5 * (
            (s1 - d1 + s2 + d2) * np.cos(kx)
            + (s1 + d1 + s2 - d2) * np.cos(ky)
        )
        dy = -4 * t1 * sx * sy
        dz = me + 0.5 * (S - D) * np.cos(kx) + 0.5 * (S + D) * np.cos(ky)
    elif spin == "down":
        d0 = 0.5 * (
            (s2 - d2 + s1 + d1) * np.cos(kx)
            + (s2 + d2 + s1 - d1) * np.cos(ky)
        )
        dy = 4 * t1 * sx * sy
        dz = -me - 0.5 * (S + D) * np.cos(kx) + 0.5 * (-S + D) * np.cos(ky)
    else:
        raise ValueError("spin must be 'up' or 'down'")
    return float(d0), float(dx), float(dy), float(dz)


def block_from_dvector(kx: float, ky: float, row: pd.Series | dict, spin: str) -> np.ndarray:
    d0, dx, dy, dz = dvector(kx, ky, row, spin)
    return np.array(
        [[d0 + dz, dx - 1j * dy], [dx + 1j * dy, d0 - dz]], dtype=complex
    )


def reconstruction_audit(
    df: pd.DataFrame, n_samples: int, n_kpoints: int, seed: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    if len(df) == 0:
        raise ValueError("No samples available for reconstruction audit")
    chosen = df.sample(min(n_samples, len(df)), random_state=seed)
    errors: list[float] = []
    for _, row in chosen.iterrows():
        for _ in range(n_kpoints):
            kx, ky = rng.uniform(-np.pi, np.pi, size=2)
            full = h_atomic(float(kx), float(ky), row)
            for spin, indices in (("up", [0, 2]), ("down", [1, 3])):
                exact = full[np.ix_(indices, indices)]
                reconstructed = block_from_dvector(float(kx), float(ky), row, spin)
                errors.append(float(np.max(np.abs(exact - reconstructed))))
    return {
        "n_samples": int(len(chosen)),
        "n_kpoints_per_sample": int(n_kpoints),
        "n_spin_block_comparisons": int(len(errors)),
        "max_abs_error": float(np.max(errors)),
        "mean_abs_error": float(np.mean(errors)),
    }


# -----------------------------------------------------------------------------
# Analytic Chern formula
# -----------------------------------------------------------------------------


def sign_with_boundary(values: Iterable[float], tol: float) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    out = np.zeros(arr.shape, dtype=int)
    out[arr > tol] = 1
    out[arr < -tol] = -1
    return out


def sector_from_chern(cup: int, cdn: int, ctot: int) -> str:
    if (cup, cdn, ctot) == (1, -1, 0):
        return "typeII_QSH_Cup_plus"
    if (cup, cdn, ctot) == (-1, 1, 0):
        return "typeII_QSH_Cup_minus"
    if (cup, cdn, ctot) == (0, 0, 0):
        return "trivial_spin_chern"
    return "other_or_unresolved"


def add_analytic_chern(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    out = df.copy()
    # Recompute masses independently, rather than trusting stored derived columns.
    out["M_plus_recomputed"] = out["m_e"] + out["d1"] + out["d2"]
    out["M_minus_recomputed"] = -out["m_e"] + out["d1"] + out["d2"]
    out["mass_recompute_max_error"] = np.maximum(
        np.abs(out["M_plus"] - out["M_plus_recomputed"]),
        np.abs(out["M_minus"] - out["M_minus_recomputed"]),
    )

    sign_mp = sign_with_boundary(out["M_plus_recomputed"], tol)
    sign_mm = sign_with_boundary(out["M_minus_recomputed"], tol)
    sign_t1 = sign_with_boundary(out["t1"], tol)
    sign_t2 = sign_with_boundary(out["t2"], tol)
    sign_tp = sign_t1 * sign_t2

    out["sign_M_plus"] = sign_mp
    out["sign_M_minus"] = sign_mm
    out["sign_t1"] = sign_t1
    out["sign_t2"] = sign_t2
    out["sign_t1t2"] = sign_tp
    out["formula_boundary"] = (
        (sign_mp == 0) | (sign_mm == 0) | (sign_t1 == 0) | (sign_t2 == 0)
    ).astype(int)

    cup = -0.5 * sign_tp * (sign_mp + sign_mm)
    cup = np.rint(cup).astype(int)
    cdn = -cup
    ctot = cup + cdn
    cup[out["formula_boundary"].to_numpy(dtype=bool)] = 99
    cdn[out["formula_boundary"].to_numpy(dtype=bool)] = 99
    ctot[out["formula_boundary"].to_numpy(dtype=bool)] = 99

    out["chern_up_analytic"] = cup
    out["chern_down_analytic"] = cdn
    out["chern_total_analytic"] = ctot
    out["numeric_sector"] = [
        sector_from_chern(int(u), int(d), int(t))
        for u, d, t in zip(out["chern_up_int"], out["chern_down_int"], out["chern_total_int"])
    ]
    out["analytic_sector"] = [
        sector_from_chern(int(u), int(d), int(t)) if int(u) != 99 else "formula_boundary"
        for u, d, t in zip(cup, cdn, ctot)
    ]
    valid = out["formula_boundary"] == 0
    out["chern_formula_match"] = 0
    out.loc[valid, "chern_formula_match"] = (
        (out.loc[valid, "chern_up_int"].astype(int) == out.loc[valid, "chern_up_analytic"])
        & (out.loc[valid, "chern_down_int"].astype(int) == out.loc[valid, "chern_down_analytic"])
        & (out.loc[valid, "chern_total_int"].astype(int) == out.loc[valid, "chern_total_analytic"])
    ).astype(int)

    out["topology_formula_prediction"] = (
        out["M_plus_recomputed"] * out["M_minus_recomputed"] > 0
    ).astype(int)
    out["numeric_typeII"] = (out["phase_label"] == "typeII_QSH").astype(int)
    out["topology_formula_match"] = (
        out["topology_formula_prediction"] == out["numeric_typeII"]
    ).astype(int)
    return out


def classification_metrics(df: pd.DataFrame) -> dict:
    valid = df[df["formula_boundary"] == 0].copy()
    matches = valid["chern_formula_match"].astype(int)
    datasets = {}
    for name, part in valid.groupby("source_dataset"):
        datasets[str(name)] = {
            "n": int(len(part)),
            "exact_chern_matches": int(part["chern_formula_match"].sum()),
            "exact_chern_accuracy": float(part["chern_formula_match"].mean()),
            "topology_accuracy": float(part["topology_formula_match"].mean()),
            "sector_counts_numeric": {
                str(k): int(v) for k, v in part["numeric_sector"].value_counts().to_dict().items()
            },
        }
    return {
        "formula": (
            "C_up = -sgn(t1*t2)/2 * [sgn(M_plus)+sgn(M_minus)]; "
            "C_down=-C_up; C_total=0"
        ),
        "domain": "gapped spin-conserving model with t1*t2*M_plus*M_minus != 0",
        "n_total_labeled": int(len(df)),
        "n_formula_boundary": int(df["formula_boundary"].sum()),
        "n_formula_valid": int(len(valid)),
        "n_exact_matches": int(matches.sum()),
        "exact_chern_accuracy": float(matches.mean()) if len(matches) else float("nan"),
        "n_mismatches": int((1 - matches).sum()),
        "topology_accuracy": float(valid["topology_formula_match"].mean()),
        "max_mass_recompute_error": float(df["mass_recompute_max_error"].max()),
        "by_dataset": datasets,
    }


def dirac_catalog() -> pd.DataFrame:
    rows = [
        {
            "spin": "up", "point": "X", "kx": "pi", "ky": "0",
            "mass": "M_plus", "jacobian_chirality": "+sgn(t1*t2)",
            "lower_band_contribution": "-1/2 sgn(t1*t2) sgn(M_plus)",
        },
        {
            "spin": "up", "point": "Y", "kx": "0", "ky": "pi",
            "mass": "-M_minus", "jacobian_chirality": "-sgn(t1*t2)",
            "lower_band_contribution": "-1/2 sgn(t1*t2) sgn(M_minus)",
        },
        {
            "spin": "down", "point": "X", "kx": "pi", "ky": "0",
            "mass": "M_minus", "jacobian_chirality": "-sgn(t1*t2)",
            "lower_band_contribution": "+1/2 sgn(t1*t2) sgn(M_minus)",
        },
        {
            "spin": "down", "point": "Y", "kx": "0", "ky": "pi",
            "mass": "-M_plus", "jacobian_chirality": "+sgn(t1*t2)",
            "lower_band_contribution": "+1/2 sgn(t1*t2) sgn(M_plus)",
        },
    ]
    return pd.DataFrame(rows)


def formula_payload(reconstruction: dict) -> dict:
    return {
        "basis": ["|1,up>", "|1,down>", "|2,up>", "|2,down>"],
        "spin_block_indices": {"up": [0, 2], "down": [1, 3]},
        "definitions": {
            "S": "s1-s2",
            "D": "d1+d2",
            "M_plus": "m_e+d1+d2",
            "M_minus": "-m_e+d1+d2",
        },
        "up_block": {
            "d_x": "4*t2*cos(kx/2)*cos(ky/2)",
            "d_y": "-4*t1*sin(kx/2)*sin(ky/2)",
            "d_z": "m_e + (S-D)/2*cos(kx) + (S+D)/2*cos(ky)",
            "mass_X": "M_plus",
            "mass_Y": "-M_minus",
        },
        "down_block": {
            "d_x": "4*t2*cos(kx/2)*cos(ky/2)",
            "d_y": "+4*t1*sin(kx/2)*sin(ky/2)",
            "d_z": "-m_e -(S+D)/2*cos(kx) + (-S+D)/2*cos(ky)",
            "mass_X": "M_minus",
            "mass_Y": "-M_plus",
        },
        "simultaneous_zero_statement": (
            "For t1*t2 != 0, d_x=d_y=0 only at X and Y (up to periodic equivalents)."
        ),
        "analytic_chern_formula": {
            "C_up": "-sgn(t1*t2)/2 * [sgn(M_plus)+sgn(M_minus)]",
            "C_down": "+sgn(t1*t2)/2 * [sgn(M_plus)+sgn(M_minus)]",
            "C_total": "0",
            "typeII_condition": "M_plus*M_minus > 0",
            "trivial_condition": "M_plus*M_minus < 0",
            "chirality_control": "sgn(C_up)=-sgn(t1*t2)*sgn(M_plus) in the type-II sectors",
        },
        "restricted_model_reduction": {
            "restriction": "d1=d2=v-u and m_e=m",
            "mass_product": "M_plus*M_minus=4*(v-u)^2-m^2",
            "condition": "2*abs(u-v)>abs(m)",
        },
        "hamiltonian_reconstruction_audit": reconstruction,
    }


def latex_derivation() -> str:
    return r"""% Auto-generated by Step 09
\section*{Analytic spin-Chern formula for the MSG 123.342 Lieb model}
Define
\begin{equation}
S=s_1-s_2,\qquad D=d_1+d_2,
\end{equation}
and
\begin{equation}
M_+=m_e+D,\qquad M_-=-m_e+D.
\end{equation}
In the basis $(|1,\uparrow\rangle,|2,\uparrow\rangle)$, the spin-up block is
\begin{equation}
h_\uparrow(\mathbf{k})=d_{0,\uparrow}(\mathbf{k})\sigma_0+
\mathbf d_\uparrow(\mathbf{k})\cdot\boldsymbol\sigma,
\end{equation}
with
\begin{align}
d_{x,\uparrow}&=4t_2\cos\frac{k_x}{2}\cos\frac{k_y}{2},\\
d_{y,\uparrow}&=-4t_1\sin\frac{k_x}{2}\sin\frac{k_y}{2},\\
d_{z,\uparrow}&=m_e+\frac{S-D}{2}\cos k_x+\frac{S+D}{2}\cos k_y.
\end{align}
For $t_1t_2\neq0$, the simultaneous zeros of $d_x$ and $d_y$ are only
$X=(\pi,0)$ and $Y=(0,\pi)$, modulo reciprocal-lattice equivalence. Their
masses are
\begin{equation}
m_{\uparrow,X}=M_+,\qquad m_{\uparrow,Y}=-M_-.
\end{equation}
The corresponding Jacobian chiralities are
\begin{equation}
\chi_{\uparrow,X}=\operatorname{sgn}(t_1t_2),\qquad
\chi_{\uparrow,Y}=-\operatorname{sgn}(t_1t_2).
\end{equation}
Using the lower-band massive-Dirac contribution
$C_i=-\chi_i\operatorname{sgn}(m_i)/2$, one obtains
\begin{equation}
\boxed{C_\uparrow=-\frac{\operatorname{sgn}(t_1t_2)}{2}
\left[\operatorname{sgn}(M_+)+\operatorname{sgn}(M_-)\right].}
\end{equation}
The spin-down block gives
\begin{equation}
\boxed{C_\downarrow=-C_\uparrow,\qquad C_{\mathrm{total}}=0.}
\end{equation}
Therefore,
\begin{equation}
M_+M_->0\Longleftrightarrow |C_\uparrow|=|C_\downarrow|=1,
\end{equation}
whereas
\begin{equation}
M_+M_-<0\Longleftrightarrow C_\uparrow=C_\downarrow=0.
\end{equation}
Within a type-II QSH sector,
\begin{equation}
\operatorname{sgn}(C_\uparrow)=-\operatorname{sgn}(t_1t_2)
\operatorname{sgn}(M_+).
\end{equation}
For the restricted parametrization $d_1=d_2=v-u$ and $m_e=m$,
\begin{equation}
M_+M_-=4(v-u)^2-m^2,
\end{equation}
so the nontrivial condition reduces to
\begin{equation}
2|u-v|>|m|.
\end{equation}
"""


def make_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    # Three-sector M_- / M_+ map.
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sectors = ["trivial_spin_chern", "typeII_QSH_Cup_plus", "typeII_QSH_Cup_minus"]
    markers = ["o", "^", "s"]
    for sector, marker in zip(sectors, markers):
        part = df[df["numeric_sector"] == sector]
        ax.scatter(
            part["M_minus_recomputed"], part["M_plus_recomputed"],
            s=18, alpha=0.65, marker=marker, label=f"{sector} (n={len(part)})"
        )
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(r"$M_-$")
    ax.set_ylabel(r"$M_+$")
    ax.set_title("Three spin-Chern sectors in the analytic mass plane")
    ax.legend(fontsize=8)
    save_figure(fig, figures_dir / "step09_01_three_sector_Mplus_Mminus_map")

    # Numeric vs analytic C_up.
    valid = df[df["formula_boundary"] == 0]
    matrix = pd.crosstab(valid["chern_up_int"].astype(int), valid["chern_up_analytic"].astype(int))
    labels = [-1, 0, 1]
    values = matrix.reindex(index=labels, columns=labels, fill_value=0).to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(values, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel(r"Analytic $C_\uparrow$")
    ax.set_ylabel(r"Numerical $C_\uparrow$")
    ax.set_title("Analytic versus numerical spin Chern number")
    fig.colorbar(image, ax=ax, label="Count")
    save_figure(fig, figures_dir / "step09_02_analytic_numeric_Cup_confusion")

    # Chirality mechanism table as scatter.
    topo = valid[valid["chern_up_int"].astype(int) != 0].copy()
    topo["mass_common_sign"] = np.sign(topo["M_plus_recomputed"]).astype(int)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    jitter = np.linspace(-0.08, 0.08, max(len(topo), 2))[: len(topo)]
    x = topo["sign_t1t2"].to_numpy(dtype=float) + jitter
    y = topo["mass_common_sign"].to_numpy(dtype=float)
    scatter = ax.scatter(x, y, c=topo["chern_up_int"], s=18, alpha=0.7)
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_xlabel(r"$\operatorname{sgn}(t_1t_2)$")
    ax.set_ylabel(r"Common sign of $M_+$ and $M_-$")
    ax.set_title(r"Spin-Chern chirality: $\mathrm{sgn}(C_\uparrow)=-\mathrm{sgn}(t_1t_2)\mathrm{sgn}(M_+)$")
    fig.colorbar(scatter, ax=ax, label=r"Numerical $C_\uparrow$")
    save_figure(fig, figures_dir / "step09_03_chirality_control")

    # Gap versus distance to analytic phase boundaries.
    distance = np.minimum(np.abs(valid["M_plus_recomputed"]), np.abs(valid["M_minus_recomputed"]))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    scatter = ax.scatter(distance, valid["direct_gap"], c=valid["chern_up_int"], s=18, alpha=0.65)
    ax.set_xlabel(r"$\min(|M_+|,|M_-|)$")
    ax.set_ylabel("Audited direct gap")
    ax.set_title("Gap robustness relative to the analytic mass boundaries")
    fig.colorbar(scatter, ax=ax, label=r"$C_\uparrow$")
    save_figure(fig, figures_dir / "step09_04_gap_vs_mass_boundary_distance")


def research_significance_manifest(metrics: dict) -> dict:
    return {
        "validated_increment": [
            "Generalizes the known restricted type-II-QSH inversion condition to the full seven-independent-parameter form of the onsite+NN+NNN MSG 123.342 Lieb Hamiltonian.",
            "Separates topology existence (the signs of M_plus and M_minus) from spin-Chern chirality (the sign of t1*t2).",
            "Provides a closed three-sector classification: trivial, C_up=+1, and C_up=-1.",
            "Closes the ML-to-physics loop: strict full-BZ/Chern labels -> interpretable feature discovery -> analytic Dirac proof -> edge-state verification.",
        ],
        "practical_meaning": [
            "Within the model domain, the topological sector can be screened algebraically without a new BZ Berry-curvature integration for every parameter point.",
            "The phase-boundary equations identify which microscopic hopping or onsite combinations must be tuned to switch topology or reverse edge-state chirality.",
            "The formula gives a compact bridge from fitted MagneticTB/Wannier parameters to a topological design rule.",
        ],
        "scope_limits": [
            "The result is for the spin-conserving four-band model at half filling with onsite, nearest-neighbor, and next-nearest-neighbor terms.",
            "It is not a classifier for all topological insulators, non-spin-conserving SOC, or higher-order topology.",
            "Additional symmetry-allowed spin-mixing or longer-range terms require a new invariant analysis.",
        ],
        "current_validation": metrics,
    }


def write_readme(output_dir: Path, metrics: dict) -> None:
    text = f"""# Step 09 — Analytic Chern Formula and Three-Sector Map

## Input mode

This program reads an **existing Step 08 output directory directly**. It does not
read or extract ZIP archives.

## Main formula

For a gapped parameter point with

```text
t1*t2*M_plus*M_minus != 0
```

and

```text
M_plus  =  m_e + d1 + d2
M_minus = -m_e + d1 + d2
```

the audited analytic formula is

```text
C_up   = -sign(t1*t2)/2 * [sign(M_plus) + sign(M_minus)]
C_down = -C_up
C_total = 0
```

Hence:

```text
M_plus*M_minus > 0  -> type-II QSH
M_plus*M_minus < 0  -> trivial spin-Chern insulator
```

Inside the type-II phase:

```text
sign(C_up) = -sign(t1*t2)*sign(M_plus)
```

## Validation result

- Total reliable Step 08 labels: **{metrics['n_total_labeled']}**
- Formula-valid nonboundary samples: **{metrics['n_formula_valid']}**
- Exact spin-resolved Chern matches: **{metrics['n_exact_matches']}**
- Mismatches: **{metrics['n_mismatches']}**
- Exact accuracy: **{metrics['exact_chern_accuracy']:.6f}**

## Run

Place this script beside the existing Step 08 output directory and run:

```bash
python 09_Lieb8_Analytic_Chern_Formula_and_Three_Sector_Map.py
```

Or specify the directory explicitly:

```bash
python 09_Lieb8_Analytic_Chern_Formula_and_Three_Sector_Map.py \\
  --step08-dir outputs_step08_analytic_boundary_wanniertools_edge_<run_tag>
```

## Important outputs

- `step09_01_spin_block_dvector_formulas.json`
- `step09_01_spin_block_dvector_formulas.tex`
- `step09_02_dirac_point_mass_chirality_catalog.csv`
- `step09_03_analytic_chern_prediction_master.csv`
- `step09_03_analytic_formula_metrics.json`
- `step09_04_formula_mismatch_samples.csv`
- `step09_05_three_sector_counts.csv`
- `step09_06_research_significance_manifest.json`
- `figures/`
"""
    (output_dir / "README_Step09.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = make_paths(args)
    print(f"[Step09] Step 08 directory: {paths.step08_dir}")
    print(f"[Step09] Output directory : {paths.output_dir}")

    base, target, loaded = load_step08_data(paths)
    combined = loaded["combined"]
    print(f"[Step09] Global reliable samples   : {len(base)}")
    print(f"[Step09] Targeted reliable samples : {len(target)}")
    print(f"[Step09] Combined labels           : {len(combined)}")

    reconstruction = reconstruction_audit(
        combined,
        n_samples=args.validation_samples,
        n_kpoints=args.validation_kpoints,
        seed=args.seed,
    )
    write_json(paths.output_dir / "step09_01_hamiltonian_reconstruction_audit.json", reconstruction)

    payload = formula_payload(reconstruction)
    write_json(paths.output_dir / "step09_01_spin_block_dvector_formulas.json", payload)
    (paths.output_dir / "step09_01_spin_block_dvector_formulas.tex").write_text(
        latex_derivation(), encoding="utf-8"
    )

    catalog = dirac_catalog()
    catalog.to_csv(paths.output_dir / "step09_02_dirac_point_mass_chirality_catalog.csv", index=False)

    master = add_analytic_chern(combined, tol=float(args.boundary_tol))
    master.to_csv(paths.output_dir / "step09_03_analytic_chern_prediction_master.csv", index=False)
    metrics = classification_metrics(master)
    write_json(paths.output_dir / "step09_03_analytic_formula_metrics.json", metrics)

    mismatches = master[
        (master["formula_boundary"] == 0) & (master["chern_formula_match"] == 0)
    ].copy()
    mismatches.to_csv(paths.output_dir / "step09_04_formula_mismatch_samples.csv", index=False)

    counts = (
        master.groupby(["source_dataset", "numeric_sector"], dropna=False)
        .size().rename("count").reset_index()
    )
    counts.to_csv(paths.output_dir / "step09_05_three_sector_counts.csv", index=False)

    sign_table = (
        master[master["formula_boundary"] == 0]
        .groupby(["sign_M_plus", "sign_M_minus", "sign_t1t2", "chern_up_int"])
        .size().rename("count").reset_index()
    )
    sign_table.to_csv(paths.output_dir / "step09_05_sign_sector_truth_table.csv", index=False)

    make_figures(master, paths.figures_dir)

    significance = research_significance_manifest(metrics)
    write_json(paths.output_dir / "step09_06_research_significance_manifest.json", significance)

    run_summary = {
        "step08_dir": str(paths.step08_dir),
        "input_mode": "existing_step08_directory_only",
        "zip_reading_enabled": False,
        "output_dir": str(paths.output_dir),
        "n_global": int(len(base)),
        "n_targeted": int(len(target)),
        "n_combined": int(len(master)),
        "formula_metrics": metrics,
        "hamiltonian_reconstruction": reconstruction,
        "next_recommended_step": (
            "Compare the full seven-parameter phase formula against the restricted "
            "literature subspace and then test controlled symmetry-allowed spin-mixing "
            "terms as a separate Step 10."
        ),
    }
    write_json(paths.output_dir / "step09_00_run_summary.json", run_summary)
    write_readme(paths.output_dir, metrics)

    registry_rows = []
    for path in sorted(paths.output_dir.rglob("*")):
        if path.is_file():
            registry_rows.append(
                {
                    "relative_path": str(path.relative_to(paths.output_dir)),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    pd.DataFrame(registry_rows).to_csv(
        paths.output_dir / "step09_00_output_file_registry.csv", index=False
    )

    print("[Step09] Formula:", metrics["formula"])
    print(
        f"[Step09] Exact matches: {metrics['n_exact_matches']}/"
        f"{metrics['n_formula_valid']}  mismatches={metrics['n_mismatches']}"
    )
    print(f"[Step09] Reconstruction max error: {reconstruction['max_abs_error']:.3e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[Step09][ERROR] {exc}", file=sys.stderr)
        raise
