from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CODE_VERSION = "TTS_STEP16M_MASS_COORDINATE_PHASE_MAPS_V1_20260726"

# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman:bold"
plt.rcParams["mathtext.it"] = "Times New Roman:italic:bold"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["mathtext.sf"] = "Times New Roman:bold"
plt.rcParams["mathtext.tt"] = "Times New Roman:bold"
plt.rcParams["mathtext.fallback"] = None
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PHASE_COLORS = {
    -2: "#4F81BD",
     0: "#D9D9D9",
     2: "#CF5A47",
}


@dataclass
class Step16Config:
    output_dir: Path = Path("outputs_tts_step16M_mass_coordinate_phase_maps")

    # Local windows are used only to select the already-certified data.
    left_r3_min: float = -0.665
    left_r3_max: float = -0.525
    left_r4_min: float = -0.180
    left_r4_max: float = -0.135

    right_r3_min: float = 0.054
    right_r3_max: float = 0.066
    right_r4_min: float = 0.128
    right_r4_max: float = 0.134

    # Certified left local-mass formula:
    # M_L = r4 + a_L r3^2 + b_L r3 + c_L
    left_a: float = 0.4489248315
    left_b: float = 0.7973688821
    left_c: float = 0.4712000815

    # Certified right four-valley mass:
    # M_4v = r3 - alpha_4v r4
    alpha_4v: float = 0.4597785904

    # Certified closure anchors. If Step12M is provided, these are replaced by
    # the values read from step12M_06_critical_closure_parameters.csv.
    closure_A_r3: float = 0.0600000000
    closure_A_r4: float = 0.1304975940
    closure_B_r3: float = 0.0600000000
    closure_B_r4: float = 0.1316899010

    # Plot limits in normalized mass coordinates. None means determine from data.
    left_mass_padding: float = 1.15
    right_mass_padding: float = 1.15
    figure_size_single: tuple[float, float] = (7.6, 7.6)
    figure_size_combined: tuple[float, float] = (14.6, 6.8)
    point_size: float = 58.0


# -----------------------------------------------------------------------------
# ZIP-stream readers: no extraction, avoids Windows WinError 206.
# -----------------------------------------------------------------------------
def _find_member_by_basename(zf: zipfile.ZipFile, filename: str) -> str:
    matches = [name for name in zf.namelist() if Path(name).name == filename]
    if not matches:
        raise FileNotFoundError(f"{filename} was not found inside the ZIP archive")
    matches.sort(key=lambda x: (len(Path(x).parts), len(x), x))
    return matches[0]


def read_csv_any(source: str | Path, filename: str) -> pd.DataFrame:
    source = Path(source)
    if source.is_dir():
        matches = list(source.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"{filename} was not found under {source}")
        matches.sort(key=lambda p: (len(p.parts), str(p)))
        return pd.read_csv(matches[0], low_memory=False)
    if source.is_file() and source.suffix.lower() == ".csv":
        return pd.read_csv(source, low_memory=False)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            member = _find_member_by_basename(zf, filename)
            return pd.read_csv(io.BytesIO(zf.read(member)), low_memory=False)
    raise FileNotFoundError(f"Unsupported or missing source: {source}")


def optional_csv(source: str | Path | None, filename: str) -> Optional[pd.DataFrame]:
    if source is None:
        return None
    path = Path(source)
    if not path.exists():
        return None
    try:
        return read_csv_any(path, filename)
    except FileNotFoundError:
        return None


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def in_window(df: pd.DataFrame, x0: float, x1: float, y0: float, y1: float) -> pd.Series:
    return (
        df["r3"].between(x0, x1, inclusive="both")
        & df["r4"].between(y0, y1, inclusive="both")
    )


def robust_scale(values: np.ndarray, quantile: float = 0.95) -> float:
    values = np.asarray(values, dtype=float)
    scale = float(np.quantile(np.abs(values[np.isfinite(values)]), quantile))
    if not np.isfinite(scale) or scale <= 1e-14:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-14:
        scale = 1.0
    return scale


def style_axis(ax, title: str):
    ax.set_box_aspect(1)
    ax.tick_params(direction="in", length=7, width=1.7, top=True, right=True, labelsize=15)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
    ax.set_title(title, fontsize=21, fontweight="bold", pad=12)


def phase_handles(include_zero: bool = True):
    handles = [
        Patch(facecolor=PHASE_COLORS[-2], edgecolor="black", linewidth=0.8,
              label=r"$\mathbf{\mathit{C}}_{\uparrow}=-2$"),
        Patch(facecolor=PHASE_COLORS[2], edgecolor="black", linewidth=0.8,
              label=r"$\mathbf{\mathit{C}}_{\uparrow}=2$"),
    ]
    if include_zero:
        handles.insert(1, Patch(facecolor=PHASE_COLORS[0], edgecolor="black", linewidth=0.8,
                                label=r"$\mathbf{\mathit{C}}_{\uparrow}=0$"))
    return handles


# -----------------------------------------------------------------------------
# Left mass coordinates
# -----------------------------------------------------------------------------
def left_boundary_r4(r3: np.ndarray, cfg: Step16Config) -> np.ndarray:
    return -(cfg.left_a * r3**2 + cfg.left_b * r3 + cfg.left_c)


def left_boundary_slope(r3: np.ndarray, cfg: Step16Config) -> np.ndarray:
    return -(2.0 * cfg.left_a * r3 + cfg.left_b)


def left_arc_coordinate(r3_values: np.ndarray, cfg: Step16Config) -> np.ndarray:
    """Signed arc length along the fitted left boundary, referenced to its center."""
    r3_values = np.asarray(r3_values, dtype=float)
    lo = min(float(np.min(r3_values)), cfg.left_r3_min)
    hi = max(float(np.max(r3_values)), cfg.left_r3_max)
    grid = np.linspace(lo, hi, 10001)
    slope = left_boundary_slope(grid, cfg)
    integrand = np.sqrt(1.0 + slope**2)
    dg = np.diff(grid)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dg)])
    ref_r3 = 0.5 * (cfg.left_r3_min + cfg.left_r3_max)
    ref_s = float(np.interp(ref_r3, grid, cumulative))
    return np.interp(r3_values, grid, cumulative) - ref_s


def transform_left(df: pd.DataFrame, cfg: Step16Config) -> tuple[pd.DataFrame, dict]:
    left = df[
        in_window(df, cfg.left_r3_min, cfg.left_r3_max, cfg.left_r4_min, cfg.left_r4_max)
        & df["strict_phase_code"].isin([-2, 0, 2])
    ].copy()
    if left.empty:
        raise RuntimeError("No certified points were found in the left local window")

    slope = left_boundary_slope(left["r3"].to_numpy(), cfg)
    raw_mass = (
        left["r4"].to_numpy()
        + cfg.left_a * left["r3"].to_numpy() ** 2
        + cfg.left_b * left["r3"].to_numpy()
        + cfg.left_c
    )
    # Geometric signed normal distance to M_L=0.
    normal_mass = raw_mass / np.sqrt(1.0 + slope**2)
    tangent = left_arc_coordinate(left["r3"].to_numpy(), cfg)

    mass_scale = robust_scale(normal_mass)
    tangent_scale = robust_scale(tangent)
    left["M_L_raw"] = raw_mass
    left["M_L_normal"] = normal_mass
    left["S_L_arc"] = tangent
    left["M_L_tilde"] = normal_mass / mass_scale
    left["S_L_tilde"] = tangent / tangent_scale

    nonzero = left[left["strict_phase_code"].isin([-2, 2])]
    sign_pred = np.where(nonzero["M_L_normal"].to_numpy() > 0, 2, -2)
    sign_accuracy = float(np.mean(sign_pred == nonzero["strict_phase_code"].to_numpy()))

    meta = {
        "formula": (
            f"M_L = r4 + ({cfg.left_a:.10g}) r3^2 + "
            f"({cfg.left_b:.10g}) r3 + ({cfg.left_c:.10g})"
        ),
        "normal_mass_definition": "M_L_normal = M_L / sqrt(1 + (dr4_boundary/dr3)^2)",
        "tangent_coordinate_definition": "S_L is signed arc length along M_L=0",
        "mass_scale": mass_scale,
        "tangent_scale": tangent_scale,
        "nonzero_phase_sign_accuracy": sign_accuracy,
        "n_points": int(len(left)),
        "n_minus2": int((left["strict_phase_code"] == -2).sum()),
        "n_zero": int((left["strict_phase_code"] == 0).sum()),
        "n_plus2": int((left["strict_phase_code"] == 2).sum()),
    }
    return left, meta


# -----------------------------------------------------------------------------
# Right double-mass coordinates
# -----------------------------------------------------------------------------
def detect_plus2_zero_transition_midpoints(right: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r3, group in right.groupby("r3"):
        group = group.sort_values("r4")
        arr = group[["r4", "strict_phase_code"]].to_numpy()
        candidates = []
        for (y1, c1), (y2, c2) in zip(arr[:-1], arr[1:]):
            if {int(c1), int(c2)} == {0, 2}:
                candidates.append(0.5 * (float(y1) + float(y2)))
        if candidates:
            # The right-junction data contain one relevant +2/0 transition per r3.
            rows.append({"r3": float(r3), "r4_mid": float(np.median(candidates))})
    return pd.DataFrame(rows)


def fit_anchored_sigma_boundary(
    transition_points: pd.DataFrame,
    anchor_r3: float,
    anchor_r4: float,
) -> dict:
    """Fit an anchored local boundary r4 = anchor_r4 + beta (r3-anchor_r3).

    A linear anchored fit is deliberately used because it is the simplest local
    mass formula and, for the current certified junction data, classifies all
    strict -2/0/+2 points consistently when paired with M_4v.
    """
    if len(transition_points) < 2:
        raise RuntimeError("Too few +2/0 transition midpoints to fit M_Sigma")
    x = transition_points["r3"].to_numpy(dtype=float) - anchor_r3
    y = transition_points["r4_mid"].to_numpy(dtype=float) - anchor_r4
    denom = float(np.dot(x, x))
    if denom <= 1e-20:
        raise RuntimeError("Degenerate transition-point geometry")
    beta = float(np.dot(x, y) / denom)
    predicted = anchor_r4 + beta * (transition_points["r3"].to_numpy() - anchor_r3)
    residual = transition_points["r4_mid"].to_numpy() - predicted
    return {
        "beta": beta,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "n_boundary_midpoints": int(len(transition_points)),
    }


def transform_right(df: pd.DataFrame, cfg: Step16Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    right = df[
        in_window(df, cfg.right_r3_min, cfg.right_r3_max, cfg.right_r4_min, cfg.right_r4_max)
        & df["strict_phase_code"].isin([-2, 0, 2])
    ].copy()
    if right.empty:
        raise RuntimeError("No certified points were found in the right junction window")

    transitions = detect_plus2_zero_transition_midpoints(right)
    sigma_fit = fit_anchored_sigma_boundary(
        transitions,
        cfg.closure_B_r3,
        cfg.closure_B_r4,
    )
    beta = sigma_fit["beta"]

    r3 = right["r3"].to_numpy(dtype=float)
    r4 = right["r4"].to_numpy(dtype=float)

    # Signed normal distance to the certified four-valley boundary.
    m4_raw = r3 - cfg.alpha_4v * r4
    m4_normal = m4_raw / math.sqrt(1.0 + cfg.alpha_4v**2)

    # Signed normal distance to the locally fitted Sigma' two-valley boundary.
    sigma_boundary = cfg.closure_B_r4 + beta * (r3 - cfg.closure_B_r3)
    ms_raw = r4 - sigma_boundary
    ms_normal = ms_raw / math.sqrt(1.0 + beta**2)

    m4_scale = robust_scale(m4_normal)
    ms_scale = robust_scale(ms_normal)

    right["M_4v_raw"] = m4_raw
    right["M_4v_normal"] = m4_normal
    right["M_Sigma_raw"] = ms_raw
    right["M_Sigma_normal"] = ms_normal
    right["M_4v_tilde"] = m4_normal / m4_scale
    right["M_Sigma_tilde"] = ms_normal / ms_scale

    # Mechanism-sector rule inferred from the ordered -2 -> +2 -> 0 path:
    #   M_Sigma > 0                 -> C_up = 0
    #   M_Sigma < 0 and M_4v > 0   -> C_up = -2
    #   M_Sigma < 0 and M_4v < 0   -> C_up = +2
    predicted = np.where(
        right["M_Sigma_normal"].to_numpy() > 0,
        0,
        np.where(right["M_4v_normal"].to_numpy() > 0, -2, 2),
    )
    right["mass_sector_prediction"] = predicted
    right["mass_sector_match"] = predicted == right["strict_phase_code"].to_numpy()

    labels = [-2, 0, 2]
    recall = {}
    for label in labels:
        mask = right["strict_phase_code"].to_numpy() == label
        recall[str(label)] = float(np.mean(predicted[mask] == label)) if np.any(mask) else None
    balanced_accuracy = float(np.mean([v for v in recall.values() if v is not None]))
    accuracy = float(np.mean(right["mass_sector_match"]))

    meta = {
        "M_4v_formula": (
            f"M_4v = r3 - ({cfg.alpha_4v:.10g}) r4"
        ),
        "M_Sigma_formula": (
            f"M_Sigma = r4 - [{cfg.closure_B_r4:.10g} + "
            f"({beta:.10g})(r3 - {cfg.closure_B_r3:.10g})]"
        ),
        "M_4v_normal_definition": "M_4v / sqrt(1 + alpha_4v^2)",
        "M_Sigma_normal_definition": "M_Sigma / sqrt(1 + beta^2)",
        "sigma_boundary_fit": sigma_fit,
        "M_4v_scale": m4_scale,
        "M_Sigma_scale": ms_scale,
        "sector_accuracy": accuracy,
        "sector_balanced_accuracy": balanced_accuracy,
        "per_class_recall": recall,
        "n_points": int(len(right)),
        "n_minus2": int((right["strict_phase_code"] == -2).sum()),
        "n_zero": int((right["strict_phase_code"] == 0).sum()),
        "n_plus2": int((right["strict_phase_code"] == 2).sum()),
        "interpretation": {
            "M_Sigma_positive": "C_up = 0",
            "M_Sigma_negative_and_M_4v_positive": "C_up = -2",
            "M_Sigma_negative_and_M_4v_negative": "C_up = +2",
        },
    }
    return right, transitions, meta


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def scatter_phases(ax, data: pd.DataFrame, xcol: str, ycol: str, size: float):
    for phase in [-2, 0, 2]:
        sub = data[data["strict_phase_code"] == phase]
        if sub.empty:
            continue
        ax.scatter(
            sub[xcol], sub[ycol],
            s=size,
            facecolors="white",
            edgecolors=PHASE_COLORS[phase],
            linewidths=1.5,
            zorder=5,
        )


def plot_left(left: pd.DataFrame, meta: dict, cfg: Step16Config, output_dir: Path):
    x = left["M_L_tilde"].to_numpy()
    y = left["S_L_tilde"].to_numpy()
    xlim = (-cfg.left_mass_padding * max(abs(x.min()), abs(x.max())),
             cfg.left_mass_padding * max(abs(x.min()), abs(x.max())))
    ylim = (-cfg.left_mass_padding * max(abs(y.min()), abs(y.max())),
             cfg.left_mass_padding * max(abs(y.min()), abs(y.max())))

    fig, ax = plt.subplots(figsize=cfg.figure_size_single)
    ax.axvspan(xlim[0], 0.0, facecolor=PHASE_COLORS[-2], alpha=0.25, zorder=0)
    ax.axvspan(0.0, xlim[1], facecolor=PHASE_COLORS[2], alpha=0.25, zorder=0)
    ax.axvline(0.0, color="black", linewidth=1.9, zorder=3)
    scatter_phases(ax, left, "M_L_tilde", "S_L_tilde", cfg.point_size)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\widetilde{\mathbf{\mathit{M}}}_{\mathrm{L}}$", fontsize=24, fontweight="bold")
    ax.set_ylabel(r"$\widetilde{\mathbf{\mathit{S}}}_{\mathrm{L}}$", fontsize=24, fontweight="bold")
    style_axis(ax, "Left four-valley boundary in local mass coordinates")

    handles = phase_handles(include_zero=True) + [
        Line2D([0], [0], color="black", linewidth=1.9,
               label=r"$\mathbf{\mathit{M}}_{\mathrm{L}}=0$"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=13, frameon=True, ncol=2)
    leg.get_frame().set_linewidth(1.1)
    leg.get_frame().set_edgecolor("black")

    ax.text(
        0.03, 0.025,
        f"sign-rule accuracy = {meta['nonzero_phase_sign_accuracy']:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, boxstyle="round,pad=0.3"),
    )
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"step16M_left_mass_coordinate_phase_map.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_right(right: pd.DataFrame, meta: dict, cfg: Step16Config, output_dir: Path):
    x = right["M_4v_tilde"].to_numpy()
    y = right["M_Sigma_tilde"].to_numpy()
    xmax = cfg.right_mass_padding * max(abs(x.min()), abs(x.max()))
    ymax = cfg.right_mass_padding * max(abs(y.min()), abs(y.max()))
    xlim = (-xmax, xmax)
    ylim = (-ymax, ymax)

    fig, ax = plt.subplots(figsize=cfg.figure_size_single)
    # Clean mechanism sectors in the double-mass plane.
    ax.axhspan(0.0, ylim[1], facecolor=PHASE_COLORS[0], alpha=0.45, zorder=0)
    ax.fill_between([xlim[0], 0.0], ylim[0], 0.0, color=PHASE_COLORS[2], alpha=0.35, zorder=0)
    ax.fill_between([0.0, xlim[1]], ylim[0], 0.0, color=PHASE_COLORS[-2], alpha=0.35, zorder=0)
    ax.axvline(0.0, color="black", linewidth=1.8, zorder=3)
    ax.axhline(0.0, color="black", linewidth=1.8, zorder=3)

    scatter_phases(ax, right, "M_4v_tilde", "M_Sigma_tilde", cfg.point_size)
    mismatch = right[~right["mass_sector_match"]]
    if not mismatch.empty:
        ax.scatter(mismatch["M_4v_tilde"], mismatch["M_Sigma_tilde"], marker="x",
                   s=90, color="black", linewidths=1.8, zorder=8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\widetilde{\mathbf{\mathit{M}}}_{4\mathrm{v}}$", fontsize=24, fontweight="bold")
    ax.set_ylabel(r"$\widetilde{\mathbf{\mathit{M}}}_{\Sigma'}$", fontsize=24, fontweight="bold")
    style_axis(ax, "Right junction in double-mass coordinates")

    handles = phase_handles(include_zero=True) + [
        Line2D([0], [0], color="black", linewidth=1.8,
               label=r"$\mathbf{\mathit{M}}_{4\mathrm{v}}=0$ or $\mathbf{\mathit{M}}_{\Sigma'}=0$"),
    ]
    if not mismatch.empty:
        handles.append(Line2D([0], [0], marker="x", linestyle="None", color="black",
                              markersize=9, label="mass-rule mismatch"))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=12.5, frameon=True, ncol=2)
    leg.get_frame().set_linewidth(1.1)
    leg.get_frame().set_edgecolor("black")

    ax.text(
        0.03, 0.025,
        f"sector accuracy = {meta['sector_accuracy']:.3f}\n"
        f"balanced accuracy = {meta['sector_balanced_accuracy']:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, boxstyle="round,pad=0.3"),
    )
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"step16M_right_double_mass_phase_map.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_combined(left: pd.DataFrame, left_meta: dict, right: pd.DataFrame, right_meta: dict,
                  cfg: Step16Config, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=cfg.figure_size_combined)

    # Left panel
    ax = axes[0]
    x = left["M_L_tilde"].to_numpy(); y = left["S_L_tilde"].to_numpy()
    xmax = cfg.left_mass_padding * max(abs(x.min()), abs(x.max()))
    ymax = cfg.left_mass_padding * max(abs(y.min()), abs(y.max()))
    ax.axvspan(-xmax, 0.0, facecolor=PHASE_COLORS[-2], alpha=0.25)
    ax.axvspan(0.0, xmax, facecolor=PHASE_COLORS[2], alpha=0.25)
    ax.axvline(0.0, color="black", linewidth=1.8)
    scatter_phases(ax, left, "M_L_tilde", "S_L_tilde", cfg.point_size * 0.85)
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(r"$\widetilde{\mathbf{\mathit{M}}}_{\mathrm{L}}$", fontsize=22, fontweight="bold")
    ax.set_ylabel(r"$\widetilde{\mathbf{\mathit{S}}}_{\mathrm{L}}$", fontsize=22, fontweight="bold")
    style_axis(ax, "(a) Left local mass plane")

    # Right panel
    ax = axes[1]
    x = right["M_4v_tilde"].to_numpy(); y = right["M_Sigma_tilde"].to_numpy()
    xmax = cfg.right_mass_padding * max(abs(x.min()), abs(x.max()))
    ymax = cfg.right_mass_padding * max(abs(y.min()), abs(y.max()))
    ax.axhspan(0.0, ymax, facecolor=PHASE_COLORS[0], alpha=0.45)
    ax.fill_between([-xmax, 0.0], -ymax, 0.0, color=PHASE_COLORS[2], alpha=0.35)
    ax.fill_between([0.0, xmax], -ymax, 0.0, color=PHASE_COLORS[-2], alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=1.8)
    scatter_phases(ax, right, "M_4v_tilde", "M_Sigma_tilde", cfg.point_size * 0.85)
    mismatch = right[~right["mass_sector_match"]]
    if not mismatch.empty:
        ax.scatter(mismatch["M_4v_tilde"], mismatch["M_Sigma_tilde"], marker="x",
                   s=80, color="black", linewidths=1.7, zorder=8)
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(r"$\widetilde{\mathbf{\mathit{M}}}_{4\mathrm{v}}$", fontsize=22, fontweight="bold")
    ax.set_ylabel(r"$\widetilde{\mathbf{\mathit{M}}}_{\Sigma'}$", fontsize=22, fontweight="bold")
    style_axis(ax, "(b) Right double-mass plane")

    handles = phase_handles(include_zero=True)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=3, fontsize=14, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"step16M_combined_mass_coordinate_phase_maps.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_step16M(
    phase_source: str | Path = "outputs_tts_step15C_repair_and_redraw_phase_maps.zip",
    step11_source: str | Path | None = "outputs_tts_step11M_lieb_aligned_fixed_slice.zip",
    step12_source: str | Path | None = "outputs_tts_step12M_junction_multiclosure_repair.zip",
    output_dir: str | Path = "outputs_tts_step16M_mass_coordinate_phase_maps",
    config: Step16Config | None = None,
):
    if config is None:
        config = Step16Config(output_dir=Path(output_dir))
    config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    phase = read_csv_any(phase_source, "step15C_00_final_phase_points_repaired.csv")

    # Use exact closure coordinates from Step12M when available.
    closure = optional_csv(step12_source, "step12M_06_critical_closure_parameters.csv")
    if closure is not None and len(closure) >= 2:
        closure = closure.sort_values("critical_lambda").reset_index(drop=True)
        config.closure_A_r3 = float(closure.loc[0, "r3"])
        config.closure_A_r4 = float(closure.loc[0, "r4"])
        config.closure_B_r3 = float(closure.loc[1, "r3"])
        config.closure_B_r4 = float(closure.loc[1, "r4"])

    left, left_meta = transform_left(phase, config)
    right, transition_points, right_meta = transform_right(phase, config)

    # Optional certified points from Step11M are transformed and stored for audit.
    step11 = optional_csv(step11_source, "step11M_07_completed_certified_boundary_points.csv")
    if step11 is not None:
        left_cert = step11[step11["branch_hint"].eq("generic_left_lower")].copy()
        if not left_cert.empty:
            slope = left_boundary_slope(left_cert["r3"].to_numpy(), config)
            mass = (
                left_cert["r4"].to_numpy()
                + config.left_a * left_cert["r3"].to_numpy() ** 2
                + config.left_b * left_cert["r3"].to_numpy()
                + config.left_c
            ) / np.sqrt(1.0 + slope**2)
            tangent = left_arc_coordinate(left_cert["r3"].to_numpy(), config)
            left_cert["M_L_normal"] = mass
            left_cert["S_L_arc"] = tangent
            left_cert.to_csv(config.output_dir / "step16M_03_left_certified_boundary_points_mass_coordinates.csv", index=False)

    transformed = pd.concat([
        left.assign(mass_map_region="left_local"),
        right.assign(mass_map_region="right_junction"),
    ], ignore_index=True, sort=False)
    transformed.to_csv(config.output_dir / "step16M_00_transformed_mass_coordinate_points.csv", index=False)
    transition_points.to_csv(config.output_dir / "step16M_01_sigma_transition_midpoints.csv", index=False)

    plot_left(left, left_meta, config, config.output_dir)
    plot_right(right, right_meta, config, config.output_dir)
    plot_combined(left, left_meta, right, right_meta, config, config.output_dir)

    report = {
        "code_version": CODE_VERSION,
        "configuration": asdict(config),
        "left_mass_coordinates": left_meta,
        "right_double_mass_coordinates": right_meta,
        "evidence_scope": {
            "left": "Local formula valid only near the certified left four-valley branch.",
            "right": (
                "M_4v is the certified local four-valley mass. M_Sigma is an exploratory "
                "local mass coordinate fitted from the certified +2/0 boundary and anchored "
                "at the Sigma' two-valley closure. It is not yet a globally derived k.p mass."
            ),
            "global_warning": (
                "These local mass coordinates must not be used as a global reparameterization "
                "of the entire fixed-background r3-r4 phase map."
            ),
        },
        "generated_figures": [
            "step16M_left_mass_coordinate_phase_map.pdf",
            "step16M_right_double_mass_phase_map.pdf",
            "step16M_combined_mass_coordinate_phase_maps.pdf",
        ],
    }
    with open(config.output_dir / "step16M_02_mass_coordinate_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md = rf"""# TTS Step16M mass-coordinate phase maps

## Left local coordinates

\[
M_{{\mathrm L}}=r_4+{config.left_a:.10g}r_3^2+{config.left_b:.10g}r_3+{config.left_c:.10g}.
\]

The plotted horizontal coordinate is the normalized signed normal distance
\(\widetilde M_{{\mathrm L}}\); the vertical coordinate
\(\widetilde S_{{\mathrm L}}\) is normalized arc length along \(M_{{\mathrm L}}=0\).
The sign rule reproduces the certified \(C_\uparrow=\pm2\) labels with accuracy
**{left_meta['nonzero_phase_sign_accuracy']:.6f}**.

## Right double-mass coordinates

\[
M_{{4v}}=r_3-{config.alpha_4v:.10g}r_4,
\]

\[
M_{{\Sigma'}}=r_4-\left[{config.closure_B_r4:.10g}
+{right_meta['sigma_boundary_fit']['beta']:.10g}(r_3-{config.closure_B_r3:.10g})\right].
\]

The mechanism-sector rule gives accuracy **{right_meta['sector_accuracy']:.6f}**
and balanced accuracy **{right_meta['sector_balanced_accuracy']:.6f}**.

## Scope

The left and right coordinates are **local mechanism coordinates**, not a global
coordinate transformation. In particular, \(M_{{\Sigma'}}\) is fitted from the
certified local \(+2/0\) boundary and anchored to the certified two-valley closure.
It should be described as an exploratory effective mass coordinate until an
independent low-energy \(k\cdot p\) derivation is completed.
"""
    (config.output_dir / "step16M_02_mass_coordinate_report.md").write_text(md, encoding="utf-8")

    print(json.dumps({
        "code_version": CODE_VERSION,
        "output_dir": str(config.output_dir),
        "left_sign_accuracy": left_meta["nonzero_phase_sign_accuracy"],
        "right_sector_accuracy": right_meta["sector_accuracy"],
        "right_sector_balanced_accuracy": right_meta["sector_balanced_accuracy"],
        "M_Sigma_beta": right_meta["sigma_boundary_fit"]["beta"],
    }, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_step16M()
