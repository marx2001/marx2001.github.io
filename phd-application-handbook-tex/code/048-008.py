#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FES u_eff-B_eff phase map with Lieb-style random sample circles in each region.

This version keeps the current u_eff-B_eff phase-map style and color mapping,
then overlays small open circles randomly distributed inside each topological
region, in a way similar to the Lieb display style.

The random circles are schematic display points only. They are not newly
computed physical data points.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_VERSION = "FES_UEFF_BEFF_CURRENT_COLORS_RANDOM_REGION_CIRCLES_20260729"

CURRENT_UB_POINTS = "fes_uB_points_all_selected_source.csv"
CURRENT_UB_STRICT = "fes_uB_points_strict.csv"
CURRENT_UB_SUMMARY = "fes_uB_phase_map_summary.json"

CURRENT_MASS_ALL = "fes_signed_mass_plane_points_all.csv"
CURRENT_MASS_STRICT = "fes_signed_mass_plane_points_strict.csv"
CURRENT_MASS_SUMMARY = "fes_mass_plane_summary.json"

ORIGINAL_STRICT_TABLES = (
    "fes_step07_strict_formula_audit.csv",
    "fes_step06_formula_validation_strict.csv",
    "fes_step05_strict_analytic_audit.csv",
)
ORIGINAL_COARSE_TABLES = (
    "fes_step06_formula_validation_coarse.csv",
    "fes_step05_all_coarse_analytic_audit.csv",
)
ORIGINAL_SUMMARIES = (
    "fes_step07_run_summary.json",
    "fes_step06_run_summary.json",
)

COLOR_RED = (169 / 255.0, 36 / 255.0, 37 / 255.0)
COLOR_BLUE = (63 / 255.0, 99 / 255.0, 173 / 255.0)
COLOR_GREY = (211 / 255.0, 211 / 255.0, 211 / 255.0)

def lighten(color: tuple[float, float, float], factor: float):
    base = np.asarray(color, dtype=float)
    return tuple(np.clip(base * (1.0 - factor) + factor, 0.0, 1.0))

REGION_FILL = {
    +1: lighten(COLOR_RED, 0.70),
    0: lighten(COLOR_GREY, 0.15),
    -1: lighten(COLOR_BLUE, 0.68),
}
POINT_EDGE = {
    +1: COLOR_RED,
    0: (0.25, 0.25, 0.25),
    -1: COLOR_BLUE,
}


def configure_plot_style() -> None:
    available = {item.name for item in font_manager.fontManager.ttflist}
    font_name = (
        "Times New Roman"
        if "Times New Roman" in available
        else "DejaVu Serif"
    )
    rcParams["font.family"] = font_name
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 9
    rcParams["axes.labelsize"] = 9
    rcParams["axes.titlesize"] = 9
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["legend.fontsize"] = 8
    rcParams["axes.linewidth"] = 0.8
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.top"] = True
    rcParams["ytick.right"] = True
    rcParams["svg.fonttype"] = "none"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["axes.unicode_minus"] = False


@dataclass
class DataSource:
    path: Path

    def __post_init__(self):
        self.path = Path(self.path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_file() and self.path.suffix.lower() != ".zip":
            raise ValueError(f"Input must be a ZIP or extracted directory: {self.path}")

    @property
    def is_zip(self) -> bool:
        return self.path.is_file()

    def names(self) -> list[str]:
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                return [n for n in archive.namelist() if not n.endswith("/")]
        return [str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file()]

    def find_basename(self, candidates: Sequence[str]) -> str | None:
        names = self.names()
        for basename in candidates:
            matches = [n for n in names if Path(n).name == basename]
            if matches:
                return sorted(matches, key=lambda x: (len(Path(x).parts), len(x)))[0]
        return None

    def read_csv(self, candidates: Sequence[str], required: bool):
        selected = self.find_basename(candidates)
        if selected is None:
            if required:
                raise FileNotFoundError(
                    f"No compatible CSV found in {self.path}.\nExpected one of: {list(candidates)}"
                )
            return None, None
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(selected) as stream:
                    return pd.read_csv(stream, low_memory=False), Path(selected).name
        return pd.read_csv(self.path / selected, low_memory=False), Path(selected).name

    def read_json_optional(self, candidates: Sequence[str]):
        selected = self.find_basename(candidates)
        if selected is None:
            return {}, None
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                return json.loads(archive.read(selected).decode("utf-8")), Path(selected).name
        return json.loads((self.path / selected).read_text(encoding="utf-8")), Path(selected).name


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def choose_chern_column(frame: pd.DataFrame) -> str:
    for column in ("chern_up_plot", "verified_chern_up_int", "chern_up_int"):
        if column in frame.columns:
            return column
    raise KeyError("No Chern column found.")


def choose_gap_column(frame: pd.DataFrame) -> str | None:
    for column in ("verified_indirect_gap", "indirect_gap"):
        if column in frame.columns:
            return column
    return None


def prepare_points(frame: pd.DataFrame, role: str):
    out = frame.copy()

    if "M_Gamma_FES" in out.columns:
        out["M_Gamma_FES"] = numeric(out, "M_Gamma_FES")
    elif "mass_Gamma_signed" in out.columns:
        out["M_Gamma_FES"] = numeric(out, "mass_Gamma_signed")
    else:
        raise KeyError("Missing M_Gamma_FES/mass_Gamma_signed.")

    if "M_M_FES" in out.columns:
        out["M_M_FES"] = numeric(out, "M_M_FES")
    elif "mass_M_signed" in out.columns:
        out["M_M_FES"] = numeric(out, "mass_M_signed")
    else:
        raise KeyError("Missing M_M_FES/mass_M_signed.")

    if "u_eff" not in out.columns:
        out["u_eff"] = 0.5 * (out["M_Gamma_FES"] - out["M_M_FES"])
    else:
        out["u_eff"] = numeric(out, "u_eff")

    if "B_eff" not in out.columns:
        out["B_eff"] = 0.25 * (out["M_Gamma_FES"] + out["M_M_FES"])
    else:
        out["B_eff"] = numeric(out, "B_eff")

    chern_col = choose_chern_column(out)
    gap_col = choose_gap_column(out)
    out["chern_up_plot"] = numeric(out, chern_col)

    if "chi_plot" in out.columns:
        out["chi_plot"] = numeric(out, "chi_plot")
    elif {"r1", "r2"}.issubset(out.columns):
        out["chi_plot"] = np.sign(numeric(out, "r1") * numeric(out, "r2"))
    else:
        raise KeyError("Missing chi_plot or r1/r2.")

    out = out.dropna(
        subset=["M_Gamma_FES", "M_M_FES", "u_eff", "B_eff", "chern_up_plot", "chi_plot"]
    ).copy()
    out["chern_up_plot"] = out["chern_up_plot"].round().astype(int)
    out["chi_plot"] = out["chi_plot"].round().astype(int)
    out["source_table_role"] = role
    return out, chern_col, gap_col


def calibrate_prefactor(strict: pd.DataFrame, summary: dict):
    prefactor_json = None
    if isinstance(summary.get("prefactor_audit"), dict):
        prefactor_json = summary["prefactor_audit"].get("prefactor")
    if prefactor_json not in (-1, 1):
        prefactor_json = summary.get("calibrated_formula_prefactor")

    try:
        prefactor_json = int(prefactor_json)
    except (TypeError, ValueError):
        prefactor_json = 0

    base = (
        strict["chi_plot"].to_numpy(float)
        * 0.5
        * (np.sign(strict["M_Gamma_FES"].to_numpy(float)) + np.sign(strict["M_M_FES"].to_numpy(float)))
    )
    actual = strict["chern_up_plot"].to_numpy(float)
    valid = np.isfinite(base) & np.isfinite(actual) & (np.abs(base) > 0)

    if valid.any():
        acc_plus = float(np.mean(base[valid] == actual[valid]))
        acc_minus = float(np.mean(-base[valid] == actual[valid]))
        data_prefactor = 1 if acc_plus >= acc_minus else -1
    else:
        acc_plus = float("nan")
        acc_minus = float("nan")
        data_prefactor = 1

    if prefactor_json in (-1, 1):
        prefactor = prefactor_json
        source = "input_summary"
    else:
        prefactor = data_prefactor
        source = "strict_data_calibration"

    return int(prefactor), {
        "prefactor": int(prefactor),
        "source": source,
        "strict_nonzero_rows_used": int(valid.sum()),
        "accuracy_if_prefactor_plus": acc_plus,
        "accuracy_if_prefactor_minus": acc_minus,
    }


def load_data(input_source: Path | str, point_source: str):
    source = DataSource(Path(input_source))

    ub_points_raw, ub_points_name = source.read_csv((CURRENT_UB_POINTS,), required=False)
    ub_strict_raw, ub_strict_name = source.read_csv((CURRENT_UB_STRICT,), required=False)

    if ub_strict_raw is not None:
        summary, _ = source.read_json_optional((CURRENT_UB_SUMMARY,))
        strict, strict_chern_col, strict_gap_col = prepare_points(ub_strict_raw, "strict")
        if point_source == "coarse" and ub_points_raw is not None:
            points, point_chern_col, point_gap_col = prepare_points(ub_points_raw, "coarse")
            point_name = ub_points_name
        else:
            points, point_chern_col, point_gap_col = strict.copy(), strict_chern_col, strict_gap_col
            point_name = ub_strict_name
        prefactor, prefactor_audit = calibrate_prefactor(strict, summary)
        return {
            "source": source.path,
            "strict": strict,
            "points": points,
            "point_gap_column": point_gap_col,
            "prefactor": prefactor,
            "prefactor_audit": prefactor_audit,
            "point_table_name": point_name,
        }

    mass_all_raw, mass_all_name = source.read_csv((CURRENT_MASS_ALL,), required=False)
    mass_strict_raw, mass_strict_name = source.read_csv((CURRENT_MASS_STRICT,), required=False)
    if mass_strict_raw is not None:
        summary, _ = source.read_json_optional((CURRENT_MASS_SUMMARY,))
        strict, strict_chern_col, strict_gap_col = prepare_points(mass_strict_raw, "strict")
        if point_source == "coarse" and mass_all_raw is not None:
            points, point_chern_col, point_gap_col = prepare_points(mass_all_raw, "coarse")
            point_name = mass_all_name
        else:
            points, point_chern_col, point_gap_col = strict.copy(), strict_chern_col, strict_gap_col
            point_name = mass_strict_name
        prefactor, prefactor_audit = calibrate_prefactor(strict, summary)
        return {
            "source": source.path,
            "strict": strict,
            "points": points,
            "point_gap_column": point_gap_col,
            "prefactor": prefactor,
            "prefactor_audit": prefactor_audit,
            "point_table_name": point_name,
        }

    strict_raw, strict_name = source.read_csv(ORIGINAL_STRICT_TABLES, required=True)
    all_raw, all_name = source.read_csv(ORIGINAL_COARSE_TABLES, required=False)
    summary, _ = source.read_json_optional(ORIGINAL_SUMMARIES)

    strict, strict_chern_col, strict_gap_col = prepare_points(strict_raw, "strict")
    if point_source == "coarse" and all_raw is not None:
        points, point_chern_col, point_gap_col = prepare_points(all_raw, "coarse")
        point_name = all_name
    else:
        points, point_chern_col, point_gap_col = strict.copy(), strict_chern_col, strict_gap_col
        point_name = strict_name

    prefactor, prefactor_audit = calibrate_prefactor(strict, summary)
    return {
        "source": source.path,
        "strict": strict,
        "points": points,
        "point_gap_column": point_gap_col,
        "prefactor": prefactor,
        "prefactor_audit": prefactor_audit,
        "point_table_name": point_name,
    }


def axis_extent(values: pd.Series, minimum: float = 0.5, padding: float = 1.10, quantile: float = 0.995):
    arr = np.abs(pd.to_numeric(values, errors="coerce").to_numpy(float))
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return minimum
    return max(minimum, padding * float(np.quantile(arr, quantile)))


def deterministic_stratified_subset(frame: pd.DataFrame, maximum: int | None, seed: int):
    if maximum is None or len(frame) <= maximum:
        return frame.copy()
    rng = np.random.default_rng(seed)
    selected = []
    groups = list(frame.groupby("chern_up_plot", sort=True))
    total = len(frame)
    for idx, (_, group) in enumerate(groups):
        if idx == len(groups) - 1:
            target = maximum - len(selected)
        else:
            target = max(1, int(round(maximum * len(group) / total)))
            target = min(target, maximum - len(selected))
        choices = rng.choice(group.index.to_numpy(), size=min(target, len(group)), replace=False)
        selected.extend(int(i) for i in choices)
    return frame.loc[selected].copy()


def add_region_labels(ax: plt.Axes, u_limit: float, B_limit: float, chirality: int, prefactor: int):
    c_top = prefactor * chirality
    c_bottom = -prefactor * chirality
    box = dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.72)
    ax.text(0.0, 0.62 * B_limit, rf"$C_\uparrow={c_top:+d}$", ha="center", va="center", bbox=box)
    ax.text(0.0, -0.62 * B_limit, rf"$C_\uparrow={c_bottom:+d}$", ha="center", va="center", bbox=box)
    ax.text(-0.64 * u_limit, 0.0, r"$C_\uparrow=0$", ha="center", va="center", bbox=box)
    ax.text(0.64 * u_limit, 0.0, r"$C_\uparrow=0$", ha="center", va="center", bbox=box)


def classify_region(u: np.ndarray, B: np.ndarray, chirality: int, prefactor: int) -> np.ndarray:
    M_gamma = u + 2.0 * B
    M_M = -u + 2.0 * B
    return (prefactor * chirality * 0.5 * (np.sign(M_gamma) + np.sign(M_M))).astype(int)


def random_points_in_region(
    target_chern: int,
    chirality: int,
    prefactor: int,
    u_limit: float,
    B_limit: float,
    n_points: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    accepted_u = []
    accepted_B = []

    # For C=0, enforce coverage on both left and right side regions.
    if target_chern == 0:
        parts = [("left", n_points // 2), ("right", n_points - n_points // 2)]
    else:
        parts = [(None, n_points)]

    for side_name, need in parts:
        got = 0
        attempts = 0
        while got < need and attempts < 200000:
            batch = min(5000, max(1000, 20 * (need - got)))
            attempts += batch

            if side_name == "left":
                u = rng.uniform(-u_limit, 0.0, size=batch)
            elif side_name == "right":
                u = rng.uniform(0.0, u_limit, size=batch)
            else:
                u = rng.uniform(-u_limit, u_limit, size=batch)
            B = rng.uniform(-B_limit, B_limit, size=batch)

            labels = classify_region(u, B, chirality=chirality, prefactor=prefactor)
            mask = labels == target_chern
            u_sel = u[mask]
            B_sel = B[mask]

            take = min(need - got, len(u_sel))
            if take > 0:
                accepted_u.extend(u_sel[:take].tolist())
                accepted_B.extend(B_sel[:take].tolist())
                got += take

        if got < need:
            raise RuntimeError(
                f"Failed to sample enough random points for C={target_chern} "
                f"(got {got}, need {need})."
            )

    return pd.DataFrame(
        {
            "u_eff": np.asarray(accepted_u, dtype=float),
            "B_eff": np.asarray(accepted_B, dtype=float),
            "chern_up_plot": int(target_chern),
            "chi_plot": int(chirality),
        }
    )


def generate_random_region_points(
    chirality: int,
    prefactor: int,
    u_limit: float,
    B_limit: float,
    n_points_per_region: int,
    seed: int,
):
    c_top = int(prefactor * chirality)
    c_bottom = int(-prefactor * chirality)

    frames = [
        random_points_in_region(
            target_chern=c_top,
            chirality=chirality,
            prefactor=prefactor,
            u_limit=u_limit,
            B_limit=B_limit,
            n_points=n_points_per_region,
            seed=seed + 11,
        ),
        random_points_in_region(
            target_chern=0,
            chirality=chirality,
            prefactor=prefactor,
            u_limit=u_limit,
            B_limit=B_limit,
            n_points=n_points_per_region,
            seed=seed + 29,
        ),
        random_points_in_region(
            target_chern=c_bottom,
            chirality=chirality,
            prefactor=prefactor,
            u_limit=u_limit,
            B_limit=B_limit,
            n_points=n_points_per_region,
            seed=seed + 47,
        ),
    ]
    out = pd.concat(frames, ignore_index=True)
    out["sample_kind"] = "random_region_circle"
    return out


def save_figure(fig: plt.Figure, stem: Path, png_dpi: int):
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": stem.with_suffix(".png"),
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
    }
    fig.savefig(paths["png"], dpi=png_dpi, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_uB_map(
    points: pd.DataFrame,
    strict: pd.DataFrame,
    chirality: int,
    prefactor: int,
    output_dir: Path | str,
    png_dpi: int,
    maximum_display_points: int | None,
    show_gap_status: bool,
    gap_column: str | None,
    show_original_points: bool,
    n_random_points_per_region: int,
    random_seed: int,
):
    subset_all = points[points["chi_plot"] == chirality].copy()
    strict_subset = strict[strict["chi_plot"] == chirality].copy()
    if subset_all.empty:
        raise ValueError(f"No samples for chi={chirality:+d}.")

    subset = deterministic_stratified_subset(
        subset_all,
        maximum=maximum_display_points,
        seed=20260729 + chirality,
    )

    u_limit = axis_extent(pd.concat([subset_all["u_eff"], strict_subset["u_eff"]], ignore_index=True))
    B_limit = axis_extent(pd.concat([subset_all["B_eff"], strict_subset["B_eff"]], ignore_index=True))

    random_region_points = generate_random_region_points(
        chirality=chirality,
        prefactor=prefactor,
        u_limit=u_limit,
        B_limit=B_limit,
        n_points_per_region=n_random_points_per_region,
        seed=random_seed + 100 * (chirality + 5),
    )

    u_axis = np.linspace(-u_limit, u_limit, 701)
    B_axis = np.linspace(-B_limit, B_limit, 701)
    U, B = np.meshgrid(u_axis, B_axis)
    M_gamma = U + 2.0 * B
    M_M = -U + 2.0 * B
    chern_grid = prefactor * chirality * 0.5 * (np.sign(M_gamma) + np.sign(M_M))

    region_rgb = np.empty(chern_grid.shape + (3,), dtype=float)
    region_rgb[...] = REGION_FILL[0]
    for value, color in REGION_FILL.items():
        region_rgb[np.isclose(chern_grid, value)] = color

    fig, ax = plt.subplots(figsize=(8.0 / 2.54, 7.0 / 2.54))
    ax.imshow(
        region_rgb,
        extent=[-u_limit, u_limit, -B_limit, B_limit],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )

    line_u = np.linspace(-u_limit, u_limit, 900)
    ax.plot(line_u, 0.5 * line_u, linestyle="--", linewidth=1.0, color="black", zorder=2)
    ax.plot(line_u, -0.5 * line_u, linestyle="--", linewidth=1.0, color="black", zorder=2)

    if show_original_points:
        for c_value in (-1, 0, +1):
            group = subset[subset["chern_up_plot"] == c_value]
            if group.empty:
                continue

            if show_gap_status and gap_column is not None and gap_column in group.columns:
                gap = pd.to_numeric(group[gap_column], errors="coerce")
                positive = group[gap > 0]
                nonpositive = group[gap <= 0]

                if len(positive):
                    ax.scatter(
                        positive["u_eff"], positive["B_eff"],
                        s=18, facecolors="white", edgecolors=[POINT_EDGE[c_value]],
                        linewidths=0.80, alpha=0.95, zorder=3
                    )
                if len(nonpositive):
                    ax.scatter(
                        nonpositive["u_eff"], nonpositive["B_eff"],
                        s=17, marker="x", color=[POINT_EDGE[c_value]],
                        linewidths=0.80, alpha=0.88, zorder=3
                    )
            else:
                ax.scatter(
                    group["u_eff"], group["B_eff"],
                    s=18, facecolors="white", edgecolors=[POINT_EDGE[c_value]],
                    linewidths=0.80, alpha=0.95, zorder=3
                )

    # Overlay Lieb-style random open circles in each region.
    for c_value in (-1, 0, +1):
        group = random_region_points[random_region_points["chern_up_plot"] == c_value]
        if group.empty:
            continue
        ax.scatter(
            group["u_eff"], group["B_eff"],
            s=24, facecolors="white", edgecolors=[POINT_EDGE[c_value]],
            linewidths=0.95, alpha=0.98, zorder=4
        )

    add_region_labels(ax, u_limit=u_limit, B_limit=B_limit, chirality=chirality, prefactor=prefactor)

    ax.set_xlim(-u_limit, u_limit)
    ax.set_ylim(-B_limit, B_limit)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$u_{\mathrm{eff}}$")
    ax.set_ylabel(r"$B_{\mathrm{eff}}$")

    ax.text(
        0.03, 0.97,
        rf"$\chi_{{\mathrm{{FES}}}}=\mathrm{{sgn}}(r_1r_2)={chirality:+d}$",
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.82),
    )

    region_handles = [
        Patch(facecolor=REGION_FILL[+1], edgecolor="none", label=r"Region: $C_\uparrow=+1$"),
        Patch(facecolor=REGION_FILL[0], edgecolor="none", label=r"Region: $C_\uparrow=0$"),
        Patch(facecolor=REGION_FILL[-1], edgecolor="none", label=r"Region: $C_\uparrow=-1$"),
    ]
    region_legend = ax.legend(
        handles=region_handles, loc="lower left",
        frameon=True, framealpha=0.88, borderpad=0.35
    )
    ax.add_artist(region_legend)

    point_handles = []
    if show_original_points:
        if show_gap_status and gap_column is not None:
            point_handles.extend([
                Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
                       markeredgecolor="black", markeredgewidth=0.9, markersize=5,
                       label=r"Original: $E_g^{\mathrm{ind}}>0$"),
                Line2D([0], [0], marker="x", linestyle="None", color="black",
                       markeredgewidth=0.9, markersize=5,
                       label=r"Original: $E_g^{\mathrm{ind}}\leq0$"),
            ])
        else:
            for c_value in (+1, 0, -1):
                label = rf"Original: $C_\uparrow={c_value:+d}$" if c_value != 0 else r"Original: $C_\uparrow=0$"
                point_handles.append(
                    Line2D([0], [0], marker="o", linestyle="None",
                           markerfacecolor="white", markeredgecolor=POINT_EDGE[c_value],
                           markeredgewidth=0.9, markersize=5, label=label)
                )
    point_handles.append(
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=0.9, markersize=5,
               label=f"Random display circles: {n_random_points_per_region} per region")
    )

    ax.legend(handles=point_handles, loc="upper right", frameon=True, framealpha=0.88, borderpad=0.35)

    fig.tight_layout()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_gap_status" if show_gap_status else ""
    stem = output_dir / f"fes_uB_phase_map_chi_{chirality:+d}_with_random_region_circles{suffix}"
    paths = save_figure(fig, stem=stem, png_dpi=png_dpi)

    subset.to_csv(output_dir / f"fes_uB_display_points_chi_{chirality:+d}_original{suffix}.csv", index=False)
    random_region_points.to_csv(output_dir / f"fes_uB_random_region_circles_chi_{chirality:+d}.csv", index=False)

    return {
        "paths": paths,
        "n_available_points": int(len(subset_all)),
        "n_original_displayed_points": int(len(subset)),
        "n_random_region_points": int(len(random_region_points)),
    }


def run_fes_uB_map_with_random_region_circles(
    input_source: Path | str,
    output_dir: Path | str = "fes_uB_phase_map_with_random_region_circles_outputs",
    point_source: str = "strict",
    png_dpi: int = 2400,
    generate_chi_minus: bool = True,
    maximum_display_points: int | None = 1400,
    generate_gap_status_version: bool = False,
    show_original_points: bool = False,
    n_random_points_per_region: int = 20,
    random_seed: int = 20260729,
):
    configure_plot_style()

    loaded = load_data(input_source=input_source, point_source=point_source)
    points = loaded["points"]
    strict = loaded["strict"]
    prefactor = int(loaded["prefactor"])

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(output_dir / "fes_uB_points_all_selected_source.csv", index=False)
    strict.to_csv(output_dir / "fes_uB_points_strict.csv", index=False)

    figures = {}
    for chirality in ([1, -1] if generate_chi_minus else [1]):
        if not (points["chi_plot"] == chirality).any():
            continue
        standard = plot_uB_map(
            points=points, strict=strict, chirality=chirality, prefactor=prefactor,
            output_dir=output_dir, png_dpi=png_dpi, maximum_display_points=maximum_display_points,
            show_gap_status=False, gap_column=loaded["point_gap_column"],
            show_original_points=show_original_points,
            n_random_points_per_region=n_random_points_per_region,
            random_seed=random_seed,
        )
        figures[chirality] = {"standard": standard}

        if generate_gap_status_version and loaded["point_gap_column"] is not None:
            gap_version = plot_uB_map(
                points=points, strict=strict, chirality=chirality, prefactor=prefactor,
                output_dir=output_dir, png_dpi=png_dpi, maximum_display_points=maximum_display_points,
                show_gap_status=True, gap_column=loaded["point_gap_column"],
                show_original_points=show_original_points,
                n_random_points_per_region=n_random_points_per_region,
                random_seed=random_seed,
            )
            figures[chirality]["gap_status"] = gap_version

    summary = {
        "script_version": SCRIPT_VERSION,
        "plotting_only": True,
        "input_source": str(loaded["source"]),
        "point_source": point_source,
        "prefactor_audit": loaded["prefactor_audit"],
        "axis_x": "u_eff = (M_Gamma_FES - M_M_FES) / 2",
        "axis_y": "B_eff = (M_Gamma_FES + M_M_FES) / 4",
        "phase_boundaries": ["B_eff = +u_eff/2", "B_eff = -u_eff/2"],
        "show_original_points": bool(show_original_points),
        "n_random_points_per_region": int(n_random_points_per_region),
        "random_seed": int(random_seed),
        "note": (
            "Random display circles are schematic visual points used to mimic the "
            "Lieb-style sampling display. They are not newly computed physical data."
        ),
    }
    summary_path = output_dir / "fes_uB_random_region_circles_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "input_source": loaded["source"],
        "output_dir": output_dir,
        "points": points,
        "strict": strict,
        "prefactor": prefactor,
        "figures": figures,
        "summary": summary,
        "summary_path": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FES u_eff-B_eff phase map with Lieb-style random region circles."
    )
    parser.add_argument("--input-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("fes_uB_phase_map_with_random_region_circles_outputs"))
    parser.add_argument("--point-source", choices=["strict", "coarse"], default="strict")
    parser.add_argument("--png-dpi", type=int, default=2400)
    parser.add_argument("--maximum-display-points", type=int, default=1400)
    parser.add_argument("--n-random-points-per-region", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260729)
    parser.add_argument("--show-original-points", action="store_true")
    parser.add_argument("--gap-status-version", action="store_true")
    parser.add_argument("--no-chi-minus", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_fes_uB_map_with_random_region_circles(
        input_source=args.input_source,
        output_dir=args.output_dir,
        point_source=args.point_source,
        png_dpi=args.png_dpi,
        generate_chi_minus=not args.no_chi_minus,
        maximum_display_points=args.maximum_display_points,
        generate_gap_status_version=args.gap_status_version,
        show_original_points=args.show_original_points,
        n_random_points_per_region=args.n_random_points_per_region,
        random_seed=args.random_seed,
    )
    print("Input:", result["input_source"])
    print("Point source:", args.point_source)
    print("Points:", len(result["points"]))
    print("Strict points:", len(result["strict"]))
    print("Show original points:", args.show_original_points)
    print("Random points per region:", args.n_random_points_per_region)
    print("Output:", result["output_dir"])
