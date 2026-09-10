#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a Lieb-style four-quadrant signed-mass phase map for the FES model.

Axes
----
x = M_Gamma^FES = mass_Gamma_signed
y = M_M^FES     = mass_M_signed

Formula
-------
C_up = eta * chi_FES / 2
       * [sgn(M_Gamma^FES) + sgn(M_M^FES)]

chi_FES = sgn(r1*r2).

The program calibrates eta from the strict FES Step07/Step06 audit.
For the current frozen Step07 result, eta = +1.

Important
---------
The nonzero FES regions are formal spin-Chern sectors. Under the minimal-model
indirect-gap no-go, they must not be called QSH-insulator regions.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_VERSION = "FES_LIEB_STYLE_FOUR_QUADRANT_20260729"

STRICT_TABLES = (
    "fes_step07_strict_formula_audit.csv",
    "fes_step06_formula_validation_strict.csv",
    "fes_step05_strict_analytic_audit.csv",
)

COARSE_TABLES = (
    "fes_step06_formula_validation_coarse.csv",
    "fes_step05_all_coarse_analytic_audit.csv",
)

SUMMARY_JSONS = (
    "fes_step07_run_summary.json",
    "fes_step06_run_summary.json",
)

REQUIRED_COLUMNS = (
    "mass_Gamma_signed",
    "mass_M_signed",
    "r1",
    "r2",
)

# Exact colors used by the Lieb paper-style figure.
COLOR_RED = (169 / 255.0, 36 / 255.0, 37 / 255.0)
COLOR_BLUE = (63 / 255.0, 99 / 255.0, 173 / 255.0)
COLOR_GREY = (211 / 255.0, 211 / 255.0, 211 / 255.0)


def lighten(
    color: tuple[float, float, float],
    factor: float,
) -> tuple[float, float, float]:
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

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_file() and self.path.suffix.lower() != ".zip":
            raise ValueError(
                f"Input must be a ZIP or an extracted directory: {self.path}"
            )

    @property
    def is_zip(self) -> bool:
        return self.path.is_file()

    def names(self) -> list[str]:
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                return [
                    name
                    for name in archive.namelist()
                    if not name.endswith("/")
                ]

        return [
            str(path.relative_to(self.path))
            for path in self.path.rglob("*")
            if path.is_file()
        ]

    def find_basename(self, candidates: Sequence[str]) -> str | None:
        names = self.names()
        for basename in candidates:
            matches = [
                name for name in names
                if Path(name).name == basename
            ]
            if matches:
                return sorted(
                    matches,
                    key=lambda item: (
                        len(Path(item).parts),
                        len(item),
                    ),
                )[0]
        return None

    def read_csv(
        self,
        candidates: Sequence[str],
        required: bool,
    ) -> tuple[pd.DataFrame | None, str | None]:
        selected = self.find_basename(candidates)
        if selected is None:
            if required:
                raise FileNotFoundError(
                    f"No compatible CSV was found in {self.path}.\n"
                    f"Expected one of: {list(candidates)}"
                )
            return None, None

        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(selected) as stream:
                    return (
                        pd.read_csv(stream, low_memory=False),
                        Path(selected).name,
                    )

        return (
            pd.read_csv(self.path / selected, low_memory=False),
            Path(selected).name,
        )

    def read_json_optional(
        self,
        candidates: Sequence[str],
    ) -> tuple[dict, str | None]:
        selected = self.find_basename(candidates)
        if selected is None:
            return {}, None

        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                return (
                    json.loads(
                        archive.read(selected).decode("utf-8")
                    ),
                    Path(selected).name,
                )

        return (
            json.loads(
                (self.path / selected).read_text(encoding="utf-8")
            ),
            Path(selected).name,
        )


def discover_fes_input(root: Path | str = ".") -> Path:
    root = Path(root).expanduser().resolve()
    patterns = (
        "outputs_fes6_step07_symbolic*.zip",
        "outputs_fes6_step07_symbolic*",
        "outputs_fes6_step06_kp*.zip",
        "outputs_fes6_step06_kp*",
        "outputs_fes6_step05_no_go*.zip",
        "outputs_fes6_step05_no_go*",
    )

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))

    valid: list[Path] = []
    for candidate in candidates:
        try:
            source = DataSource(candidate)
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
        if source.find_basename(STRICT_TABLES) is not None:
            valid.append(candidate)

    if not valid:
        raise FileNotFoundError(
            "No FES Step07/Step06/Step05 output was found. "
            "Please set input_source explicitly."
        )

    return max(valid, key=lambda path: path.stat().st_mtime)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{table_name} is missing columns: {missing}\n"
            f"Available columns: {list(frame.columns)}"
        )


def choose_chern_column(frame: pd.DataFrame) -> str:
    for column in (
        "verified_chern_up_int",
        "chern_up_int",
    ):
        if column in frame.columns:
            return column
    raise KeyError(
        "FES table lacks verified_chern_up_int/chern_up_int."
    )


def choose_gap_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "verified_indirect_gap",
        "indirect_gap",
    ):
        if column in frame.columns:
            return column
    return None


def prepare_table(
    frame: pd.DataFrame,
    role: str,
) -> tuple[pd.DataFrame, str, str | None]:
    require_columns(frame, REQUIRED_COLUMNS, role)
    chern_column = choose_chern_column(frame)
    gap_column = choose_gap_column(frame)

    out = frame.copy()
    out["M_Gamma_FES"] = numeric(out, "mass_Gamma_signed")
    out["M_M_FES"] = numeric(out, "mass_M_signed")
    out["chern_up_plot"] = numeric(out, chern_column)
    out["chi_plot"] = np.sign(
        numeric(out, "r1") * numeric(out, "r2")
    )

    out = out.dropna(
        subset=[
            "M_Gamma_FES",
            "M_M_FES",
            "chern_up_plot",
            "chi_plot",
        ]
    ).copy()

    out["chi_plot"] = out["chi_plot"].astype(int)
    out["chern_up_plot"] = out["chern_up_plot"].round().astype(int)
    out["source_table_role"] = role

    return out, chern_column, gap_column


def calibrate_prefactor(
    strict: pd.DataFrame,
    summary: dict,
) -> tuple[int, dict]:
    try:
        json_prefactor = int(
            summary.get("calibrated_formula_prefactor", 0)
        )
    except (TypeError, ValueError):
        json_prefactor = 0

    base = (
        strict["chi_plot"].to_numpy(float)
        * 0.5
        * (
            np.sign(strict["M_Gamma_FES"].to_numpy(float))
            + np.sign(strict["M_M_FES"].to_numpy(float))
        )
    )
    actual = strict["chern_up_plot"].to_numpy(float)

    valid = (
        np.isfinite(base)
        & np.isfinite(actual)
        & (np.abs(base) > 0)
    )

    if valid.any():
        accuracy_plus = float(
            np.mean(base[valid] == actual[valid])
        )
        accuracy_minus = float(
            np.mean(-base[valid] == actual[valid])
        )
        data_prefactor = (
            1 if accuracy_plus >= accuracy_minus else -1
        )
    else:
        accuracy_plus = float("nan")
        accuracy_minus = float("nan")
        data_prefactor = 1

    if json_prefactor in (-1, 1):
        prefactor = json_prefactor
        source = "summary_json"
    else:
        prefactor = data_prefactor
        source = "strict_data_calibration"

    return int(prefactor), {
        "prefactor": int(prefactor),
        "prefactor_source": source,
        "strict_nonzero_rows_used": int(valid.sum()),
        "accuracy_if_prefactor_plus": accuracy_plus,
        "accuracy_if_prefactor_minus": accuracy_minus,
    }


def load_fes_data(
    input_source: Path | str,
    point_source: str,
) -> dict:
    source = DataSource(Path(input_source))

    strict_raw, strict_name = source.read_csv(
        STRICT_TABLES,
        required=True,
    )
    assert strict_raw is not None
    strict, strict_chern_column, strict_gap_column = prepare_table(
        strict_raw,
        "strict",
    )

    coarse_raw, coarse_name = source.read_csv(
        COARSE_TABLES,
        required=False,
    )

    if point_source == "coarse" and coarse_raw is not None:
        points, point_chern_column, point_gap_column = prepare_table(
            coarse_raw,
            "coarse",
        )
        point_table_name = coarse_name
    else:
        points = strict.copy()
        point_chern_column = strict_chern_column
        point_gap_column = strict_gap_column
        point_table_name = strict_name

    summary, summary_name = source.read_json_optional(
        SUMMARY_JSONS
    )
    prefactor, prefactor_audit = calibrate_prefactor(
        strict,
        summary,
    )

    for frame in (strict, points):
        frame["chern_up_analytic"] = (
            prefactor
            * frame["chi_plot"]
            * 0.5
            * (
                np.sign(frame["M_Gamma_FES"])
                + np.sign(frame["M_M_FES"])
            )
        ).round().astype(int)

        frame["formula_match_plot"] = (
            frame["chern_up_analytic"]
            == frame["chern_up_plot"]
        ).astype(int)

    return {
        "source": source.path,
        "strict": strict,
        "points": points,
        "strict_table_name": strict_name,
        "point_table_name": point_table_name,
        "summary_name": summary_name,
        "strict_chern_column": strict_chern_column,
        "point_chern_column": point_chern_column,
        "strict_gap_column": strict_gap_column,
        "point_gap_column": point_gap_column,
        "prefactor": prefactor,
        "prefactor_audit": prefactor_audit,
    }


def symmetric_limit(
    *series: Iterable[float],
    padding: float = 1.08,
    quantile: float = 0.998,
) -> float:
    arrays: list[np.ndarray] = []
    for values in series:
        arr = pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        ).to_numpy(float)
        arr = np.abs(arr[np.isfinite(arr)])
        if len(arr):
            arrays.append(arr)

    if not arrays:
        return 1.0

    combined = np.concatenate(arrays)
    base = float(np.quantile(combined, quantile))
    return max(base * padding, 1.0e-6)


def deterministic_stratified_subset(
    frame: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.copy()

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    groups = list(frame.groupby("chern_up_plot", sort=True))
    total = len(frame)

    for group_index, (_, group) in enumerate(groups):
        if group_index == len(groups) - 1:
            target = maximum - len(selected)
        else:
            target = max(
                1,
                int(round(maximum * len(group) / total)),
            )
            target = min(
                target,
                maximum - len(selected),
            )

        choices = rng.choice(
            group.index.to_numpy(),
            size=min(target, len(group)),
            replace=False,
        )
        selected.extend(int(item) for item in choices)

    return frame.loc[selected].copy()


def add_region_labels(
    ax: plt.Axes,
    limit: float,
    chirality: int,
    prefactor: int,
) -> None:
    c_top_right = prefactor * chirality
    c_bottom_left = -prefactor * chirality

    box = dict(
        boxstyle="round,pad=0.20",
        facecolor="white",
        edgecolor="none",
        alpha=0.72,
    )

    ax.text(
        0.58 * limit,
        0.58 * limit,
        rf"$C_\uparrow={c_top_right:+d}$",
        ha="center",
        va="center",
        bbox=box,
    )
    ax.text(
        -0.58 * limit,
        -0.58 * limit,
        rf"$C_\uparrow={c_bottom_left:+d}$",
        ha="center",
        va="center",
        bbox=box,
    )
    ax.text(
        -0.58 * limit,
        0.58 * limit,
        r"$C_\uparrow=0$",
        ha="center",
        va="center",
        bbox=box,
    )
    ax.text(
        0.58 * limit,
        -0.58 * limit,
        r"$C_\uparrow=0$",
        ha="center",
        va="center",
        bbox=box,
    )


def save_figure(
    fig: plt.Figure,
    stem: Path,
    png_dpi: int,
) -> dict[str, Path]:
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


def plot_fes_mass_plane(
    points: pd.DataFrame,
    strict: pd.DataFrame,
    chirality: int,
    prefactor: int,
    output_dir: Path | str,
    png_dpi: int,
    maximum_display_points: int | None,
    show_gap_status: bool,
    gap_column: str | None,
) -> dict:
    subset_all = points[points["chi_plot"] == chirality].copy()
    strict_subset = strict[strict["chi_plot"] == chirality].copy()
    if subset_all.empty:
        raise ValueError(f"No FES samples for chi={chirality:+d}.")

    subset = deterministic_stratified_subset(
        subset_all,
        maximum=maximum_display_points,
        seed=20260729 + chirality,
    )

    limit = symmetric_limit(
        subset_all["M_Gamma_FES"],
        subset_all["M_M_FES"],
        strict_subset["M_Gamma_FES"],
        strict_subset["M_M_FES"],
    )

    axis = np.linspace(-limit, limit, 401)
    m_gamma_grid, m_m_grid = np.meshgrid(axis, axis)
    chern_grid = (
        prefactor
        * chirality
        * 0.5
        * (
            np.sign(m_gamma_grid)
            + np.sign(m_m_grid)
        )
    )

    # Initialize boundary pixels with the neutral region color.
    # At M_Gamma=0 or M_M=0, the sign formula can produce half-integer
    # intermediate values; these pixels are phase boundaries and are later
    # covered by black dashed lines.
    region_rgb = np.empty(
        chern_grid.shape + (3,),
        dtype=float,
    )
    region_rgb[...] = REGION_FILL[0]
    for value, color in REGION_FILL.items():
        region_rgb[np.isclose(chern_grid, value)] = color

    fig, ax = plt.subplots(
        figsize=(8.0 / 2.54, 7.0 / 2.54)
    )

    ax.imshow(
        region_rgb,
        extent=[-limit, limit, -limit, limit],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=0,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        zorder=2,
    )
    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        zorder=2,
    )

    for c_value in (-1, 0, +1):
        group = subset[subset["chern_up_plot"] == c_value]
        if group.empty:
            continue

        if (
            show_gap_status
            and gap_column is not None
            and gap_column in group.columns
        ):
            gap = pd.to_numeric(group[gap_column], errors="coerce")
            positive = group[gap > 0]
            nonpositive = group[gap <= 0]

            if len(positive):
                ax.scatter(
                    positive["M_Gamma_FES"],
                    positive["M_M_FES"],
                    s=17,
                    facecolors="white",
                    edgecolors=[POINT_EDGE[c_value]],
                    linewidths=0.75,
                    alpha=0.92,
                    zorder=3,
                )
            if len(nonpositive):
                ax.scatter(
                    nonpositive["M_Gamma_FES"],
                    nonpositive["M_M_FES"],
                    s=15,
                    marker="x",
                    color=[POINT_EDGE[c_value]],
                    linewidths=0.70,
                    alpha=0.82,
                    zorder=3,
                )
        else:
            ax.scatter(
                group["M_Gamma_FES"],
                group["M_M_FES"],
                s=14,
                facecolors="white",
                edgecolors=[POINT_EDGE[c_value]],
                linewidths=0.55,
                alpha=0.88,
                zorder=3,
            )

    add_region_labels(
        ax,
        limit=limit,
        chirality=chirality,
        prefactor=prefactor,
    )

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(
        r"$M_{\Gamma}^{\mathrm{FES}}"
        r"=\mu_{\Gamma}^{\mathrm{signed}}$"
    )
    ax.set_ylabel(
        r"$M_{M}^{\mathrm{FES}}"
        r"=\mu_{M}^{\mathrm{signed}}$"
    )

    ax.text(
        0.03,
        0.97,
        rf"$\chi_{{\mathrm{{FES}}}}"
        rf"=\mathrm{{sgn}}(r_1r_2)"
        rf"={chirality:+d}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
        ),
    )

    region_handles = [
        Patch(
            facecolor=REGION_FILL[+1],
            edgecolor="none",
            label=r"Region: $C_\uparrow=+1$",
        ),
        Patch(
            facecolor=REGION_FILL[0],
            edgecolor="none",
            label=r"Region: $C_\uparrow=0$",
        ),
        Patch(
            facecolor=REGION_FILL[-1],
            edgecolor="none",
            label=r"Region: $C_\uparrow=-1$",
        ),
    ]
    region_legend = ax.legend(
        handles=region_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
    )
    ax.add_artist(region_legend)

    if show_gap_status and gap_column is not None:
        point_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=5,
                label=r"Numerical: $E_g^{\mathrm{ind}}>0$",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                color="black",
                markeredgewidth=0.9,
                markersize=5,
                label=r"Numerical: $E_g^{\mathrm{ind}}\leq0$",
            ),
        ]
    else:
        point_handles = []
        for c_value in (+1, 0, -1):
            label = (
                rf"Numerical: $C_\uparrow={c_value:+d}$"
                if c_value != 0
                else r"Numerical: $C_\uparrow=0$"
            )
            point_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="white",
                    markeredgecolor=POINT_EDGE[c_value],
                    markeredgewidth=0.9,
                    markersize=5,
                    label=label,
                )
            )

    ax.legend(
        handles=point_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
    )

    fig.tight_layout()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_gap_status" if show_gap_status else ""
    stem = (
        output_dir
        / f"fes_mass_plane_chi_{chirality:+d}{suffix}"
    )
    paths = save_figure(
        fig,
        stem=stem,
        png_dpi=png_dpi,
    )

    display_csv = (
        output_dir
        / f"fes_mass_plane_display_points_chi_{chirality:+d}{suffix}.csv"
    )
    subset.to_csv(display_csv, index=False)

    return {
        "paths": paths,
        "display_csv": display_csv,
        "n_available_points": int(len(subset_all)),
        "n_displayed_points": int(len(subset)),
        "n_strict_points": int(len(strict_subset)),
    }


def run_fes_mass_plane(
    input_source: Path | str | None = None,
    project_root: Path | str = ".",
    output_dir: Path | str = "fes_mass_plane_paper",
    point_source: str = "coarse",
    png_dpi: int = 2400,
    generate_chi_minus: bool = True,
    maximum_display_points: int | None = 1400,
    generate_gap_status_version: bool = True,
) -> dict:
    configure_plot_style()

    if point_source not in {"coarse", "strict"}:
        raise ValueError(
            "point_source must be 'coarse' or 'strict'."
        )

    if input_source is None:
        input_source = discover_fes_input(project_root)

    loaded = load_fes_data(
        input_source=input_source,
        point_source=point_source,
    )

    points = loaded["points"]
    strict = loaded["strict"]
    prefactor = int(loaded["prefactor"])

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points.to_csv(
        output_dir / "fes_signed_mass_plane_points_all.csv",
        index=False,
    )
    strict.to_csv(
        output_dir / "fes_signed_mass_plane_points_strict.csv",
        index=False,
    )

    mismatches = points[
        points["formula_match_plot"] == 0
    ].copy()
    mismatches.to_csv(
        output_dir / "fes_signed_mass_formula_mismatches.csv",
        index=False,
    )

    figures: dict[int, dict] = {}
    chiralities = [1, -1] if generate_chi_minus else [1]

    for chirality in chiralities:
        if not (points["chi_plot"] == chirality).any():
            continue

        standard = plot_fes_mass_plane(
            points=points,
            strict=strict,
            chirality=chirality,
            prefactor=prefactor,
            output_dir=output_dir,
            png_dpi=png_dpi,
            maximum_display_points=maximum_display_points,
            show_gap_status=False,
            gap_column=loaded["point_gap_column"],
        )
        figures[chirality] = {"standard": standard}

        if (
            generate_gap_status_version
            and loaded["point_gap_column"] is not None
        ):
            gap_version = plot_fes_mass_plane(
                points=points,
                strict=strict,
                chirality=chirality,
                prefactor=prefactor,
                output_dir=output_dir,
                png_dpi=png_dpi,
                maximum_display_points=maximum_display_points,
                show_gap_status=True,
                gap_column=loaded["point_gap_column"],
            )
            figures[chirality]["gap_status"] = gap_version

    summary = {
        "script_version": SCRIPT_VERSION,
        "input_source": str(loaded["source"]),
        "strict_table_name": loaded["strict_table_name"],
        "point_table_name": loaded["point_table_name"],
        "point_source": point_source,
        "formula": (
            "C_up = eta * chi_FES / 2 * "
            "[sgn(M_Gamma_FES) + sgn(M_M_FES)]"
        ),
        "chi_definition": "chi_FES = sgn(r1*r2)",
        "axis_x": "M_Gamma_FES = mass_Gamma_signed",
        "axis_y": "M_M_FES = mass_M_signed",
        "prefactor_audit": loaded["prefactor_audit"],
        "n_points_total": int(len(points)),
        "n_strict_points": int(len(strict)),
        "n_formula_mismatches_in_point_table": int(len(mismatches)),
        "point_formula_accuracy": float(
            points["formula_match_plot"].mean()
        ),
        "interpretation": (
            "Nonzero FES regions are formal spin-Chern sectors. "
            "Under the minimal-model indirect-gap no-go they are not "
            "QSH-insulator regions."
        ),
    }

    summary_path = output_dir / "fes_mass_plane_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
        description=(
            "Generate a Lieb-style four-quadrant FES signed-mass "
            "phase diagram."
        )
    )
    parser.add_argument(
        "--input-source",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fes_mass_plane_paper"),
    )
    parser.add_argument(
        "--point-source",
        choices=["coarse", "strict"],
        default="coarse",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=2400,
    )
    parser.add_argument(
        "--maximum-display-points",
        type=int,
        default=1400,
    )
    parser.add_argument(
        "--no-chi-minus",
        action="store_true",
    )
    parser.add_argument(
        "--no-gap-status-version",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_fes_mass_plane(
        input_source=args.input_source,
        project_root=args.project_root,
        output_dir=args.output_dir,
        point_source=args.point_source,
        png_dpi=args.png_dpi,
        generate_chi_minus=not args.no_chi_minus,
        maximum_display_points=args.maximum_display_points,
        generate_gap_status_version=not args.no_gap_status_version,
    )

    print("Input source:", result["input_source"])
    print("Formula prefactor eta:", result["prefactor"])
    print("Processed points:", len(result["points"]))
    print("Strict points:", len(result["strict"]))
    print("Output directory:", result["output_dir"])
