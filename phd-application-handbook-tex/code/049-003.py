from __future__ import annotations

"""
Paper-ready M+–M− phase diagram for the Lieb model.

Main purpose
------------
Generate the signed-mass phase diagram in a publication-friendly style,
using the user's requested color scheme and Times New Roman font.

Topological masses
------------------
    M_plus  = m_e + d1 + d2
    M_minus = -m_e + d1 + d2

For a fixed chirality chi = sgn(t1*t2),
    C_up = -chi/2 * [sgn(M_plus) + sgn(M_minus)]

This script is designed for the main-text figure, especially chi = +1.
It can still generate both chirality sectors if needed.

Input
-----
The script reads an existing Lieb Step09 result directory or ZIP:
    outputs_step09_analytic_chern_formula_<run_tag>/
        step09_03_analytic_chern_prediction_master.csv
or the corresponding ZIP file.

If Step09 is unavailable, Step08 can be used as a fallback:
    outputs_step08_analytic_boundary_wanniertools_edge_<run_tag>/
        step08_01_analytic_feature_master.csv

Output
------
PNG (high dpi), PDF and SVG.
"""

import argparse
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.patches import Patch
from matplotlib.text import Text


# ============================================================
# 1. Constants and requested colors
# ============================================================
STEP09_PREFIX = "outputs_step09_analytic_chern_formula_"
STEP08_PREFIX = "outputs_step08_analytic_boundary_wanniertools_edge_"
STEP09_CSV = "step09_03_analytic_chern_prediction_master.csv"
STEP08_CSV = "step08_01_analytic_feature_master.csv"
REQUIRED_COLUMNS = ("m_e", "d1", "d2", "t1", "t2", "chern_up_int")

# User-requested region colors
COLOR_RED = (169 / 255.0, 36 / 255.0, 37 / 255.0)     # C_up = +1
COLOR_BLUE = (63 / 255.0, 99 / 255.0, 173 / 255.0)    # C_up = -1
COLOR_GREY = (211 / 255.0, 211 / 255.0, 211 / 255.0)  # C_up = 0


def lighten(color: tuple[float, float, float], factor: float = 0.78) -> tuple[float, float, float]:
    """Blend color toward white for region backgrounds."""
    base = np.array(color, dtype=float)
    white = np.ones(3, dtype=float)
    mixed = base * (1 - factor) + white * factor
    return tuple(np.clip(mixed, 0.0, 1.0))


REGION_FILL = {
    +1: lighten(COLOR_RED, 0.70),
    0: lighten(COLOR_GREY, 0.15),
    -1: lighten(COLOR_BLUE, 0.68),
}

# Points should be visually distinct from the regions.
# Use white-filled markers with colored edges.
POINT_EDGE = {
    +1: COLOR_RED,
    0: (0.25, 0.25, 0.25),
    -1: COLOR_BLUE,
}


# ============================================================
# 2. Plot style
# ============================================================
def configure_plot_style() -> str:
    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    font_name = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"

    rcParams["font.family"] = font_name
    rcParams["font.weight"] = "bold"
    rcParams["font.size"] = 9
    rcParams["axes.labelsize"] = 9
    rcParams["axes.labelweight"] = "bold"
    rcParams["axes.titlesize"] = 9
    rcParams["axes.titleweight"] = "bold"
    rcParams["figure.titleweight"] = "bold"
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["legend.fontsize"] = 8
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.rm"] = f"{font_name}:bold"
    rcParams["mathtext.it"] = f"{font_name}:italic:bold"
    rcParams["mathtext.bf"] = f"{font_name}:bold"
    rcParams["mathtext.sf"] = f"{font_name}:bold"
    rcParams["mathtext.tt"] = f"{font_name}:bold"
    rcParams["mathtext.fallback"] = None
    rcParams["axes.linewidth"] = 0.8
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.top"] = True
    rcParams["ytick.right"] = True
    rcParams["svg.fonttype"] = "none"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    return font_name


def force_bold_times_text(fig: plt.Figure, font_name: str) -> None:
    """Apply the requested font family and weight to every text artist."""
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily(font_name)
        artist.set_fontweight("bold")


# ============================================================
# 3. Data discovery and reading
# ============================================================
def zip_contains(archive_path: Path | str, basename: str) -> bool:
    archive_path = Path(archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            return any(Path(name).name == basename for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def candidate_roots(project_root: Path) -> list[Path]:
    roots = [
        project_root,
        project_root / "lieb",
        Path.cwd(),
        Path.cwd() / "lieb",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "lieb",
    ]
    for path in list(roots):
        roots.append(path.parent)

    unique = []
    seen = set()
    for path in roots:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def discover_lieb_input(project_root: Path | str = ".", allow_step08_fallback: bool = True) -> Path:
    project_root = Path(project_root).expanduser().resolve()
    roots = candidate_roots(project_root)

    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted(root.glob(f"{STEP09_PREFIX}*")):
            if directory.is_dir() and (directory / STEP09_CSV).is_file():
                return directory
        for archive in sorted(root.glob(f"{STEP09_PREFIX}*.zip")):
            if archive.is_file() and zip_contains(archive, STEP09_CSV):
                return archive

    if project_root.is_dir():
        step09_csv_matches = sorted(project_root.rglob(STEP09_CSV))
        if step09_csv_matches:
            return step09_csv_matches[0].parent
        step09_zip_matches = sorted(project_root.rglob(f"{STEP09_PREFIX}*.zip"))
        for archive in step09_zip_matches:
            if zip_contains(archive, STEP09_CSV):
                return archive

    if allow_step08_fallback:
        for root in roots:
            if not root.is_dir():
                continue
            for directory in sorted(root.glob(f"{STEP08_PREFIX}*")):
                if directory.is_dir() and (directory / STEP08_CSV).is_file():
                    return directory
            for archive in sorted(root.glob(f"{STEP08_PREFIX}*.zip")):
                if archive.is_file() and zip_contains(archive, STEP08_CSV):
                    return archive

        if project_root.is_dir():
            step08_csv_matches = sorted(project_root.rglob(STEP08_CSV))
            if step08_csv_matches:
                return step08_csv_matches[0].parent
            step08_zip_matches = sorted(project_root.rglob(f"{STEP08_PREFIX}*.zip"))
            for archive in step08_zip_matches:
                if zip_contains(archive, STEP08_CSV):
                    return archive

    raise FileNotFoundError(
        "No valid Lieb Step09/Step08 output was found.\n"
        "Expected Step09 file: step09_03_analytic_chern_prediction_master.csv"
    )


def _read_csv_from_zip(archive_path: Path, preferred_names: Iterable[str]) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for basename in preferred_names:
            matches = [name for name in names if Path(name).name == basename]
            if matches:
                selected = sorted(matches, key=lambda name: (len(Path(name).parts), len(name)))[0]
                with archive.open(selected) as stream:
                    return pd.read_csv(stream, low_memory=False), basename
    raise FileNotFoundError(f"Could not find any of {list(preferred_names)} in {archive_path}")


def read_lieb_output(input_source: Path | str) -> tuple[pd.DataFrame, dict]:
    source = Path(input_source).expanduser().resolve()
    preferred = (STEP09_CSV, STEP08_CSV)

    if source.is_file() and source.suffix.lower() == ".zip":
        frame, csv_name = _read_csv_from_zip(source, preferred)
        return frame, {"input_source": str(source), "input_kind": "zip", "csv_name": csv_name}

    if source.is_dir():
        for csv_name in preferred:
            direct = source / csv_name
            if direct.is_file():
                return pd.read_csv(direct, low_memory=False), {
                    "input_source": str(source),
                    "input_kind": "directory",
                    "csv_name": csv_name,
                    "csv_path": str(direct),
                }
        for csv_name in preferred:
            matches = sorted(source.rglob(csv_name))
            if matches:
                return pd.read_csv(matches[0], low_memory=False), {
                    "input_source": str(source),
                    "input_kind": "directory_recursive",
                    "csv_name": csv_name,
                    "csv_path": str(matches[0]),
                }

    raise FileNotFoundError(f"Cannot read a valid Lieb output from: {source}")


# ============================================================
# 4. Data preparation
# ============================================================
def prepare_lieb_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

    out = df.copy()
    for column in REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=list(REQUIRED_COLUMNS)).copy()

    out["M_plus_plot"] = out["m_e"] + out["d1"] + out["d2"]
    out["M_minus_plot"] = -out["m_e"] + out["d1"] + out["d2"]
    out["chi_plot"] = np.sign(out["t1"] * out["t2"]).astype(int)
    out["chern_up_analytic_plot"] = (
        -0.5 * out["chi_plot"] * (np.sign(out["M_plus_plot"]) + np.sign(out["M_minus_plot"]))
    )
    out["chern_up_int"] = out["chern_up_int"].astype(int)
    return out


def symmetric_limit(*series: Iterable[float], padding: float = 1.08) -> float:
    arrays = []
    for values in series:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        numeric = numeric[np.isfinite(numeric)]
        if len(numeric):
            arrays.append(np.abs(numeric.to_numpy(float)))
    if not arrays:
        return 1.0
    return padding * float(np.max(np.concatenate(arrays)))


# ============================================================
# 5. Plotting helpers
# ============================================================
def add_region_labels(ax: plt.Axes, limit: float, chirality: int) -> None:
    """Add compact publication-style annotations."""
    # For chi = +1:
    #   top-right  -> C=-1 (blue)
    #   bottom-left-> C=+1 (red)
    # For chi = -1 signs are reversed.
    c_tr = -chirality
    c_bl = chirality

    text_box = dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.72)

    ax.text(0.58 * limit, 0.58 * limit, rf"$C_\uparrow={c_tr:+d}$", ha="center", va="center", bbox=text_box)
    ax.text(-0.58 * limit, -0.58 * limit, rf"$C_\uparrow={c_bl:+d}$", ha="center", va="center", bbox=text_box)
    ax.text(-0.58 * limit, 0.58 * limit, r"$C_\uparrow=0$", ha="center", va="center", bbox=text_box)
    ax.text(0.58 * limit, -0.58 * limit, r"$C_\uparrow=0$", ha="center", va="center", bbox=text_box)


# ============================================================
# 6. Main plotting function
# ============================================================
def plot_mass_plane(
    data: pd.DataFrame,
    chirality: int,
    output_dir: Path | str,
    png_dpi: int = 2400,
    show_title: bool = False,
    font_name: str = "Times New Roman",
) -> dict[str, Path]:
    if chirality not in (-1, 1):
        raise ValueError("chirality must be +1 or -1")

    subset = data[data["chi_plot"] == chirality].copy()
    if subset.empty:
        raise ValueError(f"No samples for chi={chirality:+d}")

    limit = symmetric_limit(subset["M_plus_plot"], subset["M_minus_plot"])
    fig, ax = plt.subplots(figsize=(8.0 / 2.54, 7.0 / 2.54))

    # Background regions using the requested paper colors.
    # Determine which Chern sector occupies each quadrant for this chirality.
    # Use filled rectangles via ax.axvspan/ax.axhspan combos by contour-like pcolormesh.
    axis = np.linspace(-limit, limit, 401)
    m_plus_grid, m_minus_grid = np.meshgrid(axis, axis)
    chern_grid = -0.5 * chirality * (np.sign(m_plus_grid) + np.sign(m_minus_grid))

    # Manual facecolor map ensures exact requested hues.
    region_rgb = np.empty(chern_grid.shape + (3,), dtype=float)
    for value, color in REGION_FILL.items():
        mask = np.isclose(chern_grid, value)
        region_rgb[mask] = color

    ax.imshow(
        region_rgb,
        extent=[-limit, limit, -limit, limit],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=0,
    )

    # Boundary lines
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="black", zorder=2)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black", zorder=2)

    # Points: use white fill + colored edges for strong contrast with the colored regions.
    for c_val in (-1, 0, 1):
        group = subset[subset["chern_up_int"] == c_val]
        if len(group) == 0:
            continue
        ax.scatter(
            group["M_plus_plot"],
            group["M_minus_plot"],
            s=18,
            facecolors="white",
            edgecolors=[POINT_EDGE[c_val]],
            linewidths=0.8,
            alpha=1.0,
            zorder=3,
        )

    add_region_labels(ax, limit, chirality)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$M_{+}=m_e+d_1+d_2$")
    ax.set_ylabel(r"$M_{-}=-m_e+d_1+d_2$")

    # Main-text friendly expression: no long title; only a compact chi tag.
    if show_title:
        ax.set_title(rf"Signed-mass phase diagram ($\chi={chirality:+d}$)")
    else:
        ax.text(
            0.03,
            0.97,
            rf"$\chi={chirality:+d}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.82),
        )

    # Compact legend with clear separation between region meaning and numerical points.
    patches = [
        Patch(facecolor=REGION_FILL[+1], edgecolor="none", label=r"Region: $C_\uparrow=+1$"),
        Patch(facecolor=REGION_FILL[0], edgecolor="none", label=r"Region: $C_\uparrow=0$"),
        Patch(facecolor=REGION_FILL[-1], edgecolor="none", label=r"Region: $C_\uparrow=-1$"),
    ]
    leg1 = ax.legend(handles=patches, loc="lower left", frameon=True, framealpha=0.88, borderpad=0.35)
    ax.add_artist(leg1)

    # Dummy handles for point style legend.
    point_handles = []
    point_labels = []
    for c_val, label in [(+1, r"Numerical: $C_\uparrow=+1$"), (0, r"Numerical: $C_\uparrow=0$"), (-1, r"Numerical: $C_\uparrow=-1$")]:
        point_handles.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markerfacecolor='white',
                markeredgecolor=POINT_EDGE[c_val],
                markeredgewidth=0.9,
                markersize=5,
            )
        )
        point_labels.append(label)
    ax.legend(point_handles, point_labels, loc="upper right", frameon=True, framealpha=0.88, borderpad=0.35)

    force_bold_times_text(fig, font_name)
    fig.tight_layout()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"lieb_mass_plane_chi_{chirality:+d}"

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


# ============================================================
# 7. Workflow
# ============================================================
def run_paper_mass_plane(input_source: Path | str | None = None, project_root: Path | str = ".", output_dir: Path | str = "lieb_mass_plane_paper", png_dpi: int = 2400, generate_chi_minus: bool = True) -> dict:
    font_name = configure_plot_style()

    if input_source is None:
        input_source = discover_lieb_input(project_root)
    input_source = Path(input_source).expanduser().resolve()

    raw, metadata = read_lieb_output(input_source)
    data = prepare_lieb_data(raw)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_dir / "lieb_signed_mass_plane_points.csv", index=False)

    # Main-text figure: chi = +1
    figures = {
        +1: plot_mass_plane(
            data=data,
            chirality=+1,
            output_dir=output_dir,
            png_dpi=png_dpi,
            show_title=False,
            font_name=font_name,
        )
    }

    if generate_chi_minus and (data["chi_plot"] == -1).any():
        figures[-1] = plot_mass_plane(
            data=data,
            chirality=-1,
            output_dir=output_dir,
            png_dpi=png_dpi,
            show_title=False,
            font_name=font_name,
        )

    return {
        "metadata": metadata,
        "input_source": input_source,
        "data": data,
        "figures": figures,
        "output_dir": output_dir,
    }


# ============================================================
# 8. CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a paper-ready Lieb M+–M− phase diagram.")
    parser.add_argument("--input-source", type=Path, default=None, help="Step09/Step08 output directory or ZIP. If omitted, search automatically.")
    parser.add_argument("--project-root", type=Path, default=Path('.'), help="Root for automatic search when input is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("lieb_mass_plane_paper"))
    parser.add_argument("--png-dpi", type=int, default=2400)
    parser.add_argument("--no-chi-minus", action="store_true", help="Generate only chi=+1, recommended for the main-text figure.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_paper_mass_plane(
        input_source=args.input_source,
        project_root=args.project_root,
        output_dir=args.output_dir,
        png_dpi=args.png_dpi,
        generate_chi_minus=not args.no_chi_minus,
    )
    print("Input source:", result["input_source"])
    print("Detected CSV:", result["metadata"].get("csv_name"))
    print("Processed samples:", len(result["data"]))
    print("Output directory:", result["output_dir"])
