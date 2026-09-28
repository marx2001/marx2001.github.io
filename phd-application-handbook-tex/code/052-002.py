#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 19: main-text package with four non-trivial edge/Hall examples.

The main-text selection is intentionally compact:

    (a) Lieb, C_s = +1
    (b) FES,  C_s = +1
    (c) TTS,  C_s = +1
    (d) TTS,  C_s = +2

Critical and trivial examples from Step 18 remain available as supplementary
material and are not repeated here.  Each main-text case contains a
WannierTools-style semi-infinite edge spectral function, spin-resolved Hall
conductivity, bulk-band data, the exact parameter vector and validation
metadata.  Plot colors are inherited from the phase-map palettes in Step 18.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import base64
import json
import re
import struct
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

import TTS_step18_three_system_wanniertools_style_edge_ahc_examples as core


CODE_VERSION = "TTS_STEP19_MAINTEXT_FOUR_TOPOLOGICAL_V1_20260730"
DEFAULT_OUTPUT = (
    core.DEFAULT_OUTPUT / "main_text_four_topological"
)
CASE_ORDER = (
    "a_lieb_Cs_plus1",
    "b_fes_Cs_plus1",
    "c_tts_Cs_plus1",
    "d_tts_Cs_plus2",
)
PANEL_LABEL = {
    "a_lieb_Cs_plus1": "(a)",
    "b_fes_Cs_plus1": "(b)",
    "c_tts_Cs_plus1": "(c)",
    "d_tts_Cs_plus2": "(d)",
}
DISPLAY_TITLE = {
    "a_lieb_Cs_plus1": r"Lieb:  $C_s=+1$",
    "b_fes_Cs_plus1": r"FES:  $C_s=+1$",
    "c_tts_Cs_plus1": r"TTS:  $C_s=+1$",
    "d_tts_Cs_plus2": r"TTS:  $C_s=+2$",
}


def configure_step19_plot_style() -> None:
    """Times New Roman throughout, with bold italic/roman math preserved."""
    core.configure_plot_style()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman:bold",
            "mathtext.it": "Times New Roman:italic:bold",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman:bold",
            "mathtext.tt": "Times New Roman:bold",
            "mathtext.fallback": "stix",
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
        }
    )


def save_figure_inkscape_safe(
    fig: plt.Figure,
    stem: Path,
    dpi: int,
    primary_svg_editable: bool = False,
) -> list[str]:
    """Save a complete SVG plus a separately editable Matplotlib SVG.

    Some Inkscape builds intermittently omit one or more of Matplotlib's
    embedded raster layers when a figure contains several surface-spectrum
    images and inset colour bars.  The primary SVG therefore contains one
    flattened, self-contained PNG layer and also carries a relative-file
    fallback for older Inkscape versions.  The companion ``*_editable.svg``
    retains vector text, axes, and bulk-band curves.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    editable_svg_path = stem.with_name(stem.name + "_editable").with_suffix(
        ".svg"
    )
    save_kwargs = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": 0.04,
        "facecolor": "white",
    }
    fig.savefig(png_path, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(editable_svg_path, **save_kwargs)

    if primary_svg_editable:
        fig.savefig(svg_path, **save_kwargs)
        make_svg_raster_layers_inkscape_compatible(svg_path)
    else:
        png_bytes = png_path.read_bytes()
        if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Unexpected PNG header: {png_path}")
        width, height = struct.unpack(">II", png_bytes[16:24])
        encoded = base64.b64encode(png_bytes).decode("ascii")
        absolute_png = str(png_path.resolve()).replace("\\", "/")
        svg_text = (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg"\n'
            '     xmlns:xlink="http://www.w3.org/1999/xlink"\n'
            '     xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/'
            'sodipodi-0.dtd"\n'
            f'     width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '  <title>Inkscape-compatible complete figure</title>\n'
            f'  <image x="0" y="0" width="{width}" height="{height}"\n'
            '         preserveAspectRatio="xMidYMid meet"\n'
            f'         href="data:image/png;base64,{encoded}"\n'
            f'         xlink:href="{png_path.name}"\n'
            f'         sodipodi:absref="{absolute_png}"/>\n'
            '</svg>\n'
        )
        svg_path.write_text(svg_text, encoding="utf-8")
    plt.close(fig)
    return [
        str(png_path),
        str(pdf_path),
        str(svg_path),
        str(editable_svg_path),
    ]


def make_svg_raster_layers_inkscape_compatible(svg_path: Path) -> list[Path]:
    """Give each scientific raster layer an external fallback for Inkscape."""
    svg_text = svg_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'xlink:href="data:image/png;base64,\s*([^"]+)"',
        flags=re.DOTALL,
    )
    layer_paths: list[Path] = []
    layer_index = 0

    def replace_layer(match: re.Match[str]) -> str:
        nonlocal layer_index
        layer_index += 1
        encoded = re.sub(r"\s+", "", match.group(1))
        layer_path = svg_path.with_name(
            f"{svg_path.stem}_raster_layer_{layer_index:02d}.png"
        )
        layer_path.write_bytes(base64.b64decode(encoded))
        layer_paths.append(layer_path)
        return (
            f'href="data:image/png;base64,{encoded}" '
            f'xlink:href="{layer_path.name}"'
        )

    updated = pattern.sub(replace_layer, svg_text)
    if layer_index == 0:
        raise RuntimeError(f"No raster layer found in editable SVG: {svg_path}")
    # Keep STIXGeneral on MathText accent/symbol glyphs.  Matplotlib encodes the
    # overbar and parallel sign as STIX private-use glyphs; changing their font
    # family makes Inkscape display a circled stroke or a short minus instead.
    svg_path.write_text(updated, encoding="utf-8")
    return layer_paths


def save_editable_svg_only(
    fig: plt.Figure,
    svg_path: Path,
    dpi: int,
) -> list[str]:
    """Update only one editable SVG, leaving PNG/PDF/other figures untouched."""
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        svg_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    make_svg_raster_layers_inkscape_compatible(svg_path)
    plt.close(fig)
    return [str(svg_path)]


def clean_params(values: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    return {name: float(values[name]) for name in names}


def select_lieb_main(lieb: Any) -> core.SelectedState:
    master_path = (
        core.LIEB_ROOT
        / "outputs_step08_analytic_boundary_wanniertools_edge_msg123342_lieb8_from_step06_n1024_pairs4_seed20260711"
        / "step08_01_analytic_feature_master.csv"
    )
    master = pd.read_csv(master_path)
    candidates = master[
        (pd.to_numeric(master["chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(master["gap_convergence_ok"], errors="coerce") == 1)
        & (pd.to_numeric(master["chern_up_int"], errors="coerce") == 1)
        & (pd.to_numeric(master["final_indirect_gap"], errors="coerce") > 0)
    ].copy()
    if candidates.empty:
        raise RuntimeError("No reliable insulating Lieb C_s=+1 candidate")
    # The largest-gap candidate (about 1.88 eV) makes the bulk/edge panel look
    # artificially empty.  Select a clean, converged intermediate-gap example
    # close to the roughly 0.5 eV gap scale in the user's reference figure.
    target_gap = 0.50
    candidates["display_gap_score"] = (
        (
            pd.to_numeric(candidates["final_indirect_gap"], errors="coerce")
            - target_gap
        ).abs()
        + 0.25
        * (
            pd.to_numeric(candidates["final_direct_gap"], errors="coerce")
            - pd.to_numeric(
                candidates["final_indirect_gap"], errors="coerce"
            )
        ).abs()
    )
    row = candidates.sort_values(
        ["display_gap_score", "final_direct_gap"],
        ascending=[True, False],
    ).iloc[0]
    physical = [float(row[name]) for name in lieb.PHYS7]
    params = clean_params(lieb.phys7_to_raw8(physical), tuple(lieb.RAW8))
    return core.SelectedState(
        system="lieb",
        state="topological",
        source_id=str(row["sample_id"]),
        path_id="lieb_step08_intermediate_gap_Cs_plus1_near_0p5eV",
        params=params,
        expected_chern_up=1,
        expected_chern_down=-1,
        critical_kx=float(row["refined_direct_kx"]),
        critical_ky=float(row["refined_direct_ky"]),
        selection_note=(
            "Reliable converged Lieb C_s=+1 insulator selected near a 0.5 eV "
            "indirect gap for visual comparability with the reference panel; "
            "the previous maximum-gap example was intentionally not used."
        ),
    )


def select_fes_main(fes: Any) -> core.SelectedState:
    strict_path = (
        core.FES_ROOT
        / "outputs_fes6_step04_boundary"
        / "fes_step04_strict_verified_results.csv"
    )
    strict = pd.read_csv(strict_path)
    candidates = strict[
        (pd.to_numeric(strict["verified_chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(strict["verified_chern_up_int"], errors="coerce") == 1)
    ].copy()
    if candidates.empty:
        raise RuntimeError("No reliable FES C_s=+1 candidate")
    row = candidates.sort_values(
        ["verified_min_direct_gap", "verified_indirect_gap"],
        ascending=False,
    ).iloc[0]
    reduced = {
        name: float(row[name])
        for name in ("m_e", "t1", "t2", "r1", "r2")
    }
    params = clean_params(fes.raw6_from_reduced5(reduced), tuple(fes.RAW6))
    return core.SelectedState(
        system="fes",
        state="topological",
        source_id=str(row["point_id"]),
        path_id="fes_step04_largest_verified_direct_gap_Cs_plus1",
        params=params,
        expected_chern_up=1,
        expected_chern_down=-1,
        critical_kx=float(row["verified_direct_gap_kx"]),
        critical_ky=float(row["verified_direct_gap_ky"]),
        selection_note=(
            "Largest verified direct gap among reliable FES C_s=+1 samples. "
            "Its negative indirect gap makes it a spin-Chern band metal, not "
            "a globally insulating quantized Hall plateau."
        ),
    )


def select_tts_main(tts: Any, chern_up: int) -> core.SelectedState:
    representatives_path = (
        core.TTS_ROOT
        / "outputs_tts_step07_observable_validation"
        / "step07_01_selected_observable_representatives.csv"
    )
    representatives = pd.read_csv(representatives_path)
    candidates = representatives[
        pd.to_numeric(
            representatives["expected_chern_up"], errors="coerce"
        ) == int(chern_up)
    ]
    if candidates.empty:
        raise RuntimeError(f"No TTS C_s={chern_up:+d} representative")
    row = candidates.sort_values(
        ["indirect_gap", "min_direct_gap"],
        ascending=False,
    ).iloc[0]
    reduced = {
        name: float(row[name])
        for name in tts.REDUCED7
    }
    params = clean_params(tts.raw8_from_reduced7(reduced), tuple(tts.RAW8))
    return core.SelectedState(
        system="tts",
        state="topological",
        source_id=str(row["sample_id"]),
        path_id=f"tts_step07_largest_strict_indirect_gap_Cs_{chern_up:+d}",
        params=params,
        expected_chern_up=int(chern_up),
        expected_chern_down=-int(chern_up),
        critical_kx=float(row["direct_gap_kx"]),
        critical_ky=float(row["direct_gap_ky"]),
        selection_note=str(row["selection_reason"]),
    )


def select_main_cases(modules: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [
        {
            "case_id": "a_lieb_Cs_plus1",
            "selected": select_lieb_main(modules["lieb"]),
        },
        {
            "case_id": "b_fes_Cs_plus1",
            "selected": select_fes_main(modules["fes"]),
        },
        {
            "case_id": "c_tts_Cs_plus1",
            "selected": select_tts_main(modules["tts"], +1),
        },
        {
            "case_id": "d_tts_Cs_plus2",
            "selected": select_tts_main(modules["tts"], +2),
        },
    ]
    if tuple(case["case_id"] for case in cases) != CASE_ORDER:
        raise RuntimeError("Unexpected main-text case ordering")
    return cases


def choose_case_window(system: str, direct_gap: float) -> float:
    if system == "lieb":
        return float(np.clip(0.62 * direct_gap + 0.08, 0.55, 1.20))
    if system == "fes":
        return 0.10
    return 0.60


def classification(audit: dict[str, Any]) -> str:
    if int(audit["is_global_insulator"]) == 1:
        return "spin_chern_insulator"
    if int(audit["is_direct_gapped"]) == 1:
        return "spin_chern_band_metal"
    return "gapless"


def title_for(case_id: str, panel: bool = True) -> str:
    prefix = PANEL_LABEL[case_id] + "  " if panel else ""
    return prefix + DISPLAY_TITLE[case_id]


def draw_surface(
    ax: plt.Axes,
    result: dict[str, Any],
    specs: dict[str, core.ModelSpec],
    title: str,
) -> None:
    selected = result["selected"]
    spec = specs[selected.system]
    rgb = core.surface_rgb(spec, selected, result["surface"])
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
    ax.axhline(0.0, color=core.BLACK, lw=1.0, ls="--")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-result["window"], result["window"])
    ax.set_xticks([-np.pi, 0.0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_xlabel(r"$k_{\parallel}$", fontweight="bold")
    ax.set_ylabel(
        r"$(E-E_{\mathrm{ref}})\ \mathrm{(eV)}$",
        fontweight="bold",
    )
    ax.set_title(title, fontweight="bold")
    core.style_axis(ax)
    # A taller-than-wide box shortens the physical horizontal axis without
    # changing the displayed momentum interval [-pi, pi].
    ax.set_box_aspect(1.08)


def draw_hall(
    ax: plt.Axes,
    result: dict[str, Any],
    specs: dict[str, core.ModelSpec],
    title: str,
    show_legend: bool,
) -> None:
    selected = result["selected"]
    spec = specs[selected.system]
    hall = result["hall"]
    color_up, color_down = core.spin_colors(spec, selected)
    phase = core.phase_color(spec, selected.expected_chern_up)
    audit = result["audit"]
    ef = float(audit["energy_reference"])
    if int(audit["is_global_insulator"]) == 1:
        interval_low = float(audit["vbm"]) - ef
        interval_high = float(audit["cbm"]) - ef
    else:
        # FES is an indirect-overlap metal.  Highlight only its local direct
        # gap around the reference valley; do not imply a global bulk gap.
        half_local_gap = 0.5 * float(audit["critical_valley_gap"])
        interval_low = -half_local_gap
        interval_high = +half_local_gap
    ax.axvspan(
        interval_low,
        interval_high,
        color=core.lighten(phase, 0.76),
        alpha=0.65,
        zorder=0,
    )
    ax.plot(
        hall["energy"],
        hall["sigma_xy_up_e2_over_h"],
        color=color_up,
        lw=2.0,
        label=r"$\sigma_{xy}^{\uparrow}$",
    )
    ax.plot(
        hall["energy"],
        hall["sigma_xy_down_e2_over_h"],
        color=color_down,
        lw=2.0,
        label=r"$\sigma_{xy}^{\downarrow}$",
    )
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.axvline(0.0, color=core.BLACK, lw=0.9, ls=":")
    ax.set_xlim(-result["window"], result["window"])
    ax.set_ylim(-2.35, 2.35)
    ax.set_xlabel(r"$E-E_{\mathrm{ref}}$", fontweight="bold")
    ax.set_ylabel(r"$\sigma_{xy}\ (e^2/h)$", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    if show_legend:
        ax.legend(loc="best", frameon=False, handlelength=2.0)
    core.style_axis(ax)
    ax.set_box_aspect(1.08)


def plot_individual_pair(
    result: dict[str, Any],
    specs: dict[str, core.ModelSpec],
    stem: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(5.85, 3.35))
    case_id = result["case_id"]
    draw_surface(axes[0], result, specs, title_for(case_id, panel=False))
    draw_hall(axes[1], result, specs, "Spin-resolved Hall response", True)
    fig.tight_layout(w_pad=1.25)
    return save_figure_inkscape_safe(fig, stem, dpi)


def plot_surface_four(
    results: list[dict[str, Any]],
    specs: dict[str, core.ModelSpec],
    stem: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(5.85, 6.65))
    for ax, result in zip(axes.flat, results):
        draw_surface(
            ax,
            result,
            specs,
            title_for(result["case_id"]),
        )
    fig.tight_layout(h_pad=1.0, w_pad=1.05)
    return save_figure_inkscape_safe(fig, stem, dpi)


def plot_hall_four(
    results: list[dict[str, Any]],
    specs: dict[str, core.ModelSpec],
    stem: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(5.85, 6.65), sharey=True)
    for index, (ax, result) in enumerate(zip(axes.flat, results)):
        draw_hall(
            ax,
            result,
            specs,
            title_for(result["case_id"]),
            show_legend=(index == 0),
        )
    fig.tight_layout(h_pad=1.0, w_pad=1.05)
    return save_figure_inkscape_safe(fig, stem, dpi)


def principal_edge_window(result: dict[str, Any]) -> float:
    """Return an energy window focused on the edge states near the main gap."""
    audit = result["audit"]
    selected = result["selected"]
    surface_energy = np.asarray(result["surface"]["energy"], dtype=float)
    full_window = float(np.max(np.abs(surface_energy)))
    if selected.system in ("lieb", "tts"):
        return min(1.5, full_window)
    if int(audit["is_global_insulator"]) == 1:
        ef = float(audit["energy_reference"])
        half_gap_extent = max(
            abs(float(audit["vbm"]) - ef),
            abs(float(audit["cbm"]) - ef),
        )
        if result["selected"].system == "lieb":
            focused = 1.25 * half_gap_extent
        else:
            # Include appreciable projected bulk-band background outside the
            # topological gap while keeping the edge branches visually central.
            focused = max(0.42, 2.05 * half_gap_extent)
    else:
        # The FES example is an indirect-overlap metal.  Focus on the local
        # direct-gap neighborhood plus enough surrounding projected bulk bands.
        focused = max(0.075, 12.0 * float(audit["critical_valley_gap"]))
    return float(min(full_window, focused))


def edge_spin_colors(
    selected: core.SelectedState | None = None,
) -> tuple[str, str]:
    """Use a fixed red/blue convention for spin-up/spin-down edge spectra."""
    del selected
    return core.LIEB_FES_COLORS[+1], core.LIEB_FES_COLORS[-1]


def edge_spin_colormap() -> LinearSegmentedColormap:
    """WannierTools-style spin map: blue -1, white 0, red +1."""
    color_up, color_down = edge_spin_colors()
    return LinearSegmentedColormap.from_list(
        "edge_spin_down_white_up",
        [color_down, "white", color_up],
        N=256,
    )


def edge_surface_rgb(
    selected: core.SelectedState,
    surface: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Use the project's WannierTools-style spin-resolved spectral mixing."""
    visible_up = np.log1p(
        np.asarray(surface["surface_dos_up"], dtype=float)
    )
    visible_down = np.log1p(
        np.asarray(surface["surface_dos_down"], dtype=float)
    )
    scale = max(
        float(np.max(visible_up)),
        float(np.max(visible_down)),
        1.0e-14,
    )
    visible_up /= scale
    visible_down /= scale
    intensity = np.maximum(visible_up, visible_down)
    total = np.maximum(visible_up + visible_down, 1.0e-14)
    color_up, color_down = edge_spin_colors(selected)
    rgb_up = np.asarray(to_rgb(color_up), dtype=float)
    rgb_down = np.asarray(to_rgb(color_down), dtype=float)
    mixed = (
        visible_up[..., None] * rgb_up[None, None, :]
        + visible_down[..., None] * rgb_down[None, None, :]
    ) / total[..., None]
    white = np.ones_like(mixed)
    rgb = white * (1.0 - intensity[..., None]) + mixed * intensity[..., None]
    return np.clip(rgb, 0.0, 1.0)


def add_spin_colorbar(ax: plt.Axes) -> None:
    """Add an editable vector blue-white-red spin-polarization colour bar."""
    color_up, color_down = edge_spin_colors()
    cmap = edge_spin_colormap()
    cax = inset_axes(
        ax,
        width="3.2%",
        height="94%",
        loc="lower left",
        bbox_to_anchor=(1.025, 0.03, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    strip_count = 96
    for index in range(strip_count):
        y0 = index / float(strip_count)
        y1 = (index + 1) / float(strip_count)
        color = cmap((index + 0.5) / float(strip_count))
        cax.add_patch(
            Rectangle(
                (0.0, y0),
                1.0,
                y1 - y0 + 1.0e-5,
                transform=cax.transAxes,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
            )
        )
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(0.0, 1.0)
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_color(core.BLACK)
        spine.set_linewidth(1.25)
    cax.set_ylabel(
        r"$P_z$",
        rotation=90,
        labelpad=7,
        fontweight="bold",
    )
    cax.yaxis.set_label_position("right")
    cax.text(
        1.55,
        1.0,
        r"$\uparrow$",
        color=color_up,
        ha="left",
        va="center",
        transform=cax.transAxes,
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    cax.text(
        1.55,
        0.0,
        r"$\downarrow$",
        color=color_down,
        ha="left",
        va="center",
        transform=cax.transAxes,
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )


def draw_edge_only(
    ax: plt.Axes,
    result: dict[str, Any],
    specs: dict[str, core.ModelSpec],
    title: str,
    show_legend: bool = True,
) -> None:
    selected = result["selected"]
    spec = specs[selected.system]
    surface = result["surface"]
    rgb = edge_surface_rgb(selected, surface)
    # Retain the established WannierTools-like continuum, but reduce its
    # contrast slightly relative to the earlier 1.28 enhancement.
    rgb = np.clip(1.0 - 1.15 * (1.0 - rgb), 0.0, 1.0)
    full_window = float(
        np.max(np.abs(np.asarray(surface["energy"], dtype=float)))
    )
    focused_window = principal_edge_window(result)
    ax.imshow(
        rgb,
        origin="lower",
        extent=[-np.pi, np.pi, -full_window, full_window],
        aspect="auto",
        interpolation="nearest",
    )
    ax.axhline(0.0, color=core.BLACK, lw=1.0, ls="--")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-focused_window, focused_window)
    ax.set_xticks([-np.pi, 0.0, np.pi])
    # The calculation uses A1=(1,0) as the periodic direction and A2=(0,1)
    # as the open/surface-normal direction.  Thus k_parallel=kx and the
    # projected 1D edge BZ follows X-Gamma-X.  A Y-Gamma-Y path
    # would instead require an x-normal edge and a new surface calculation.
    edge_tick_labels = ax.set_xticklabels(
        ["X", "\u0393", "X"]
    )
    for tick_label in edge_tick_labels:
        tick_label.set_fontfamily("Times New Roman")
        tick_label.set_fontweight("bold")
        tick_label.set_fontstyle("normal")
    ax.set_xlabel(r"$k_{\parallel}$", fontweight="bold")
    ax.set_ylabel(
        r"$(E-E_{\mathrm{ref}})\ \mathrm{(eV)}$",
        fontweight="bold",
    )
    ax.set_title(title + r"  $(010)$ edge", fontweight="bold")
    if show_legend:
        color_up, color_down = edge_spin_colors(selected)
        cup = int(selected.expected_chern_up)
        cdown = int(selected.expected_chern_down)
        handles = [
            Line2D(
                [0],
                [0],
                color=color_up,
                lw=3.0,
                label=rf"$C_{{\uparrow}}={cup:+d}$",
            ),
            Line2D(
                [0],
                [0],
                color=color_down,
                lw=3.0,
                label=rf"$C_{{\downarrow}}={cdown:+d}$",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.72,
            borderaxespad=0.35,
            handlelength=1.7,
        )
    add_spin_colorbar(ax)
    core.style_axis(ax)
    # Preserve k_parallel in [-pi, pi], but make the printed axis physically
    # shorter and emphasize the gap-region dispersion.
    ax.set_box_aspect(1.12)


def plot_edge_only_individual(
    result: dict[str, Any],
    specs: dict[str, core.ModelSpec],
    stem: Path,
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(3.15, 3.45))
    draw_edge_only(
        ax,
        result,
        specs,
        specs[result["selected"].system].label,
        show_legend=True,
    )
    fig.tight_layout()
    return save_figure_inkscape_safe(fig, stem, dpi)


def plot_edge_only_three_systems(
    results: list[dict[str, Any]],
    specs: dict[str, core.ModelSpec],
    stem: Path,
    dpi: int,
) -> list[str]:
    cs1_results = [
        result
        for result in results
        if int(result["selected"].expected_chern_up) == 1
    ]
    if [result["selected"].system for result in cs1_results] != [
        "lieb",
        "fes",
        "tts",
    ]:
        raise RuntimeError("Expected Lieb/FES/TTS C_s=+1 edge cases")
    fig, axes = plt.subplots(1, 3, figsize=(8.55, 3.65))
    for index, (ax, result) in enumerate(zip(axes, cs1_results)):
        draw_edge_only(
            ax,
            result,
            specs,
            f"({chr(ord('a') + index)})  "
            + specs[result["selected"].system].label,
            show_legend=True,
        )
    fig.tight_layout(w_pad=2.1)
    return save_figure_inkscape_safe(fig, stem, dpi)


def square_bulk_path(
    points_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    """Full square-BZ path Gamma-X-M-Y-Gamma."""
    named = [
        (r"$\Gamma$", (0.0, 0.0)),
        ("X", (np.pi, 0.0)),
        ("M", (np.pi, np.pi)),
        ("Y", (0.0, np.pi)),
        (r"$\Gamma$", (0.0, 0.0)),
    ]
    klist: list[np.ndarray] = []
    distance: list[float] = []
    ticks = [0.0]
    labels = [named[0][0]]
    current = 0.0
    previous: np.ndarray | None = None
    for segment in range(len(named) - 1):
        start = np.asarray(named[segment][1], dtype=float)
        stop = np.asarray(named[segment + 1][1], dtype=float)
        for index in range(int(points_per_segment) + 1):
            if segment > 0 and index == 0:
                continue
            fraction = index / float(points_per_segment)
            kpoint = (1.0 - fraction) * start + fraction * stop
            if previous is not None:
                current += float(np.linalg.norm(kpoint - previous))
            klist.append(kpoint)
            distance.append(current)
            previous = kpoint
        ticks.append(current)
        labels.append(named[segment + 1][0])
    return np.asarray(klist), np.asarray(distance), ticks, labels


def tts_altermagnetic_bulk_path(
    points_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    """TTS path containing Sigma and Sigma-prime spin-split diagonals."""
    named = [
        (r"$\Gamma$", (0.0, 0.0)),
        ("X", (np.pi, 0.0)),
        ("M", (np.pi, np.pi)),
        (r"$\Gamma$", (0.0, 0.0)),
        (r"$M^\prime$", (-np.pi, np.pi)),
        ("Y", (0.0, np.pi)),
        (r"$\Gamma$", (0.0, 0.0)),
    ]
    klist: list[np.ndarray] = []
    distance: list[float] = []
    ticks = [0.0]
    labels = [named[0][0]]
    current = 0.0
    previous: np.ndarray | None = None
    for segment in range(len(named) - 1):
        start = np.asarray(named[segment][1], dtype=float)
        stop = np.asarray(named[segment + 1][1], dtype=float)
        for index in range(int(points_per_segment) + 1):
            if segment > 0 and index == 0:
                continue
            fraction = index / float(points_per_segment)
            kpoint = (1.0 - fraction) * start + fraction * stop
            if previous is not None:
                current += float(np.linalg.norm(kpoint - previous))
            klist.append(kpoint)
            distance.append(current)
            previous = kpoint
        ticks.append(current)
        labels.append(named[segment + 1][0])
    return np.asarray(klist), np.asarray(distance), ticks, labels


def display_bulk_path(
    system: str,
    points_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    if system == "tts":
        return tts_altermagnetic_bulk_path(points_per_segment)
    return square_bulk_path(points_per_segment)


def calculate_square_bulk_bands(
    spec: core.ModelSpec,
    selected: core.SelectedState,
    energy_reference: float,
    points_per_segment: int,
) -> tuple[pd.DataFrame, list[float], list[str]]:
    """Spin-resolved bands on the system-appropriate display path."""
    klist, distance, ticks, labels = display_bulk_path(
        spec.key,
        points_per_segment,
    )
    spin_blocks = (
        ("up", np.asarray(spec.spin_up_indices, dtype=int), 1.0),
        ("down", np.asarray(spec.spin_down_indices, dtype=int), -1.0),
    )
    rows: list[dict[str, float | int | str]] = []
    for k_index, ((kx, ky), coordinate) in enumerate(
        zip(klist, distance)
    ):
        hamiltonian = spec.h_atomic(
            float(kx), float(ky), selected.params
        )
        for spin_sector, indices, spin_z in spin_blocks:
            spin_hamiltonian = hamiltonian[np.ix_(indices, indices)]
            energies = np.linalg.eigvalsh(spin_hamiltonian)
            for band_index, energy in enumerate(energies):
                rows.append(
                    {
                        "k_index": int(k_index),
                        "path_coordinate": float(coordinate),
                        "kx": float(kx),
                        "ky": float(ky),
                        "spin_sector": spin_sector,
                        "band": int(band_index + 1),
                        "energy": float(energy - energy_reference),
                        "spin_z": float(spin_z),
                    }
                )
    return pd.DataFrame(rows), ticks, labels


def draw_square_bulk(
    ax: plt.Axes,
    result: dict[str, Any],
    specs: dict[str, core.ModelSpec],
    title: str,
) -> None:
    selected = result["selected"]
    spec = specs[selected.system]
    bulk = result["bulk_square"]
    color_up, color_down = edge_spin_colors(selected)
    up_bands = {
        int(index): band
        for index, band in bulk[bulk["spin_sector"] == "up"].groupby("band")
    }
    down_bands = {
        int(index): band
        for index, band in bulk[bulk["spin_sector"] == "down"].groupby("band")
    }
    for band_index in sorted(set(up_bands) | set(down_bands)):
        up_band = up_bands.get(band_index)
        down_band = down_bands.get(band_index)
        degenerate = (
            up_band is not None
            and down_band is not None
            and len(up_band) == len(down_band)
            and np.allclose(
                up_band["energy"].to_numpy(),
                down_band["energy"].to_numpy(),
                rtol=0.0,
                atol=1.0e-10,
            )
        )
        if up_band is not None:
            ax.plot(
                up_band["path_coordinate"],
                up_band["energy"],
                color=color_up,
                ls="-",
                lw=2.25 if degenerate else 1.60,
                alpha=0.98,
                zorder=2,
            )
        if down_band is not None:
            ax.plot(
                down_band["path_coordinate"],
                down_band["energy"],
                color=color_down,
                ls="-",
                lw=1.15 if degenerate else 1.60,
                alpha=0.98,
                zorder=3,
            )
    for tick in result["bulk_square_ticks"]:
        ax.axvline(tick, color="#AFAFAF", lw=0.75, ls="--", zorder=0)
    ax.axhline(0.0, color=core.BLACK, lw=1.0, ls="--")
    ax.set_xlim(
        float(result["bulk_square_ticks"][0]),
        float(result["bulk_square_ticks"][-1]),
    )
    ax.set_ylim(
        -principal_edge_window(result),
        principal_edge_window(result),
    )
    ax.set_xticks(result["bulk_square_ticks"])
    ax.set_xticklabels(result["bulk_square_labels"])
    ax.set_xlabel("Bulk momentum path", fontweight="bold")
    ax.set_ylabel(
        r"$(E-E_{\mathrm{ref}})\ \mathrm{(eV)}$",
        fontweight="bold",
    )
    ax.set_title(title, fontweight="bold")
    core.style_axis(ax)
    ax.set_box_aspect(1.12)


def plot_bulk_edge_individual(
    result: dict[str, Any],
    specs: dict[str, core.ModelSpec],
    stem: Path,
    dpi: int,
) -> list[str]:
    label = specs[result["selected"].system].label
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 3.55))
    draw_square_bulk(
        axes[0],
        result,
        specs,
        label + ": bulk",
    )
    draw_edge_only(
        axes[1],
        result,
        specs,
        label,
        show_legend=True,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.93,
        bottom=0.18,
        top=0.88,
        wspace=0.52,
    )
    return save_figure_inkscape_safe(fig, stem, dpi)


def plot_bulk_edge_rows(
    results: list[dict[str, Any]],
    specs: dict[str, core.ModelSpec],
    stem: Path,
    dpi: int,
    primary_svg_editable: bool = False,
    svg_only: bool = False,
) -> list[str]:
    """Reference-style rows: bulk on the left, matching edge on the right."""
    row_count = len(results)
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(6.85, 3.35 * row_count),
        squeeze=False,
    )
    for row_index, result in enumerate(results):
        system_label = specs[result["selected"].system].label
        is_band_metal = (
            result["selected"].system == "fes"
            and int(result["audit"]["is_global_insulator"]) == 0
        )
        bulk_letter = chr(ord("a") + 2 * row_index)
        edge_letter = chr(ord("a") + 2 * row_index + 1)
        cs_value = int(result["selected"].expected_chern_up)
        bulk_title = (
            rf"({bulk_letter})  {system_label}: bulk, "
            rf"$C_s={cs_value:+d}$"
        )
        edge_title = rf"({edge_letter})  {system_label}"
        if is_band_metal:
            bulk_title += "  (band metal)"
            edge_title += "  (band metal)"
        draw_square_bulk(
            axes[row_index, 0],
            result,
            specs,
            bulk_title,
        )
        draw_edge_only(
            axes[row_index, 1],
            result,
            specs,
            edge_title,
            show_legend=True,
        )
    fig.subplots_adjust(
        left=0.11,
        right=0.92,
        bottom=0.07,
        top=0.97,
        hspace=0.48,
        wspace=0.54,
    )
    if svg_only:
        return save_editable_svg_only(fig, stem.with_suffix(".svg"), dpi)
    return save_figure_inkscape_safe(
        fig,
        stem,
        dpi,
        primary_svg_editable=primary_svg_editable,
    )


def parameter_record(case_id: str, selected: core.SelectedState) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case_id,
        "system": selected.system,
        "source_id": selected.source_id,
        "C_s": selected.expected_chern_up,
        "C_up": selected.expected_chern_up,
        "C_down": selected.expected_chern_down,
    }
    record.update(selected.params)
    return record


def run(output_dir: Path, quick: bool = False) -> dict[str, Any]:
    started = time.time()
    configure_step19_plot_style()
    settings = core.build_settings(quick)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    specs, modules = core.build_model_specs()
    cases = select_main_cases(modules)
    results: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    hall_records: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []

    print("[1/4] Four main-text representatives", flush=True)
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        selected: core.SelectedState = case["selected"]
        spec = specs[selected.system]
        sample_dir = data_dir / case_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  [{index}/4] {case_id}: gap/Chern/Fourier",
            flush=True,
        )

        gap = core.gap_audit(spec, selected, settings.gap_nk)
        chern = core.chern_audit(spec, selected, settings.chern_nk)
        audit = {
            "case_id": case_id,
            "panel": PANEL_LABEL[case_id],
            "system": selected.system,
            "source_id": selected.source_id,
            "path_id": selected.path_id,
            "expected_C_s": selected.expected_chern_up,
            "expected_chern_up": selected.expected_chern_up,
            "expected_chern_down": selected.expected_chern_down,
            "selection_note": selected.selection_note,
            **gap,
            **chern,
        }
        audit["C_s_numeric"] = (
            float(chern["chern_up"]) - float(chern["chern_down"])
        ) / 2.0
        audit["chern_matches_selection"] = int(
            int(chern["chern_up_int"]) == selected.expected_chern_up
            and int(chern["chern_down_int"]) == selected.expected_chern_down
        )
        audit["classification"] = classification(audit)
        if selected.system == "tts":
            magnetic_audit = modules["tts"].model_tests(
                selected.params,
                n_random=12,
            )
            audit["sigma_max_spin_splitting"] = float(
                magnetic_audit[
                    "sigma_kx_eq_ky_max_spin_splitting"
                ]
            )
            audit["sigma_prime_max_spin_splitting"] = float(
                magnetic_audit[
                    "sigma_prime_minus_kx_eq_ky_max_spin_splitting"
                ]
            )
            audit["symmetry_axis_max_spin_splitting"] = float(
                max(
                    magnetic_audit[
                        "delta_ky0_spin_degeneracy_error"
                    ],
                    magnetic_audit[
                        "delta_prime_kx0_spin_degeneracy_error"
                    ],
                    magnetic_audit[
                        "Z_kxpi_spin_degeneracy_error"
                    ],
                    magnetic_audit[
                        "Z_prime_kypi_spin_degeneracy_error"
                    ],
                )
            )
        audit_records.append(audit)
        parameter_records.append(parameter_record(case_id, selected))

        window = choose_case_window(
            selected.system,
            float(gap["min_direct_gap"]),
        )
        ef = float(gap["energy_reference"])
        hoppings = core.extract_hoppings(
            spec,
            selected.params,
            nfft=settings.fourier_n,
        )
        hopping_rows = [
            {
                "rx": rx,
                "ry": ry,
                "max_abs_hopping": float(np.max(np.abs(matrix))),
            }
            for (rx, ry), matrix in sorted(hoppings.items())
        ]
        pd.DataFrame(hopping_rows).to_csv(
            sample_dir / "wannier_hopping_summary.csv",
            index=False,
        )

        bulk, bulk_ticks, bulk_labels = core.calculate_bulk_bands(
            spec,
            selected,
            ef,
            settings.band_points_per_segment,
        )
        bulk.to_csv(sample_dir / "bulk_bands.csv", index=False)
        bulk_square, bulk_square_ticks, bulk_square_labels = (
            calculate_square_bulk_bands(
                spec,
                selected,
                ef,
                settings.band_points_per_segment,
            )
        )
        bulk_square.to_csv(
            sample_dir / "bulk_bands_display_path.csv",
            index=False,
        )

        eta = max(
            5.0e-4 if selected.system == "fes" else 1.5e-3,
            0.006 * window,
        )
        print(
            f"  [{index}/4] {case_id}: semi-infinite edge "
            f"(window={window:.4f}, eta={eta:.3e})",
            flush=True,
        )
        surface = core.calculate_surface(
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

        hall_nk = core.hall_mesh_for_state(selected, settings)
        print(
            f"  [{index}/4] {case_id}: Hall response nk={hall_nk}",
            flush=True,
        )
        hall, hall_summary = core.calculate_hall(
            spec,
            hoppings,
            ef,
            window,
            hall_nk,
            settings.response_energy_points,
        )
        hall.to_csv(
            sample_dir / "wanniertools_style_hall.csv",
            index=False,
        )
        hall_record = {
            "case_id": case_id,
            "panel": PANEL_LABEL[case_id],
            "system": selected.system,
            "source_id": selected.source_id,
            "expected_C_s": selected.expected_chern_up,
            "classification": audit["classification"],
            **hall_summary,
        }
        hall_records.append(hall_record)
        (sample_dir / "parameters_and_reference.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "code_version": CODE_VERSION,
                    "selected_state": {
                        "system": selected.system,
                        "source_id": selected.source_id,
                        "path_id": selected.path_id,
                        "C_s": selected.expected_chern_up,
                        "C_up": selected.expected_chern_up,
                        "C_down": selected.expected_chern_down,
                        "selection_note": selected.selection_note,
                    },
                    "parameters": selected.params,
                    "energy_reference": ef,
                    "energy_reference_kind": gap["energy_reference_kind"],
                    "plot_window": window,
                    "surface_eta": eta,
                    "classification": audit["classification"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        results.append(
            {
                "case_id": case_id,
                "selected": selected,
                "audit": audit,
                "window": window,
                "bulk": bulk,
                "bulk_ticks": bulk_ticks,
                "bulk_labels": bulk_labels,
                "bulk_square": bulk_square,
                "bulk_square_ticks": bulk_square_ticks,
                "bulk_square_labels": bulk_square_labels,
                "surface": surface,
                "hall": hall,
            }
        )
        print(
            f"       direct={float(gap['min_direct_gap']):.6g}, "
            f"indirect={float(gap['indirect_gap']):.6g}, "
            f"C_s={float(audit['C_s_numeric']):+.6f}",
            flush=True,
        )

    print("[2/4] Summary tables", flush=True)
    audit_df = pd.DataFrame(audit_records)
    hall_df = pd.DataFrame(hall_records)
    params_df = pd.DataFrame(parameter_records)
    audit_df.to_csv(
        output_dir / "step19_01_maintext_case_summary.csv",
        index=False,
    )
    hall_df.to_csv(
        output_dir / "step19_02_hall_summary.csv",
        index=False,
    )
    params_df.to_csv(
        output_dir / "step19_03_parameter_table.csv",
        index=False,
    )

    print("[3/4] Four individual figures and two optional composites", flush=True)
    manifest: list[dict[str, str]] = []
    for result in results:
        case_id = result["case_id"]
        files = plot_individual_pair(
            result,
            specs,
            figures_dir / f"Fig_maintext_{case_id}_edge_and_hall",
            settings.dpi,
        )
        for path in files:
            manifest.append(
                {
                    "figure_role": "individual_maintext_edge_and_hall",
                    "case_id": case_id,
                    "path": path,
                }
            )
        edge_files = plot_edge_only_individual(
            result,
            specs,
            figures_dir / f"Fig_edge_only_{case_id}_Cup_Cdown",
            settings.dpi,
        )
        for path in edge_files:
            manifest.append(
                {
                    "figure_role": "individual_edge_only_Cup_Cdown",
                    "case_id": case_id,
                    "path": path,
                }
            )
        bulk_edge_files = plot_bulk_edge_individual(
            result,
            specs,
            figures_dir / f"Fig_bulk_edge_pair_{case_id}",
            settings.dpi,
        )
        for path in bulk_edge_files:
            manifest.append(
                {
                    "figure_role": "individual_bulk_edge_pair",
                    "case_id": case_id,
                    "path": path,
                }
            )
    for path in plot_edge_only_three_systems(
        results,
        specs,
        figures_dir / "Fig_edge_only_three_systems_Cup_Cdown",
        settings.dpi,
    ):
        manifest.append(
            {
                "figure_role": "three_system_edge_only_Cup_Cdown",
                "case_id": "lieb_fes_tts_Cs_plus1",
                "path": path,
            }
        )
    cs1_results = [
        result
        for result in results
        if int(result["selected"].expected_chern_up) == 1
    ]
    for role, row_results, stem_name in (
        (
            "three_system_bulk_edge_rows",
            cs1_results,
            "Fig_bulk_edge_three_systems_Cs_plus1",
        ),
        (
            "four_case_bulk_edge_rows",
            results,
            "Fig_bulk_edge_four_topological_cases",
        ),
    ):
        for path in plot_bulk_edge_rows(
            row_results,
            specs,
            figures_dir / stem_name,
            settings.dpi,
            primary_svg_editable=(
                stem_name == "Fig_bulk_edge_four_topological_cases"
            ),
        ):
            manifest.append(
                {
                    "figure_role": role,
                    "case_id": "multiple",
                    "path": path,
                }
            )
    for role, files in (
        (
            "optional_four_panel_edge_composite",
            plot_surface_four(
                results,
                specs,
                figures_dir / "Fig_maintext_four_edge_spectra",
                settings.dpi,
            ),
        ),
        (
            "optional_four_panel_hall_composite",
            plot_hall_four(
                results,
                specs,
                figures_dir / "Fig_maintext_four_hall_conductivity",
                settings.dpi,
            ),
        ),
    ):
        for path in files:
            manifest.append(
                {"figure_role": role, "case_id": "all_four", "path": path}
            )
    pd.DataFrame(manifest).to_csv(
        output_dir / "step19_04_figure_manifest.csv",
        index=False,
    )

    print("[4/4] Validation certificate", flush=True)
    hall_by_case = {
        str(row["case_id"]): row
        for row in hall_records
    }
    insulating_case_ids = [
        row["case_id"]
        for row in audit_records
        if row["classification"] == "spin_chern_insulator"
    ]
    plateau_errors = {
        case_id: abs(
            float(hall_by_case[case_id]["mid_reference_sigma_up"])
            - float(
                next(
                    row["expected_C_s"]
                    for row in audit_records
                    if row["case_id"] == case_id
                )
            )
        )
        for case_id in insulating_case_ids
    }
    certificate = {
        "code_version": CODE_VERSION,
        "generated_utc": core.utc_now(),
        "quick_mode": bool(quick),
        "case_count": len(results),
        "case_order": list(CASE_ORDER),
        "all_chern_match": bool(
            all(int(row["chern_matches_selection"]) == 1 for row in audit_records)
        ),
        "all_direct_gapped": bool(
            all(int(row["is_direct_gapped"]) == 1 for row in audit_records)
        ),
        "global_insulator_count": int(
            sum(int(row["is_global_insulator"]) for row in audit_records)
        ),
        "band_metal_cases": [
            row["case_id"]
            for row in audit_records
            if row["classification"] == "spin_chern_band_metal"
        ],
        "insulating_midgap_hall_plateau_abs_errors": plateau_errors,
        "insulating_hall_plateaus_pass": bool(
            all(error < (0.12 if quick else 0.035) for error in plateau_errors.values())
        ),
        "mid_reference_charge_cancellation_pass": bool(
            all(
                abs(float(row["mid_reference_sigma_charge"])) < 5.0e-6
                for row in hall_records
            )
        ),
        "fes_interpretation": (
            "The selected FES C_s=+1 state is direct-gapped but has a negative "
            "indirect gap. It is therefore reported as a spin-Chern band metal; "
            "its Fermi-level Hall value is not claimed as a quantized plateau."
        ),
        "supplementary_relation": (
            "The Step18 nine-state topological/critical/Chern-changed triptychs "
            "remain unchanged and are intended for supplementary material."
        ),
        "runtime_seconds": float(time.time() - started),
    }
    (output_dir / "step19_05_validation_certificate.json").write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    readme = (
        "# Main-text four-topology package\n\n"
        "The four individual `Fig_maintext_*_edge_and_hall` files are the "
        "recommended main-text choices. Each combines the semi-infinite edge "
        "spectral function and the matching spin-resolved Hall response. "
        "`Fig_maintext_four_edge_spectra` and "
        "`Fig_maintext_four_hall_conductivity` are optional 2x2 composites.\n\n"
        "Cases: (a) Lieb C_s=+1; (b) FES C_s=+1; "
        "(c) TTS C_s=+1; (d) TTS C_s=+2.\n\n"
        "Important: the FES case is a direct-gapped spin-Chern band metal "
        "because its indirect gap is negative. Its Hall response is therefore "
        "not described as a quantized insulating plateau.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    if not certificate["all_chern_match"]:
        raise RuntimeError("At least one main-text Chern number failed")
    if not certificate["all_direct_gapped"]:
        raise RuntimeError("At least one main-text case is not direct-gapped")
    if not certificate["insulating_hall_plateaus_pass"]:
        raise RuntimeError("An insulating Hall plateau did not converge")
    if not certificate["mid_reference_charge_cancellation_pass"]:
        raise RuntimeError("Time-reversal charge Hall cancellation failed")
    print(
        f"Done: {output_dir} ({certificate['runtime_seconds']:.1f} s)",
        flush=True,
    )
    return certificate


def refresh_wide_edge_surfaces(output_dir: Path) -> dict[str, Any]:
    """Calculate only the wide-energy surface spectra needed by edge figures."""
    configure_step19_plot_style()
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    specs, modules = core.build_model_specs()
    cases = select_main_cases(modules)
    settings = core.build_settings(False)
    refreshed: list[str] = []
    for case in cases:
        selected: core.SelectedState = case["selected"]
        if selected.system not in ("lieb", "tts"):
            continue
        case_id = str(case["case_id"])
        sample_dir = data_dir / case_id
        metadata = json.loads(
            (sample_dir / "parameters_and_reference.json").read_text(
                encoding="utf-8"
            )
        )
        ef = float(metadata["energy_reference"])
        hoppings = core.extract_hoppings(
            specs[selected.system],
            selected.params,
            nfft=settings.fourier_n,
        )
        eta = 0.0072 if selected.system == "lieb" else 0.0045
        print(
            f"Wide edge surface {case_id}: E=[-1.5,1.5], "
            f"nk={settings.surface_k_points}, nE=361, eta={eta:.4g}",
            flush=True,
        )
        surface = core.calculate_surface(
            specs[selected.system],
            hoppings,
            ef,
            1.5,
            settings.surface_k_points,
            361,
            eta,
        )
        np.savez_compressed(
            sample_dir
            / "wanniertools_style_surface_spectrum_wide_Eminus1p5_to_1p5.npz",
            **surface,
        )
        refreshed.append(case_id)
    certificate = replot_existing(output_dir)
    certificate_path = output_dir / "step19_05_validation_certificate.json"
    certificate["wide_edge_surface_refresh_utc"] = core.utc_now()
    certificate["wide_edge_surface_cases"] = refreshed
    certificate["wide_edge_surface_energy_range"] = [-1.5, 1.5]
    certificate["edge_spin_color_policy"] = (
        "C_up is red and C_down is blue for all edge-only figures."
    )
    certificate_path.write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return certificate


def refresh_four_case_editable_svg(output_dir: Path) -> Path:
    """Regenerate only the four-case SVG as an Inkscape-editable hybrid."""
    configure_step19_plot_style()
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    audit_path = output_dir / "step19_01_maintext_case_summary.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    specs, modules = core.build_model_specs()
    cases = select_main_cases(modules)
    settings = core.build_settings(False)
    audit_df = pd.read_csv(audit_path)
    results: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        selected: core.SelectedState = case["selected"]
        sample_dir = data_dir / case_id
        metadata = json.loads(
            (sample_dir / "parameters_and_reference.json").read_text(
                encoding="utf-8"
            )
        )
        wide_surface_path = (
            sample_dir
            / "wanniertools_style_surface_spectrum_wide_Eminus1p5_to_1p5.npz"
        )
        surface_path = (
            wide_surface_path
            if wide_surface_path.is_file()
            else sample_dir / "wanniertools_style_surface_spectrum.npz"
        )
        with np.load(surface_path, allow_pickle=False) as stored:
            surface = {name: stored[name] for name in stored.files}
        bulk_square, bulk_square_ticks, bulk_square_labels = (
            calculate_square_bulk_bands(
                specs[selected.system],
                selected,
                float(metadata["energy_reference"]),
                settings.band_points_per_segment,
            )
        )
        audit = audit_df[
            audit_df["case_id"].astype(str) == case_id
        ].iloc[0].to_dict()
        results.append(
            {
                "case_id": case_id,
                "selected": selected,
                "audit": audit,
                "window": float(metadata["plot_window"]),
                "surface": surface,
                "bulk_square": bulk_square,
                "bulk_square_ticks": bulk_square_ticks,
                "bulk_square_labels": bulk_square_labels,
            }
        )
    svg_path = (
        figures_dir / "Fig_bulk_edge_four_topological_cases.svg"
    )
    plot_bulk_edge_rows(
        results,
        specs,
        svg_path.with_suffix(""),
        settings.dpi,
        primary_svg_editable=True,
        svg_only=True,
    )
    print(f"Editable four-case SVG refreshed: {svg_path}", flush=True)
    return svg_path


def replot_existing(output_dir: Path) -> dict[str, Any]:
    """Rebuild figures from saved numerical data without recalculation."""
    configure_step19_plot_style()
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    audit_path = output_dir / "step19_01_maintext_case_summary.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    specs, modules = core.build_model_specs()
    cases = select_main_cases(modules)
    settings = core.build_settings(False)
    audit_df = pd.read_csv(audit_path)
    results: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        selected: core.SelectedState = case["selected"]
        sample_dir = data_dir / case_id
        metadata = json.loads(
            (sample_dir / "parameters_and_reference.json").read_text(
                encoding="utf-8"
            )
        )
        wide_surface_path = (
            sample_dir
            / "wanniertools_style_surface_spectrum_wide_Eminus1p5_to_1p5.npz"
        )
        surface_path = (
            wide_surface_path
            if wide_surface_path.is_file()
            else sample_dir / "wanniertools_style_surface_spectrum.npz"
        )
        with np.load(surface_path, allow_pickle=False) as stored:
            surface = {name: stored[name] for name in stored.files}
        hall = pd.read_csv(sample_dir / "wanniertools_style_hall.csv")
        bulk_square_path = (
            sample_dir / "bulk_bands_display_path.csv"
        )
        bulk_square, bulk_square_ticks, bulk_square_labels = (
            calculate_square_bulk_bands(
                specs[selected.system],
                selected,
                float(metadata["energy_reference"]),
                settings.band_points_per_segment,
            )
        )
        bulk_square.to_csv(bulk_square_path, index=False)
        audit = audit_df[
            audit_df["case_id"].astype(str) == case_id
        ].iloc[0].to_dict()
        results.append(
            {
                "case_id": case_id,
                "selected": selected,
                "audit": audit,
                "window": float(metadata["plot_window"]),
                "surface": surface,
                "hall": hall,
                "bulk_square": bulk_square,
                "bulk_square_ticks": bulk_square_ticks,
                "bulk_square_labels": bulk_square_labels,
            }
        )

    dpi = settings.dpi
    manifest: list[dict[str, str]] = []
    for result in results:
        case_id = result["case_id"]
        files = plot_individual_pair(
            result,
            specs,
            figures_dir / f"Fig_maintext_{case_id}_edge_and_hall",
            dpi,
        )
        for path in files:
            manifest.append(
                {
                    "figure_role": "individual_spin_resolved_edge_and_hall",
                    "case_id": case_id,
                    "path": path,
                }
            )
        edge_files = plot_edge_only_individual(
            result,
            specs,
            figures_dir / f"Fig_edge_only_{case_id}_Cup_Cdown",
            dpi,
        )
        for path in edge_files:
            manifest.append(
                {
                    "figure_role": "individual_edge_only_Cup_Cdown",
                    "case_id": case_id,
                    "path": path,
                }
            )
        bulk_edge_files = plot_bulk_edge_individual(
            result,
            specs,
            figures_dir / f"Fig_bulk_edge_pair_{case_id}",
            dpi,
        )
        for path in bulk_edge_files:
            manifest.append(
                {
                    "figure_role": "individual_bulk_edge_pair",
                    "case_id": case_id,
                    "path": path,
                }
            )
    for path in plot_edge_only_three_systems(
        results,
        specs,
        figures_dir / "Fig_edge_only_three_systems_Cup_Cdown",
        dpi,
    ):
        manifest.append(
            {
                "figure_role": "three_system_edge_only_Cup_Cdown",
                "case_id": "lieb_fes_tts_Cs_plus1",
                "path": path,
            }
        )
    cs1_results = [
        result
        for result in results
        if int(result["selected"].expected_chern_up) == 1
    ]
    for role, row_results, stem_name in (
        (
            "three_system_bulk_edge_rows",
            cs1_results,
            "Fig_bulk_edge_three_systems_Cs_plus1",
        ),
        (
            "four_case_bulk_edge_rows",
            results,
            "Fig_bulk_edge_four_topological_cases",
        ),
    ):
        for path in plot_bulk_edge_rows(
            row_results,
            specs,
            figures_dir / stem_name,
            dpi,
            primary_svg_editable=(
                stem_name == "Fig_bulk_edge_four_topological_cases"
            ),
        ):
            manifest.append(
                {
                    "figure_role": role,
                    "case_id": "multiple",
                    "path": path,
                }
            )
    for role, files in (
        (
            "four_panel_spin_resolved_edge_composite",
            plot_surface_four(
                results,
                specs,
                figures_dir / "Fig_maintext_four_edge_spectra",
                dpi,
            ),
        ),
        (
            "four_panel_spin_resolved_hall_composite",
            plot_hall_four(
                results,
                specs,
                figures_dir / "Fig_maintext_four_hall_conductivity",
                dpi,
            ),
        ),
    ):
        for path in files:
            manifest.append(
                {"figure_role": role, "case_id": "all_four", "path": path}
            )
    pd.DataFrame(manifest).to_csv(
        output_dir / "step19_04_figure_manifest.csv",
        index=False,
    )
    certificate_path = output_dir / "step19_05_validation_certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["figure_revision_utc"] = core.utc_now()
    certificate["figure_revision"] = (
        "Bulk and edge spectra are paired case by case. All spin-up bulk "
        "bands are solid red and all spin-down bulk bands are solid blue; "
        "exactly degenerate red/blue curves use nested solid strokes so both "
        "remain visible. Primary SVG files are complete Inkscape-compatible "
        "single-layer figures, with companion *_editable.svg files retaining "
        "vector text, axes, and bulk curves."
    )
    certificate_path.write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Replotted without numerical recalculation: {figures_dir}",
        flush=True,
    )
    return certificate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small numerical grids for pipeline testing only.",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate figures from saved Step19 data without recalculation.",
    )
    parser.add_argument(
        "--refresh-wide-edge-only",
        action="store_true",
        help=(
            "Calculate Lieb/TTS surface spectra on E in [-1.5,1.5] and "
            "regenerate edge figures; Hall and bulk data are not recalculated."
        ),
    )
    parser.add_argument(
        "--editable-four-case-svg-only",
        action="store_true",
        help=(
            "Regenerate only Fig_bulk_edge_four_topological_cases.svg with "
            "editable vector axes, curves, text, legends, and colour bars."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.editable_four_case_svg_only:
        svg_path = refresh_four_case_editable_svg(args.output_dir)
        print(
            json.dumps(
                {"editable_four_case_svg": str(svg_path)},
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    if args.refresh_wide_edge_only:
        certificate = refresh_wide_edge_surfaces(args.output_dir)
    elif args.replot_only:
        certificate = replot_existing(args.output_dir)
    else:
        certificate = run(args.output_dir, quick=bool(args.quick))
    print(json.dumps(certificate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
