from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lxml import etree


ROOT = Path(r"D:\1_ML\tts")
SOURCE_SVG = Path(r"D:\1_ML\picture\Fig4.svg")
OUTPUT = ROOT / "outputs_tts_step31_fig4_path_berry_curvature"
OUTPUT_SVG = OUTPUT / "Fig4_with_path_Berry_curvature.svg"
DATA_DIR = OUTPUT / "path_berry_curvature_data"

CORE_PATH = ROOT / "TTS_step18_three_system_wanniertools_style_edge_ahc_examples.py"
STEP19_PATH = ROOT / "TTS_step19_maintext_four_topological_edge_hall.py"
STEP22_PATH = ROOT / "TTS_step22_user_selection_dual_color_render.py"

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
NS = {"svg": SVG_NS}
GREEN = "#009E55"

# The four bulk-band axes in the supplied Fig4.svg.  The identifiers are
# Matplotlib clip paths already present in that exact source file.
CASE_CLIPS = {
    "a_lieb_Cs_plus1": "pfa3bb03856",
    "b_fes_Cs_plus1": "pd58dd54c92",
    "c_tts_Cs_plus1": "pa7505dfcc6",
    "d_tts_Cs_plus2": "p91e0cc4bd4",
}
PATH_POINTS_PER_SEGMENT = {
    "lieb": 120,
    "fes": 2400,
    "tts": 120,
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_transform(value: str | None) -> np.ndarray:
    """Parse the simple affine transforms used by the supplied SVG."""
    result = np.eye(3, dtype=float)
    if not value:
        return result
    pattern = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
    for name, payload in pattern.findall(value):
        numbers = [
            float(token)
            for token in re.split(r"[\s,]+", payload.strip())
            if token
        ]
        transform = np.eye(3, dtype=float)
        if name == "matrix" and len(numbers) == 6:
            a, b, c, d, e, f = numbers
            transform = np.asarray(
                [[a, c, e], [b, d, f], [0.0, 0.0, 1.0]],
                dtype=float,
            )
        elif name == "translate" and len(numbers) in (1, 2):
            transform[0, 2] = numbers[0]
            transform[1, 2] = numbers[1] if len(numbers) == 2 else 0.0
        elif name == "scale" and len(numbers) in (1, 2):
            transform[0, 0] = numbers[0]
            transform[1, 1] = numbers[1] if len(numbers) == 2 else numbers[0]
        else:
            raise ValueError(f"Unsupported SVG transform: {name}({payload})")
        result = result @ transform
    return result


def transform_to_root(node: etree._Element) -> np.ndarray:
    chain: list[etree._Element] = []
    current: etree._Element | None = node
    while current is not None:
        chain.append(current)
        current = current.getparent()
    result = np.eye(3, dtype=float)
    for element in reversed(chain):
        result = result @ parse_transform(element.get("transform"))
    return result


def find_panel_bounds(
    root: etree._Element,
    clip_id: str,
) -> tuple[float, float, float, float]:
    clips = root.xpath(
        ".//svg:clipPath[@id=$clip_id]",
        namespaces=NS,
        clip_id=clip_id,
    )
    if len(clips) != 1:
        raise RuntimeError(f"Expected one clipPath {clip_id}, found {len(clips)}")
    rectangles = clips[0].xpath("./svg:rect", namespaces=NS)
    if len(rectangles) != 1:
        raise RuntimeError(f"Clip {clip_id} is not a single rectangle")
    rect = rectangles[0]

    candidates = root.xpath(
        ".//svg:path[contains(@clip-path, $clip_id)]",
        namespaces=NS,
        clip_id=clip_id,
    )
    candidates = [
        node
        for node in candidates
        if (
            ("#a92425" in node.get("style", "").lower())
            or ("#3f63ad" in node.get("style", "").lower())
        )
        and len(node.get("d", "")) > 100
    ]
    if not candidates:
        raise RuntimeError(f"No transformed band path found for {clip_id}")
    # Some panels contain an old, displaced duplicate path that shares the
    # same clipPath.  The visible publication band is the densely encoded
    # path (largest d attribute); using the first XML match displaced the FES
    # overlay by one imported-group translation.
    visible_band = max(candidates, key=lambda node: len(node.get("d", "")))
    matrix = transform_to_root(visible_band)

    x = float(rect.get("x"))
    y = float(rect.get("y"))
    width = float(rect.get("width"))
    height = float(rect.get("height"))
    corners = np.asarray(
        [
            [x, y, 1.0],
            [x + width, y, 1.0],
            [x, y + height, 1.0],
            [x + width, y + height, 1.0],
        ],
        dtype=float,
    )
    mapped = (matrix @ corners.T).T
    x0 = float(np.min(mapped[:, 0]))
    x1 = float(np.max(mapped[:, 0]))
    y0 = float(np.min(mapped[:, 1]))
    y1 = float(np.max(mapped[:, 1]))
    return x0, y0, x1, y1


def nice_limit(maximum: float) -> float:
    """Symmetric panel limit with a small amount of headroom."""
    target = max(float(maximum) * 1.06, 1.0e-12)
    exponent = math.floor(math.log10(target))
    scale = 10.0**exponent
    fraction = target / scale
    for candidate in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.2, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= candidate + 1.0e-12:
            return float(candidate * scale)
    return float(10.0 * scale)


def format_tick(value: float) -> str:
    absolute = abs(float(value))
    if absolute >= 100.0:
        return f"{absolute:.0f}"
    if absolute >= 10.0:
        return f"{absolute:.0f}"
    if absolute >= 1.0:
        return f"{absolute:.1f}".rstrip("0").rstrip(".")
    if absolute >= 0.1:
        return f"{absolute:.2f}".rstrip("0").rstrip(".")
    return f"{absolute:.2g}"


def calculate_path_curvature(
    core: Any,
    step19: Any,
    step22: Any,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    specs, modules = core.build_model_specs()
    curves: dict[str, pd.DataFrame] = {}
    audit: list[dict[str, Any]] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for case_id, item in step22.SELECTIONS.items():
        selected = step22.make_selected(core, modules, item)
        spec = specs[selected.system]
        hoppings = core.extract_hoppings(spec, selected.params, nfft=8)
        klist, distance, ticks, labels = step19.display_bulk_path(
            spec.key,
            points_per_segment=PATH_POINTS_PER_SEGMENT[spec.key],
        )
        up_indices = np.asarray(spec.spin_up_indices, dtype=int)
        down_indices = np.asarray(spec.spin_down_indices, dtype=int)
        rows: list[dict[str, float | int | str]] = []

        for index, ((kx, ky), coordinate) in enumerate(zip(klist, distance)):
            hamiltonian, velocity_x, velocity_y = core.bloch_from_hoppings(
                float(kx),
                float(ky),
                hoppings,
            )
            _, omega_up_bands = core.band_berry_curvature(
                hamiltonian[np.ix_(up_indices, up_indices)],
                velocity_x[np.ix_(up_indices, up_indices)],
                velocity_y[np.ix_(up_indices, up_indices)],
            )
            _, omega_down_bands = core.band_berry_curvature(
                hamiltonian[np.ix_(down_indices, down_indices)],
                velocity_x[np.ix_(down_indices, down_indices)],
                velocity_y[np.ix_(down_indices, down_indices)],
            )
            omega_up = float(np.sum(omega_up_bands[: spec.n_occ_spin]))
            omega_down = float(np.sum(omega_down_bands[: spec.n_occ_spin]))
            rows.append(
                {
                    "k_index": int(index),
                    "path_coordinate": float(coordinate),
                    "kx": float(kx),
                    "ky": float(ky),
                    "omega_z_up": omega_up,
                    "omega_z_down": omega_down,
                    "omega_z_total": omega_up + omega_down,
                }
            )

        frame = pd.DataFrame(rows)
        curves[case_id] = frame
        frame.to_csv(
            DATA_DIR / f"{case_id}_path_berry_curvature.csv",
            index=False,
        )
        maximum = float(np.max(np.abs(frame["omega_z_total"].to_numpy())))
        limit = nice_limit(maximum)
        audit.append(
            {
                "case_id": case_id,
                "system": selected.system,
                "source_id": selected.source_id,
                "expected_chern_up": selected.expected_chern_up,
                "expected_chern_down": selected.expected_chern_down,
                "path_labels": " | ".join(
                    label.replace("$", "")
                    .replace("\\Gamma", "Γ")
                    .replace("M^\\prime", "M′")
                    for label in labels
                ),
                "path_ticks": json.dumps([float(value) for value in ticks]),
                "omega_z_total_min": float(frame["omega_z_total"].min()),
                "omega_z_total_max": float(frame["omega_z_total"].max()),
                "right_axis_limit": limit,
                "path_points_per_segment": PATH_POINTS_PER_SEGMENT[spec.key],
            }
        )
    return curves, audit


def svg_path(
    frame: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    limit: float,
) -> str:
    x0, y0, x1, y1 = bounds
    coordinate = frame["path_coordinate"].to_numpy(dtype=float)
    omega = frame["omega_z_total"].to_numpy(dtype=float)
    x = x0 + (coordinate - coordinate[0]) / (
        coordinate[-1] - coordinate[0]
    ) * (x1 - x0)
    y = y0 + (limit - omega) / (2.0 * limit) * (y1 - y0)
    commands = [f"M {x[0]:.6f},{y[0]:.6f}"]
    commands.extend(
        f"L {x_value:.6f},{y_value:.6f}"
        for x_value, y_value in zip(x[1:], y[1:])
    )
    return " ".join(commands)


def add_text(
    parent: etree._Element,
    *,
    x: float,
    y: float,
    text: str,
    anchor: str = "start",
    size: float = 3.0,
    element_id: str,
) -> etree._Element:
    node = etree.SubElement(
        parent,
        f"{{{SVG_NS}}}text",
        {
            "id": element_id,
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "text-anchor": anchor,
            "style": (
                f"font-family:'Times New Roman';font-size:{size:.3f}px;"
                f"font-weight:bold;font-style:normal;fill:{GREEN};"
                "stroke:none"
            ),
        },
    )
    node.text = text
    return node


def add_omega_label(
    parent: etree._Element,
    *,
    x: float,
    y: float,
    element_id: str,
) -> None:
    text = etree.SubElement(
        parent,
        f"{{{SVG_NS}}}text",
        {
            "id": element_id,
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "style": (
                "font-family:'Times New Roman';font-size:3.45px;"
                f"font-weight:bold;fill:{GREEN};stroke:none"
            ),
        },
    )
    omega = etree.SubElement(
        text,
        f"{{{SVG_NS}}}tspan",
        {"style": "font-style:italic"},
    )
    omega.text = "Ω"
    subscript = etree.SubElement(
        text,
        f"{{{SVG_NS}}}tspan",
        {
            "style": "font-style:italic;font-size:70%",
            "baseline-shift": "sub",
        },
    )
    subscript.text = "z"


def add_svg_overlays(
    curves: dict[str, pd.DataFrame],
    audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parser = etree.XMLParser(huge_tree=True, remove_blank_text=False)
    tree = etree.parse(str(SOURCE_SVG), parser)
    root = tree.getroot()
    if root.xpath(".//*[@id='berry-curvature-overlays']"):
        raise RuntimeError("Source SVG already contains a Berry-curvature overlay")

    defs_nodes = root.xpath("./svg:defs", namespaces=NS)
    if defs_nodes:
        defs = defs_nodes[0]
    else:
        defs = etree.Element(f"{{{SVG_NS}}}defs")
        root.insert(0, defs)

    group = etree.SubElement(
        root,
        f"{{{SVG_NS}}}g",
        {
            "id": "berry-curvature-overlays",
            f"{{{INKSCAPE_NS}}}groupmode": "layer",
            f"{{{INKSCAPE_NS}}}label": "Path Berry curvature Omega_z",
        },
    )
    audit_by_case = {str(record["case_id"]): record for record in audit}
    overlay_records: list[dict[str, Any]] = []

    for case_id, clip_id in CASE_CLIPS.items():
        bounds = find_panel_bounds(root, clip_id)
        x0, y0, x1, y1 = bounds
        limit = float(audit_by_case[case_id]["right_axis_limit"])
        overlay_clip_id = f"berry-clip-{case_id}"
        clip = etree.SubElement(
            defs,
            f"{{{SVG_NS}}}clipPath",
            {
                "id": overlay_clip_id,
                "clipPathUnits": "userSpaceOnUse",
            },
        )
        etree.SubElement(
            clip,
            f"{{{SVG_NS}}}rect",
            {
                "x": f"{x0:.6f}",
                "y": f"{y0:.6f}",
                "width": f"{x1 - x0:.6f}",
                "height": f"{y1 - y0:.6f}",
            },
        )

        panel_group = etree.SubElement(
            group,
            f"{{{SVG_NS}}}g",
            {
                "id": f"berry-overlay-{case_id}",
                f"{{{INKSCAPE_NS}}}label": f"{case_id}: total Omega_z",
            },
        )
        curve_width = 0.50 if case_id == "b_fes_Cs_plus1" else 0.58
        etree.SubElement(
            panel_group,
            f"{{{SVG_NS}}}path",
            {
                "id": f"berry-curve-{case_id}",
                "d": svg_path(curves[case_id], bounds, limit),
                "clip-path": f"url(#{overlay_clip_id})",
                "style": (
                    f"fill:none;stroke:{GREEN};stroke-width:{curve_width:.2f};"
                    "stroke-linecap:round;stroke-linejoin:round;"
                    "stroke-opacity:1"
                ),
            },
        )

        # The right axis is deliberately minimal, matching the cited paper:
        # a green spine, endpoint ticks/numbers, and Omega_z at the top.
        etree.SubElement(
            panel_group,
            f"{{{SVG_NS}}}path",
            {
                "id": f"berry-right-spine-{case_id}",
                "d": f"M {x1:.6f},{y0:.6f} L {x1:.6f},{y1:.6f}",
                "style": f"fill:none;stroke:{GREEN};stroke-width:0.55",
            },
        )
        tick_length = 1.25
        if case_id == "b_fes_Cs_plus1":
            # Keep the numbers clear of the final Γ label.  The curve still
            # uses the full ±25 range; only the labelled ticks are at ±20.
            tick_value = 0.8 * limit
            tick_top_y = y0 + (limit - tick_value) / (2.0 * limit) * (y1 - y0)
            tick_bottom_y = (
                y0 + (limit + tick_value) / (2.0 * limit) * (y1 - y0)
            )
        else:
            tick_value = limit
            tick_top_y = y0
            tick_bottom_y = y1
        for suffix, y in (("top", tick_top_y), ("bottom", tick_bottom_y)):
            etree.SubElement(
                panel_group,
                f"{{{SVG_NS}}}path",
                {
                    "id": f"berry-right-tick-{suffix}-{case_id}",
                    "d": f"M {x1:.6f},{y:.6f} L {x1 + tick_length:.6f},{y:.6f}",
                    "style": f"fill:none;stroke:{GREEN};stroke-width:0.48",
                },
            )

        tick_text = format_tick(tick_value)
        if case_id == "b_fes_Cs_plus1":
            label_top_y = tick_top_y + 0.9
            label_bottom_y = tick_bottom_y + 0.9
            label_size = 2.55
        else:
            label_top_y = y0 + 3.0
            label_bottom_y = y1 - 0.8
            label_size = 2.75
        add_text(
            panel_group,
            x=x1 + 1.55,
            y=label_top_y,
            text=tick_text,
            size=label_size,
            element_id=f"berry-right-label-top-{case_id}",
        )
        add_text(
            panel_group,
            x=x1 + 1.55,
            y=label_bottom_y,
            text=f"−{tick_text}",
            size=label_size,
            element_id=f"berry-right-label-bottom-{case_id}",
        )
        add_omega_label(
            panel_group,
            x=x1 + 1.35,
            y=y0 - 1.4,
            element_id=f"berry-right-title-{case_id}",
        )
        overlay_records.append(
            {
                "case_id": case_id,
                "source_clip_id": clip_id,
                "overlay_clip_id": overlay_clip_id,
                "panel_x0": x0,
                "panel_y0": y0,
                "panel_x1": x1,
                "panel_y1": y1,
                "right_axis_limit": limit,
                "right_axis_labelled_tick": tick_value,
                "curve_id": f"berry-curve-{case_id}",
            }
        )

    tree.write(
        str(OUTPUT_SVG),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )
    return overlay_records


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256(SOURCE_SVG)
    core = load_module("step31_core", CORE_PATH)
    step19 = load_module("step31_plot", STEP19_PATH)
    step22 = load_module("step31_selection", STEP22_PATH)

    curves, audit = calculate_path_curvature(core, step19, step22)
    overlay_records = add_svg_overlays(curves, audit)
    source_hash_after = sha256(SOURCE_SVG)
    if source_hash_before != source_hash_after:
        raise RuntimeError("The source Fig4.svg was unexpectedly modified")

    pd.DataFrame(audit).to_csv(
        OUTPUT / "step31_01_path_berry_curvature_audit.csv",
        index=False,
    )
    pd.DataFrame(overlay_records).to_csv(
        OUTPUT / "step31_02_svg_overlay_manifest.csv",
        index=False,
    )
    manifest = {
        "source_svg": str(SOURCE_SVG),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_preserved": source_hash_before == source_hash_after,
        "output_svg": str(OUTPUT_SVG),
        "quantity_plotted": "Omega_z_total(k) = Omega_z_up(k) + Omega_z_down(k)",
        "right_axis": "Berry curvature along the same high-symmetry path",
        "not_plotted": (
            "Anomalous Hall conductivity sigma_xy is a Brillouin-zone "
            "integral and is not a function of this one-dimensional k path."
        ),
        "curve_color": GREEN,
        "svg_editability": (
            "Each curve, right spine, tick, and label is an independent "
            "vector object in the 'Path Berry curvature Omega_z' layer."
        ),
        "cases": overlay_records,
    }
    (OUTPUT / "step31_03_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Source preserved: {source_hash_before == source_hash_after}")
    print(f"Output SVG: {OUTPUT_SVG}")
    print(f"Data: {DATA_DIR}")


if __name__ == "__main__":
    main()
