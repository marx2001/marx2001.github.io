from __future__ import annotations

import base64
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image
from scipy import ndimage


ROOT = Path(r"D:\1_ML\tts")
OUTPUT = ROOT / "outputs_tts_step22_user_selected_dual_color"
FINAL_DATA = OUTPUT / "final_selected_data"
STEP20_DATA = (
    ROOT
    / "outputs_tts_step20_edge_representative_rescreen"
    / "final_selected_data"
)
LIEB_MASTER = Path(
    r"D:\1_ML\lieb"
    r"\outputs_step08_analytic_boundary_wanniertools_edge_msg123342_lieb8_from_step06_n1024_pairs4_seed20260711"
    r"\step08_01_analytic_feature_master.csv"
)
FES_STRICT = Path(
    r"D:\1_ML\fes\outputs_fes6_step04_boundary"
    r"\fes_step04_strict_verified_results.csv"
)
TTS_STRICT = (
    ROOT
    / "outputs_tts_stepA_pm1_screening_v2"
    / "tts_stepA_01_all_normalized_strict_points.csv"
)

SELECTIONS = {
    "a_lieb_Cs_plus1": {
        "system": "lieb",
        "source_id": "sobol_000544",
        "chern_up": +1,
        "eta": 0.005686,
        "surface_source": "calculate",
    },
    "b_fes_Cs_plus1": {
        "system": "fes",
        "source_id": "rs-rd_i0031_j0044",
        "chern_up": +1,
        "eta": 0.00045,
        "surface_source": "step20",
    },
    "c_tts_Cs_plus1": {
        "system": "tts",
        "source_id": "sobol_train_000456",
        "chern_up": +1,
        "eta": 0.0045,
        "surface_source": "step20",
    },
    "d_tts_Cs_plus2": {
        "system": "tts",
        "source_id": "sobol_external_000117",
        "chern_up": +2,
        "eta": 0.0045,
        "surface_source": "step20",
    },
}
WINDOWS = {
    "a_lieb_Cs_plus1": 1.5,
    "b_fes_Cs_plus1": 0.075,
    "c_tts_Cs_plus1": 1.5,
    "d_tts_Cs_plus2": 1.5,
}
RED = np.asarray((231, 36, 28), dtype=float) / 255.0
BLUE = np.asarray((44, 88, 167), dtype=float) / 255.0


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_selected(
    core: Any,
    modules: dict[str, Any],
    item: dict[str, Any],
) -> Any:
    system = str(item["system"])
    source_id = str(item["source_id"])
    if system == "lieb":
        table = pd.read_csv(LIEB_MASTER)
        row = table[table["sample_id"] == source_id].iloc[0]
        module = modules["lieb"]
        params = module.phys7_to_raw8(
            [float(row[name]) for name in module.PHYS7]
        )
        kx = float(row["refined_direct_kx"])
        ky = float(row["refined_direct_ky"])
    elif system == "fes":
        table = pd.read_csv(FES_STRICT)
        row = table[table["point_id"] == source_id].iloc[0]
        module = modules["fes"]
        params = module.raw6_from_reduced5(
            {
                name: float(row[name])
                for name in ("m_e", "t1", "t2", "r1", "r2")
            }
        )
        kx = float(row["verified_direct_gap_kx"])
        ky = float(row["verified_direct_gap_ky"])
    else:
        table = pd.read_csv(TTS_STRICT)
        row = table[table["point_id"] == source_id].iloc[0]
        module = modules["tts"]
        params = module.raw8_from_reduced7(
            {name: float(row[name]) for name in module.REDUCED7}
        )
        kx = float(row["direct_gap_kx"])
        ky = float(row["direct_gap_ky"])
    chern_up = int(item["chern_up"])
    return core.SelectedState(
        system=system,
        state="topological",
        source_id=source_id,
        path_id="step22_user_selected",
        params={key: float(value) for key, value in params.items()},
        expected_chern_up=chern_up,
        expected_chern_down=-chern_up,
        critical_kx=kx,
        critical_ky=ky,
        selection_note="Representative explicitly selected by the user.",
    )


def prepare_data() -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FINAL_DATA.mkdir(parents=True, exist_ok=True)
    core = load_module(
        "step22_core",
        ROOT / "TTS_step18_three_system_wanniertools_style_edge_ahc_examples.py",
    )
    step19 = load_module(
        "step22_plot",
        ROOT / "TTS_step19_maintext_four_topological_edge_hall.py",
    )
    specs, modules = core.build_model_specs()
    records = []
    manifest = {}
    for index, (case_id, item) in enumerate(SELECTIONS.items(), start=1):
        selected = make_selected(core, modules, item)
        spec = specs[selected.system]
        print(f"[{index}/4] {case_id} {selected.source_id}", flush=True)
        gap = core.gap_audit(spec, selected, nk=72)
        chern = core.chern_audit(spec, selected, nk=48)
        if (
            int(chern["chern_up_int"]) != selected.expected_chern_up
            or int(chern["chern_down_int"]) != selected.expected_chern_down
        ):
            raise RuntimeError(f"Chern mismatch for {case_id}: {chern}")
        bulk, ticks, labels = step19.calculate_square_bulk_bands(
            spec,
            selected,
            energy_reference=float(gap["energy_reference"]),
            points_per_segment=120,
        )
        case_dir = FINAL_DATA / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        bulk_path = case_dir / "bulk_bands_final.csv"
        surface_path = case_dir / "surface_spectrum_final.npz"
        bulk.to_csv(bulk_path, index=False)
        if item["surface_source"] == "step20":
            source_surface = (
                STEP20_DATA / case_id / "surface_spectrum_final.npz"
            )
            shutil.copy2(source_surface, surface_path)
        else:
            hoppings = core.extract_hoppings(spec, selected.params, nfft=8)
            surface = core.calculate_surface(
                spec,
                hoppings,
                ef=float(gap["energy_reference"]),
                energy_window=WINDOWS[case_id],
                k_points=141,
                energy_points=361,
                eta=float(item["eta"]),
            )
            np.savez_compressed(surface_path, **surface)
        records.append(
            {
                "case_id": case_id,
                "system": selected.system,
                "source_id": selected.source_id,
                "eta": float(item["eta"]),
                "gap_to_eta": float(gap["min_direct_gap"]) / float(item["eta"]),
                **gap,
                **chern,
                **selected.params,
            }
        )
        manifest[case_id] = {
            "source_id": selected.source_id,
            "surface_npz": str(surface_path),
            "bulk_csv": str(bulk_path),
            "bulk_ticks": [float(value) for value in ticks],
            "bulk_labels": labels,
            "params": selected.params,
        }
    audit = pd.DataFrame(records)
    audit.to_csv(OUTPUT / "step22_01_final_physics_audit.csv", index=False)
    (OUTPUT / "step22_02_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit


def upsample_fields(
    dos_up: np.ndarray,
    dos_down: np.ndarray,
    height: int = 1420,
    width: int = 1268,
) -> tuple[np.ndarray, np.ndarray]:
    up = np.log1p(np.asarray(dos_up, dtype=float))
    down = np.log1p(np.asarray(dos_down, dtype=float))
    scale = max(float(np.quantile(np.maximum(up, down), 0.997)), 1.0e-14)
    zoom = (height / up.shape[0], width / up.shape[1])
    up = ndimage.zoom(up / scale, zoom=zoom, order=3, mode="nearest")
    down = ndimage.zoom(down / scale, zoom=zoom, order=3, mode="nearest")
    up = ndimage.gaussian_filter(np.clip(up, 0.0, 1.3), sigma=(1.0, 3.0))
    down = ndimage.gaussian_filter(
        np.clip(down, 0.0, 1.3), sigma=(1.0, 3.0)
    )
    return up, down


def white_overlap_rgb(
    dos_up: np.ndarray,
    dos_down: np.ndarray,
) -> np.ndarray:
    up, down = upsample_fields(dos_up, dos_down)
    intensity = np.maximum(up, down)
    polarization = (up - down) / np.maximum(up + down, 1.0e-14)
    effective = np.clip(
        np.power(intensity, 0.85)
        * np.power(np.abs(polarization), 0.60),
        0.0,
        0.96,
    )
    endpoint = np.where(
        (polarization >= 0.0)[..., None],
        RED[None, None, :],
        BLUE[None, None, :],
    )
    return np.clip(
        1.0 - effective[..., None] * (1.0 - endpoint),
        0.0,
        1.0,
    )


def direct_mix_rgb(
    dos_up: np.ndarray,
    dos_down: np.ndarray,
) -> np.ndarray:
    up, down = upsample_fields(dos_up, dos_down)
    total = np.maximum(up + down, 1.0e-14)
    mixed = (
        up[..., None] * RED[None, None, :]
        + down[..., None] * BLUE[None, None, :]
    ) / total[..., None]
    intensity = np.clip(np.power(np.maximum(up, down), 0.85), 0.0, 0.96)
    return np.clip(
        1.0 - intensity[..., None] * (1.0 - mixed),
        0.0,
        1.0,
    )


def encode_png(rgb: np.ndarray) -> tuple[str, bytes]:
    array = np.asarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    payload = buffer.getvalue()
    return (
        "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
        payload,
    )


if __name__ == "__main__":
    prepare_data()
