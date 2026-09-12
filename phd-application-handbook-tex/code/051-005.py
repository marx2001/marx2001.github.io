from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"D:\1_ML\tts")
OUTPUT = ROOT / "outputs_tts_step21_small_gap_edge_rescreen"
FINAL_DIR = OUTPUT / "final_selected_data"
LIEB_MASTER = Path(
    r"D:\1_ML\lieb"
    r"\outputs_step08_analytic_boundary_wanniertools_edge_msg123342_lieb8_from_step06_n1024_pairs4_seed20260711"
    r"\step08_01_analytic_feature_master.csv"
)
TTS_STRICT = (
    ROOT
    / "outputs_tts_stepA_pm1_screening_v2"
    / "tts_stepA_01_all_normalized_strict_points.csv"
)
SELECTIONS = {
    "a_lieb_Cs_plus1": {
        "system": "lieb",
        "source_id": "sobol_000367",
        "chern_up": +1,
        "eta": 0.0072,
    },
    "c_tts_Cs_plus1": {
        "system": "tts",
        "source_id": "sobol_train_000343",
        "chern_up": +1,
        "eta": 0.002121,
    },
    "d_tts_Cs_plus2": {
        "system": "tts",
        "source_id": "sobol_train_000376",
        "chern_up": +2,
        "eta": 0.002863,
    },
}


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
    case_id: str,
    item: dict[str, Any],
) -> Any:
    if item["system"] == "lieb":
        table = pd.read_csv(LIEB_MASTER)
        row = table[table["sample_id"] == item["source_id"]].iloc[0]
        module = modules["lieb"]
        params = module.phys7_to_raw8(
            [float(row[name]) for name in module.PHYS7]
        )
        kx = float(row["refined_direct_kx"])
        ky = float(row["refined_direct_ky"])
    else:
        table = pd.read_csv(TTS_STRICT)
        row = table[table["point_id"] == item["source_id"]].iloc[0]
        module = modules["tts"]
        params = module.raw8_from_reduced7(
            {name: float(row[name]) for name in module.REDUCED7}
        )
        kx = float(row["direct_gap_kx"])
        ky = float(row["direct_gap_ky"])
    return core.SelectedState(
        system=str(item["system"]),
        state="topological",
        source_id=str(item["source_id"]),
        path_id="step21_final_small_gap_selection",
        params={key: float(value) for key, value in params.items()},
        expected_chern_up=int(item["chern_up"]),
        expected_chern_down=-int(item["chern_up"]),
        critical_kx=kx,
        critical_ky=ky,
        selection_note=(
            "Small-gap phase-interior point selected to reduce projected-bulk "
            "color occupancy while retaining resolvable topological branches."
        ),
    )


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    core = load_module(
        "step21_final_core",
        ROOT / "TTS_step18_three_system_wanniertools_style_edge_ahc_examples.py",
    )
    step19 = load_module(
        "step21_final_plot",
        ROOT / "TTS_step19_maintext_four_topological_edge_hall.py",
    )
    specs, modules = core.build_model_specs()
    audit_records = []
    manifest = {}
    for index, (case_id, item) in enumerate(SELECTIONS.items(), start=1):
        selected = make_selected(core, modules, case_id, item)
        spec = specs[selected.system]
        print(
            f"[{index}/3] {case_id} {selected.source_id}",
            flush=True,
        )
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
        hoppings = core.extract_hoppings(spec, selected.params, nfft=8)
        surface = core.calculate_surface(
            spec,
            hoppings,
            ef=float(gap["energy_reference"]),
            energy_window=1.5,
            k_points=141,
            energy_points=361,
            eta=float(item["eta"]),
        )
        case_dir = FINAL_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        surface_path = case_dir / "surface_spectrum_final.npz"
        bulk_path = case_dir / "bulk_bands_final.csv"
        np.savez_compressed(surface_path, **surface)
        bulk.to_csv(bulk_path, index=False)
        audit_records.append(
            {
                "case_id": case_id,
                "system": selected.system,
                "source_id": selected.source_id,
                "expected_chern_up": selected.expected_chern_up,
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
    pd.DataFrame(audit_records).to_csv(
        OUTPUT / "step21_04_final_physics_audit.csv", index=False
    )
    (OUTPUT / "step21_05_final_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
