from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\1_ML\tts")
OUTPUT = ROOT / "outputs_tts_step21_small_gap_edge_rescreen"
STEP20_OUTPUT = ROOT / "outputs_tts_step20_edge_representative_rescreen"
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

WINDOWS = {
    "a_lieb_Cs_plus1": 1.5,
    "c_tts_Cs_plus1": 1.5,
    "d_tts_Cs_plus2": 1.5,
}
BASE_ETA = {
    "a_lieb_Cs_plus1": 0.0072,
    "c_tts_Cs_plus1": 0.0045,
    "d_tts_Cs_plus2": 0.0045,
}
BENCHMARK_IDS = {
    "a_lieb_Cs_plus1": "sobol_000555",
    "c_tts_Cs_plus1": "sobol_train_000456",
    "d_tts_Cs_plus2": "sobol_external_000117",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected(
    core: Any,
    system: str,
    source_id: str,
    params: dict[str, float],
    chern_up: int,
    kx: float,
    ky: float,
) -> Any:
    return core.SelectedState(
        system=system,
        state="topological",
        source_id=source_id,
        path_id="step21_small_gap_visual_rescreen",
        params={key: float(value) for key, value in params.items()},
        expected_chern_up=chern_up,
        expected_chern_down=-chern_up,
        critical_kx=float(kx),
        critical_ky=float(ky),
        selection_note="Small-gap visual rescreen candidate.",
    )


def build_lieb_candidates(
    core: Any,
    lieb: Any,
) -> list[Any]:
    master = pd.read_csv(LIEB_MASTER)
    candidates = master[
        (pd.to_numeric(master["chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(master["gap_convergence_ok"], errors="coerce") == 1)
        & (pd.to_numeric(master["chern_up_int"], errors="coerce") == 1)
        & (
            pd.to_numeric(master["final_indirect_gap"], errors="coerce")
            .between(0.07, 0.25)
        )
        & (
            pd.to_numeric(
                master["distance_to_any_analytic_boundary"], errors="coerce"
            )
            > 0.05
        )
    ].copy()
    candidates["prefilter"] = (
        (
            pd.to_numeric(candidates["final_indirect_gap"], errors="coerce")
            - 0.14
        ).abs()
        + 0.45
        * (
            pd.to_numeric(candidates["final_direct_gap"], errors="coerce")
            - pd.to_numeric(
                candidates["final_indirect_gap"], errors="coerce"
            )
        ).abs()
        - 0.18
        * pd.to_numeric(
            candidates["distance_to_any_analytic_boundary"], errors="coerce"
        )
    )
    rows = candidates.sort_values("prefilter").head(8)
    output: list[Any] = []
    for _, row in rows.iterrows():
        params = lieb.phys7_to_raw8(
            [float(row[name]) for name in lieb.PHYS7]
        )
        output.append(
            selected(
                core,
                "lieb",
                str(row["sample_id"]),
                params,
                +1,
                float(row["refined_direct_kx"]),
                float(row["refined_direct_ky"]),
            )
        )
    benchmark = master[master["sample_id"] == BENCHMARK_IDS["a_lieb_Cs_plus1"]]
    row = benchmark.iloc[0]
    output.append(
        selected(
            core,
            "lieb",
            str(row["sample_id"]),
            lieb.phys7_to_raw8([float(row[name]) for name in lieb.PHYS7]),
            +1,
            float(row["refined_direct_kx"]),
            float(row["refined_direct_ky"]),
        )
    )
    return output


def build_tts_candidates(
    core: Any,
    tts: Any,
    chern_up: int,
) -> list[Any]:
    table = pd.read_csv(TTS_STRICT)
    if chern_up == 1:
        low, high = 0.018, 0.12
        case_id = "c_tts_Cs_plus1"
    else:
        low, high = 0.03, 0.20
        case_id = "d_tts_Cs_plus2"
    rows = table[
        (pd.to_numeric(table["chern_up_int"], errors="coerce") == chern_up)
        & (pd.to_numeric(table["strict_gap_verified"], errors="coerce") == 1)
        & (
            pd.to_numeric(table["strict_chern_verified"], errors="coerce")
            == 1
        )
        & (
            pd.to_numeric(table["indirect_gap"], errors="coerce").between(
                low, high
            )
        )
    ].sort_values("indirect_gap")
    output: list[Any] = []
    for _, row in rows.iterrows():
        params = tts.raw8_from_reduced7(
            {name: float(row[name]) for name in tts.REDUCED7}
        )
        output.append(
            selected(
                core,
                "tts",
                str(row["point_id"]),
                params,
                chern_up,
                float(row["direct_gap_kx"]),
                float(row["direct_gap_ky"]),
            )
        )
    benchmark = table[table["point_id"] == BENCHMARK_IDS[case_id]].iloc[0]
    output.append(
        selected(
            core,
            "tts",
            str(benchmark["point_id"]),
            tts.raw8_from_reduced7(
                {name: float(benchmark[name]) for name in tts.REDUCED7}
            ),
            chern_up,
            float(benchmark["direct_gap_kx"]),
            float(benchmark["direct_gap_ky"]),
        )
    )
    return output


def effective_eta(case_id: str, direct_gap: float) -> float:
    floor = 8.0e-4
    return float(
        min(
            BASE_ETA[case_id],
            max(floor, float(direct_gap) / 18.0),
        )
    )


def visual_score(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    ratio = np.clip(group["min_direct_gap"] / group["eta"], 0.0, 20.0) / 20.0
    rank = lambda column, ascending: group[column].rank(
        pct=True, ascending=ascending, method="average"
    )
    group["small_gap_visual_score"] = (
        0.27 * rank("surface_colored_fraction", ascending=False)
        + 0.18 * rank("surface_dirty_overlap", ascending=False)
        + 0.15 * rank("surface_mean_polarization", ascending=True)
        + 0.17 * rank("surface_ridge_strength", ascending=True)
        + 0.10 * rank("surface_ridge_smoothness", ascending=False)
        + 0.13 * ratio
    )
    return group.sort_values("small_gap_visual_score", ascending=False)


def calculate_candidates(
    core: Any,
    step20: Any,
    specs: dict[str, Any],
    candidates: dict[str, list[Any]],
) -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_dir = OUTPUT / "surface_candidates"
    data_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = sum(len(items) for items in candidates.values())
    count = 0
    for case_id, items in candidates.items():
        for item in items:
            count += 1
            spec = specs[item.system]
            gap = core.gap_audit(spec, item, nk=48)
            eta = effective_eta(case_id, float(gap["min_direct_gap"]))
            print(
                f"[{count:02d}/{total:02d}] {case_id} {item.source_id} "
                f"gap={gap['indirect_gap']:.6f} eta={eta:.6f}",
                flush=True,
            )
            hoppings = core.extract_hoppings(spec, item.params, nfft=8)
            surface = core.calculate_surface(
                spec,
                hoppings,
                ef=float(gap["energy_reference"]),
                energy_window=WINDOWS[case_id],
                k_points=81,
                energy_points=241,
                eta=eta,
            )
            case_dir = data_dir / f"{case_id}__{item.source_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            npz_path = case_dir / "surface_preview.npz"
            np.savez_compressed(npz_path, **surface)
            records.append(
                {
                    "case_id": case_id,
                    "system": item.system,
                    "source_id": item.source_id,
                    "is_step20_benchmark": int(
                        item.source_id == BENCHMARK_IDS[case_id]
                    ),
                    "expected_chern_up": item.expected_chern_up,
                    "min_direct_gap": float(gap["min_direct_gap"]),
                    "indirect_gap": float(gap["indirect_gap"]),
                    "energy_reference": float(gap["energy_reference"]),
                    "eta": eta,
                    "gap_to_eta": float(gap["min_direct_gap"]) / eta,
                    **step20.surface_metrics(surface),
                    "params_json": json.dumps(item.params, sort_keys=True),
                    "critical_kx": item.critical_kx,
                    "critical_ky": item.critical_ky,
                    "surface_npz": str(npz_path),
                }
            )
    audit = pd.DataFrame(records)
    audit = pd.concat(
        [visual_score(group) for _, group in audit.groupby("case_id")],
        ignore_index=True,
    ).sort_values(
        ["case_id", "small_gap_visual_score"],
        ascending=[True, False],
    )
    audit.to_csv(OUTPUT / "step21_01_small_gap_surface_audit.csv", index=False)
    return audit


def draw_case_sheets(step20: Any, audit: pd.DataFrame) -> None:
    for case_id, group in audit.groupby("case_id", sort=False):
        group = group.sort_values(
            "small_gap_visual_score", ascending=False
        ).reset_index(drop=True)
        columns = 3
        rows = int(np.ceil(len(group) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(8.1, 2.65 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        for index, (_, record) in enumerate(group.iterrows()):
            ax = axes.flat[index]
            with np.load(record["surface_npz"]) as data:
                energy = np.asarray(data["energy"], dtype=float)
                rgb = step20.clean_spin_rgb(
                    data["surface_dos_up"],
                    data["surface_dos_down"],
                    height=900,
                    width=720,
                )
            ax.imshow(
                rgb,
                origin="lower",
                extent=(-np.pi, np.pi, energy[0], energy[-1]),
                aspect="auto",
                interpolation="bicubic",
            )
            ax.set_xticks((-np.pi, 0.0, np.pi), ("X", r"$\Gamma$", "X"))
            benchmark = " [current]" if record["is_step20_benchmark"] else ""
            ax.set_title(
                f"{record['source_id']}{benchmark}\n"
                f"gap={record['indirect_gap']:.4f}, "
                f"score={record['small_gap_visual_score']:.3f}",
                fontsize=8,
            )
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)
        for ax in axes.flat[len(group):]:
            ax.axis("off")
        fig.savefig(
            OUTPUT / f"step21_02_{case_id}_comparison.png",
            dpi=320,
            facecolor="white",
        )
        plt.close(fig)


def write_provisional_selection(audit: pd.DataFrame) -> None:
    selection = {}
    for case_id, group in audit.groupby("case_id", sort=False):
        row = group.sort_values(
            "small_gap_visual_score", ascending=False
        ).iloc[0]
        selection[case_id] = {
            key: (
                float(row[key])
                if isinstance(row[key], (float, np.floating))
                else int(row[key])
                if isinstance(row[key], (int, np.integer))
                else row[key]
            )
            for key in (
                "source_id",
                "system",
                "expected_chern_up",
                "min_direct_gap",
                "indirect_gap",
                "eta",
                "gap_to_eta",
                "small_gap_visual_score",
                "surface_npz",
                "params_json",
                "critical_kx",
                "critical_ky",
            )
        }
    (OUTPUT / "step21_03_provisional_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    core = load_module(
        "step21_core",
        ROOT / "TTS_step18_three_system_wanniertools_style_edge_ahc_examples.py",
    )
    step20 = load_module(
        "step21_render",
        ROOT / "TTS_step20_edge_representative_rescreen.py",
    )
    specs, modules = core.build_model_specs()
    candidates = {
        "a_lieb_Cs_plus1": build_lieb_candidates(
            core, modules["lieb"]
        ),
        "c_tts_Cs_plus1": build_tts_candidates(
            core, modules["tts"], +1
        ),
        "d_tts_Cs_plus2": build_tts_candidates(
            core, modules["tts"], +2
        ),
    }
    print(
        "Candidate inventory:",
        {key: len(value) for key, value in candidates.items()},
        flush=True,
    )
    audit = calculate_candidates(core, step20, specs, candidates)
    draw_case_sheets(step20, audit)
    write_provisional_selection(audit)


if __name__ == "__main__":
    main()
