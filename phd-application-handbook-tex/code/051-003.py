from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


ROOT = Path(r"D:\1_ML\tts")
CORE_PATH = ROOT / "TTS_step18_three_system_wanniertools_style_edge_ahc_examples.py"
DEFAULT_OUTPUT = ROOT / "outputs_tts_step20_edge_representative_rescreen"

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

CASE_ORDER = (
    "a_lieb_Cs_plus1",
    "b_fes_Cs_plus1",
    "c_tts_Cs_plus1",
    "d_tts_Cs_plus2",
)
CASE_LABELS = {
    "a_lieb_Cs_plus1": r"Lieb  $C_{\uparrow}=+1$",
    "b_fes_Cs_plus1": r"FES  $C_{\uparrow}=+1$",
    "c_tts_Cs_plus1": r"TTS  $C_{\uparrow}=+1$",
    "d_tts_Cs_plus2": r"TTS  $C_{\uparrow}=+2$",
}
WINDOWS = {
    "a_lieb_Cs_plus1": 1.5,
    "b_fes_Cs_plus1": 0.075,
    "c_tts_Cs_plus1": 1.5,
    "d_tts_Cs_plus2": 1.5,
}
ETAS = {
    "a_lieb_Cs_plus1": 0.0072,
    "b_fes_Cs_plus1": 0.00045,
    "c_tts_Cs_plus1": 0.0045,
    "d_tts_Cs_plus2": 0.0045,
}
TARGET_GAPS = {
    "a_lieb_Cs_plus1": 0.50,
    "b_fes_Cs_plus1": 0.0025,
    "c_tts_Cs_plus1": 0.32,
    "d_tts_Cs_plus2": 0.25,
}

RED = np.asarray((231, 36, 28), dtype=float) / 255.0
BLUE = np.asarray((44, 88, 167), dtype=float) / 255.0


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_from_row(
    core: Any,
    system: str,
    source_id: str,
    params: dict[str, float],
    chern_up: int,
    kx: float,
    ky: float,
    note: str,
) -> Any:
    return core.SelectedState(
        system=system,
        state="topological",
        source_id=str(source_id),
        path_id="step20_visual_rescreen",
        params={key: float(value) for key, value in params.items()},
        expected_chern_up=int(chern_up),
        expected_chern_down=-int(chern_up),
        critical_kx=float(kx),
        critical_ky=float(ky),
        selection_note=note,
    )


def build_candidates(core: Any, modules: dict[str, Any]) -> dict[str, list[Any]]:
    lieb = modules["lieb"]
    fes = modules["fes"]
    tts = modules["tts"]

    lieb_master = pd.read_csv(LIEB_MASTER)
    lieb_rows = lieb_master[
        (pd.to_numeric(lieb_master["chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(lieb_master["gap_convergence_ok"], errors="coerce") == 1)
        & (pd.to_numeric(lieb_master["chern_up_int"], errors="coerce") == 1)
        & (pd.to_numeric(lieb_master["final_indirect_gap"], errors="coerce") > 0)
        & (
            pd.to_numeric(
                lieb_master["distance_to_any_analytic_boundary"],
                errors="coerce",
            )
            > 0.06
        )
    ].copy()
    lieb_rows["prefilter"] = (
        (
            pd.to_numeric(lieb_rows["final_indirect_gap"], errors="coerce")
            - TARGET_GAPS["a_lieb_Cs_plus1"]
        ).abs()
        + 0.20
        * (
            pd.to_numeric(lieb_rows["final_direct_gap"], errors="coerce")
            - pd.to_numeric(lieb_rows["final_indirect_gap"], errors="coerce")
        ).abs()
        - 0.12
        * pd.to_numeric(
            lieb_rows["distance_to_any_analytic_boundary"], errors="coerce"
        )
    )
    lieb_rows = lieb_rows.sort_values("prefilter").head(36)
    lieb_candidates: list[Any] = []
    for _, row in lieb_rows.iterrows():
        physical = [float(row[name]) for name in lieb.PHYS7]
        params = lieb.phys7_to_raw8(physical)
        lieb_candidates.append(
            selected_from_row(
                core,
                "lieb",
                str(row["sample_id"]),
                params,
                +1,
                float(row["refined_direct_kx"]),
                float(row["refined_direct_ky"]),
                "Reliable converged phase-interior Lieb C_s=+1 point.",
            )
        )

    fes_rows = pd.read_csv(FES_STRICT)
    fes_rows = fes_rows[
        (pd.to_numeric(fes_rows["verified_chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(fes_rows["verified_chern_up_int"], errors="coerce") == 1)
        & (
            pd.to_numeric(
                fes_rows["verified_min_direct_gap"], errors="coerce"
            )
            > 8.0e-4
        )
    ].copy()
    fes_candidates: list[Any] = []
    for _, row in fes_rows.iterrows():
        reduced = {
            name: float(row[name])
            for name in ("m_e", "t1", "t2", "r1", "r2")
        }
        params = fes.raw6_from_reduced5(reduced)
        fes_candidates.append(
            selected_from_row(
                core,
                "fes",
                str(row["point_id"]),
                params,
                +1,
                float(row["verified_direct_gap_kx"]),
                float(row["verified_direct_gap_ky"]),
                "Strictly verified FES C_s=+1 phase-interior point.",
            )
        )

    tts_rows = pd.read_csv(TTS_STRICT)
    tts_candidates: dict[int, list[Any]] = {+1: [], +2: []}
    for chern_up in (+1, +2):
        subset = tts_rows[
            (pd.to_numeric(tts_rows["chern_up_int"], errors="coerce") == chern_up)
            & (pd.to_numeric(tts_rows["strict_gap_verified"], errors="coerce") == 1)
            & (
                pd.to_numeric(
                    tts_rows["strict_chern_verified"], errors="coerce"
                )
                == 1
            )
            & (pd.to_numeric(tts_rows["indirect_gap"], errors="coerce") > 0.01)
        ]
        for _, row in subset.iterrows():
            reduced = {name: float(row[name]) for name in tts.REDUCED7}
            params = tts.raw8_from_reduced7(reduced)
            tts_candidates[chern_up].append(
                selected_from_row(
                    core,
                    "tts",
                    str(row["point_id"]),
                    params,
                    chern_up,
                    float(row["direct_gap_kx"]),
                    float(row["direct_gap_ky"]),
                    f"Strictly verified TTS C_s={chern_up:+d} phase-interior point.",
                )
            )

    return {
        "a_lieb_Cs_plus1": lieb_candidates,
        "b_fes_Cs_plus1": fes_candidates,
        "c_tts_Cs_plus1": tts_candidates[+1],
        "d_tts_Cs_plus2": tts_candidates[+2],
    }


def projected_bulk_metrics(
    spec: Any,
    selected: Any,
    ef: float,
    window: float,
    nk: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    k_parallel = np.linspace(-np.pi, np.pi, int(nk), endpoint=True)
    k_normal = np.linspace(-np.pi, np.pi, int(nk), endpoint=False)
    valence = np.empty_like(k_parallel)
    conduction = np.empty_like(k_parallel)
    all_relative: list[np.ndarray] = []
    for ix, kx in enumerate(k_parallel):
        eigenvalues = np.asarray(
            [
                np.linalg.eigvalsh(
                    spec.h_periodic(float(kx), float(ky), selected.params)
                )
                for ky in k_normal
            ],
            dtype=float,
        )
        eigenvalues -= float(ef)
        valence[ix] = float(
            np.max(eigenvalues[:, spec.n_occ_total - 1])
        )
        conduction[ix] = float(
            np.min(eigenvalues[:, spec.n_occ_total])
        )
        all_relative.append(eigenvalues.ravel())

    all_energy = np.concatenate(all_relative)
    projected_gap = conduction - valence
    gap_center = 0.5 * (conduction + valence)
    energy_scale = max(float(window), 1.0e-8)
    envelope_scale = max(
        float(np.ptp(valence) + np.ptp(conduction)),
        1.0e-8,
    )
    symmetry_error = float(
        np.mean(np.abs(valence - valence[::-1]))
        + np.mean(np.abs(conduction - conduction[::-1]))
    ) / envelope_scale
    roughness = float(
        np.mean(np.abs(np.diff(valence, n=2)))
        + np.mean(np.abs(np.diff(conduction, n=2)))
    ) / envelope_scale
    positive_gap = np.maximum(projected_gap, 0.0)
    overlap = np.maximum(-projected_gap, 0.0)
    metrics = {
        "projected_gap_fraction": float(np.mean(projected_gap > 0.0)),
        "projected_gap_median": float(np.median(positive_gap)),
        "projected_overlap_mean": float(np.mean(overlap)),
        "projected_gap_center_abs": float(np.median(np.abs(gap_center))),
        "projected_symmetry_error": symmetry_error,
        "projected_envelope_roughness": roughness,
        "bulk_near_reference_fraction": float(
            np.mean(np.abs(all_energy) < 0.20 * energy_scale)
        ),
    }
    arrays = {
        "k_parallel": k_parallel,
        "projected_valence": valence,
        "projected_conduction": conduction,
    }
    return metrics, arrays


def percentile_rank(values: pd.Series, high_is_good: bool) -> pd.Series:
    return values.rank(pct=True, ascending=high_is_good, method="average")


def score_coarse(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    target = float(TARGET_GAPS[str(group["case_id"].iloc[0])])
    gap_match = np.exp(
        -np.abs(group["indirect_gap"] - target) / max(target, 1.0e-6)
    )
    group["coarse_score"] = (
        0.25 * percentile_rank(group["projected_gap_fraction"], True)
        + 0.18 * percentile_rank(group["projected_gap_median"], True)
        + 0.12 * percentile_rank(group["projected_overlap_mean"], False)
        + 0.14 * percentile_rank(group["projected_symmetry_error"], False)
        + 0.14 * percentile_rank(group["projected_envelope_roughness"], False)
        + 0.07
        * percentile_rank(group["bulk_near_reference_fraction"], False)
        + 0.10 * gap_match
    )
    return group.sort_values("coarse_score", ascending=False)


def run_coarse(
    core: Any,
    specs: dict[str, Any],
    candidates: dict[str, list[Any]],
    output_dir: Path,
    nk_gap: int,
    nk_projected: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    envelope_dir = output_dir / "projected_bulk_envelopes"
    envelope_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(items) for items in candidates.values())
    done = 0
    for case_id in CASE_ORDER:
        for selected in candidates[case_id]:
            done += 1
            print(
                f"[coarse {done:02d}/{total:02d}] "
                f"{case_id} {selected.source_id}",
                flush=True,
            )
            spec = specs[selected.system]
            gap = core.gap_audit(spec, selected, nk=nk_gap)
            metrics, arrays = projected_bulk_metrics(
                spec,
                selected,
                ef=float(gap["energy_reference"]),
                window=WINDOWS[case_id],
                nk=nk_projected,
            )
            record = {
                "case_id": case_id,
                "system": selected.system,
                "source_id": selected.source_id,
                "expected_chern_up": selected.expected_chern_up,
                **{key: float(value) for key, value in gap.items()
                   if isinstance(value, (int, float, np.integer, np.floating))},
                **metrics,
                "params_json": json.dumps(selected.params, sort_keys=True),
                "critical_kx": float(selected.critical_kx),
                "critical_ky": float(selected.critical_ky),
            }
            records.append(record)
            np.savez_compressed(
                envelope_dir / f"{case_id}__{selected.source_id}.npz",
                **arrays,
            )
    coarse = pd.DataFrame.from_records(records)
    scored = pd.concat(
        [score_coarse(group) for _, group in coarse.groupby("case_id")],
        ignore_index=True,
    )
    scored = scored.sort_values(
        ["case_id", "coarse_score"], ascending=[True, False]
    )
    scored.to_csv(output_dir / "step20_01_coarse_candidate_audit.csv", index=False)
    return scored


def reconstruct_selected(core: Any, row: pd.Series) -> Any:
    return selected_from_row(
        core,
        str(row["system"]),
        str(row["source_id"]),
        json.loads(str(row["params_json"])),
        int(row["expected_chern_up"]),
        float(row["critical_kx"]),
        float(row["critical_ky"]),
        "Step20 visual rescreen shortlist.",
    )


def surface_metrics(surface: dict[str, Any]) -> dict[str, float]:
    up = np.log1p(np.asarray(surface["surface_dos_up"], dtype=float))
    down = np.log1p(np.asarray(surface["surface_dos_down"], dtype=float))
    scale = max(float(np.quantile(np.maximum(up, down), 0.997)), 1.0e-14)
    up = np.clip(up / scale, 0.0, 1.5)
    down = np.clip(down / scale, 0.0, 1.5)
    intensity = np.maximum(up, down)
    polarization = (up - down) / np.maximum(up + down, 1.0e-14)
    purity = np.abs(polarization)
    effective = np.clip(
        np.power(intensity, 0.85) * np.power(purity, 0.60),
        0.0,
        1.0,
    )
    weight = intensity / max(float(np.sum(intensity)), 1.0e-14)
    dirty_overlap = float(np.sum(weight * (1.0 - purity)))
    ridge_strength = np.max(effective, axis=0)
    ridge_index = np.argmax(effective, axis=0)
    energy = np.asarray(surface["energy"], dtype=float)
    ridge_energy = energy[ridge_index]
    strong = ridge_strength > 0.15
    if np.count_nonzero(strong) >= 5:
        smoothness = float(
            np.median(np.abs(np.diff(ridge_energy[strong], n=2)))
        ) / max(float(np.ptp(energy)), 1.0e-12)
    else:
        smoothness = 1.0
    return {
        "surface_dirty_overlap": dirty_overlap,
        "surface_mean_polarization": float(np.sum(weight * purity)),
        "surface_ridge_coverage": float(np.mean(strong)),
        "surface_ridge_strength": float(np.mean(ridge_strength)),
        "surface_ridge_smoothness": smoothness,
        "surface_colored_fraction": float(np.mean(effective > 0.14)),
    }


def clean_spin_rgb(
    dos_up: np.ndarray,
    dos_down: np.ndarray,
    height: int = 900,
    width: int = 720,
) -> np.ndarray:
    up = np.log1p(np.asarray(dos_up, dtype=float))
    down = np.log1p(np.asarray(dos_down, dtype=float))
    scale = max(float(np.quantile(np.maximum(up, down), 0.997)), 1.0e-14)
    zoom = (height / up.shape[0], width / up.shape[1])
    up = ndimage.zoom(up / scale, zoom=zoom, order=3, mode="nearest")
    down = ndimage.zoom(down / scale, zoom=zoom, order=3, mode="nearest")
    up = ndimage.gaussian_filter(np.clip(up, 0.0, 1.3), sigma=(0.8, 1.6))
    down = ndimage.gaussian_filter(
        np.clip(down, 0.0, 1.3), sigma=(0.8, 1.6)
    )
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


def score_surfaces(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["surface_score"] = (
        0.25 * percentile_rank(group["surface_dirty_overlap"], False)
        + 0.20 * percentile_rank(group["surface_mean_polarization"], True)
        + 0.18 * percentile_rank(group["surface_ridge_coverage"], True)
        + 0.15 * percentile_rank(group["surface_ridge_strength"], True)
        + 0.12 * percentile_rank(group["surface_ridge_smoothness"], False)
        + 0.10 * percentile_rank(group["surface_colored_fraction"], False)
    )
    group["combined_score"] = (
        0.35 * group["coarse_score"] + 0.65 * group["surface_score"]
    )
    return group.sort_values("combined_score", ascending=False)


def draw_contact_sheet(audit: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(CASE_ORDER),
        3,
        figsize=(7.8, 10.0),
        constrained_layout=True,
    )
    for row_index, case_id in enumerate(CASE_ORDER):
        subset = audit[audit["case_id"] == case_id].sort_values(
            "combined_score", ascending=False
        )
        for col_index, (_, row) in enumerate(subset.head(3).iterrows()):
            ax = axes[row_index, col_index]
            with np.load(str(row["surface_npz"])) as data:
                energy = np.asarray(data["energy"], dtype=float)
                rgb = clean_spin_rgb(
                    data["surface_dos_up"],
                    data["surface_dos_down"],
                )
            ax.imshow(
                rgb,
                origin="lower",
                extent=(-np.pi, np.pi, energy[0], energy[-1]),
                aspect="auto",
                interpolation="bicubic",
            )
            ax.set_xticks((-np.pi, 0.0, np.pi), ("X", r"$\Gamma$", "X"))
            if col_index == 0:
                ax.set_ylabel(CASE_LABELS[case_id] + "\nEnergy (eV)")
            ax.set_title(
                f"{row['source_id']}\nscore={row['combined_score']:.3f}",
                fontsize=8,
            )
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)
    fig.savefig(output_dir / "step20_03_surface_shortlist_contact_sheet.png", dpi=300)
    fig.savefig(output_dir / "step20_03_surface_shortlist_contact_sheet.svg")
    plt.close(fig)


def run_surface_shortlist(
    core: Any,
    specs: dict[str, Any],
    coarse: pd.DataFrame,
    output_dir: Path,
    shortlist: int,
    k_points: int,
    energy_points: int,
    fourier_n: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    data_dir = output_dir / "surface_shortlist"
    data_dir.mkdir(parents=True, exist_ok=True)
    chosen_rows: list[pd.DataFrame] = []
    for case_id in CASE_ORDER:
        group = coarse[coarse["case_id"] == case_id].sort_values(
            "coarse_score", ascending=False
        )
        target = TARGET_GAPS[case_id]
        priority_indices: list[int] = list(group.head(2).index)
        priority_indices.append(int(group["indirect_gap"].idxmax()))
        priority_indices.append(
            int((group["indirect_gap"] - target).abs().idxmin())
        )
        priority_indices.append(int(group["min_direct_gap"].idxmax()))
        unique_indices: list[int] = []
        for row_index in priority_indices + list(group.index):
            if row_index not in unique_indices:
                unique_indices.append(row_index)
            if len(unique_indices) >= int(shortlist):
                break
        chosen_rows.append(group.loc[unique_indices])
    short = pd.concat(chosen_rows, ignore_index=False)
    total = len(short)
    for index, (_, row) in enumerate(short.iterrows(), start=1):
        case_id = str(row["case_id"])
        selected = reconstruct_selected(core, row)
        spec = specs[selected.system]
        print(
            f"[surface {index:02d}/{total:02d}] "
            f"{case_id} {selected.source_id}",
            flush=True,
        )
        hoppings = core.extract_hoppings(
            spec, selected.params, nfft=int(fourier_n)
        )
        surface = core.calculate_surface(
            spec,
            hoppings,
            ef=float(row["energy_reference"]),
            energy_window=WINDOWS[case_id],
            k_points=int(k_points),
            energy_points=int(energy_points),
            eta=ETAS[case_id],
        )
        case_dir = data_dir / f"{case_id}__{selected.source_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        npz_path = case_dir / "surface_preview.npz"
        np.savez_compressed(npz_path, **surface)
        records.append(
            {
                **row.to_dict(),
                **surface_metrics(surface),
                "surface_npz": str(npz_path),
            }
        )
    audit = pd.DataFrame.from_records(records)
    audit = pd.concat(
        [score_surfaces(group) for _, group in audit.groupby("case_id")],
        ignore_index=True,
    ).sort_values(["case_id", "combined_score"], ascending=[True, False])
    audit.to_csv(output_dir / "step20_02_surface_shortlist_audit.csv", index=False)
    draw_contact_sheet(audit, output_dir)
    return audit


def write_selection_json(audit: pd.DataFrame, output_dir: Path) -> None:
    selections: dict[str, Any] = {}
    for case_id in CASE_ORDER:
        row = (
            audit[audit["case_id"] == case_id]
            .sort_values("combined_score", ascending=False)
            .iloc[0]
        )
        selections[case_id] = {
            "source_id": str(row["source_id"]),
            "system": str(row["system"]),
            "expected_chern_up": int(row["expected_chern_up"]),
            "min_direct_gap": float(row["min_direct_gap"]),
            "indirect_gap": float(row["indirect_gap"]),
            "coarse_score": float(row["coarse_score"]),
            "surface_score": float(row["surface_score"]),
            "combined_score": float(row["combined_score"]),
            "surface_npz": str(row["surface_npz"]),
            "params": json.loads(str(row["params_json"])),
            "critical_kx": float(row["critical_kx"]),
            "critical_ky": float(row["critical_ky"]),
        }
    (output_dir / "step20_04_provisional_best_points.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_final(
    core: Any,
    specs: dict[str, Any],
    output_dir: Path,
    k_points: int,
    energy_points: int,
    fourier_n: int,
) -> list[dict[str, Any]]:
    selection_path = output_dir / "step20_04_provisional_best_points.json"
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    step19 = load_module(
        "step20_plot_support",
        ROOT / "TTS_step19_maintext_four_topological_edge_hall.py",
    )
    final_dir = output_dir / "final_selected_data"
    final_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for index, case_id in enumerate(CASE_ORDER, start=1):
        item = selections[case_id]
        selected = selected_from_row(
            core,
            str(item["system"]),
            str(item["source_id"]),
            item["params"],
            int(item["expected_chern_up"]),
            float(item["critical_kx"]),
            float(item["critical_ky"]),
            "Final Step20 visually screened representative.",
        )
        spec = specs[selected.system]
        print(
            f"[final {index}/4] {case_id} {selected.source_id}",
            flush=True,
        )
        gap = core.gap_audit(spec, selected, nk=72)
        chern = core.chern_audit(spec, selected, nk=48)
        if (
            int(chern["chern_up_int"]) != int(selected.expected_chern_up)
            or int(chern["chern_down_int"])
            != int(selected.expected_chern_down)
        ):
            raise RuntimeError(
                f"Chern verification failed for {case_id}: {chern}"
            )
        bulk, ticks, labels = step19.calculate_square_bulk_bands(
            spec,
            selected,
            energy_reference=float(gap["energy_reference"]),
            points_per_segment=120,
        )
        hoppings = core.extract_hoppings(
            spec, selected.params, nfft=int(fourier_n)
        )
        surface = core.calculate_surface(
            spec,
            hoppings,
            ef=float(gap["energy_reference"]),
            energy_window=WINDOWS[case_id],
            k_points=int(k_points),
            energy_points=int(energy_points),
            eta=ETAS[case_id],
        )
        case_dir = final_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        surface_path = case_dir / "surface_spectrum_final.npz"
        bulk_path = case_dir / "bulk_bands_final.csv"
        np.savez_compressed(surface_path, **surface)
        bulk.to_csv(bulk_path, index=False)
        result = {
            "case_id": case_id,
            "selected": selected,
            "gap": gap,
            "chern": chern,
            "bulk": bulk,
            "bulk_ticks": ticks,
            "bulk_labels": labels,
            "surface": surface,
            "surface_path": str(surface_path),
            "bulk_path": str(bulk_path),
        }
        results.append(result)
        audit_records.append(
            {
                "case_id": case_id,
                "system": selected.system,
                "source_id": selected.source_id,
                "expected_chern_up": selected.expected_chern_up,
                **gap,
                **chern,
                **selected.params,
            }
        )
    pd.DataFrame(audit_records).to_csv(
        output_dir / "step20_05_final_physics_audit.csv", index=False
    )
    draw_final_preview(results, output_dir)
    return results


def draw_final_preview(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "font.size": 9.5,
            "axes.linewidth": 1.3,
        }
    )
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(6.8, 12.4),
        constrained_layout=False,
    )
    cmap = LinearSegmentedColormap.from_list(
        "spin_down_white_up", [BLUE, np.ones(3), RED], N=512
    )
    for row_index, result in enumerate(results):
        case_id = result["case_id"]
        selected = result["selected"]
        ax_bulk, ax_edge = axes[row_index]
        bulk = result["bulk"]
        for spin_sector, color in (("up", RED), ("down", BLUE)):
            subset = bulk[bulk["spin_sector"] == spin_sector]
            for _, band in subset.groupby("band"):
                ax_bulk.plot(
                    band["path_coordinate"],
                    band["energy"],
                    color=color,
                    lw=1.25,
                    ls="-",
                    alpha=0.98,
                )
        ax_bulk.set_xlim(
            float(result["bulk_ticks"][0]),
            float(result["bulk_ticks"][-1]),
        )
        ax_bulk.set_ylim(-WINDOWS[case_id], WINDOWS[case_id])
        ax_bulk.set_xticks(result["bulk_ticks"])
        ax_bulk.set_xticklabels(result["bulk_labels"])
        ax_bulk.set_ylabel(r"$E-E_{\mathrm{ref}}$ (eV)")
        ax_bulk.set_title(
            f"({chr(ord('a') + 2 * row_index)})  "
            + CASE_LABELS[case_id]
            + " bulk",
            fontsize=10,
        )

        surface = result["surface"]
        energy = np.asarray(surface["energy"], dtype=float)
        rgb = clean_spin_rgb(
            surface["surface_dos_up"],
            surface["surface_dos_down"],
            height=1200,
            width=900,
        )
        ax_edge.imshow(
            rgb,
            origin="lower",
            extent=(-np.pi, np.pi, energy[0], energy[-1]),
            aspect="auto",
            interpolation="bicubic",
        )
        ax_edge.set_xlim(-np.pi, np.pi)
        ax_edge.set_ylim(-WINDOWS[case_id], WINDOWS[case_id])
        ax_edge.set_xticks((-np.pi, 0.0, np.pi), ("X", r"$\Gamma$", "X"))
        ax_edge.set_ylabel(r"$E-E_{\mathrm{ref}}$ (eV)")
        ax_edge.set_title(
            f"({chr(ord('b') + 2 * row_index)})  "
            + CASE_LABELS[case_id]
            + " edge",
            fontsize=10,
        )
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(-1.0, 1.0), cmap=cmap),
            ax=ax_edge,
            fraction=0.045,
            pad=0.025,
        )
        colorbar.set_ticks((-1.0, 0.0, 1.0))
        colorbar.set_ticklabels(
            (r"$\downarrow$", r"$P_z=0$", r"$\uparrow$")
        )
        for axis in (ax_bulk, ax_edge):
            axis.tick_params(
                direction="in", top=True, right=True, width=1.1, length=4
            )
            for spine in axis.spines.values():
                spine.set_linewidth(1.3)
    fig.subplots_adjust(
        left=0.10,
        right=0.92,
        bottom=0.045,
        top=0.97,
        wspace=0.40,
        hspace=0.47,
    )
    fig.savefig(
        output_dir / "step20_06_rescreened_bulk_edge_preview.png",
        dpi=450,
        facecolor="white",
    )
    fig.savefig(
        output_dir / "step20_06_rescreened_bulk_edge_preview.svg",
        dpi=450,
        facecolor="white",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("coarse", "surface", "final", "all"),
        default="all",
    )
    parser.add_argument("--gap-nk", type=int, default=40)
    parser.add_argument("--projected-nk", type=int, default=49)
    parser.add_argument("--shortlist", type=int, default=3)
    parser.add_argument("--surface-k", type=int, default=81)
    parser.add_argument("--surface-energy", type=int, default=241)
    parser.add_argument("--fourier-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = load_module("step20_core", CORE_PATH)
    specs, modules = core.build_model_specs()
    if args.stage in ("surface", "final"):
        coarse = pd.read_csv(
            args.output_dir / "step20_01_coarse_candidate_audit.csv"
        )
    else:
        candidates = build_candidates(core, modules)
        inventory = {
            case_id: len(candidates[case_id]) for case_id in CASE_ORDER
        }
        print("Candidate inventory:", inventory, flush=True)
        coarse = run_coarse(
            core,
            specs,
            candidates,
            args.output_dir,
            nk_gap=args.gap_nk,
            nk_projected=args.projected_nk,
        )
    if args.stage == "coarse":
        return
    if args.stage == "final":
        run_final(
            core,
            specs,
            args.output_dir,
            k_points=141,
            energy_points=361,
            fourier_n=args.fourier_n,
        )
        return
    audit = run_surface_shortlist(
        core,
        specs,
        coarse,
        args.output_dir,
        shortlist=args.shortlist,
        k_points=args.surface_k,
        energy_points=args.surface_energy,
        fourier_n=args.fourier_n,
    )
    write_selection_json(audit, args.output_dir)
    if args.stage == "all":
        run_final(
            core,
            specs,
            args.output_dir,
            k_points=141,
            energy_points=361,
            fourier_n=args.fourier_n,
        )


if __name__ == "__main__":
    main()
