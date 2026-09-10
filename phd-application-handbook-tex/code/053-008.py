from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import TTS_step18_three_system_wanniertools_style_edge_ahc_examples as core
import TTS_step19_maintext_four_topological_edge_hall as step19
import TTS_step22_user_selection_dual_color_render as step22


ROOT = Path(r"D:\1_ML\tts")
OUTPUT = ROOT / "outputs_submission_hall_current_fig4"
FIGURES = OUTPUT / "figures"
DATA = OUTPUT / "data"
STEP22_AUDIT = (
    ROOT
    / "outputs_tts_step22_user_selected_dual_color"
    / "step22_01_final_physics_audit.csv"
)

CASE_ORDER = tuple(step22.SELECTIONS)
PANEL_TITLE = {
    "a_lieb_Cs_plus1": r"(a) Lieb, $C_s=+1$",
    "b_fes_Cs_plus1": r"(b) FES, fixed-index $C_s=+1$ metal",
    "c_tts_Cs_plus1": r"(c) TTS, $C_s=+1$",
    "d_tts_Cs_plus2": r"(d) TTS, $C_s=+2$",
}
ENERGY_WINDOW = {
    "a_lieb_Cs_plus1": 0.55,
    "b_fes_Cs_plus1": 0.10,
    "c_tts_Cs_plus1": 0.60,
    "d_tts_Cs_plus2": 0.60,
}
MAIN_MESH = {
    "a_lieb_Cs_plus1": 121,
    "b_fes_Cs_plus1": 161,
    "c_tts_Cs_plus1": 81,
    "d_tts_Cs_plus2": 81,
}
CHECK_MESH = {
    "a_lieb_Cs_plus1": 101,
    "b_fes_Cs_plus1": 141,
    "c_tts_Cs_plus1": 61,
    "d_tts_Cs_plus2": 61,
}
ENERGY_POINTS = 241
FOURIER_N = 8


def value_at_reference(curve: pd.DataFrame, column: str) -> float:
    index = int(np.argmin(np.abs(curve["energy"].to_numpy(float))))
    return float(curve.iloc[index][column])


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    step19.configure_step19_plot_style()
    specs, modules = core.build_model_specs()
    audit_table = pd.read_csv(STEP22_AUDIT).set_index("case_id")

    results: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for case_id in CASE_ORDER:
        item = step22.SELECTIONS[case_id]
        selected = step22.make_selected(core, modules, item)
        spec = specs[selected.system]
        audit = audit_table.loc[case_id]
        energy_reference = float(audit["energy_reference"])
        window = float(ENERGY_WINDOW[case_id])
        hoppings = core.extract_hoppings(
            spec, selected.params, nfft=FOURIER_N
        )

        curves: dict[int, pd.DataFrame] = {}
        for nk in (CHECK_MESH[case_id], MAIN_MESH[case_id]):
            print(f"{case_id}: Hall mesh {nk} x {nk}", flush=True)
            curve, _ = core.calculate_hall(
                spec,
                hoppings,
                energy_reference,
                window,
                nk,
                ENERGY_POINTS,
            )
            curves[nk] = curve

        final_curve = curves[MAIN_MESH[case_id]]
        check_curve = curves[CHECK_MESH[case_id]]
        case_dir = DATA / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        final_curve.to_csv(case_dir / "spin_resolved_hall.csv", index=False)

        up = value_at_reference(final_curve, "sigma_xy_up_e2_over_h")
        down = value_at_reference(final_curve, "sigma_xy_down_e2_over_h")
        charge = value_at_reference(final_curve, "sigma_xy_charge_e2_over_h")
        check_up = value_at_reference(check_curve, "sigma_xy_up_e2_over_h")
        check_down = value_at_reference(check_curve, "sigma_xy_down_e2_over_h")
        expected = int(selected.expected_chern_up)
        is_insulator = int(audit["is_global_insulator"]) == 1
        summaries.append(
            {
                "case_id": case_id,
                "system": selected.system,
                "source_id": selected.source_id,
                "C_up": expected,
                "C_down": -expected,
                "is_global_insulator": int(is_insulator),
                "energy_reference_eV": energy_reference,
                "energy_min_eV": -window,
                "energy_max_eV": window,
                "energy_points": ENERGY_POINTS,
                "main_mesh": MAIN_MESH[case_id],
                "check_mesh": CHECK_MESH[case_id],
                "sigma_up_at_reference_e2_over_h": up,
                "sigma_down_at_reference_e2_over_h": down,
                "sigma_charge_at_reference_e2_over_h": charge,
                "mesh_change_up": abs(up - check_up),
                "mesh_change_down": abs(down - check_down),
                "quantized_plateau_error_up": (
                    abs(up - expected) if is_insulator else np.nan
                ),
                "quantized_plateau_error_down": (
                    abs(down + expected) if is_insulator else np.nan
                ),
            }
        )
        results.append(
            {
                "case_id": case_id,
                "selected": selected,
                "audit": audit,
                "curve": final_curve,
                "window": window,
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "step35_01_hall_convergence_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(6.10, 6.25))
    for index, (ax, result) in enumerate(zip(axes.flat, results)):
        case_id = str(result["case_id"])
        selected = result["selected"]
        audit = result["audit"]
        curve = result["curve"]
        window = float(result["window"])
        spec = specs[selected.system]
        color_up, color_down = core.spin_colors(spec, selected)
        phase_color = core.phase_color(spec, selected.expected_chern_up)

        if int(audit["is_global_insulator"]) == 1:
            low = float(audit["vbm"]) - float(audit["energy_reference"])
            high = float(audit["cbm"]) - float(audit["energy_reference"])
            hatch = None
        else:
            half_gap = 0.5 * float(audit["critical_valley_gap"])
            low, high = -half_gap, half_gap
            hatch = "////"
        ax.axvspan(
            low,
            high,
            color=core.lighten(phase_color, 0.78),
            alpha=0.65,
            hatch=hatch,
            edgecolor=phase_color if hatch else "none",
            linewidth=0.0,
            zorder=0,
        )
        ax.plot(
            curve["energy"],
            curve["sigma_xy_up_e2_over_h"],
            color=color_up,
            lw=1.9,
            label=r"$\sigma_{xy}^{(s=\uparrow)}$",
        )
        ax.plot(
            curve["energy"],
            curve["sigma_xy_down_e2_over_h"],
            color=color_down,
            lw=1.9,
            label=r"$\sigma_{xy}^{(s=\downarrow)}$",
        )
        ax.axhline(0.0, color="#777777", lw=0.7)
        ax.axvline(0.0, color="#111111", lw=0.8, ls=":")
        ax.set_xlim(-window, window)
        ax.set_ylim(-2.35, 2.35)
        ax.set_xlabel(r"$\mu-E_{\mathrm{ref}}\ \mathrm{(eV)}$")
        ax.set_ylabel(r"$\sigma_{xy}^{(s)}\ (e^2/h)$")
        ax.set_title(PANEL_TITLE[case_id])
        if index == 0:
            ax.legend(loc="best", frameon=False, fontsize=8)
        core.style_axis(ax)
        ax.set_box_aspect(0.92)

    fig.tight_layout(h_pad=1.15, w_pad=1.05)
    stem = FIGURES / "FigS6_spin_resolved_hall_current_fig4"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "calculation": "zero-temperature occupation-weighted Kubo-Berry integral",
        "k_mesh": "uniform midpoint mesh in [-pi, pi)^2",
        "energy_points": ENERGY_POINTS,
        "fourier_reconstruction_n": FOURIER_N,
        "figure": str(stem.with_suffix(".pdf")),
        "cases": summaries,
    }
    (OUTPUT / "step35_02_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    insulating = summary[summary["is_global_insulator"] == 1]
    if float(insulating["quantized_plateau_error_up"].max()) >= 0.035:
        raise RuntimeError("An insulating spin-up Hall plateau failed tolerance")
    if float(insulating["quantized_plateau_error_down"].max()) >= 0.035:
        raise RuntimeError("An insulating spin-down Hall plateau failed tolerance")
    if float(summary["sigma_charge_at_reference_e2_over_h"].abs().max()) >= 5e-6:
        raise RuntimeError("Charge Hall cancellation failed")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
