#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FES Step 05 — Γ/M 解析本征值、branch-switching 与 no-go 条件审计

目标
----
1. 对 fes 六参数八带模型的单自旋 4x4 块，在 Γ 与 M 点解析因式分解；
2. 给出四条解析本征值，并验证其与数值 Hamiltonian 完全一致；
3. 检测半填充时 Γ/M 之间的 B+ 分支 occupied/unoccupied 交换；
4. 计算严格上界

       E_g^ind <= -|E_{Γ,B+} - E_{M,B+}|
               = -4 |r1^2-r2^2| / (B1+B2) <= 0,

   其中 B_i = sqrt((m_e+t_d)^2 + 4 r_i^2)；
5. 审计 Step 04 的所有严格拓扑点和所有粗筛拓扑点；
6. 补充严格复核 Step 04 因队列上限未覆盖的粗筛 TI 候选；
7. 沿 Step 04 两个锚点之间的路径，解析定位 Γ/M 直接闭隙与
   r1^2=r2^2（间接带隙 no-go 上界为零）的位置顺序。

本步骤用于建立“最小六参数 fes 模型中是否存在 type-II QSH 绝缘窗口”
的解析证据。它不会仅凭有限样本自动宣称数学定理；输出会明确区分：

- exact_identity：解析恒等式；
- verified_numerical_evidence：严格数值证据；
- conditional_no_go：在 B+ partner-switching 分支排序下的严格 no-go。

运行示例
--------
快速测试：
    python FES_step05_analytic_no_go.py \
      --step04-input outputs_fes6_step04_boundary.zip \
      --output-dir outputs_fes6_step05_no_go \
      --test --overwrite

正式运行：
    python FES_step05_analytic_no_go.py \
      --step04-input outputs_fes6_step04_boundary.zip \
      --output-dir outputs_fes6_step05_no_go \
      --workers 4 --overwrite
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Iterable, Sequence
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import sympy as sp

# The matching Step 02 core must be in the same directory or importable.
import FES_step02_sobol_scan as core

np.set_printoptions(precision=12, suppress=True)

CODE_VERSION = "FES_STEP05_ANALYTIC_NOGO_V1_20260713"
RESULT_SCHEMA_VERSION = "fes_step05_analytic_nogo_schema_v1"

PARAMS5 = ("m_e", "t1", "t2", "r1", "r2")
BRANCHES = ("G_A_minus", "G_A_plus", "G_B_minus", "G_B_plus",
            "M_A_minus", "M_A_plus", "M_B_minus", "M_B_plus")


@dataclass
class NoGoConfig:
    step04_input: str = "outputs_fes6_step04_boundary.zip"
    output_dir: str = "outputs_fes6_step05_no_go"
    workers: int = max(1, min(4, os.cpu_count() or 1))

    validation_samples: int = 128
    validation_seed: int = 20260713
    dense_path_points: int = 5001

    gap_tol: float = 1.0e-3
    chern_tol: float = 0.08
    min_det_tol: float = 1.0e-7
    verify_gap_grids: tuple[int, ...] = (51, 71)
    verify_gap_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    )
    verify_chern_grids: tuple[int, ...] = (21, 31, 41)
    verify_chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )

    reverify_all_coarse_ti: bool = False
    overwrite: bool = False
    resume: bool = False
    test_mode: bool = False


# =============================================================================
# 1. 输入、输出与版本管理
# =============================================================================


def script_sha256() -> str | None:
    if "__file__" not in globals():
        return None
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def prepare_output(config: NoGoConfig) -> Path:
    out = Path(config.output_dir).resolve()
    if out.exists() and config.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strict_details").mkdir(exist_ok=True)
    return out


def _find_result_root(root: Path) -> Path:
    required = {
        "fes_step04_all_coarse_scan.csv",
        "fes_step04_strict_verified_results.csv",
        "fes_step04_anchor_points.csv",
    }
    if required.issubset({p.name for p in root.iterdir() if p.is_file()}):
        return root
    candidates = []
    for path in root.rglob("fes_step04_all_coarse_scan.csv"):
        parent = path.parent
        if required.issubset({p.name for p in parent.iterdir() if p.is_file()}):
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(
            f"无法在 {root} 中找到完整 Step 04 结果目录；必须包含 {sorted(required)}"
        )
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def locate_step04_directory(input_path: str | Path, output_dir: Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return _find_result_root(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("--step04-input 必须是目录或 ZIP 文件")
    extract_root = output_dir / "_step04_extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_root)
    return _find_result_root(extract_root)


def write_metadata(config: NoGoConfig, output: Path, step04_dir: Path) -> None:
    metadata = {
        "code_version": CODE_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": script_sha256(),
        "step02_core_version": getattr(core, "CODE_VERSION", None),
        "step02_core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "step04_directory": str(step04_dir),
        "config": asdict(config),
    }
    (output / "fes_step05_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =============================================================================
# 2. Γ/M 解析本征值与特征多项式
# =============================================================================


def analytic_branches(reduced: Dict[str, float]) -> Dict[str, float]:
    """Return exact closed-form Γ/M eigenvalue branches for one spin block."""
    m = float(reduced["m_e"])
    t1 = float(reduced["t1"])
    t2 = float(reduced["t2"])
    r1 = float(reduced["r1"])
    r2 = float(reduced["r2"])

    t_s = 0.5 * (t1 + t2)
    t_d = 0.5 * (t1 - t2)

    A1 = math.sqrt((m - t_d) ** 2 + 4.0 * r1 * r1)
    A2 = math.sqrt((m - t_d) ** 2 + 4.0 * r2 * r2)
    B1 = math.sqrt((m + t_d) ** 2 + 4.0 * r1 * r1)
    B2 = math.sqrt((m + t_d) ** 2 + 4.0 * r2 * r2)

    values = {
        "t_s_half": t_s,
        "t_d_half": t_d,
        "A1": A1,
        "A2": A2,
        "B1": B1,
        "B2": B2,
        "G_A_minus": -t_s - A1,
        "G_A_plus": -t_s + A1,
        "G_B_minus": +t_s - B2,
        "G_B_plus": +t_s + B2,
        "M_A_minus": -t_s - A2,
        "M_A_plus": -t_s + A2,
        "M_B_minus": +t_s - B1,
        "M_B_plus": +t_s + B1,
    }
    return values


def _ordered_special(branch: Dict[str, float], prefix: str) -> list[tuple[str, float]]:
    keys = [name for name in BRANCHES if name.startswith(prefix + "_")]
    return sorted(((name, float(branch[name])) for name in keys), key=lambda item: item[1])


def analytic_diagnostics(reduced: Dict[str, float]) -> Dict[str, object]:
    b = analytic_branches(reduced)
    G = _ordered_special(b, "G")
    M = _ordered_special(b, "M")

    G_rank = {name: rank for rank, (name, _) in enumerate(G)}
    M_rank = {name: rank for rank, (name, _) in enumerate(M)}

    gamma_v_name, gamma_v = G[1]
    gamma_c_name, gamma_c = G[2]
    M_v_name, M_v = M[1]
    M_c_name, M_c = M[2]

    gamma_gap = gamma_c - gamma_v
    M_gap = M_c - M_v
    gamma_M_gap = min(gamma_c, M_c) - max(gamma_v, M_v)

    # Signed Gamma/M inversion masses for the A- and B+ branches.
    # mass_Gamma > 0 means B+ lies below A- at Gamma.
    # mass_M > 0 means A- lies below B+ at M.
    mass_Gamma = b["G_A_minus"] - b["G_B_plus"]
    mass_M = b["M_B_plus"] - b["M_A_minus"]
    mass_product = mass_Gamma * mass_M

    # The B+ branch is the branch that empirically switches occupancy in all
    # verified topological points of Steps 02–04.
    bplus_G_rank = G_rank["G_B_plus"]
    bplus_M_rank = M_rank["M_B_plus"]
    bplus_partner_switch = {bplus_G_rank, bplus_M_rank} == {1, 2}

    delta_Bplus = b["M_B_plus"] - b["G_B_plus"]  # B1 - B2
    denominator = b["B1"] + b["B2"]
    rationalized_delta = (
        4.0 * (float(reduced["r1"]) ** 2 - float(reduced["r2"]) ** 2) / denominator
        if denominator > 0.0 else 0.0
    )
    no_go_bound = -abs(delta_Bplus) if bplus_partner_switch else float("nan")

    active_pattern = (
        f"G:{gamma_v_name}->{gamma_c_name};M:{M_v_name}->{M_c_name}"
    )

    result: Dict[str, object] = dict(b)
    result.update(
        {
            "G_e1": G[0][1], "G_e2": G[1][1], "G_e3": G[2][1], "G_e4": G[3][1],
            "M_e1": M[0][1], "M_e2": M[1][1], "M_e3": M[2][1], "M_e4": M[3][1],
            "gamma_valence_branch": gamma_v_name,
            "gamma_conduction_branch": gamma_c_name,
            "M_valence_branch": M_v_name,
            "M_conduction_branch": M_c_name,
            "gamma_local_gap_analytic": gamma_gap,
            "M_local_gap_analytic": M_gap,
            "gamma_M_indirect_analytic": gamma_M_gap,
            "mass_Gamma_signed": mass_Gamma,
            "mass_M_signed": mass_M,
            "mass_product": mass_product,
            "mass_same_sign": int(mass_product > 0.0),
            "mass_boundary_distance": min(abs(mass_Gamma), abs(mass_M)),
            "active_branch_pattern": active_pattern,
            "G_Bplus_rank": bplus_G_rank,
            "M_Bplus_rank": bplus_M_rank,
            "Bplus_partner_switch": int(bplus_partner_switch),
            "delta_Bplus_M_minus_G": delta_Bplus,
            "delta_Bplus_rationalized": rationalized_delta,
            "Bplus_identity_error": delta_Bplus - rationalized_delta,
            "conditional_no_go_bound": no_go_bound,
            "r_square_difference": float(reduced["r1"]) ** 2 - float(reduced["r2"]) ** 2,
            "r_square_equal": int(abs(float(reduced["r1"]) ** 2 - float(reduced["r2"]) ** 2) < 1e-12),
        }
    )
    return result


def symbolic_derivation() -> Dict[str, str]:
    """Derive and return symbolic characteristic polynomials and formulas."""
    lam, m, t1, t2, r1, r2 = sp.symbols("lambda m_e t_1 t_2 r_1 r_2", real=True)
    ts = (t1 + t2) / 2
    td = (t1 - t2) / 2

    A1 = sp.sqrt((m - td) ** 2 + 4 * r1 ** 2)
    A2 = sp.sqrt((m - td) ** 2 + 4 * r2 ** 2)
    B1 = sp.sqrt((m + td) ** 2 + 4 * r1 ** 2)
    B2 = sp.sqrt((m + td) ** 2 + 4 * r2 ** 2)

    # Factorized forms derived from the 4x4 spin-up matrices.
    pG = sp.expand(((lam + ts) ** 2 - A1 ** 2) * ((lam - ts) ** 2 - B2 ** 2))
    pM = sp.expand(((lam + ts) ** 2 - A2 ** 2) * ((lam - ts) ** 2 - B1 ** 2))

    delta_B = sp.simplify(B1 - B2)
    delta_B_rational = sp.simplify(4 * (r1 ** 2 - r2 ** 2) / (B1 + B2))
    identity = sp.simplify(delta_B - delta_B_rational)

    formulas = {
        "p_Gamma_factorized": sp.sstr(sp.factor(pG)),
        "p_M_factorized": sp.sstr(sp.factor(pM)),
        "E_Gamma_A_pm": sp.sstr(-ts) + " ± " + sp.sstr(A1),
        "E_Gamma_B_pm": sp.sstr(ts) + " ± " + sp.sstr(B2),
        "E_M_A_pm": sp.sstr(-ts) + " ± " + sp.sstr(A2),
        "E_M_B_pm": sp.sstr(ts) + " ± " + sp.sstr(B1),
        "delta_Bplus": sp.sstr(delta_B),
        "delta_Bplus_rationalized": sp.sstr(delta_B_rational),
        "delta_identity_simplified": sp.sstr(identity),
        "conditional_no_go": (
            "If E_{Gamma,B+} and E_{M,B+} exchange occupied/unoccupied rank at half filling, "
            "then E_g^ind <= -|B1-B2| = -4|r1^2-r2^2|/(B1+B2) <= 0."
        ),
    }
    return formulas


def write_symbolic_outputs(output: Path) -> Dict[str, str]:
    formulas = symbolic_derivation()
    lines = [
        "fes six-parameter model: Gamma/M exact formulas",
        "=" * 72,
        "Definitions:",
        "  t_s=(t1+t2)/2, t_d=(t1-t2)/2",
        "  A_i=sqrt((m_e-t_d)^2+4 r_i^2)",
        "  B_i=sqrt((m_e+t_d)^2+4 r_i^2)",
        "",
        "Gamma eigenvalues:",
        "  E_Gamma,A± = -t_s ± A1",
        "  E_Gamma,B± = +t_s ± B2",
        "M eigenvalues:",
        "  E_M,A± = -t_s ± A2",
        "  E_M,B± = +t_s ± B1",
        "",
        "Exact identity:",
        "  E_M,B+ - E_Gamma,B+ = B1-B2",
        "  B1-B2 = 4(r1^2-r2^2)/(B1+B2)",
        "",
        "Conditional no-go:",
        "  If B+ changes occupied/unoccupied rank between Gamma and M,",
        "  E_g^ind <= -|B1-B2|",
        "          = -4|r1^2-r2^2|/(B1+B2) <= 0.",
        "  It is strictly negative when r1^2 != r2^2.",
        "",
        "SymPy output:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in formulas.items())
    (output / "fes_step05_symbolic_formulas.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    latex = r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}
\section*{fes $\Gamma/M$ analytic spectrum and conditional no-go}
Define
\[
 t_s=\frac{t_1+t_2}{2},\qquad t_d=\frac{t_1-t_2}{2},
\]
\[
 A_i=\sqrt{(m_e-t_d)^2+4r_i^2},\qquad
 B_i=\sqrt{(m_e+t_d)^2+4r_i^2}.
\]
The one-spin eigenvalues are
\[
 E_{\Gamma,A}^{\pm}=-t_s\pm A_1,\qquad
 E_{\Gamma,B}^{\pm}= t_s\pm B_2,
\]
\[
 E_{M,A}^{\pm}=-t_s\pm A_2,\qquad
 E_{M,B}^{\pm}= t_s\pm B_1.
\]
The $B^+$ branch difference obeys the exact identity
\[
 E_{M,B^+}-E_{\Gamma,B^+}=B_1-B_2
 =\frac{4(r_1^2-r_2^2)}{B_1+B_2}.
\]
If $B^+$ changes from occupied to unoccupied (or vice versa) between
$\Gamma$ and $M$ at half filling, the global indirect gap satisfies
\[
 E_g^{\mathrm{ind}}\le -\left|B_1-B_2\right|
 =-\frac{4\left|r_1^2-r_2^2\right|}{B_1+B_2}\le0.
\]
For $r_1^2\ne r_2^2$ the upper bound is strictly negative.
\end{document}
"""
    (output / "fes_step05_symbolic_formulas.tex").write_text(latex, encoding="utf-8")
    (output / "fes_step05_symbolic_formulas.json").write_text(
        json.dumps(formulas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return formulas


# =============================================================================
# 3. 解析公式数值验证
# =============================================================================


def _row_reduced(row: pd.Series | Dict[str, object]) -> Dict[str, float]:
    return {name: float(row[name]) for name in PARAMS5}


def validate_analytic_spectrum(
    source_df: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    if len(source_df) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    n = min(int(n_samples), len(source_df))
    indices = rng.choice(len(source_df), size=n, replace=False)
    rows = []
    for idx in indices:
        row = source_df.iloc[int(idx)]
        reduced = _row_reduced(row)
        raw6 = core.raw6_from_reduced5(reduced)
        analytic = analytic_branches(reduced)

        H_G = core.h_fes_atomic(0.0, 0.0, raw6)
        H_M = core.h_fes_atomic(math.pi, math.pi, raw6)
        up = core.SPIN_UP_INDICES
        G_num = np.linalg.eigvalsh(H_G[np.ix_(up, up)])
        M_num = np.linalg.eigvalsh(H_M[np.ix_(up, up)])
        G_ana = np.sort([analytic[name] for name in BRANCHES if name.startswith("G_")])
        M_ana = np.sort([analytic[name] for name in BRANCHES if name.startswith("M_")])

        rows.append(
            {
                "source_row": int(idx),
                **reduced,
                "max_abs_error_Gamma": float(np.max(np.abs(G_num - G_ana))),
                "max_abs_error_M": float(np.max(np.abs(M_num - M_ana))),
                "max_abs_error": float(max(np.max(np.abs(G_num - G_ana)), np.max(np.abs(M_num - M_ana)))),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 4. Step 04 数据解析审计
# =============================================================================


def augment_with_analytic(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    records = []
    for _, row in df.iterrows():
        records.append(analytic_diagnostics(_row_reduced(row)))
    analytic_df = pd.DataFrame(records, index=df.index)
    overlap = [col for col in analytic_df.columns if col in df.columns]
    if overlap:
        analytic_df = analytic_df.rename(columns={col: f"analytic_{col}" for col in overlap})
    return pd.concat([df.reset_index(drop=True), analytic_df.reset_index(drop=True)], axis=1)


def add_no_go_certificate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    verified_gap_col = (
        "verified_indirect_gap" if "verified_indirect_gap" in out.columns else "indirect_gap"
    )
    observed = pd.to_numeric(out.get(verified_gap_col), errors="coerce")
    bound = pd.to_numeric(out.get("conditional_no_go_bound"), errors="coerce")
    out["observed_indirect_for_certificate"] = observed
    out["no_go_bound_residual_observed_minus_bound"] = observed - bound
    out["conditional_no_go_certificate"] = (
        (pd.to_numeric(out["Bplus_partner_switch"], errors="coerce") == 1)
        & observed.notna()
        & bound.notna()
        & (observed <= bound + 5.0e-8)
        & (bound <= 1.0e-12)
    ).astype(int)
    out["gamma_M_exactness_error"] = observed - pd.to_numeric(
        out["gamma_M_indirect_analytic"], errors="coerce"
    )
    return out


# =============================================================================
# 5. 完整复核所有粗筛 TI 候选（修复 Step 04 队列上限遗漏）
# =============================================================================


def strict_verify_row(
    row: pd.Series,
    config: NoGoConfig,
    details_dir: Path,
) -> Dict[str, object]:
    point_id = str(row["point_id"])
    reduced = _row_reduced(row)
    raw6 = core.raw6_from_reduced5(reduced)
    result: Dict[str, object] = dict(row)
    result["strict_error"] = ""

    try:
        gap_summary, gap_table = core.consensus_gap_audit(
            raw6,
            config.verify_gap_grids,
            config.verify_gap_shifts,
            config.gap_tol,
        )
        result.update(gap_summary)
        gap_table.insert(0, "point_id", point_id)
        gap_table.to_csv(
            details_dir / f"{point_id}_gap_consensus.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        result["strict_phase_label"] = "numeric_error"
        result["strict_error"] = f"gap_failed: {repr(exc)}"
        return result

    if not bool(int(result.get("verified_is_direct_gapped", 0))):
        result["strict_phase_label"] = "noninsulating_or_gap_closing"
        return result
    if not bool(int(result.get("verified_is_balanced_spin_sector", 0))):
        result["strict_phase_label"] = "spin_sector_filling_mismatch"
        return result

    try:
        ch_summary, ch_table = core.consensus_chern_audit(
            raw6,
            config.verify_chern_grids,
            config.verify_chern_shifts,
            config.chern_tol,
            config.min_det_tol,
            include_total=True,
        )
        result.update(ch_summary)
        ch_table.insert(0, "point_id", point_id)
        ch_table.to_csv(
            details_dir / f"{point_id}_chern_consensus.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        result["strict_phase_label"] = "chern_unreliable"
        result["strict_error"] = f"chern_failed: {repr(exc)}"
        return result

    reliable = bool(int(result.get("verified_chern_reliable", 0)))
    cu = result.get("verified_chern_up_int")
    cd = result.get("verified_chern_down_int")
    ct = result.get("verified_chern_total_int")
    insulator = bool(int(result.get("verified_is_physical_insulator", 0)))

    topological = reliable and cu is not None and cd is not None and ct == 0 and cu == -cd and cu != 0
    if not reliable:
        label = "chern_unreliable"
    elif topological and insulator:
        label = "spin_chern_TI_candidate"
    elif topological:
        label = "spin_chern_band_metal"
    elif insulator and cu == 0 and cd == 0 and ct == 0:
        label = "trivial_insulator"
    elif not insulator:
        label = "indirect_overlap"
    else:
        label = "other_gapped_phase"
    result["strict_phase_label"] = label
    return result


def complete_coarse_ti_verification(
    coarse: pd.DataFrame,
    existing_strict: pd.DataFrame,
    config: NoGoConfig,
    output: Path,
) -> pd.DataFrame:
    candidates = coarse[coarse["phase_label"] == "spin_chern_TI_candidate"].copy()
    existing_ids = set(existing_strict["point_id"].astype(str)) if len(existing_strict) else set()

    if config.test_mode:
        # Quick test validates parsing, symbolic formulas, analytic spectra and
        # output generation. It deliberately reuses the existing strict Step 04
        # table so reduced test grids cannot create scientific false positives.
        return existing_strict.copy()

    if config.reverify_all_coarse_ti:
        queue = candidates
    else:
        queue = candidates[~candidates["point_id"].astype(str).isin(existing_ids)]

    details_dir = output / "strict_details"
    new_rows = [strict_verify_row(row, config, details_dir) for _, row in queue.iterrows()]
    new_df = pd.DataFrame(new_rows)

    combined = existing_strict.copy()
    if len(new_df):
        combined = pd.concat([combined, new_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["point_id"], keep="last")
    return combined


# =============================================================================
# 6. 锚点路径的稠密解析边界定位
# =============================================================================


def interpolate_reduced(a: Dict[str, float], b: Dict[str, float], lam: float) -> Dict[str, float]:
    return {name: (1.0 - lam) * a[name] + lam * b[name] for name in PARAMS5}


def _roots_from_grid(x: np.ndarray, y: np.ndarray) -> list[float]:
    roots: list[float] = []
    for i in range(len(x) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0.0:
            roots.append(float(x[i]))
            continue
        if y0 * y1 < 0.0:
            # Linear interpolation is sufficient for bracketing summary; the
            # dense grid makes the residual negligible.
            roots.append(float(x[i] - y0 * (x[i + 1] - x[i]) / (y1 - y0)))
    return roots


def dense_anchor_path(anchor_df: pd.DataFrame, n_points: int) -> tuple[pd.DataFrame, Dict[str, object]]:
    if len(anchor_df) < 2:
        return pd.DataFrame(), {}
    topo_row = anchor_df[anchor_df["anchor_role"] == "best_topological"]
    pos_row = anchor_df[anchor_df["anchor_role"] == "positive_gap_anchor"]
    if len(topo_row) == 0 or len(pos_row) == 0:
        topo_row = anchor_df.iloc[[0]]
        pos_row = anchor_df.iloc[[1]]
    a = _row_reduced(topo_row.iloc[0])
    b = _row_reduced(pos_row.iloc[0])

    lambdas = np.linspace(0.0, 1.0, int(n_points))
    rows = []
    for lam in lambdas:
        reduced = interpolate_reduced(a, b, float(lam))
        diag = analytic_diagnostics(reduced)
        rows.append({"path_lambda": float(lam), **reduced, **diag})
    df = pd.DataFrame(rows)

    summary = {
        "mass_Gamma_zero_roots": _roots_from_grid(
            df["path_lambda"].to_numpy(), df["mass_Gamma_signed"].to_numpy()
        ),
        "mass_M_zero_roots": _roots_from_grid(
            df["path_lambda"].to_numpy(), df["mass_M_signed"].to_numpy()
        ),
        "min_gamma_local_gap": float(df["gamma_local_gap_analytic"].min()),
        "lambda_min_gamma_local_gap": float(df.loc[df["gamma_local_gap_analytic"].idxmin(), "path_lambda"]),
        "min_M_local_gap": float(df["M_local_gap_analytic"].min()),
        "lambda_min_M_local_gap": float(df.loc[df["M_local_gap_analytic"].idxmin(), "path_lambda"]),
        "gamma_M_indirect_zero_roots": _roots_from_grid(
            df["path_lambda"].to_numpy(), df["gamma_M_indirect_analytic"].to_numpy()
        ),
        "delta_Bplus_zero_roots": _roots_from_grid(
            df["path_lambda"].to_numpy(), df["delta_Bplus_M_minus_G"].to_numpy()
        ),
        "min_gamma_M_indirect": float(df["gamma_M_indirect_analytic"].min()),
        "max_gamma_M_indirect": float(df["gamma_M_indirect_analytic"].max()),
    }
    return df, summary


def plot_dense_path(df: pd.DataFrame, output_path: Path) -> None:
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = df["path_lambda"].to_numpy()
    ax.plot(x, df["mass_Gamma_signed"], label=r"$\mu_\Gamma$")
    ax.plot(x, df["mass_M_signed"], label=r"$\mu_M$")
    ax.plot(x, df["gamma_M_indirect_analytic"], label=r"$g_{\Gamma M}^{ind}$")
    ax.plot(x, df["conditional_no_go_bound"], linestyle="--", label="B+ no-go bound")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"Path coordinate $\lambda$")
    ax.set_ylabel("Energy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# =============================================================================
# 7. 汇总与主流程
# =============================================================================


def build_branch_pattern_counts(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    cols = [label_col, "active_branch_pattern", "Bplus_partner_switch"]
    return (
        df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([label_col, "count"], ascending=[True, False])
    )


def run_pipeline(config: NoGoConfig) -> Dict[str, Path]:
    output = prepare_output(config)
    step04_dir = locate_step04_directory(config.step04_input, output)
    write_metadata(config, output, step04_dir)
    write_symbolic_outputs(output)

    coarse = pd.read_csv(step04_dir / "fes_step04_all_coarse_scan.csv")
    strict = pd.read_csv(step04_dir / "fes_step04_strict_verified_results.csv")
    anchors = pd.read_csv(step04_dir / "fes_step04_anchor_points.csv")

    # Validate exact analytic spectrum against numerical matrices.
    validation_n = 16 if config.test_mode else config.validation_samples
    validation = validate_analytic_spectrum(coarse, validation_n, config.validation_seed)
    validation_path = output / "fes_step05_eigenvalue_validation.csv"
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

    # Complete every coarse TI candidate; default only fills missing Step 04 rows.
    strict_complete = complete_coarse_ti_verification(coarse, strict, config, output)
    strict_complete_path = output / "fes_step05_all_coarse_TI_strict_verification.csv"
    strict_complete.to_csv(strict_complete_path, index=False, encoding="utf-8-sig")

    coarse_aug = add_no_go_certificate(augment_with_analytic(coarse))
    coarse_aug_path = output / "fes_step05_all_coarse_analytic_audit.csv"
    coarse_aug.to_csv(coarse_aug_path, index=False, encoding="utf-8-sig")

    strict_aug = add_no_go_certificate(augment_with_analytic(strict_complete))
    strict_aug_path = output / "fes_step05_strict_analytic_audit.csv"
    strict_aug.to_csv(strict_aug_path, index=False, encoding="utf-8-sig")

    topological = strict_aug[
        strict_aug["strict_phase_label"].isin(["spin_chern_band_metal", "spin_chern_TI_candidate"])
    ].copy()
    topological_path = output / "fes_step05_verified_topological_no_go_certificates.csv"
    topological.to_csv(topological_path, index=False, encoding="utf-8-sig")

    pattern_counts = build_branch_pattern_counts(strict_aug, "strict_phase_label")
    pattern_path = output / "fes_step05_branch_pattern_counts.csv"
    pattern_counts.to_csv(pattern_path, index=False, encoding="utf-8-sig")

    dense_n = 501 if config.test_mode else config.dense_path_points
    dense_path, path_summary = dense_anchor_path(anchors, dense_n)
    dense_path_path = output / "fes_step05_anchor_path_analytic_dense.csv"
    dense_path.to_csv(dense_path_path, index=False, encoding="utf-8-sig")
    plot_dense_path(dense_path, output / "fes_step05_anchor_path_analytic.png")

    n_top = len(topological)
    n_cert = int(topological.get("conditional_no_go_certificate", pd.Series(dtype=int)).sum())
    n_switch = int(topological.get("Bplus_partner_switch", pd.Series(dtype=int)).sum())
    n_strict_ti = int((strict_aug["strict_phase_label"] == "spin_chern_TI_candidate").sum())

    reliable_mask = (
        pd.to_numeric(strict_aug.get("verified_chern_reliable"), errors="coerce") == 1
    ) & pd.to_numeric(strict_aug.get("verified_chern_up_int"), errors="coerce").notna()
    reliable = strict_aug[reliable_mask].copy()
    if len(reliable):
        actual_top = pd.to_numeric(reliable["verified_chern_up_int"], errors="coerce").abs() > 0
        predicted_top = pd.to_numeric(reliable["mass_product"], errors="coerce") > 0
        mass_accuracy = float((actual_top.to_numpy() == predicted_top.to_numpy()).mean())
        mass_false_positive = int((~actual_top & predicted_top).sum())
        mass_false_negative = int((actual_top & ~predicted_top).sum())
    else:
        mass_accuracy = None
        mass_false_positive = 0
        mass_false_negative = 0

    summary = {
        "code_version": CODE_VERSION,
        "step04_total_coarse_points": int(len(coarse)),
        "step04_coarse_TI_candidates": int((coarse["phase_label"] == "spin_chern_TI_candidate").sum()),
        "strict_rows_after_completion": int(len(strict_complete)),
        "strict_spin_chern_TI_candidates": n_strict_ti,
        "strict_topological_points": n_top,
        "topological_Bplus_partner_switch_count": n_switch,
        "topological_conditional_no_go_certificate_count": n_cert,
        "topological_certificate_fraction": float(n_cert / n_top) if n_top else None,
        "reliable_chern_rows_for_mass_test": int(len(reliable)),
        "mass_same_sign_topology_accuracy": mass_accuracy,
        "mass_same_sign_false_positive_count": mass_false_positive,
        "mass_same_sign_false_negative_count": mass_false_negative,
        "candidate_topology_rule": "|C_up|=1 iff mass_Gamma*mass_M>0 (empirically tested, not yet analytically proven)",
        "max_verified_topological_indirect_gap": (
            float(pd.to_numeric(topological["verified_indirect_gap"], errors="coerce").max())
            if n_top and "verified_indirect_gap" in topological else None
        ),
        "max_abs_Gamma_eigenvalue_validation_error": (
            float(validation["max_abs_error_Gamma"].max()) if len(validation) else None
        ),
        "max_abs_M_eigenvalue_validation_error": (
            float(validation["max_abs_error_M"].max()) if len(validation) else None
        ),
        "max_Bplus_identity_error": (
            float(np.nanmax(np.abs(pd.to_numeric(strict_aug["Bplus_identity_error"], errors="coerce"))))
            if len(strict_aug) else None
        ),
        "max_topological_gamma_M_exactness_error": (
            float(np.nanmax(np.abs(pd.to_numeric(topological["gamma_M_exactness_error"], errors="coerce"))))
            if n_top else None
        ),
        "conditional_no_go_statement": (
            "Within the half-filled branch regime where B+ exchanges occupied/unoccupied rank "
            "between Gamma and M, E_ind <= -4|r1^2-r2^2|/(B1+B2) <= 0."
        ),
        "scope_warning": (
            "This is an exact conditional statement for the detected branch ordering. "
            "A complete model-wide theorem additionally requires proving that every nonzero-spin-Chern "
            "phase of the six-parameter Hamiltonian necessarily has this branch ordering."
        ),
        "dense_anchor_path": path_summary,
    }

    summary_path = output / "fes_step05_no_go_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "FES Step 05 analytic no-go audit",
        "=" * 72,
        f"Strict topological points: {n_top}",
        f"B+ partner-switching points: {n_switch}/{n_top}",
        f"Conditional no-go certificates: {n_cert}/{n_top}",
        f"Strict spin-Chern TI candidates: {n_strict_ti}",
        f"Best verified topological indirect gap: {summary['max_verified_topological_indirect_gap']}",
        "",
        "Candidate topology rule from strict data:",
        f"  |C_up|=1 iff mu_Gamma*mu_M>0 accuracy: {mass_accuracy}",
        f"  false positives / false negatives: {mass_false_positive} / {mass_false_negative}",
        "",
        "Exact conditional result:",
        "  If B+ exchanges occupied/unoccupied rank between Gamma and M,",
        "  E_ind <= -|B1-B2| = -4|r1^2-r2^2|/(B1+B2) <= 0.",
        "",
        "Logical status:",
        "  - Gamma/M eigenvalue formulas: exact.",
        "  - B1-B2 rationalization: exact.",
        "  - No-go under B+ partner switching: exact.",
        "  - All currently verified topological points satisfying partner switching: numerical evidence.",
        "  - Model-wide no-go theorem: requires a separate proof that nonzero Chern always implies this ordering.",
    ]
    (output / "fes_step05_no_go_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "output_dir": output,
        "summary": summary_path,
        "validation": validation_path,
        "strict_audit": strict_aug_path,
        "topological_certificates": topological_path,
        "dense_path": dense_path_path,
        "report": output / "fes_step05_no_go_report.txt",
    }


# =============================================================================
# 8. CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step04-input", default="outputs_fes6_step04_boundary.zip")
    parser.add_argument("--output-dir", default="outputs_fes6_step05_no_go")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--dense-path-points", type=int, default=5001)
    parser.add_argument("--reverify-all-coarse-ti", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite 与 --resume 不能同时使用")

    if args.test:
        config = NoGoConfig(
            step04_input=args.step04_input,
            output_dir=args.output_dir,
            workers=1,
            validation_samples=16,
            dense_path_points=501,
            verify_gap_grids=(17, 21),
            verify_gap_shifts=((0.0, 0.0), (0.5, 0.5)),
            verify_chern_grids=(11, 15),
            verify_chern_shifts=((0.0, 0.0),),
            reverify_all_coarse_ti=False,
            overwrite=bool(args.overwrite),
            resume=bool(args.resume),
            test_mode=True,
        )
    else:
        config = NoGoConfig(
            step04_input=args.step04_input,
            output_dir=args.output_dir,
            workers=max(1, args.workers),
            validation_samples=max(1, args.validation_samples),
            dense_path_points=max(101, args.dense_path_points),
            reverify_all_coarse_ti=bool(args.reverify_all_coarse_ti),
            overwrite=bool(args.overwrite),
            resume=bool(args.resume),
            test_mode=False,
        )

    print("=" * 78)
    print("FES Step 05: analytic Gamma/M no-go audit")
    print("Version:", CODE_VERSION)
    print("Core   :", getattr(core, "CODE_VERSION", "unknown"))
    print("Input  :", Path(config.step04_input).resolve())
    print("Output :", Path(config.output_dir).resolve())
    print("=" * 78)

    paths = run_pipeline(config)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
