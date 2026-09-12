#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 02 v3 — 稳健物理标签、Hamiltonian 指纹与可解释机器学习

本脚本对应用户提供的 Lieb 工作流中的：
    02v3_Lieb8_Robust_Hamiltonian_fingerprint_and_ML.ipynb

但模型、标签与特征已经改写为 tts 八带交错磁体系：
- tts tessellation: 3.3.4.3.4
- MSG: P4'/mbm' (BNS 127.391)
- 8 bands, half filling = 4 occupied bands
- each conserved-spin block is 4x4 with 2 occupied bands
- frozen Step 01 core: TTS_step01_model_and_label_audit_v2.py

本阶段的研究主线：
    四来源参数采样
    -> 完整 BZ 稳健物理标签
    -> 去冗余 Hamiltonian fingerprint
    -> 采样来源捷径诊断
    -> 重复交叉验证与 global_random 外部测试
    -> 分组 permutation importance

与此前报错版本不同：
1. 全部物理批量计算严格串行；
2. sklearn/joblib 默认 n_jobs=1；
3. 不使用 ProcessPoolExecutor，因此可直接在 Windows Jupyter 中运行；
4. 先保存 gap，再决定是否计算 Chern，近闭隙不会被误记为 numeric_error；
5. fingerprint 使用 4x4 自旋块的谱不变量和占据子空间投影量，
   不把 tts 错误压缩成 Lieb 的 2x2 Pauli 模型。

注意：
- spin_chern_TI_candidate 仍是模型级候选；边缘态、DOS、SHC 在后续步骤验证。
- 本阶段用于发现候选控制变量，不用于直接宣称最终解析机制。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence
import argparse
import hashlib
import json
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import (
        RepeatedStratifiedKFold,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Step 02 需要 scikit-learn 与 joblib。请运行：\n"
        "  conda install scikit-learn joblib\n"
        "或：\n"
        "  pip install scikit-learn joblib"
    ) from exc

try:
    import TTS_step01_model_and_label_audit_v2 as core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未找到 TTS_step01_model_and_label_audit_v2.py。\n"
        "请把 Step 01 v2 的 py 文件与本脚本放在同一目录。"
    ) from exc

EXPECTED_CORE_VERSION = "TTS_STEP01_V2_20260713"
if getattr(core, "CODE_VERSION", None) != EXPECTED_CORE_VERSION:
    raise RuntimeError(
        "Step 01 核心版本不一致：\n"
        f"  expected = {EXPECTED_CORE_VERSION}\n"
        f"  loaded   = {getattr(core, 'CODE_VERSION', None)}\n"
        "请使用配套代码包中的 Step 01 v2 文件。"
    )

np.set_printoptions(precision=10, suppress=True)

CODE_VERSION = "TTS_STEP02V3_20260713"
WORKFLOW_STEP = "step02v3"
SYSTEM_TAG = "tts8_bns127391"
TASK_TAG = "robust_hamiltonian_fingerprint_ml"
REDUCED7 = core.REDUCED7.copy()
RAW8 = core.RAW8.copy()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Step02Config:
    output_dir: Path = Path(
        "outputs_tts_step02v3_robust_hamiltonian_fingerprint_ml"
    )

    # Four source regions, following the uploaded Lieb Step 02 v3 workflow.
    n_topo_local: int = 40
    n_trivial_local: int = 40
    n_bridge: int = 40
    n_global: int = 40
    include_exact_anchors: bool = True
    random_seed: int = 20260713

    # Physics labels.
    gap_nk: int = 60
    chern_grids: tuple[int, ...] = (21, 31)
    chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )
    direct_gap_skip_chern: float = 5.0e-3
    gap_tol: float = 1.0e-3
    chern_integer_tol: float = 0.08
    min_link_tol: float = 1.0e-7
    sum_rule_tol: float = 0.15
    min_chern_consensus: int = 2

    # Fingerprint.
    path_n: int = 31

    # ML evaluation.
    cv_splits: int = 5
    cv_repeats: int = 5
    validation_fraction: float = 0.25
    random_forest_estimators: int = 400
    permutation_repeats: int = 15
    ml_n_jobs: int = 1

    # Workflow controls.
    checkpoint_every: int = 10
    force_recalculate_physics: bool = False
    force_recalculate_fingerprint: bool = False

    def normalized(self) -> "Step02Config":
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)

        for name in ("n_topo_local", "n_trivial_local", "n_bridge", "n_global"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.gap_nk < 8:
            raise ValueError("gap_nk must be >= 8")
        if self.path_n < 5:
            raise ValueError("path_n must be >= 5")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if self.ml_n_jobs == 0:
            raise ValueError("ml_n_jobs cannot be 0")
        return self

    @property
    def n_requested(self) -> int:
        return (
            int(self.n_topo_local)
            + int(self.n_trivial_local)
            + int(self.n_bridge)
            + int(self.n_global)
            + (2 if self.include_exact_anchors else 0)
        )


# Reduced seven-dimensional centers. The global energy shift e0 is fixed to zero.
TOPO_CENTER = core.reduced7_from_raw8(core.TOPO_PARAMS)
TRIVIAL_CENTER = core.reduced7_from_raw8(core.PAPER_PARAMS)

# Local windows are deliberately moderate. Step 02 is a robustness/fingerprint stage,
# not the final global existence scan.
TOPO_HALF_WIDTH = {
    "m_e": 0.08,
    "t1": 0.08,
    "t2": 0.12,
    "r1": 0.10,
    "r2": 0.10,
    "r3": 0.12,
    "r4": 0.12,
}
TRIVIAL_HALF_WIDTH = {
    "m_e": 0.20,
    "t1": 0.18,
    "t2": 0.18,
    "r1": 0.20,
    "r2": 0.20,
    "r3": 0.20,
    "r4": 0.20,
}
GLOBAL_RANGES = {name: (-1.0, 1.0) for name in REDUCED7}


# =============================================================================
# Output registry and safe file helpers
# =============================================================================

def _config_tag(config: Step02Config) -> str:
    return (
        f"n{config.n_requested}_gap{config.gap_nk}"
        f"_chern{'-'.join(map(str, config.chern_grids))}"
        f"_seed{config.random_seed}"
    )


def build_output_registry(config: Step02Config) -> Dict[str, Path]:
    config = config.normalized()
    run_tag = f"{SYSTEM_TAG}_{_config_tag(config)}"
    figures = config.output_dir / "figures"
    models = config.output_dir / "models"

    def out(substep: int, content: str, ext: str, *, figure: bool = False) -> Path:
        base = figures if figure else config.output_dir
        return base / (
            f"{WORKFLOW_STEP}_{substep:02d}_{content}__{run_tag}.{ext.lstrip('.')}"
        )

    return {
        "run_config": out(0, "run_configuration", "json"),
        "registry": out(0, "output_file_registry", "csv"),
        "candidates": out(3, "candidate_parameters_four_source_regions", "csv"),
        "physics_checkpoint": out(4, "physics_labels_checkpoint", "csv"),
        "physics_final": out(4, "physics_labels_final_robust", "csv"),
        "chern_attempts": out(4, "chern_adaptive_attempt_records", "csv"),
        "phase_counts": out(4, "phase_counts", "csv"),
        "phase_counts_fig": out(4, "phase_counts", "png", figure=True),
        "fingerprint": out(5, "hamiltonian_fingerprint_independent_channels", "csv"),
        "feature_metadata": out(5, "fingerprint_feature_metadata", "csv"),
        "binary_dataset": out(6, "strict_binary_dataset_ti_vs_trivial", "csv"),
        "source_label_diagnostic": out(6, "source_label_shortcut_diagnostic", "csv"),
        "cv_metrics": out(8, "repeated_cross_validation_metrics", "csv"),
        "holdout_metrics": out(8, "holdout_metrics", "csv"),
        "external_global_metrics": out(8, "external_global_test_metrics", "csv"),
        "classification_report": out(9, "best_model_classification_report", "txt"),
        "validation_predictions": out(9, "best_model_validation_predictions", "csv"),
        "confusion_matrix": out(9, "best_model_confusion_matrix", "png", figure=True),
        "individual_importance": out(10, "individual_permutation_importance", "csv"),
        "group_importance_family": out(10, "group_importance_by_family", "csv"),
        "group_importance_kregion": out(10, "group_importance_by_kregion", "csv"),
        "group_importance_spin": out(10, "group_importance_by_spin", "csv"),
        "group_importance_family_fig": out(10, "group_importance_by_family", "png", figure=True),
        "class_feature_means": out(11, "top_feature_class_means", "csv"),
        "validation_subset": out(12, "magnetictb_validation_subset", "csv"),
        "summary": out(13, "run_summary", "json"),
        "model": models / f"{WORKFLOW_STEP}_13_best_random_forest__{run_tag}.joblib",
    }


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_metadata(config: Step02Config, outputs: Dict[str, Path]) -> None:
    core_path = Path(core.__file__).resolve()
    payload = {
        "code_version": CODE_VERSION,
        "workflow_step": WORKFLOW_STEP,
        "system_tag": SYSTEM_TAG,
        "task_tag": TASK_TAG,
        "step01_core_version": core.CODE_VERSION,
        "step01_core_file": str(core_path),
        "step01_core_sha256": file_sha256(core_path),
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "chern_shifts": [list(x) for x in config.chern_shifts],
        },
        "topological_center": TOPO_CENTER,
        "trivial_center": TRIVIAL_CENTER,
        "interpretation_warning": (
            "spin_chern_TI_candidate is a model-level candidate; "
            "edge states, DOS and SHC are not yet verified."
        ),
    }
    outputs["run_config"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    registry = pd.DataFrame(
        [
            {
                "output_key": key,
                "filename": path.name,
                "relative_path": str(path),
            }
            for key, path in outputs.items()
        ]
    )
    atomic_write_csv(registry, outputs["registry"])


# =============================================================================
# Step 3. Four-source parameter sampling
# =============================================================================

def _sample_local(
    center: Dict[str, float],
    widths: Dict[str, float],
    n: int,
    source: str,
    prefix: str,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    for i in range(int(n)):
        row = {"sample_id": f"{prefix}_{i:06d}", "source_region": source}
        for key in REDUCED7:
            row[key] = float(
                rng.uniform(center[key] - widths[key], center[key] + widths[key])
            )
        rows.append(row)
    return rows


def _sample_bridge(n: int, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for i in range(int(n)):
        alpha = float(rng.uniform(0.0, 1.0))
        row = {
            "sample_id": f"bridge_{i:06d}",
            "source_region": "bridge",
            "bridge_alpha": alpha,
        }
        for key in REDUCED7:
            center = (1.0 - alpha) * TOPO_CENTER[key] + alpha * TRIVIAL_CENTER[key]
            if key == "m_e":
                sigma = 0.06
            elif key.startswith("t"):
                sigma = 0.05
            else:
                sigma = 0.08
            row[key] = float(center + rng.normal(0.0, sigma))
        rows.append(row)
    return rows


def _sample_global(n: int, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for i in range(int(n)):
        row = {
            "sample_id": f"global_{i:06d}",
            "source_region": "global_random",
        }
        for key in REDUCED7:
            lo, hi = GLOBAL_RANGES[key]
            row[key] = float(rng.uniform(lo, hi))
        rows.append(row)
    return rows


def generate_candidates(config: Step02Config, outputs: Dict[str, Path]) -> pd.DataFrame:
    config = config.normalized()
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict] = []

    if config.include_exact_anchors:
        rows.append({
            "sample_id": "anchor_topological_exact",
            "source_region": "exact_anchor",
            **TOPO_CENTER,
        })
        rows.append({
            "sample_id": "anchor_trivial_paper_exact",
            "source_region": "exact_anchor",
            **TRIVIAL_CENTER,
        })

    rows += _sample_local(
        TOPO_CENTER,
        TOPO_HALF_WIDTH,
        config.n_topo_local,
        "topo_local",
        "topolocal",
        rng,
    )
    rows += _sample_local(
        TRIVIAL_CENTER,
        TRIVIAL_HALF_WIDTH,
        config.n_trivial_local,
        "trivial_local",
        "trivlocal",
        rng,
    )
    rows += _sample_bridge(config.n_bridge, rng)
    rows += _sample_global(config.n_global, rng)

    candidates = pd.DataFrame(rows)
    for key in REDUCED7:
        candidates[key] = pd.to_numeric(candidates[key], errors="raise")
    candidates = candidates.drop_duplicates("sample_id", keep="last").reset_index(drop=True)
    atomic_write_csv(candidates, outputs["candidates"])
    return candidates


# =============================================================================
# Step 4. Robust physical labels
# =============================================================================

def _integer_if_close(value: float, tol: float) -> int | None:
    if not np.isfinite(value):
        return None
    nearest = int(np.rint(value))
    return nearest if abs(float(value) - nearest) <= tol else None


def one_chern_attempt(
    params: Dict[str, float],
    nk: int,
    shift: tuple[float, float],
    config: Step02Config,
) -> dict:
    row: dict = {
        "chern_nk": int(nk),
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
        "attempt_reliable": 0,
        "attempt_error": "",
    }
    try:
        row.update(core.calculate_spin_cherns(params, int(nk), shift))
    except Exception as exc:
        row["attempt_error"] = repr(exc)
        return row

    cu = _integer_if_close(float(row["chern_up"]), config.chern_integer_tol)
    cd = _integer_if_close(float(row["chern_down"]), config.chern_integer_tol)
    ct = _integer_if_close(float(row["chern_total"]), config.chern_integer_tol)
    row.update({"chern_up_int": cu, "chern_down_int": cd, "chern_total_int": ct})

    reliable = bool(
        cu is not None
        and cd is not None
        and ct is not None
        and float(row["min_det_up"]) > config.min_link_tol
        and float(row["min_det_down"]) > config.min_link_tol
        and float(row["min_det_total"]) > config.min_link_tol
        and float(row["sum_rule_error"]) <= config.sum_rule_tol
    )
    row["attempt_reliable"] = int(reliable)
    return row


def adaptive_spin_chern(
    params: Dict[str, float],
    config: Step02Config,
) -> tuple[dict, list[dict]]:
    attempts: list[dict] = []
    for nk in config.chern_grids:
        for shift in config.chern_shifts:
            attempts.append(one_chern_attempt(params, int(nk), shift, config))

    reliable = [row for row in attempts if int(row.get("attempt_reliable", 0)) == 1]
    if not reliable:
        return {
            "chern_reliable": 0,
            "chern_consensus_count": 0,
            "chern_status": "unresolved",
            "chern_up": np.nan,
            "chern_down": np.nan,
            "chern_total": np.nan,
            "spin_chern": np.nan,
            "chern_up_int": np.nan,
            "chern_down_int": np.nan,
            "chern_total_int": np.nan,
            "min_link_consensus": np.nan,
        }, attempts

    tuples = [
        (
            int(row["chern_up_int"]),
            int(row["chern_down_int"]),
            int(row["chern_total_int"]),
        )
        for row in reliable
    ]
    counts = pd.Series(tuples, dtype="object").value_counts()
    consensus_tuple = counts.index[0]
    consensus_count = int(counts.iloc[0])
    consensus_ok = consensus_count >= int(config.min_chern_consensus)

    matching = [
        row for row in reliable
        if (
            int(row["chern_up_int"]),
            int(row["chern_down_int"]),
            int(row["chern_total_int"]),
        ) == consensus_tuple
    ]
    chosen = matching[-1]
    min_link = min(
        min(float(row["min_det_up"]), float(row["min_det_down"]), float(row["min_det_total"]))
        for row in matching
    )
    cu, cd, ct = consensus_tuple

    return {
        "chern_reliable": int(consensus_ok),
        "chern_consensus_count": consensus_count,
        "chern_status": "consensus" if consensus_ok else "single_valid_attempt",
        "chern_up": float(chosen["chern_up"]),
        "chern_down": float(chosen["chern_down"]),
        "chern_total": float(chosen["chern_total"]),
        "spin_chern": 0.5 * (float(chosen["chern_up"]) - float(chosen["chern_down"])),
        "chern_up_int": int(cu),
        "chern_down_int": int(cd),
        "chern_total_int": int(ct),
        "min_link_consensus": float(min_link),
    }, attempts


def evaluate_sample_physics(row: pd.Series, config: Step02Config) -> tuple[dict, list[dict]]:
    reduced = {name: float(row[name]) for name in REDUCED7}
    params = core.raw8_from_reduced7(reduced, e0=0.0)
    sample_id = str(row["sample_id"])

    out: dict = {
        "sample_id": sample_id,
        "source_region": str(row["source_region"]),
        **reduced,
        **params,
        **core.derived_features(params),
        "phase_label": "true_numeric_error",
        "error": "",
        "chern_reliable": 0,
        "is_spin_chern_topological": 0,
        "is_spin_chern_TI_candidate": 0,
        "is_typeII_QSH_confirmed": 0,
    }
    if "bridge_alpha" in row and pd.notna(row.get("bridge_alpha", np.nan)):
        out["bridge_alpha"] = float(row["bridge_alpha"])

    attempts: list[dict] = []
    try:
        # Save all gap/filling information before any Chern calculation.
        gap = core.scan_band_gaps(
            params,
            nk=int(config.gap_nk),
            shift=(0.0, 0.0),
            gap_tol=config.gap_tol,
        )
        out.update(gap)

        if float(gap["min_direct_gap"]) <= config.gap_tol:
            out["phase_label"] = "noninsulating_or_gap_closing"
            out["chern_status"] = "not_computed_for_gap_closing"
            return out, attempts

        if int(gap["is_balanced_spin_sector"]) != 1:
            out["phase_label"] = "spin_sector_filling_mismatch"
            out["chern_status"] = "not_computed_for_filling_mismatch"
            return out, attempts

        if float(gap["min_direct_gap"]) <= config.direct_gap_skip_chern:
            out["phase_label"] = "boundary_or_small_gap"
            out["chern_status"] = "skipped_small_direct_gap"
            return out, attempts

        chern, raw_attempts = adaptive_spin_chern(params, config)
        for attempt in raw_attempts:
            item = dict(attempt)
            item["sample_id"] = sample_id
            attempts.append(item)
        out.update(chern)

        if int(chern["chern_reliable"]) != 1:
            out["phase_label"] = "chern_unresolved"
            return out, attempts

        cu = int(chern["chern_up_int"])
        cd = int(chern["chern_down_int"])
        ct = int(chern["chern_total_int"])
        topological = bool(cu == -cd and abs(cu) >= 1 and ct == 0)
        indirect_positive = bool(float(gap["indirect_gap"]) > config.gap_tol)

        out["is_spin_chern_topological"] = int(topological)
        if topological and indirect_positive:
            out["phase_label"] = "spin_chern_TI_candidate"
            out["is_spin_chern_TI_candidate"] = 1
        elif topological:
            out["phase_label"] = "spin_chern_band_metal"
        elif cu == 0 and cd == 0 and ct == 0 and indirect_positive:
            out["phase_label"] = "trivial_insulator"
        elif cu == 0 and cd == 0 and ct == 0:
            out["phase_label"] = "indirect_overlap"
        elif ct != 0:
            out["phase_label"] = "Chern_insulator"
        else:
            out["phase_label"] = "other_gapped_phase"

    except Exception as exc:
        out["phase_label"] = "true_numeric_error"
        out["chern_status"] = "exception"
        out["error"] = repr(exc)

    return out, attempts


def run_physics_labels(
    config: Step02Config,
    outputs: Dict[str, Path],
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint = outputs["physics_checkpoint"]
    attempts_path = outputs["chern_attempts"]

    if checkpoint.exists() and not config.force_recalculate_physics:
        physics_df = pd.read_csv(checkpoint, low_memory=False)
        done_ids = set(physics_df["sample_id"].astype(str))
        print(f"Loaded physics checkpoint: {len(done_ids)} samples")
    else:
        physics_df = pd.DataFrame()
        done_ids: set[str] = set()

    if attempts_path.exists() and not config.force_recalculate_physics:
        attempts_df = pd.read_csv(attempts_path, low_memory=False)
    else:
        attempts_df = pd.DataFrame()

    pending: list[dict] = []
    pending_attempts: list[dict] = []
    total = len(candidates)
    started = time.time()

    for _, row in candidates.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id in done_ids:
            continue

        result, attempts = evaluate_sample_physics(row, config)
        pending.append(result)
        pending_attempts.extend(attempts)

        if len(pending) >= config.checkpoint_every:
            physics_df = pd.concat([physics_df, pd.DataFrame(pending)], ignore_index=True)
            physics_df = physics_df.drop_duplicates("sample_id", keep="last")
            atomic_write_csv(physics_df, checkpoint)
            done_ids.update(str(x["sample_id"]) for x in pending)
            pending = []

            if pending_attempts:
                attempts_df = pd.concat(
                    [attempts_df, pd.DataFrame(pending_attempts)],
                    ignore_index=True,
                )
                attempts_df = attempts_df.drop_duplicates(
                    ["sample_id", "chern_nk", "shift_x", "shift_y"],
                    keep="last",
                )
                atomic_write_csv(attempts_df, attempts_path)
                pending_attempts = []
            print(f"Physics labels: {len(done_ids)}/{total}")

    if pending:
        physics_df = pd.concat([physics_df, pd.DataFrame(pending)], ignore_index=True)
    physics_df = physics_df.drop_duplicates("sample_id", keep="last")
    atomic_write_csv(physics_df, checkpoint)
    atomic_write_csv(physics_df, outputs["physics_final"])

    if pending_attempts:
        attempts_df = pd.concat([attempts_df, pd.DataFrame(pending_attempts)], ignore_index=True)
    if not attempts_df.empty:
        attempts_df = attempts_df.drop_duplicates(
            ["sample_id", "chern_nk", "shift_x", "shift_y"],
            keep="last",
        )
    atomic_write_csv(attempts_df, attempts_path)

    counts = (
        physics_df["phase_label"]
        .value_counts(dropna=False)
        .rename_axis("phase_label")
        .reset_index(name="count")
    )
    counts["fraction"] = counts["count"] / max(len(physics_df), 1)
    atomic_write_csv(counts, outputs["phase_counts"])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(counts["phase_label"].astype(str), counts["count"])
    ax.set_ylabel("Count")
    ax.set_title("TTS Step 02 v3 robust phase labels")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(outputs["phase_counts_fig"], dpi=220)
    plt.close(fig)

    print(f"Physics stage elapsed: {time.time() - started:.1f} s")
    return physics_df, attempts_df


# =============================================================================
# Step 5. Gauge-consistent 4x4-spin-block Hamiltonian fingerprint
# =============================================================================

K_POINTS = {
    "G": (0.0, 0.0),
    "X": (np.pi, 0.0),
    "Y": (0.0, np.pi),
    "M": (np.pi, np.pi),
    "Mp": (-np.pi, np.pi),
    "S50": (0.50 * np.pi, 0.50 * np.pi),
    "Sp50": (-0.50 * np.pi, 0.50 * np.pi),
    "Z50": (np.pi, 0.50 * np.pi),
    "Zp50": (0.50 * np.pi, np.pi),
}

PATHS = {
    "GM_Sigma": (np.array([0.0, 0.0]), np.array([np.pi, np.pi])),
    "GMp_SigmaPrime": (np.array([0.0, 0.0]), np.array([-np.pi, np.pi])),
    "GX_Delta": (np.array([0.0, 0.0]), np.array([np.pi, 0.0])),
    "GY_DeltaPrime": (np.array([0.0, 0.0]), np.array([0.0, np.pi])),
    "XM_Z": (np.array([np.pi, 0.0]), np.array([np.pi, np.pi])),
    "YMp_ZPrime": (np.array([0.0, np.pi]), np.array([-np.pi, np.pi])),
}


def _spectral_moments(evals: np.ndarray) -> dict:
    e = np.asarray(evals, dtype=float)
    center = float(np.mean(e))
    centered = e - center
    return {
        "center": center,
        "bandwidth": float(e[-1] - e[0]),
        "moment2": float(np.mean(centered ** 2)),
        "moment3": float(np.mean(centered ** 3)),
    }


def _occupied_projector(h: np.ndarray, n_occ: int) -> np.ndarray:
    _, vec = np.linalg.eigh(h)
    occ = vec[:, : int(n_occ)]
    return occ @ occ.conj().T


def _projector_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q, ord="fro") / math.sqrt(2.0))


def fixed_k_features(params: Dict[str, float], klabel: str, kx: float, ky: float) -> tuple[dict, list[dict]]:
    feat: dict = {}
    meta: list[dict] = []

    spin_evals: dict[str, np.ndarray] = {}
    for spin in ("up", "down"):
        h4 = core.h_spin_block_periodic(kx, ky, params, spin)
        evals = np.linalg.eigvalsh(h4)
        spin_evals[spin] = evals
        values = {
            "occ_center": float(np.mean(evals[:2])),
            "unocc_center": float(np.mean(evals[2:])),
            "block_gap": float(evals[2] - evals[1]),
            "bandwidth": float(evals[-1] - evals[0]),
        }
        for channel, value in values.items():
            name = f"{spin}_{channel}_{klabel}"
            feat[name] = value
            meta.append({
                "feature": name,
                "spin": spin,
                "channel": channel,
                "k_region": klabel,
                "feature_family": "fixed_k_spin_spectrum",
            })

    full = np.linalg.eigvalsh(core.h_tts_periodic(kx, ky, params))
    full_values = {
        "full_direct_gap": float(full[4] - full[3]),
        "full_bandwidth": float(full[-1] - full[0]),
        "spin_spectrum_l2": float(np.linalg.norm(spin_evals["up"] - spin_evals["down"])),
        "spin_spectrum_max": float(np.max(np.abs(spin_evals["up"] - spin_evals["down"]))),
    }
    for channel, value in full_values.items():
        name = f"{channel}_{klabel}"
        feat[name] = value
        meta.append({
            "feature": name,
            "spin": "both",
            "channel": channel,
            "k_region": klabel,
            "feature_family": "fixed_k_full_spectrum",
        })
    return feat, meta


def path_summary(
    params: Dict[str, float],
    start: np.ndarray,
    end: np.ndarray,
    n: int,
) -> dict:
    t_values = np.linspace(0.0, 1.0, int(n))
    full_gaps: list[float] = []
    up_gaps: list[float] = []
    down_gaps: list[float] = []
    split_l2: list[float] = []
    split_max: list[float] = []

    p_up: list[np.ndarray] = []
    p_down: list[np.ndarray] = []
    p_full: list[np.ndarray] = []

    for t in t_values:
        k = (1.0 - t) * start + t * end
        kx, ky = float(k[0]), float(k[1])
        h_up = core.h_spin_block_periodic(kx, ky, params, "up")
        h_down = core.h_spin_block_periodic(kx, ky, params, "down")
        h_full = core.h_tts_periodic(kx, ky, params)

        eu = np.linalg.eigvalsh(h_up)
        ed = np.linalg.eigvalsh(h_down)
        ef = np.linalg.eigvalsh(h_full)
        up_gaps.append(float(eu[2] - eu[1]))
        down_gaps.append(float(ed[2] - ed[1]))
        full_gaps.append(float(ef[4] - ef[3]))
        split_l2.append(float(np.linalg.norm(eu - ed)))
        split_max.append(float(np.max(np.abs(eu - ed))))

        p_up.append(_occupied_projector(h_up, 2))
        p_down.append(_occupied_projector(h_down, 2))
        p_full.append(_occupied_projector(h_full, 4))

    full_arr = np.asarray(full_gaps)
    up_arr = np.asarray(up_gaps)
    down_arr = np.asarray(down_gaps)
    l2_arr = np.asarray(split_l2)
    max_arr = np.asarray(split_max)

    def path_length(projectors: list[np.ndarray]) -> float:
        return float(sum(
            _projector_distance(projectors[i], projectors[i + 1])
            for i in range(len(projectors) - 1)
        ))

    return {
        "full_gap_min": float(np.min(full_arr)),
        "full_gap_mean": float(np.mean(full_arr)),
        "full_gap_argmin_t": float(t_values[int(np.argmin(full_arr))]),
        "up_gap_min": float(np.min(up_arr)),
        "down_gap_min": float(np.min(down_arr)),
        "spin_split_l2_max": float(np.max(l2_arr)),
        "spin_split_l2_mean": float(np.mean(l2_arr)),
        "spin_split_max_max": float(np.max(max_arr)),
        "up_projector_endpoint_distance": _projector_distance(p_up[0], p_up[-1]),
        "down_projector_endpoint_distance": _projector_distance(p_down[0], p_down[-1]),
        "full_projector_endpoint_distance": _projector_distance(p_full[0], p_full[-1]),
        "full_projector_path_length": path_length(p_full),
    }


def extract_fingerprint(sample_id: str, params: Dict[str, float], path_n: int) -> tuple[dict, list[dict]]:
    feat: dict = {"sample_id": sample_id}
    meta: list[dict] = []

    for klabel, (kx, ky) in K_POINTS.items():
        fixed, fixed_meta = fixed_k_features(params, klabel, float(kx), float(ky))
        feat.update(fixed)
        meta.extend(fixed_meta)

    for path_name, (start, end) in PATHS.items():
        summary = path_summary(params, start, end, path_n)
        for channel, value in summary.items():
            name = f"{channel}_{path_name}"
            feat[name] = value
            spin = (
                "up" if channel.startswith("up_")
                else "down" if channel.startswith("down_")
                else "both"
            )
            feat_meta = {
                "feature": name,
                "spin": spin,
                "channel": channel,
                "k_region": path_name,
                "feature_family": (
                    "path_projector" if "projector" in channel else "path_spectrum"
                ),
            }
            meta.append(feat_meta)
    return feat, meta


def run_fingerprint(
    config: Step02Config,
    outputs: Dict[str, Path],
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp_path = outputs["fingerprint"]
    meta_path = outputs["feature_metadata"]

    if (
        fp_path.exists()
        and meta_path.exists()
        and not config.force_recalculate_fingerprint
    ):
        return (
            pd.read_csv(fp_path, low_memory=False),
            pd.read_csv(meta_path, low_memory=False),
        )

    fp_rows: list[dict] = []
    meta_rows: list[dict] = []
    started = time.time()
    total = len(candidates)

    for count, (_, row) in enumerate(candidates.iterrows(), start=1):
        reduced = {name: float(row[name]) for name in REDUCED7}
        params = core.raw8_from_reduced7(reduced, e0=0.0)
        feat, meta = extract_fingerprint(str(row["sample_id"]), params, config.path_n)
        fp_rows.append(feat)
        meta_rows.extend(meta)
        if count % max(config.checkpoint_every, 1) == 0 or count == total:
            print(f"Fingerprint: {count}/{total}")

    fingerprint_df = pd.DataFrame(fp_rows)
    metadata_df = pd.DataFrame(meta_rows).drop_duplicates("feature", keep="first")
    atomic_write_csv(fingerprint_df, fp_path)
    atomic_write_csv(metadata_df, meta_path)
    print(f"Fingerprint stage elapsed: {time.time() - started:.1f} s")
    return fingerprint_df, metadata_df


# =============================================================================
# Steps 6-13. Binary ML, shortcut diagnostics and explainability
# =============================================================================

def build_binary_dataset(
    outputs: Dict[str, Path],
    physics_df: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_df = physics_df.merge(fingerprint_df, on="sample_id", how="inner")
    binary = full_df[
        full_df["phase_label"].isin(
            ["spin_chern_TI_candidate", "trivial_insulator"]
        )
    ].copy()
    binary["target_typeII"] = (
        binary["phase_label"] == "spin_chern_TI_candidate"
    ).astype(int)
    atomic_write_csv(binary, outputs["binary_dataset"])

    if binary.empty:
        diagnostic = pd.DataFrame(columns=[
            "source_region", "max_label_fraction", "dominant_label", "shortcut_warning"
        ])
    else:
        fractions = pd.crosstab(
            binary["source_region"], binary["phase_label"], normalize="index"
        )
        rows = []
        for source, values in fractions.iterrows():
            rows.append({
                "source_region": source,
                "max_label_fraction": float(values.max()),
                "dominant_label": str(values.idxmax()),
                "shortcut_warning": int(float(values.max()) >= 0.90),
            })
        diagnostic = pd.DataFrame(rows)
    atomic_write_csv(diagnostic, outputs["source_label_diagnostic"])

    if not diagnostic.empty and diagnostic["shortcut_warning"].any():
        warnings.warn(
            "至少一个采样来源中单一标签占比 >= 90%。随机拆分可能学习采样来源捷径；"
            "必须结合 global_random 外部测试解释。"
        )
    return binary, diagnostic


def make_model(name: str, config: Step02Config, random_state: int = 42):
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.random_forest_estimators),
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=int(config.ml_n_jobs),
        )
    raise ValueError(name)


def metric_dict(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }


def _adaptive_cv(binary_df: pd.DataFrame, config: Step02Config):
    counts = binary_df["target_typeII"].value_counts()
    if len(counts) < 2:
        return None
    n_splits = min(int(config.cv_splits), int(counts.min()))
    if n_splits < 2:
        return None
    return RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=int(config.cv_repeats),
        random_state=42,
    )


def grouped_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Dict[str, list[str]],
    repeats: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = balanced_accuracy_score(y, model.predict(X))
    rows: list[dict] = []

    for group_name, features in groups.items():
        features = [f for f in features if f in X.columns]
        if not features:
            continue
        drops: list[float] = []
        for _ in range(int(repeats)):
            permuted = X.copy()
            order = rng.permutation(len(permuted))
            # Permute the group jointly to preserve within-group correlations.
            permuted.loc[:, features] = permuted.iloc[order][features].to_numpy()
            score = balanced_accuracy_score(y, model.predict(permuted))
            drops.append(float(baseline - score))
        rows.append({
            "group": group_name,
            "n_features": len(features),
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
            "baseline_balanced_accuracy": float(baseline),
        })
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def run_machine_learning(
    config: Step02Config,
    outputs: Dict[str, Path],
    binary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> dict:
    summary: dict = {
        "ml_status": "not_run",
        "n_binary": int(len(binary_df)),
        "n_topological": int(binary_df.get("target_typeII", pd.Series(dtype=int)).sum()),
        "n_trivial": int((1 - binary_df.get("target_typeII", pd.Series(dtype=int))).sum()) if len(binary_df) else 0,
    }

    empty_csvs = [
        "cv_metrics", "holdout_metrics", "external_global_metrics",
        "validation_predictions", "individual_importance",
        "group_importance_family", "group_importance_kregion",
        "group_importance_spin", "class_feature_means", "validation_subset",
    ]

    if binary_df.empty or binary_df["target_typeII"].nunique() < 2:
        summary["ml_status"] = "insufficient_two_class_data"
        for key in empty_csvs:
            atomic_write_csv(pd.DataFrame(), outputs[key])
        outputs["classification_report"].write_text(
            "ML not run: strict binary dataset does not contain both classes.\n",
            encoding="utf-8",
        )
        return summary

    fp_features = metadata_df["feature"].astype(str).tolist()
    fp_features = [f for f in fp_features if f in binary_df.columns]
    feature_sets = {
        "Reduced7": REDUCED7.copy(),
        "HamiltonianFingerprint": fp_features,
        "Combined": REDUCED7.copy() + fp_features,
    }

    cv = _adaptive_cv(binary_df, config)
    cv_rows: list[dict] = []
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    if cv is not None:
        for feature_name, features in feature_sets.items():
            X = binary_df[features].astype(float)
            y = binary_df["target_typeII"].astype(int)
            for model_name in ("logistic", "random_forest"):
                result = cross_validate(
                    make_model(model_name, config),
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=int(config.ml_n_jobs),
                    return_train_score=False,
                    error_score=np.nan,
                )
                row = {"feature_set": feature_name, "model": model_name}
                for metric in scoring:
                    values = np.asarray(result[f"test_{metric}"], dtype=float)
                    row[f"{metric}_mean"] = float(np.nanmean(values))
                    row[f"{metric}_std"] = float(np.nanstd(values))
                cv_rows.append(row)
    cv_metrics = pd.DataFrame(cv_rows)
    if not cv_metrics.empty:
        cv_metrics = cv_metrics.sort_values("balanced_accuracy_mean", ascending=False)
    atomic_write_csv(cv_metrics, outputs["cv_metrics"])

    # Holdout uses Combined features; fallback to all data if stratified split is impossible.
    best_features = feature_sets["Combined"]
    X_all = binary_df[best_features].astype(float)
    y_all = binary_df["target_typeII"].astype(int)
    class_counts = y_all.value_counts()
    can_holdout = bool(class_counts.min() >= 2 and len(binary_df) >= 8)

    if can_holdout:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=float(config.validation_fraction),
            random_state=42,
            stratify=y_all,
        )
    else:
        X_train, X_val, y_train, y_val = X_all, X_all, y_all, y_all
        warnings.warn("样本过少，holdout 使用训练集回代；该指标不能作为泛化证据。")

    best_model = make_model("random_forest", config)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    holdout = pd.DataFrame([{
        **metric_dict(y_val, y_pred),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "holdout_is_independent": int(can_holdout),
    }])
    atomic_write_csv(holdout, outputs["holdout_metrics"])

    val_rows = binary_df.loc[X_val.index, [
        "sample_id", "source_region", "phase_label", *REDUCED7
    ]].copy()
    val_rows["target_typeII"] = y_val
    val_rows["predicted_typeII"] = y_pred
    val_rows["probability_typeII"] = y_prob
    atomic_write_csv(val_rows, outputs["validation_predictions"])

    report = classification_report(
        y_val,
        y_pred,
        target_names=["trivial", "spin_chern_TI_candidate"],
        zero_division=0,
    )
    outputs["classification_report"].write_text(report, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_val, y_pred, labels=[0, 1]),
        display_labels=["trivial", "TI candidate"],
    ).plot(ax=ax, values_format="d")
    ax.set_title("TTS Step 02 v3 validation")
    fig.tight_layout()
    fig.savefig(outputs["confusion_matrix"], dpi=220)
    plt.close(fig)

    # External global_random test, trained only on non-global samples.
    train_external = binary_df[binary_df["source_region"] != "global_random"].copy()
    test_external = binary_df[binary_df["source_region"] == "global_random"].copy()
    external_rows: list[dict] = []
    if train_external["target_typeII"].nunique() == 2 and len(test_external) > 0:
        external_model = make_model("random_forest", config)
        external_model.fit(
            train_external[best_features].astype(float),
            train_external["target_typeII"].astype(int),
        )
        ext_pred = external_model.predict(test_external[best_features].astype(float))
        row = metric_dict(test_external["target_typeII"].astype(int), ext_pred)
        row.update({
            "n_train": int(len(train_external)),
            "n_test_global": int(len(test_external)),
            "n_test_topological": int(test_external["target_typeII"].sum()),
            "n_test_trivial": int((1 - test_external["target_typeII"]).sum()),
            "test_contains_both_classes": int(test_external["target_typeII"].nunique() == 2),
        })
        external_rows.append(row)
    else:
        external_rows.append({
            "n_train": int(len(train_external)),
            "n_test_global": int(len(test_external)),
            "warning": "external test unavailable or one-class training set",
        })
    atomic_write_csv(pd.DataFrame(external_rows), outputs["external_global_metrics"])

    # Individual importance.
    individual = permutation_importance(
        best_model,
        X_val,
        y_val,
        n_repeats=int(config.permutation_repeats),
        random_state=42,
        scoring="balanced_accuracy",
        n_jobs=int(config.ml_n_jobs),
    )
    individual_df = pd.DataFrame({
        "feature": best_features,
        "importance_mean": individual.importances_mean,
        "importance_std": individual.importances_std,
    }).sort_values("importance_mean", ascending=False)
    atomic_write_csv(individual_df, outputs["individual_importance"])

    meta = metadata_df.set_index("feature")
    family_groups: Dict[str, list[str]] = {"reduced7_parameters": REDUCED7.copy()}
    kregion_groups: Dict[str, list[str]] = {}
    spin_groups: Dict[str, list[str]] = {"reduced7_parameters": REDUCED7.copy()}
    for feature in fp_features:
        if feature not in meta.index:
            continue
        row = meta.loc[feature]
        family_groups.setdefault(str(row["feature_family"]), []).append(feature)
        kregion_groups.setdefault(str(row["k_region"]), []).append(feature)
        spin_groups.setdefault(str(row["spin"]), []).append(feature)

    family_imp = grouped_permutation_importance(
        best_model, X_val, y_val, family_groups, config.permutation_repeats
    )
    kregion_imp = grouped_permutation_importance(
        best_model, X_val, y_val, kregion_groups, config.permutation_repeats
    )
    spin_imp = grouped_permutation_importance(
        best_model, X_val, y_val, spin_groups, config.permutation_repeats
    )
    atomic_write_csv(family_imp, outputs["group_importance_family"])
    atomic_write_csv(kregion_imp, outputs["group_importance_kregion"])
    atomic_write_csv(spin_imp, outputs["group_importance_spin"])

    if not family_imp.empty:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        plot_df = family_imp.sort_values("importance_mean", ascending=True)
        ax.barh(plot_df["group"], plot_df["importance_mean"], xerr=plot_df["importance_std"])
        ax.set_xlabel("Balanced-accuracy decrease")
        ax.set_title("Grouped permutation importance by feature family")
        fig.tight_layout()
        fig.savefig(outputs["group_importance_family_fig"], dpi=220)
        plt.close(fig)

    # Class means for top individual features.
    top_features = individual_df.head(20)["feature"].tolist()
    class_means = binary_df.groupby("phase_label")[top_features].mean().T
    if {
        "spin_chern_TI_candidate", "trivial_insulator"
    }.issubset(class_means.columns):
        class_means["TI_minus_trivial"] = (
            class_means["spin_chern_TI_candidate"]
            - class_means["trivial_insulator"]
        )
    class_means = class_means.reset_index(names="feature")
    atomic_write_csv(class_means, outputs["class_feature_means"])

    # Representative Hamiltonians for independent MagneticTB or later strict checks.
    representatives: list[pd.DataFrame] = []
    for label in ("spin_chern_TI_candidate", "trivial_insulator"):
        part = binary_df[binary_df["phase_label"] == label].copy()
        if part.empty:
            continue
        part = part.sort_values(
            ["indirect_gap", "min_direct_gap"], ascending=False
        ).head(5)
        representatives.append(part[[
            "sample_id", "source_region", "phase_label",
            *REDUCED7, *RAW8,
            "min_direct_gap", "indirect_gap",
            "chern_up_int", "chern_down_int", "chern_total_int",
        ]])
    validation_subset = (
        pd.concat(representatives, ignore_index=True)
        if representatives else pd.DataFrame()
    )
    atomic_write_csv(validation_subset, outputs["validation_subset"])

    dump(best_model, outputs["model"])
    summary.update({
        "ml_status": "completed",
        "n_features_reduced7": len(REDUCED7),
        "n_features_fingerprint": len(fp_features),
        "n_features_combined": len(best_features),
        "holdout_balanced_accuracy": float(holdout.iloc[0]["balanced_accuracy"]),
        "holdout_f1": float(holdout.iloc[0]["f1"]),
        "holdout_is_independent": int(can_holdout),
        "best_model_path": str(outputs["model"]),
    })
    return summary


# =============================================================================
# Full workflow
# =============================================================================

def run_step02(config: Step02Config | None = None) -> Dict[str, object]:
    config = (config or Step02Config()).normalized()
    outputs = build_output_registry(config)
    write_run_metadata(config, outputs)

    print("=" * 78)
    print("TTS Step 02 v3 — robust Hamiltonian fingerprint and ML")
    print("Step 01 core:", core.CODE_VERSION)
    print("Output directory:", config.output_dir.resolve())
    print("All physics calculations: SERIAL")
    print("ML n_jobs:", config.ml_n_jobs)
    print("=" * 78)

    print("[1/7] Four-source parameter sampling")
    candidates = generate_candidates(config, outputs)
    print(candidates.groupby("source_region").size())

    print("[2/7] Robust full-BZ physics labels")
    physics_df, attempts_df = run_physics_labels(config, outputs, candidates)
    print(physics_df["phase_label"].value_counts(dropna=False))
    print(pd.crosstab(physics_df["source_region"], physics_df["phase_label"]))

    print("[3/7] Gauge-consistent 4x4-spin-block fingerprint")
    fingerprint_df, metadata_df = run_fingerprint(config, outputs, candidates)
    print("Fingerprint shape:", fingerprint_df.shape)

    print("[4/7] Strict binary dataset and source shortcut diagnosis")
    binary_df, source_diag = build_binary_dataset(outputs, physics_df, fingerprint_df)
    print(binary_df["phase_label"].value_counts(dropna=False))
    if not source_diag.empty:
        print(source_diag.to_string(index=False))

    print("[5/7] Repeated CV, holdout and external global test")
    ml_summary = run_machine_learning(config, outputs, binary_df, metadata_df)
    print(json.dumps(ml_summary, indent=2, ensure_ascii=False))

    print("[6/7] Save run summary")
    phase_counts = physics_df["phase_label"].value_counts(dropna=False).to_dict()
    summary = {
        "code_version": CODE_VERSION,
        "step01_core_version": core.CODE_VERSION,
        "n_candidates": int(len(candidates)),
        "n_physics_rows": int(len(physics_df)),
        "n_chern_attempt_rows": int(len(attempts_df)),
        "n_fingerprint_features": int(fingerprint_df.shape[1] - 1),
        "phase_counts": {str(k): int(v) for k, v in phase_counts.items()},
        "n_strict_binary": int(len(binary_df)),
        **ml_summary,
        "next_decision": (
            "If both classes are sufficiently populated, proceed to Step 03 global Sobol. "
            "Do not interpret local random-split accuracy as a final physical law."
        ),
    }
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[7/7] Done")
    print("Summary:", outputs["summary"].resolve())
    return {
        "config": config,
        "outputs": outputs,
        "candidates": candidates,
        "physics": physics_df,
        "chern_attempts": attempts_df,
        "fingerprint": fingerprint_df,
        "feature_metadata": metadata_df,
        "binary": binary_df,
        "source_diagnostic": source_diag,
        "summary": summary,
    }


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Step02Config.output_dir,
    )
    parser.add_argument("--n-topo-local", type=int, default=40)
    parser.add_argument("--n-trivial-local", type=int, default=40)
    parser.add_argument("--n-bridge", type=int, default=40)
    parser.add_argument("--n-global", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--gap-nk", type=int, default=60)
    parser.add_argument("--path-n", type=int, default=31)
    parser.add_argument("--ml-n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Step02Config:
    if args.quick:
        return Step02Config(
            output_dir=args.output_dir,
            n_topo_local=min(args.n_topo_local, 4),
            n_trivial_local=min(args.n_trivial_local, 4),
            n_bridge=min(args.n_bridge, 4),
            n_global=min(args.n_global, 4),
            random_seed=args.seed,
            gap_nk=min(args.gap_nk, 24),
            chern_grids=(11, 15),
            chern_shifts=((0.0, 0.0), (0.5, 0.5)),
            path_n=min(args.path_n, 11),
            cv_splits=3,
            cv_repeats=2,
            random_forest_estimators=80,
            permutation_repeats=2,
            ml_n_jobs=args.ml_n_jobs,
            checkpoint_every=4,
            force_recalculate_physics=args.force,
            force_recalculate_fingerprint=args.force,
        )
    return Step02Config(
        output_dir=args.output_dir,
        n_topo_local=args.n_topo_local,
        n_trivial_local=args.n_trivial_local,
        n_bridge=args.n_bridge,
        n_global=args.n_global,
        random_seed=args.seed,
        gap_nk=args.gap_nk,
        path_n=args.path_n,
        ml_n_jobs=args.ml_n_jobs,
        force_recalculate_physics=args.force,
        force_recalculate_fingerprint=args.force,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_step02(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
