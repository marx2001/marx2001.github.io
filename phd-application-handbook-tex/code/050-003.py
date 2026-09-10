#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 03 — 七维全局 Sobol、严格 spin-Chern 分扇区与分层机器学习

研究目标
--------
1. 在去除整体能量平移后的七维参数空间
   (m_e, t1, t2, r1, r2, r3, r4)
   中进行全局 Sobol 低差异采样；
2. 使用冻结的 TTS Step 01 Hamiltonian、周期规范、完整 BZ 带隙与
   非阿贝尔 spin-Chern 标签器；
3. 对所有非零 Chern 或网格不一致样本执行更严格的多网格、多 shift 复核；
4. 保留 C_up = 0, ±1, ±2, ... 的完整分扇区标签，而不是只做“拓扑/平庸”合并；
5. 建立分层模型：
      Level A: 全局绝缘体门控；
      Level B1: 严格绝缘体中的 spin-Chern 拓扑/平庸二分类；
      Level B2: C_up 精确扇区多分类（样本数足够时自动运行）；
6. 使用参数空间整簇留出与独立 Sobol 序列评估空间外推能力。

关键实现原则
------------
- 物理计算完全串行；不使用 ProcessPoolExecutor，兼容 Windows/Jupyter。
- sklearn/joblib 默认 n_jobs=1，避免 BrokenProcessPool。
- 全局 Sobol 主样本与独立 Sobol 外部样本严格分开。
- 已知 C_up=-1、C_up=±2 和文献平庸点仅作为回归测试 control，
  不参与全局相比例统计，也不参与机器学习训练或测试。
- overall energy shift e0 固定为 0。默认将 Sobol 七维向量归一化到单位范数，
  消除不改变本征态和拓扑的整体能量缩放自由度。
- spin_chern_TI_candidate 仍是模型级候选；边缘态、DOS、SHC 在后续步骤验证。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Any
import argparse
import hashlib
import json
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

try:
    from joblib import dump
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        ConfusionMatrixDisplay,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import (
        GroupShuffleSplit,
        RepeatedStratifiedKFold,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Step 03 需要 scipy、scikit-learn 与 joblib。请运行：\n"
        "  conda install scipy scikit-learn joblib\n"
        "或：\n"
        "  pip install scipy scikit-learn joblib"
    ) from exc

try:
    import TTS_step01_model_and_label_audit_v2 as core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未找到 TTS_step01_model_and_label_audit_v2.py。\n"
        "请把 Step 01 v2 的 py 文件与本脚本放在同一目录。"
    ) from exc

try:
    import TTS_step02v3_robust_hamiltonian_fingerprint_and_ML as fp_core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未找到 TTS_step02v3_robust_hamiltonian_fingerprint_and_ML.py。\n"
        "Step 03 复用 Step 02 v3 已审计的 4x4 自旋块 Hamiltonian fingerprint，\n"
        "请把配套文件放在同一目录。"
    ) from exc

EXPECTED_STEP01_VERSION = "TTS_STEP01_V2_20260713"
EXPECTED_STEP02_VERSION = "TTS_STEP02V3_20260713"
if getattr(core, "CODE_VERSION", None) != EXPECTED_STEP01_VERSION:
    raise RuntimeError(
        "Step 01 核心版本不一致：\n"
        f"  expected = {EXPECTED_STEP01_VERSION}\n"
        f"  loaded   = {getattr(core, 'CODE_VERSION', None)}"
    )
if getattr(fp_core, "CODE_VERSION", None) != EXPECTED_STEP02_VERSION:
    raise RuntimeError(
        "Step 02 fingerprint 版本不一致：\n"
        f"  expected = {EXPECTED_STEP02_VERSION}\n"
        f"  loaded   = {getattr(fp_core, 'CODE_VERSION', None)}"
    )

np.set_printoptions(precision=10, suppress=True)

CODE_VERSION = "TTS_STEP03_V1_20260713"
WORKFLOW_STEP = "step03"
SYSTEM_TAG = "tts8_bns127391"
TASK_TAG = "global_sobol_hierarchical_topology_ml"
REDUCED7 = core.REDUCED7.copy()
RAW8 = core.RAW8.copy()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Step03Config:
    output_dir: Path = Path("outputs_tts_step03_global_sobol_hierarchical_topology_ml")

    # Independent Sobol sequences.
    train_sobol_power: int = 9       # 2^9 = 512
    external_sobol_power: int = 7    # 2^7 = 128
    train_seed: int = 20260713
    external_seed: int = 20260731
    normalize_sobol_vectors: bool = True
    include_control_points: bool = True

    # Physics label: all samples first receive this full-BZ audit.
    gap_nk: int = 60
    initial_chern_grids: tuple[int, ...] = (21, 31)
    initial_chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )

    # Any nonzero Chern or inconsistent initial result is sent here.
    strict_gap_grids: tuple[int, ...] = (51, 71)
    strict_gap_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    )
    strict_chern_grids: tuple[int, ...] = (31, 41, 51)
    strict_chern_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.5),
    )

    direct_gap_skip_chern: float = 5.0e-3
    gap_tol: float = 1.0e-3
    chern_integer_tol: float = 0.08
    min_link_tol: float = 1.0e-7
    sum_rule_tol: float = 0.15

    # Hamiltonian fingerprint reused from Step 02 v3.
    fingerprint_path_n: int = 21

    # Machine learning.
    n_parameter_clusters: int = 8
    group_holdout_splits: int = 20
    group_holdout_fraction: float = 0.25
    random_validation_fraction: float = 0.25
    cv_splits: int = 5
    cv_repeats: int = 3
    random_forest_estimators: int = 500
    permutation_repeats: int = 20
    min_class_count_for_training: int = 5
    ml_n_jobs: int = 1

    # Workflow controls.
    checkpoint_every: int = 10
    force_recalculate_physics: bool = False
    force_recalculate_fingerprint: bool = False
    force_recalculate_ml: bool = False

    def normalized(self) -> "Step03Config":
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        if self.train_sobol_power < 2:
            raise ValueError("train_sobol_power must be >= 2")
        if self.external_sobol_power < 0:
            raise ValueError("external_sobol_power must be >= 0")
        if self.gap_nk < 8:
            raise ValueError("gap_nk must be >= 8")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if self.ml_n_jobs == 0:
            raise ValueError("ml_n_jobs cannot be 0")
        if self.n_parameter_clusters < 2:
            raise ValueError("n_parameter_clusters must be >= 2")
        return self

    @property
    def n_train(self) -> int:
        return 2 ** int(self.train_sobol_power)

    @property
    def n_external(self) -> int:
        return 0 if self.external_sobol_power == 0 else 2 ** int(self.external_sobol_power)


# Known points are controls only. They are excluded from all ML datasets and global fractions.
CONTROL_REDUCED7: dict[str, dict[str, float]] = {
    "control_topological_Cm1": core.reduced7_from_raw8(core.TOPO_PARAMS),
    "control_trivial_paper": core.reduced7_from_raw8(core.PAPER_PARAMS),
    "control_high_chern_Cp2": {
        "m_e": 0.227401,
        "t1": -0.325125,
        "t2": 0.494735,
        "r1": -0.970710,
        "r2": -0.332244,
        "r3": 0.448718,
        "r4": -0.575438,
    },
    "control_high_chern_Cm2": {
        "m_e": 0.251241,
        "t1": -0.907558,
        "t2": 0.610720,
        "r1": -0.609335,
        "r2": 0.657497,
        "r3": 0.597549,
        "r4": -0.532312,
    },
}
CONTROL_EXPECTED_CUP = {
    "control_topological_Cm1": -1,
    "control_trivial_paper": 0,
    "control_high_chern_Cp2": 2,
    "control_high_chern_Cm2": -2,
}


# =============================================================================
# Output registry and safe file helpers
# =============================================================================


def _run_tag(config: Step03Config) -> str:
    return (
        f"{SYSTEM_TAG}_train{config.n_train}_external{config.n_external}"
        f"_gap{config.gap_nk}"
        f"_chern{'-'.join(map(str, config.initial_chern_grids))}"
        f"_strict{'-'.join(map(str, config.strict_chern_grids))}"
        f"_seed{config.train_seed}"
    )


def build_output_registry(config: Step03Config) -> Dict[str, Path]:
    config = config.normalized()
    run_tag = _run_tag(config)
    figures = config.output_dir / "figures"
    models = config.output_dir / "models"

    def out(substep: int, content: str, ext: str, *, figure: bool = False, model: bool = False) -> Path:
        base = models if model else figures if figure else config.output_dir
        return base / f"{WORKFLOW_STEP}_{substep:02d}_{content}__{run_tag}.{ext.lstrip('.')}"

    return {
        "run_config": out(0, "run_configuration", "json"),
        "registry": out(0, "output_file_registry", "csv"),
        "parameters": out(1, "train_external_sobol_and_controls", "csv"),
        "physics_checkpoint": out(2, "physics_labels_checkpoint", "csv"),
        "physics_final": out(2, "physics_labels_final", "csv"),
        "chern_attempts": out(2, "chern_attempt_records", "csv"),
        "strict_gap_checks": out(2, "strict_gap_check_records", "csv"),
        "phase_counts": out(3, "global_phase_counts", "csv"),
        "phase_counts_source": out(3, "phase_counts_by_sobol_source", "csv"),
        "chern_sector_counts": out(3, "strict_insulator_chern_sector_counts", "csv"),
        "control_audit": out(3, "control_point_regression_audit", "csv"),
        "topological_candidates": out(3, "strict_spin_chern_TI_candidates", "csv"),
        "best_by_sector": out(3, "best_gap_candidate_by_chern_sector", "csv"),
        "phase_counts_fig": out(3, "global_phase_counts", "png", figure=True),
        "chern_sector_fig": out(3, "strict_insulator_chern_sector_counts", "png", figure=True),
        "fingerprint": out(4, "hamiltonian_fingerprint", "csv"),
        "fingerprint_metadata": out(4, "hamiltonian_fingerprint_metadata", "csv"),
        "analytic_features": out(4, "symmetry_motivated_parameter_features", "csv"),
        "combined_features": out(4, "combined_ml_feature_table", "csv"),
        "cluster_assignments": out(5, "train_parameter_space_cluster_assignments", "csv"),
        "levelA_dataset": out(5, "levelA_global_insulator_gate_dataset", "csv"),
        "levelB_binary_dataset": out(5, "levelB_binary_topological_vs_trivial_dataset", "csv"),
        "levelB_multiclass_dataset": out(5, "levelB_multiclass_chern_sector_dataset", "csv"),
        "random_cv_metrics": out(6, "random_repeated_cv_metrics", "csv"),
        "cluster_holdout_metrics": out(6, "parameter_cluster_holdout_metrics", "csv"),
        "external_metrics": out(6, "independent_sobol_external_metrics", "csv"),
        "best_model_registry": out(6, "best_model_registry", "csv"),
        "external_predictions": out(7, "independent_sobol_external_predictions", "csv"),
        "cluster_predictions": out(7, "best_cluster_holdout_predictions", "csv"),
        "classification_reports": out(7, "classification_reports", "txt"),
        "levelA_confusion": out(7, "levelA_external_confusion", "png", figure=True),
        "levelB_confusion": out(7, "levelB_binary_external_confusion", "png", figure=True),
        "levelB_multi_confusion": out(7, "levelB_multiclass_external_confusion", "png", figure=True),
        "group_importance": out(8, "levelB_binary_grouped_permutation_importance", "csv"),
        "individual_importance": out(8, "levelB_binary_individual_permutation_importance", "csv"),
        "importance_fig": out(8, "levelB_binary_grouped_permutation_importance", "png", figure=True),
        "misclassified": out(9, "independent_sobol_misclassified_hamiltonians", "csv"),
        "summary": out(10, "run_summary", "json"),
        "levelA_model": out(10, "best_levelA_model", "joblib", model=True),
        "levelB_model": out(10, "best_levelB_binary_model", "joblib", model=True),
        "levelB_multi_model": out(10, "best_levelB_multiclass_model", "joblib", model=True),
    }


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(config: Step03Config, outputs: Dict[str, Path]) -> None:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload.update({
        "code_version": CODE_VERSION,
        "step01_core_version": getattr(core, "CODE_VERSION", None),
        "step02_fingerprint_version": getattr(fp_core, "CODE_VERSION", None),
        "n_train": config.n_train,
        "n_external": config.n_external,
        "run_tag": _run_tag(config),
    })
    atomic_write_text(json.dumps(payload, indent=2, ensure_ascii=False), outputs["run_config"])

    rows = []
    for key, path in outputs.items():
        rows.append({"key": key, "path": str(path), "exists": int(path.exists())})
    atomic_write_csv(pd.DataFrame(rows), outputs["registry"])


# =============================================================================
# Step 1. Global Sobol generation
# =============================================================================


def _sobol_rows(power: int, seed: int, source: str, normalize: bool) -> list[dict]:
    if power <= 0:
        return []
    sampler = qmc.Sobol(d=len(REDUCED7), scramble=True, seed=int(seed))
    unit = sampler.random_base2(m=int(power))
    raw_vectors = 2.0 * unit - 1.0
    norms = np.linalg.norm(raw_vectors, axis=1)
    if np.any(norms < 1.0e-12):
        raise RuntimeError("Sobol generated a near-zero parameter vector; change the seed.")
    vectors = raw_vectors / norms[:, None] if normalize else raw_vectors

    rows: list[dict] = []
    prefix = "train" if source == "train_sobol" else "external"
    for index, (raw_vec, vec, norm) in enumerate(zip(raw_vectors, vectors, norms)):
        reduced = {name: float(value) for name, value in zip(REDUCED7, vec)}
        raw8 = core.raw8_from_reduced7(reduced, e0=0.0)
        row = {
            "sample_id": f"sobol_{prefix}_{index:06d}",
            "sample_source": source,
            "sobol_index": int(index),
            "is_control": 0,
            "is_ml_eligible": 1,
            "pre_normalization_norm": float(norm),
            "post_normalization_norm": float(np.linalg.norm(vec)),
            **reduced,
            **raw8,
        }
        for name, value in zip(REDUCED7, raw_vec):
            row[f"pre_norm_{name}"] = float(value)
        rows.append(row)
    return rows


def generate_parameters(config: Step03Config, outputs: Dict[str, Path]) -> pd.DataFrame:
    path = outputs["parameters"]
    if path.exists():
        existing = pd.read_csv(path, low_memory=False)
        expected = config.n_train + config.n_external + (len(CONTROL_REDUCED7) if config.include_control_points else 0)
        if len(existing) == expected:
            return existing

    rows = _sobol_rows(
        config.train_sobol_power,
        config.train_seed,
        "train_sobol",
        config.normalize_sobol_vectors,
    )
    rows.extend(_sobol_rows(
        config.external_sobol_power,
        config.external_seed,
        "external_sobol",
        config.normalize_sobol_vectors,
    ))

    if config.include_control_points:
        for sample_id, reduced in CONTROL_REDUCED7.items():
            raw8 = core.raw8_from_reduced7(reduced, e0=0.0)
            rows.append({
                "sample_id": sample_id,
                "sample_source": "control",
                "sobol_index": -1,
                "is_control": 1,
                "is_ml_eligible": 0,
                "pre_normalization_norm": float(np.linalg.norm([reduced[k] for k in REDUCED7])),
                "post_normalization_norm": float(np.linalg.norm([reduced[k] for k in REDUCED7])),
                **reduced,
                **raw8,
            })

    df = pd.DataFrame(rows)
    if df["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id generated.")
    atomic_write_csv(df, path)
    return df


# =============================================================================
# Step 2. Physics labels with strict nonzero-Chern verification
# =============================================================================


def _integer_if_close(value: float, tol: float) -> int | None:
    if not np.isfinite(value):
        return None
    nearest = int(np.rint(value))
    return nearest if abs(float(value) - nearest) <= tol else None


def one_chern_attempt(
    sample_id: str,
    params: Dict[str, float],
    nk: int,
    shift: tuple[float, float],
    stage: str,
    config: Step03Config,
) -> dict:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "stage": stage,
        "chern_nk": int(nk),
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
        "attempt_ok": 0,
        "attempt_reliable": 0,
        "error": "",
    }
    try:
        calc = core.calculate_spin_cherns(params, nk=int(nk), shift=shift)
        row.update(calc)
        cu = _integer_if_close(float(calc["chern_up"]), config.chern_integer_tol)
        cd = _integer_if_close(float(calc["chern_down"]), config.chern_integer_tol)
        ct = _integer_if_close(float(calc["chern_total"]), config.chern_integer_tol)
        reliable = bool(
            cu is not None and cd is not None and ct is not None
            and float(calc["min_det_up"]) > config.min_link_tol
            and float(calc["min_det_down"]) > config.min_link_tol
            and float(calc["min_det_total"]) > config.min_link_tol
            and float(calc["sum_rule_error"]) <= config.sum_rule_tol
        )
        row.update({
            "chern_up_int": np.nan if cu is None else int(cu),
            "chern_down_int": np.nan if cd is None else int(cd),
            "chern_total_int": np.nan if ct is None else int(ct),
            "attempt_ok": 1,
            "attempt_reliable": int(reliable),
        })
    except Exception as exc:  # numerical failures are retained as data
        row["error"] = repr(exc)
    return row


def _exact_consensus(attempts: Sequence[dict]) -> dict:
    total = len(attempts)
    reliable = [a for a in attempts if int(a.get("attempt_reliable", 0)) == 1]
    keys: list[tuple[int, int, int]] = []
    for a in reliable:
        keys.append((
            int(a["chern_up_int"]),
            int(a["chern_down_int"]),
            int(a["chern_total_int"]),
        ))
    unique = sorted(set(keys))
    exact = bool(total > 0 and len(reliable) == total and len(unique) == 1)
    if exact:
        cu, cd, ct = unique[0]
    else:
        cu = cd = ct = None
    return {
        "attempt_count": int(total),
        "reliable_count": int(len(reliable)),
        "n_unique_reliable_sectors": int(len(unique)),
        "exact_consensus": int(exact),
        "consensus_chern_up": cu,
        "consensus_chern_down": cd,
        "consensus_chern_total": ct,
    }


def _classify_from_gap_and_chern(
    gap: dict,
    consensus: dict,
    config: Step03Config,
    verification_level: str,
) -> dict:
    row: dict[str, Any] = {}
    row.update({
        "verification_level": verification_level,
        "chern_exact_consensus": int(consensus["exact_consensus"]),
        "chern_attempt_count": int(consensus["attempt_count"]),
        "chern_reliable_count": int(consensus["reliable_count"]),
        "chern_unique_sector_count": int(consensus["n_unique_reliable_sectors"]),
    })
    if not consensus["exact_consensus"]:
        row.update({
            "phase_label": "boundary_or_chern_unreliable",
            "chern_up_int": np.nan,
            "chern_down_int": np.nan,
            "chern_total_int": np.nan,
            "is_strict_insulator": 0,
            "is_spin_chern_topological": 0,
            "is_spin_chern_TI_candidate": 0,
        })
        return row

    cu = int(consensus["consensus_chern_up"])
    cd = int(consensus["consensus_chern_down"])
    ct = int(consensus["consensus_chern_total"])
    topological = bool(cu == -cd and abs(cu) >= 1 and ct == 0)
    direct_gapped = bool(float(gap["min_direct_gap"]) > config.gap_tol)
    indirect_gapped = bool(float(gap["indirect_gap"]) > config.gap_tol)
    balanced = bool(
        int(gap["spin_occupancy_mismatch_count"]) == 0
        and float(gap["min_balanced_sector_gap"]) > config.gap_tol
    )

    if not direct_gapped:
        label = "noninsulating_or_gap_closing"
    elif not balanced:
        label = "spin_sector_filling_mismatch"
    elif topological and indirect_gapped:
        label = "spin_chern_TI_candidate"
    elif topological:
        label = "spin_chern_band_metal"
    elif cu == 0 and cd == 0 and ct == 0 and indirect_gapped:
        label = "trivial_insulator"
    elif cu == 0 and cd == 0 and ct == 0:
        label = "indirect_overlap"
    else:
        label = "other_gapped_chern_sector" if indirect_gapped else "other_overlapping_chern_sector"

    row.update({
        "phase_label": label,
        "chern_up_int": cu,
        "chern_down_int": cd,
        "chern_total_int": ct,
        "spin_chern_int": int((cu - cd) // 2) if (cu - cd) % 2 == 0 else 0.5 * (cu - cd),
        "is_strict_insulator": int(label in {"spin_chern_TI_candidate", "trivial_insulator"}),
        "is_spin_chern_topological": int(topological),
        "is_spin_chern_TI_candidate": int(label == "spin_chern_TI_candidate"),
    })
    return row


def evaluate_sample_physics(
    row: pd.Series,
    config: Step03Config,
) -> tuple[dict, list[dict], list[dict]]:
    sample_id = str(row["sample_id"])
    reduced = {name: float(row[name]) for name in REDUCED7}
    params = core.raw8_from_reduced7(reduced, e0=0.0)

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_source": str(row["sample_source"]),
        "is_control": int(row["is_control"]),
        "is_ml_eligible": int(row["is_ml_eligible"]),
        **reduced,
        **params,
        **core.derived_features(params),
        "physics_error": "",
    }
    chern_rows: list[dict] = []
    strict_gap_rows: list[dict] = []

    try:
        coarse_gap = core.scan_band_gaps(
            params,
            nk=int(config.gap_nk),
            shift=(0.0, 0.0),
            gap_tol=config.gap_tol,
        )
        result.update(coarse_gap)
        result["coarse_min_direct_gap"] = float(coarse_gap["min_direct_gap"])
        result["coarse_indirect_gap"] = float(coarse_gap["indirect_gap"])

        if float(coarse_gap["min_direct_gap"]) <= config.direct_gap_skip_chern:
            result.update({
                "phase_label": "noninsulating_or_gap_closing",
                "verification_level": "gap_only",
                "chern_exact_consensus": 0,
                "chern_attempt_count": 0,
                "chern_reliable_count": 0,
                "chern_unique_sector_count": 0,
                "chern_up_int": np.nan,
                "chern_down_int": np.nan,
                "chern_total_int": np.nan,
                "is_strict_insulator": 0,
                "is_spin_chern_topological": 0,
                "is_spin_chern_TI_candidate": 0,
            })
            return result, chern_rows, strict_gap_rows

        if (
            int(coarse_gap["spin_occupancy_mismatch_count"]) > 0
            or float(coarse_gap["min_balanced_sector_gap"]) <= config.gap_tol
        ):
            result.update({
                "phase_label": "spin_sector_filling_mismatch",
                "verification_level": "gap_and_filling",
                "chern_exact_consensus": 0,
                "chern_attempt_count": 0,
                "chern_reliable_count": 0,
                "chern_unique_sector_count": 0,
                "chern_up_int": np.nan,
                "chern_down_int": np.nan,
                "chern_total_int": np.nan,
                "is_strict_insulator": 0,
                "is_spin_chern_topological": 0,
                "is_spin_chern_TI_candidate": 0,
            })
            return result, chern_rows, strict_gap_rows

        initial_attempts: list[dict] = []
        for nk in config.initial_chern_grids:
            for shift in config.initial_chern_shifts:
                attempt = one_chern_attempt(sample_id, params, int(nk), shift, "initial", config)
                initial_attempts.append(attempt)
                chern_rows.append(attempt)
        initial_consensus = _exact_consensus(initial_attempts)

        initial_nonzero = bool(
            initial_consensus["exact_consensus"]
            and int(initial_consensus["consensus_chern_up"]) != 0
        )
        needs_strict = bool(
            int(row["is_control"]) == 1
            or not initial_consensus["exact_consensus"]
            or initial_nonzero
        )

        if not needs_strict:
            result.update(_classify_from_gap_and_chern(
                coarse_gap,
                initial_consensus,
                config,
                verification_level="initial_all_grid_consensus",
            ))
            return result, chern_rows, strict_gap_rows

        # Strict gap audit for all nonzero/ambiguous/control samples.
        strict_gap_results: list[dict] = []
        for nk in config.strict_gap_grids:
            for shift in config.strict_gap_shifts:
                g = core.scan_band_gaps(params, nk=int(nk), shift=shift, gap_tol=config.gap_tol)
                g = {"sample_id": sample_id, "stage": "strict", **g}
                strict_gap_results.append(g)
                strict_gap_rows.append(g)

        strict_gap = dict(coarse_gap)
        strict_gap["min_direct_gap"] = float(min(g["min_direct_gap"] for g in strict_gap_results))
        strict_gap["indirect_gap"] = float(min(g["indirect_gap"] for g in strict_gap_results))
        strict_gap["min_balanced_sector_gap"] = float(min(g["min_balanced_sector_gap"] for g in strict_gap_results))
        strict_gap["spin_occupancy_mismatch_count"] = int(max(g["spin_occupancy_mismatch_count"] for g in strict_gap_results))
        strict_gap["min_spin_gap_up"] = float(min(g["min_spin_gap_up"] for g in strict_gap_results))
        strict_gap["min_spin_gap_down"] = float(min(g["min_spin_gap_down"] for g in strict_gap_results))
        result["min_direct_gap"] = strict_gap["min_direct_gap"]
        result["indirect_gap"] = strict_gap["indirect_gap"]
        result["min_balanced_sector_gap"] = strict_gap["min_balanced_sector_gap"]
        result["spin_occupancy_mismatch_count"] = strict_gap["spin_occupancy_mismatch_count"]
        result["strict_gap_verified"] = 1
        result["strict_gap_check_count"] = len(strict_gap_results)

        strict_attempts: list[dict] = []
        for nk in config.strict_chern_grids:
            for shift in config.strict_chern_shifts:
                attempt = one_chern_attempt(sample_id, params, int(nk), shift, "strict", config)
                strict_attempts.append(attempt)
                chern_rows.append(attempt)
        strict_consensus = _exact_consensus(strict_attempts)
        result.update(_classify_from_gap_and_chern(
            strict_gap,
            strict_consensus,
            config,
            verification_level="strict_all_grid_consensus",
        ))
        result["strict_chern_verified"] = int(strict_consensus["exact_consensus"])
        return result, chern_rows, strict_gap_rows

    except Exception as exc:
        result.update({
            "phase_label": "true_numeric_error",
            "physics_error": repr(exc),
            "verification_level": "error",
            "is_strict_insulator": 0,
            "is_spin_chern_topological": 0,
            "is_spin_chern_TI_candidate": 0,
        })
        return result, chern_rows, strict_gap_rows


def run_physics_labels(
    config: Step03Config,
    outputs: Dict[str, Path],
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_path = outputs["physics_final"]
    checkpoint_path = outputs["physics_checkpoint"]
    chern_path = outputs["chern_attempts"]
    strict_gap_path = outputs["strict_gap_checks"]

    if final_path.exists() and not config.force_recalculate_physics:
        final = pd.read_csv(final_path, low_memory=False)
        if set(final["sample_id"]) == set(parameters["sample_id"]):
            attempts = pd.read_csv(chern_path, low_memory=False) if chern_path.exists() else pd.DataFrame()
            strict_gaps = pd.read_csv(strict_gap_path, low_memory=False) if strict_gap_path.exists() else pd.DataFrame()
            return final, attempts, strict_gaps

    if checkpoint_path.exists() and not config.force_recalculate_physics:
        done_df = pd.read_csv(checkpoint_path, low_memory=False)
        done_ids = set(done_df["sample_id"].astype(str))
        result_rows = done_df.to_dict("records")
    else:
        done_ids = set()
        result_rows = []

    if chern_path.exists() and not config.force_recalculate_physics:
        chern_rows = pd.read_csv(chern_path, low_memory=False).to_dict("records")
    else:
        chern_rows = []
    if strict_gap_path.exists() and not config.force_recalculate_physics:
        strict_gap_rows = pd.read_csv(strict_gap_path, low_memory=False).to_dict("records")
    else:
        strict_gap_rows = []

    total = len(parameters)
    started = time.time()
    processed_now = 0
    for _, row in parameters.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id in done_ids:
            continue
        result, attempts, gap_checks = evaluate_sample_physics(row, config)
        result_rows.append(result)
        chern_rows.extend(attempts)
        strict_gap_rows.extend(gap_checks)
        processed_now += 1

        if processed_now % config.checkpoint_every == 0:
            atomic_write_csv(pd.DataFrame(result_rows), checkpoint_path)
            atomic_write_csv(pd.DataFrame(chern_rows), chern_path)
            atomic_write_csv(pd.DataFrame(strict_gap_rows), strict_gap_path)
            print(
                f"Physics: {len(result_rows)}/{total} | "
                f"elapsed {time.time() - started:.1f} s"
            )

    physics_df = pd.DataFrame(result_rows)
    # Restore deterministic parameter order.
    order = {sid: i for i, sid in enumerate(parameters["sample_id"].astype(str))}
    physics_df["__order"] = physics_df["sample_id"].map(order)
    physics_df = physics_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    atomic_write_csv(physics_df, final_path)
    atomic_write_csv(physics_df, checkpoint_path)
    atomic_write_csv(pd.DataFrame(chern_rows), chern_path)
    atomic_write_csv(pd.DataFrame(strict_gap_rows), strict_gap_path)
    return physics_df, pd.DataFrame(chern_rows), pd.DataFrame(strict_gap_rows)


# =============================================================================
# Step 3. Phase and Chern-sector summaries
# =============================================================================


def summarize_phases(
    config: Step03Config,
    outputs: Dict[str, Path],
    physics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    global_df = physics[physics["is_control"] == 0].copy()
    phase_counts = (
        global_df["phase_label"].value_counts(dropna=False)
        .rename_axis("phase_label").reset_index(name="count")
    )
    phase_counts["fraction"] = phase_counts["count"] / max(len(global_df), 1)
    atomic_write_csv(phase_counts, outputs["phase_counts"])

    by_source = (
        global_df.groupby(["sample_source", "phase_label"], dropna=False)
        .size().reset_index(name="count")
    )
    by_source["source_total"] = by_source.groupby("sample_source")["count"].transform("sum")
    by_source["fraction_within_source"] = by_source["count"] / by_source["source_total"]
    atomic_write_csv(by_source, outputs["phase_counts_source"])

    strict_ins = global_df[global_df["is_strict_insulator"] == 1].copy()
    sector_counts = (
        strict_ins["chern_up_int"].value_counts(dropna=False).sort_index()
        .rename_axis("chern_up_int").reset_index(name="count")
    )
    if len(sector_counts):
        sector_counts["fraction_of_strict_insulators"] = sector_counts["count"] / len(strict_ins)
    atomic_write_csv(sector_counts, outputs["chern_sector_counts"])

    topo = global_df[global_df["phase_label"] == "spin_chern_TI_candidate"].copy()
    topo = topo.sort_values(["chern_up_int", "indirect_gap"], ascending=[True, False])
    atomic_write_csv(topo, outputs["topological_candidates"])

    if len(topo):
        best = (
            topo.sort_values("indirect_gap", ascending=False)
            .groupby("chern_up_int", as_index=False).head(1)
            .sort_values("chern_up_int")
        )
    else:
        best = pd.DataFrame(columns=topo.columns)
    atomic_write_csv(best, outputs["best_by_sector"])

    controls = physics[physics["is_control"] == 1].copy()
    if len(controls):
        controls["expected_chern_up"] = controls["sample_id"].map(CONTROL_EXPECTED_CUP)
        controls["control_chern_match"] = (
            controls["chern_up_int"].fillna(9999).astype(float)
            == controls["expected_chern_up"].astype(float)
        ).astype(int)
    atomic_write_csv(controls, outputs["control_audit"])

    # Default-color plots only.
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(phase_counts):
        ax.bar(phase_counts["phase_label"].astype(str), phase_counts["count"])
        ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Count")
    ax.set_title("TTS Step 03 global Sobol phase counts")
    fig.tight_layout()
    fig.savefig(outputs["phase_counts_fig"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(sector_counts):
        ax.bar(sector_counts["chern_up_int"].astype(str), sector_counts["count"])
    ax.set_xlabel("C_up")
    ax.set_ylabel("Strict-insulator count")
    ax.set_title("Spin-Chern sectors among strict insulators")
    fig.tight_layout()
    fig.savefig(outputs["chern_sector_fig"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "phase_counts": phase_counts,
        "phase_counts_by_source": by_source,
        "chern_sector_counts": sector_counts,
        "topological_candidates": topo,
        "best_by_sector": best,
        "control_audit": controls,
    }


# =============================================================================
# Step 4. Features: symmetry combinations + Step 02 v3 fingerprint
# =============================================================================


def symmetry_parameter_features(parameters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    meta: list[dict] = []
    root2 = math.sqrt(2.0)
    for _, row in parameters.iterrows():
        p = {name: float(row[name]) for name in REDUCED7}
        me, t1, t2, r1, r2, r3, r4 = [p[k] for k in REDUCED7]
        values = {
            "sample_id": str(row["sample_id"]),
            **p,
            "abs_m_e": abs(me),
            "abs_t1": abs(t1),
            "abs_t2": abs(t2),
            "t_sum": t1 + t2,
            "t_diff": t1 - t2,
            "t_product": t1 * t2,
            "t_norm": math.sqrt(t1*t1 + t2*t2),
            "t_s": (t1 + t2) / root2,
            "t_d": (t1 - t2) / root2,
            "r13_sum": r1 + r3,
            "r13_diff": r1 - r3,
            "r24_sum": r2 + r4,
            "r24_diff": r2 - r4,
            "r12_sum": r1 + r2,
            "r34_sum": r3 + r4,
            "r_all_sum": r1 + r2 + r3 + r4,
            "r_all_norm": math.sqrt(r1*r1 + r2*r2 + r3*r3 + r4*r4),
            "r13_product": r1 * r3,
            "r24_product": r2 * r4,
            "r_cross_product": (r1 + r3) * (r2 + r4),
            "r_anisotropy_norm": math.sqrt((r1-r3)**2 + (r2-r4)**2),
            "r_pair_sum_mismatch": (r1 + r3) - (r2 + r4),
            "r_pair_diff_mismatch": (r1 - r3) - (r2 - r4),
            "m_e_t_sum": me * (t1 + t2),
            "m_e_t_diff": me * (t1 - t2),
            "m_e_r13_sum": me * (r1 + r3),
            "m_e_r24_sum": me * (r2 + r4),
            "t_product_r_cross": (t1*t2) * ((r1+r3)*(r2+r4)),
        }
        rows.append(values)

    df = pd.DataFrame(rows)
    for feature in df.columns:
        if feature == "sample_id":
            continue
        family = "raw_parameters" if feature in REDUCED7 else "symmetry_parameter_combinations"
        meta.append({
            "feature": feature,
            "feature_family": family,
            "k_region": "parameter_space",
            "spin": "none",
            "channel": feature,
        })
    return df, pd.DataFrame(meta)


def run_fingerprint(
    config: Step03Config,
    outputs: Dict[str, Path],
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp_path = outputs["fingerprint"]
    meta_path = outputs["fingerprint_metadata"]
    if fp_path.exists() and meta_path.exists() and not config.force_recalculate_fingerprint:
        fp_df = pd.read_csv(fp_path, low_memory=False)
        if set(fp_df["sample_id"]) == set(parameters["sample_id"]):
            return fp_df, pd.read_csv(meta_path, low_memory=False)

    if fp_path.exists() and not config.force_recalculate_fingerprint:
        done = pd.read_csv(fp_path, low_memory=False)
        done_ids = set(done["sample_id"].astype(str))
        fp_rows = done.to_dict("records")
    else:
        done_ids = set()
        fp_rows = []

    meta_rows: list[dict] = []
    total = len(parameters)
    started = time.time()
    processed = 0
    for _, row in parameters.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id in done_ids:
            continue
        reduced = {name: float(row[name]) for name in REDUCED7}
        params = core.raw8_from_reduced7(reduced, e0=0.0)
        feat, meta = fp_core.extract_fingerprint(sample_id, params, config.fingerprint_path_n)
        fp_rows.append(feat)
        meta_rows.extend(meta)
        processed += 1
        if processed % config.checkpoint_every == 0:
            atomic_write_csv(pd.DataFrame(fp_rows), fp_path)
            print(f"Fingerprint: {len(fp_rows)}/{total} | elapsed {time.time()-started:.1f} s")

    fp_df = pd.DataFrame(fp_rows)
    order = {sid: i for i, sid in enumerate(parameters["sample_id"].astype(str))}
    fp_df["__order"] = fp_df["sample_id"].map(order)
    fp_df = fp_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    # If resuming, regenerate metadata from one sample to guarantee a complete table.
    if not meta_rows and len(parameters):
        first = parameters.iloc[0]
        reduced = {name: float(first[name]) for name in REDUCED7}
        _, meta_rows = fp_core.extract_fingerprint(
            str(first["sample_id"]),
            core.raw8_from_reduced7(reduced, e0=0.0),
            config.fingerprint_path_n,
        )
    meta_df = pd.DataFrame(meta_rows).drop_duplicates("feature", keep="first")
    atomic_write_csv(fp_df, fp_path)
    atomic_write_csv(meta_df, meta_path)
    return fp_df, meta_df


def build_feature_table(
    outputs: Dict[str, Path],
    parameters: pd.DataFrame,
    physics: pd.DataFrame,
    fingerprint: pd.DataFrame,
    fingerprint_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analytic, analytic_meta = symmetry_parameter_features(parameters)
    atomic_write_csv(analytic, outputs["analytic_features"])
    feature_meta = pd.concat([analytic_meta, fingerprint_meta], ignore_index=True)
    feature_meta = feature_meta.drop_duplicates("feature", keep="first")

    labels = physics[[
        "sample_id", "sample_source", "is_control", "is_ml_eligible",
        "phase_label", "is_strict_insulator", "is_spin_chern_topological",
        "is_spin_chern_TI_candidate", "chern_up_int", "chern_down_int",
        "chern_total_int", "min_direct_gap", "indirect_gap",
        "min_balanced_sector_gap", "spin_occupancy_mismatch_count",
        "verification_level",
    ]].copy()
    combined = labels.merge(analytic, on="sample_id", how="inner").merge(
        fingerprint, on="sample_id", how="inner"
    )
    atomic_write_csv(combined, outputs["combined_features"])
    return combined, feature_meta


# =============================================================================
# Step 5-8. Hierarchical ML with cluster holdout and independent Sobol test
# =============================================================================


def _metric_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    both = len(np.unique(y_true)) >= 2
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "n_positive_true": int(np.sum(y_true == 1)),
        "n_positive_pred": int(np.sum(y_pred == 1)),
    }


def _metric_multiclass(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) >= 2 else np.nan,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else np.nan,
        "n_classes_true": int(len(np.unique(y_true))),
        "n_classes_pred": int(len(np.unique(y_pred))),
    }


def _make_model(model_name: str, config: Step03Config, multiclass: bool = False):
    if model_name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.random_forest_estimators),
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=int(config.ml_n_jobs),
            min_samples_leaf=2,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _feature_sets(combined: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, list[str]]:
    numeric = set(combined.select_dtypes(include=[np.number]).columns)
    raw7 = [f for f in REDUCED7 if f in numeric]
    symmetry = metadata.loc[
        metadata["feature_family"].isin(["raw_parameters", "symmetry_parameter_combinations"]),
        "feature",
    ].tolist()
    projector = metadata.loc[metadata["feature_family"] == "path_projector", "feature"].tolist()
    spectrum = metadata.loc[metadata["feature_family"].isin([
        "fixed_k_spin_spectrum", "fixed_k_full_spectrum", "path_spectrum"
    ]), "feature"].tolist()
    fingerprint_all = metadata.loc[
        ~metadata["feature_family"].isin(["raw_parameters", "symmetry_parameter_combinations"]),
        "feature",
    ].tolist()

    sets = {
        "raw7": raw7,
        "symmetry_parameters": [f for f in symmetry if f in numeric],
        "spectrum_fingerprint": [f for f in spectrum if f in numeric],
        "projector_fingerprint": [f for f in projector if f in numeric],
        "full_fingerprint": [f for f in fingerprint_all if f in numeric],
        "combined_all": [f for f in dict.fromkeys(symmetry + fingerprint_all) if f in numeric],
    }
    return {name: cols for name, cols in sets.items() if len(cols) > 0}


def _build_datasets(
    config: Step03Config,
    outputs: Dict[str, Path],
    combined: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    train = combined[(combined["sample_source"] == "train_sobol") & (combined["is_ml_eligible"] == 1)].copy()
    external = combined[(combined["sample_source"] == "external_sobol") & (combined["is_ml_eligible"] == 1)].copy()

    train["levelA_target"] = train["is_strict_insulator"].astype(int)
    external["levelA_target"] = external["is_strict_insulator"].astype(int)

    train_B = train[train["phase_label"].isin(["trivial_insulator", "spin_chern_TI_candidate"])].copy()
    external_B = external[external["phase_label"].isin(["trivial_insulator", "spin_chern_TI_candidate"])].copy()
    train_B["levelB_binary_target"] = (train_B["phase_label"] == "spin_chern_TI_candidate").astype(int)
    external_B["levelB_binary_target"] = (external_B["phase_label"] == "spin_chern_TI_candidate").astype(int)
    train_B["levelB_multiclass_target"] = train_B["chern_up_int"].round().astype(int).map(lambda x: f"C_up_{x:+d}")
    external_B["levelB_multiclass_target"] = external_B["chern_up_int"].round().astype(int).map(lambda x: f"C_up_{x:+d}")

    atomic_write_csv(train, outputs["levelA_dataset"])
    atomic_write_csv(train_B, outputs["levelB_binary_dataset"])
    atomic_write_csv(train_B, outputs["levelB_multiclass_dataset"])
    return {
        "train_A": train,
        "external_A": external,
        "train_B": train_B,
        "external_B": external_B,
    }


def _assign_parameter_clusters(
    config: Step03Config,
    outputs: Dict[str, Path],
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    train_A = datasets["train_A"].copy()
    n_clusters = min(config.n_parameter_clusters, max(2, len(train_A) // 20))
    n_clusters = min(n_clusters, max(2, len(train_A) - 1))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(train_A[REDUCED7].astype(float))
    train_A["parameter_cluster"] = labels

    # Carry the same train-sample cluster labels to Level B.
    mapping = train_A.set_index("sample_id")["parameter_cluster"]
    train_B = datasets["train_B"].copy()
    train_B["parameter_cluster"] = train_B["sample_id"].map(mapping).astype(int)

    assignments = train_A[["sample_id", *REDUCED7, "parameter_cluster"]].copy()
    atomic_write_csv(assignments, outputs["cluster_assignments"])
    datasets = dict(datasets)
    datasets["train_A"] = train_A
    datasets["train_B"] = train_B
    return datasets


def _valid_binary_dataset(df: pd.DataFrame, target: str, min_count: int) -> bool:
    counts = df[target].value_counts()
    return len(counts) == 2 and int(counts.min()) >= min_count


def _valid_multiclass_dataset(df: pd.DataFrame, target: str, min_count: int) -> bool:
    counts = df[target].value_counts()
    return len(counts) >= 2 and int(counts.min()) >= min_count


def _random_cv(
    df: pd.DataFrame,
    target: str,
    task: str,
    feature_sets: dict[str, list[str]],
    config: Step03Config,
) -> pd.DataFrame:
    rows: list[dict] = []
    multiclass = task.endswith("multiclass")
    valid = _valid_multiclass_dataset(df, target, config.min_class_count_for_training) if multiclass else _valid_binary_dataset(df, target, config.min_class_count_for_training)
    if not valid:
        return pd.DataFrame([{
            "task": task,
            "status": "skipped_insufficient_class_counts",
            "class_counts": json.dumps(df[target].value_counts().to_dict(), ensure_ascii=False),
        }])

    min_count = int(df[target].value_counts().min())
    n_splits = min(config.cv_splits, min_count)
    if n_splits < 2:
        return pd.DataFrame([{"task": task, "status": "skipped_cv_splits_lt_2"}])
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=config.cv_repeats,
        random_state=42,
    )
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1_macro" if multiclass else "f1",
    }
    for feature_name, features in feature_sets.items():
        X = df[features].astype(float)
        y = df[target]
        for model_name in ("logistic", "random_forest"):
            model = _make_model(model_name, config, multiclass=multiclass)
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=int(config.ml_n_jobs),
                error_score=np.nan,
            )
            rows.append({
                "task": task,
                "status": "ok",
                "feature_set": feature_name,
                "model": model_name,
                "n_samples": len(df),
                "n_features": len(features),
                "accuracy_mean": float(np.nanmean(scores["test_accuracy"])),
                "accuracy_std": float(np.nanstd(scores["test_accuracy"])),
                "balanced_accuracy_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
                "balanced_accuracy_std": float(np.nanstd(scores["test_balanced_accuracy"])),
                "f1_mean": float(np.nanmean(scores["test_f1"])),
                "f1_std": float(np.nanstd(scores["test_f1"])),
            })
    return pd.DataFrame(rows)


def _cluster_holdout(
    df: pd.DataFrame,
    target: str,
    task: str,
    feature_sets: dict[str, list[str]],
    config: Step03Config,
) -> tuple[pd.DataFrame, dict[tuple, dict]]:
    rows: list[dict] = []
    fitted: dict[tuple, dict] = {}
    multiclass = task.endswith("multiclass")
    valid = _valid_multiclass_dataset(df, target, config.min_class_count_for_training) if multiclass else _valid_binary_dataset(df, target, config.min_class_count_for_training)
    if not valid or "parameter_cluster" not in df.columns:
        return pd.DataFrame([{
            "task": task,
            "status": "skipped_insufficient_class_counts_or_clusters",
            "class_counts": json.dumps(df[target].value_counts().to_dict(), ensure_ascii=False),
        }]), fitted

    splitter = GroupShuffleSplit(
        n_splits=config.group_holdout_splits,
        test_size=config.group_holdout_fraction,
        random_state=42,
    )
    groups = df["parameter_cluster"].to_numpy()
    y_all = df[target]

    for split_index, (train_idx, test_idx) in enumerate(splitter.split(df, y_all, groups)):
        y_train = y_all.iloc[train_idx]
        y_test = y_all.iloc[test_idx]
        if multiclass:
            # Every class in test must be known to the training set; otherwise the split is not scoreable.
            if not set(y_test.unique()).issubset(set(y_train.unique())) or len(y_test.unique()) < 2:
                continue
        else:
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

        for feature_name, features in feature_sets.items():
            X_train = df.iloc[train_idx][features].astype(float)
            X_test = df.iloc[test_idx][features].astype(float)
            for model_name in ("logistic", "random_forest"):
                model = _make_model(model_name, config, multiclass=multiclass)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                metrics = _metric_multiclass(y_test, pred) if multiclass else _metric_binary(y_test, pred)
                record = {
                    "task": task,
                    "status": "ok",
                    "split_index": int(split_index),
                    "feature_set": feature_name,
                    "model": model_name,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_clusters": json.dumps(sorted(set(groups[train_idx].tolist()))),
                    "test_clusters": json.dumps(sorted(set(groups[test_idx].tolist()))),
                    **metrics,
                }
                rows.append(record)
                key = (task, feature_name, model_name, split_index)
                fitted[key] = {
                    "model": model,
                    "features": features,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "y_test": y_test.copy(),
                    "pred": pred,
                }
    if not rows:
        return pd.DataFrame([{"task": task, "status": "no_valid_group_split"}]), fitted
    return pd.DataFrame(rows), fitted


def _select_best_from_cluster(cluster_metrics: pd.DataFrame, task: str) -> dict | None:
    part = cluster_metrics[(cluster_metrics.get("task") == task) & (cluster_metrics.get("status") == "ok")].copy()
    if not len(part):
        return None
    score_col = "balanced_accuracy"
    f1_col = "f1_macro" if "f1_macro" in part.columns and part["f1_macro"].notna().any() else "f1"
    grouped = (
        part.groupby(["feature_set", "model"], as_index=False)
        .agg(
            balanced_accuracy_mean=(score_col, "mean"),
            balanced_accuracy_std=(score_col, "std"),
            f1_mean=(f1_col, "mean"),
            valid_splits=("split_index", "nunique"),
        )
        .sort_values(["balanced_accuracy_mean", "f1_mean", "valid_splits"], ascending=False)
    )
    if not len(grouped):
        return None
    return grouped.iloc[0].to_dict()


def _external_evaluate(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    target: str,
    task: str,
    best_spec: dict | None,
    feature_sets: dict[str, list[str]],
    config: Step03Config,
) -> tuple[dict, Any | None, pd.DataFrame]:
    if best_spec is None:
        return ({"task": task, "status": "skipped_no_best_cluster_model"}, None, pd.DataFrame())
    feature_name = str(best_spec["feature_set"])
    model_name = str(best_spec["model"])
    features = feature_sets[feature_name]
    multiclass = task.endswith("multiclass")

    if len(external_df) == 0:
        return ({"task": task, "status": "skipped_no_external_samples"}, None, pd.DataFrame())
    if multiclass:
        if not set(external_df[target].unique()).issubset(set(train_df[target].unique())):
            unknown = sorted(set(external_df[target].unique()) - set(train_df[target].unique()))
            return ({
                "task": task,
                "status": "skipped_external_contains_unseen_classes",
                "unseen_classes": json.dumps(unknown),
            }, None, pd.DataFrame())
    else:
        if len(train_df[target].unique()) < 2:
            return ({"task": task, "status": "skipped_train_has_one_class"}, None, pd.DataFrame())

    model = _make_model(model_name, config, multiclass=multiclass)
    model.fit(train_df[features].astype(float), train_df[target])
    pred = model.predict(external_df[features].astype(float))
    metrics = _metric_multiclass(external_df[target], pred) if multiclass else _metric_binary(external_df[target], pred)
    record = {
        "task": task,
        "status": "ok",
        "feature_set": feature_name,
        "model": model_name,
        "n_train": len(train_df),
        **metrics,
    }
    pred_df = external_df[[
        "sample_id", "sample_source", "phase_label", "chern_up_int",
        "min_direct_gap", "indirect_gap", *REDUCED7,
    ]].copy()
    pred_df["task"] = task
    pred_df["target_true"] = external_df[target].astype(str).to_numpy()
    pred_df["target_pred"] = np.asarray(pred).astype(str)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(external_df[features].astype(float))
        classes = [str(c) for c in model.classes_]
        for i, cls in enumerate(classes):
            pred_df[f"prob_{cls}"] = probabilities[:, i]
    return record, model, pred_df


def _plot_confusion_from_predictions(pred_df: pd.DataFrame, task: str, path: Path) -> None:
    subset = pred_df[pred_df["task"] == task]
    if not len(subset):
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ConfusionMatrixDisplay.from_predictions(
        subset["target_true"], subset["target_pred"], ax=ax, xticks_rotation=35
    )
    ax.set_title(task)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def grouped_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    config: Step03Config,
) -> pd.DataFrame:
    if len(X) == 0 or len(np.unique(y)) < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(42)
    baseline = balanced_accuracy_score(y, model.predict(X))
    rows = []
    for group_name, group_features in groups.items():
        active = [f for f in group_features if f in X.columns]
        if not active:
            continue
        drops = []
        for _ in range(config.permutation_repeats):
            perm = rng.permutation(len(X))
            xp = X.copy()
            xp.loc[:, active] = X.iloc[perm][active].to_numpy()
            drops.append(baseline - balanced_accuracy_score(y, model.predict(xp)))
        rows.append({
            "group": group_name,
            "n_features": len(active),
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
            "baseline_balanced_accuracy": float(baseline),
        })
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def run_machine_learning(
    config: Step03Config,
    outputs: Dict[str, Path],
    combined: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    datasets = _build_datasets(config, outputs, combined)
    datasets = _assign_parameter_clusters(config, outputs, datasets)
    feature_sets = _feature_sets(combined, metadata)

    tasks = [
        ("LevelA_insulator_gate", datasets["train_A"], datasets["external_A"], "levelA_target"),
        ("LevelB_binary_topology", datasets["train_B"], datasets["external_B"], "levelB_binary_target"),
        ("LevelB_multiclass_chern", datasets["train_B"], datasets["external_B"], "levelB_multiclass_target"),
    ]

    random_rows = []
    cluster_rows = []
    all_fitted: dict[tuple, dict] = {}
    best_registry_rows = []
    external_rows = []
    external_prediction_frames = []
    fitted_external_models: dict[str, Any] = {}

    for task, train_df, external_df, target in tasks:
        random_df = _random_cv(train_df, target, task, feature_sets, config)
        random_rows.append(random_df)
        cluster_df, fitted = _cluster_holdout(train_df, target, task, feature_sets, config)
        cluster_rows.append(cluster_df)
        all_fitted.update(fitted)
        best = _select_best_from_cluster(cluster_df, task)
        if best is not None:
            best_registry_rows.append({"task": task, **best})
        ext_record, ext_model, ext_predictions = _external_evaluate(
            train_df, external_df, target, task, best, feature_sets, config
        )
        external_rows.append(ext_record)
        if ext_model is not None:
            fitted_external_models[task] = ext_model
        if len(ext_predictions):
            external_prediction_frames.append(ext_predictions)

    random_metrics = pd.concat(random_rows, ignore_index=True, sort=False)
    cluster_metrics = pd.concat(cluster_rows, ignore_index=True, sort=False)
    external_metrics = pd.DataFrame(external_rows)
    best_registry = pd.DataFrame(best_registry_rows)
    external_predictions = (
        pd.concat(external_prediction_frames, ignore_index=True, sort=False)
        if external_prediction_frames else pd.DataFrame()
    )

    atomic_write_csv(random_metrics, outputs["random_cv_metrics"])
    atomic_write_csv(cluster_metrics, outputs["cluster_holdout_metrics"])
    atomic_write_csv(external_metrics, outputs["external_metrics"])
    atomic_write_csv(best_registry, outputs["best_model_registry"])
    atomic_write_csv(external_predictions, outputs["external_predictions"])

    # Save one representative best cluster split per task.
    cluster_pred_frames = []
    report_parts = []
    for _, best in best_registry.iterrows():
        task = str(best["task"])
        matches = cluster_metrics[
            (cluster_metrics["task"] == task)
            & (cluster_metrics["feature_set"] == best["feature_set"])
            & (cluster_metrics["model"] == best["model"])
            & (cluster_metrics["status"] == "ok")
        ].sort_values("balanced_accuracy", ascending=False)
        if not len(matches):
            continue
        split_index = int(matches.iloc[0]["split_index"])
        key = (task, str(best["feature_set"]), str(best["model"]), split_index)
        fit = all_fitted.get(key)
        if fit is None:
            continue
        source_df = datasets["train_A"] if task == "LevelA_insulator_gate" else datasets["train_B"]
        test_rows = source_df.iloc[fit["test_idx"]].copy()
        out = test_rows[["sample_id", "phase_label", "chern_up_int", "parameter_cluster", *REDUCED7]].copy()
        out["task"] = task
        out["target_true"] = fit["y_test"].astype(str).to_numpy()
        out["target_pred"] = np.asarray(fit["pred"]).astype(str)
        cluster_pred_frames.append(out)
        report_parts.append(
            f"===== {task} / best cluster split =====\n"
            + classification_report(out["target_true"], out["target_pred"], zero_division=0)
        )

    cluster_predictions = (
        pd.concat(cluster_pred_frames, ignore_index=True, sort=False)
        if cluster_pred_frames else pd.DataFrame()
    )
    atomic_write_csv(cluster_predictions, outputs["cluster_predictions"])

    for task in ("LevelA_insulator_gate", "LevelB_binary_topology", "LevelB_multiclass_chern"):
        sub = external_predictions[external_predictions["task"] == task] if len(external_predictions) else pd.DataFrame()
        if len(sub):
            report_parts.append(
                f"===== {task} / independent Sobol external =====\n"
                + classification_report(sub["target_true"], sub["target_pred"], zero_division=0)
            )
    atomic_write_text("\n\n".join(report_parts), outputs["classification_reports"])

    if len(external_predictions):
        _plot_confusion_from_predictions(external_predictions, "LevelA_insulator_gate", outputs["levelA_confusion"])
        _plot_confusion_from_predictions(external_predictions, "LevelB_binary_topology", outputs["levelB_confusion"])
        _plot_confusion_from_predictions(external_predictions, "LevelB_multiclass_chern", outputs["levelB_multi_confusion"])

    # Save external-fitted models.
    if "LevelA_insulator_gate" in fitted_external_models:
        dump(fitted_external_models["LevelA_insulator_gate"], outputs["levelA_model"])
    if "LevelB_binary_topology" in fitted_external_models:
        dump(fitted_external_models["LevelB_binary_topology"], outputs["levelB_model"])
    if "LevelB_multiclass_chern" in fitted_external_models:
        dump(fitted_external_models["LevelB_multiclass_chern"], outputs["levelB_multi_model"])

    # Level B binary importance, evaluated on the independent external Sobol subset.
    importance_group = pd.DataFrame()
    importance_individual = pd.DataFrame()
    best_binary = best_registry[best_registry["task"] == "LevelB_binary_topology"] if len(best_registry) else pd.DataFrame()
    model_binary = fitted_external_models.get("LevelB_binary_topology")
    ext_B = datasets["external_B"]
    if len(best_binary) and model_binary is not None and len(ext_B) and len(ext_B["levelB_binary_target"].unique()) >= 2:
        feature_name = str(best_binary.iloc[0]["feature_set"])
        features = feature_sets[feature_name]
        X_ext = ext_B[features].astype(float)
        y_ext = ext_B["levelB_binary_target"].astype(int)
        family_groups = {
            str(family): metadata.loc[metadata["feature_family"] == family, "feature"].tolist()
            for family in metadata["feature_family"].dropna().unique()
        }
        importance_group = grouped_permutation_importance(
            model_binary, X_ext, y_ext, family_groups, config
        )
        atomic_write_csv(importance_group, outputs["group_importance"])

        perm = permutation_importance(
            model_binary,
            X_ext,
            y_ext,
            scoring="balanced_accuracy",
            n_repeats=config.permutation_repeats,
            random_state=42,
            n_jobs=config.ml_n_jobs,
        )
        importance_individual = pd.DataFrame({
            "feature": features,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
        atomic_write_csv(importance_individual, outputs["individual_importance"])

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = importance_group.sort_values("importance_mean")
        ax.barh(plot_df["group"], plot_df["importance_mean"], xerr=plot_df["importance_std"])
        ax.set_xlabel("Grouped permutation importance")
        ax.set_title("Level B binary importance on independent Sobol set")
        fig.tight_layout()
        fig.savefig(outputs["importance_fig"], dpi=220, bbox_inches="tight")
        plt.close(fig)
    else:
        atomic_write_csv(importance_group, outputs["group_importance"])
        atomic_write_csv(importance_individual, outputs["individual_importance"])

    # Misclassified external Hamiltonians.
    if len(external_predictions):
        misclassified = external_predictions[
            external_predictions["target_true"] != external_predictions["target_pred"]
        ].copy()
    else:
        misclassified = pd.DataFrame()
    atomic_write_csv(misclassified, outputs["misclassified"])

    return {
        "datasets": datasets,
        "feature_sets": feature_sets,
        "random_metrics": random_metrics,
        "cluster_metrics": cluster_metrics,
        "external_metrics": external_metrics,
        "best_registry": best_registry,
        "external_predictions": external_predictions,
        "cluster_predictions": cluster_predictions,
        "group_importance": importance_group,
        "individual_importance": importance_individual,
        "misclassified": misclassified,
    }


# =============================================================================
# Step 10. Orchestration
# =============================================================================


def run_step03(config: Step03Config | None = None) -> Dict[str, Any]:
    config = (config or Step03Config()).normalized()
    outputs = build_output_registry(config)
    write_run_metadata(config, outputs)

    print("[1/7] Generate global and independent Sobol parameters")
    parameters = generate_parameters(config, outputs)
    print(f"      total rows = {len(parameters)}")

    print("[2/7] Full-BZ physics labels and strict Chern verification")
    physics, chern_attempts, strict_gap_checks = run_physics_labels(config, outputs, parameters)
    print(physics["phase_label"].value_counts(dropna=False).to_string())

    print("[3/7] Phase and exact Chern-sector summaries")
    phase_summary = summarize_phases(config, outputs, physics)

    print("[4/7] Hamiltonian fingerprint")
    fingerprint, fp_metadata = run_fingerprint(config, outputs, parameters)

    print("[5/7] Symmetry parameter combinations and combined feature table")
    combined, metadata = build_feature_table(
        outputs, parameters, physics, fingerprint, fp_metadata
    )

    print("[6/7] Hierarchical ML: random CV, cluster holdout, independent Sobol")
    ml = run_machine_learning(config, outputs, combined, metadata)

    print("[7/7] Save run summary")
    global_physics = physics[physics["is_control"] == 0]
    controls = phase_summary["control_audit"]
    summary = {
        "code_version": CODE_VERSION,
        "run_tag": _run_tag(config),
        "n_train_sobol": config.n_train,
        "n_external_sobol": config.n_external,
        "n_control": int((physics["is_control"] == 1).sum()),
        "global_phase_counts": {
            str(k): int(v) for k, v in global_physics["phase_label"].value_counts().items()
        },
        "strict_insulator_chern_sector_counts": {
            str(k): int(v)
            for k, v in global_physics.loc[
                global_physics["is_strict_insulator"] == 1, "chern_up_int"
            ].value_counts().sort_index().items()
        },
        "n_strict_TI_candidates": int((global_physics["phase_label"] == "spin_chern_TI_candidate").sum()),
        "n_high_spin_chern_TI_candidates": int((
            (global_physics["phase_label"] == "spin_chern_TI_candidate")
            & (global_physics["chern_up_int"].abs() >= 2)
        ).sum()),
        "control_all_chern_match": int(
            len(controls) > 0 and controls.get("control_chern_match", pd.Series(dtype=int)).eq(1).all()
        ),
        "best_model_registry": ml["best_registry"].to_dict("records"),
        "external_metrics": ml["external_metrics"].to_dict("records"),
        "output_dir": str(config.output_dir),
    }
    atomic_write_text(json.dumps(summary, indent=2, ensure_ascii=False), outputs["summary"])
    write_run_metadata(config, outputs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return {
        "config": config,
        "outputs": outputs,
        "parameters": parameters,
        "physics": physics,
        "chern_attempts": chern_attempts,
        "strict_gap_checks": strict_gap_checks,
        "phase_summary": phase_summary,
        "fingerprint": fingerprint,
        "combined_features": combined,
        "metadata": metadata,
        "ml": ml,
        "summary": summary,
    }


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Step03Config.output_dir)
    parser.add_argument("--train-power", type=int, default=Step03Config.train_sobol_power)
    parser.add_argument("--external-power", type=int, default=Step03Config.external_sobol_power)
    parser.add_argument("--train-seed", type=int, default=Step03Config.train_seed)
    parser.add_argument("--external-seed", type=int, default=Step03Config.external_seed)
    parser.add_argument("--gap-nk", type=int, default=Step03Config.gap_nk)
    parser.add_argument("--path-n", type=int, default=Step03Config.fingerprint_path_n)
    parser.add_argument("--ml-n-jobs", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=Step03Config.checkpoint_every)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--force-physics", action="store_true")
    parser.add_argument("--force-fingerprint", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Step03Config:
    if args.smoke_test:
        return Step03Config(
            output_dir=args.output_dir,
            train_sobol_power=3,       # 8
            external_sobol_power=2,    # 4
            train_seed=args.train_seed,
            external_seed=args.external_seed,
            normalize_sobol_vectors=not args.no_normalize,
            include_control_points=not args.no_controls,
            gap_nk=17,
            initial_chern_grids=(11, 13),
            initial_chern_shifts=((0.0, 0.0), (0.5, 0.5)),
            strict_gap_grids=(17, 21),
            strict_gap_shifts=((0.0, 0.0), (0.5, 0.5)),
            strict_chern_grids=(13, 15),
            strict_chern_shifts=((0.0, 0.0), (0.5, 0.5)),
            fingerprint_path_n=7,
            n_parameter_clusters=3,
            group_holdout_splits=4,
            cv_splits=2,
            cv_repeats=1,
            random_forest_estimators=40,
            permutation_repeats=3,
            min_class_count_for_training=2,
            checkpoint_every=2,
            ml_n_jobs=1,
            force_recalculate_physics=args.force_physics,
            force_recalculate_fingerprint=args.force_fingerprint,
        )
    return Step03Config(
        output_dir=args.output_dir,
        train_sobol_power=args.train_power,
        external_sobol_power=args.external_power,
        train_seed=args.train_seed,
        external_seed=args.external_seed,
        normalize_sobol_vectors=not args.no_normalize,
        include_control_points=not args.no_controls,
        gap_nk=args.gap_nk,
        fingerprint_path_n=args.path_n,
        ml_n_jobs=args.ml_n_jobs,
        checkpoint_every=args.checkpoint_every,
        force_recalculate_physics=args.force_physics,
        force_recalculate_fingerprint=args.force_fingerprint,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_step03(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
