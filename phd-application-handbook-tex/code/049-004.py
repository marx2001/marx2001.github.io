#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 04 — Chern 扇区定向扩充、拓扑—平庸边界扫描与闭隙谷追踪

研究目标
--------
1. 从 TTS Step 03 的严格结果中分别选择 C_up = -2, -1, +1, +2
   （以及数据中实际存在的其他非零整数扇区）的代表性拓扑锚点；
2. 围绕每个 Chern 扇区进行局部 Sobol 定向扩充，确认拓扑区域不是孤立点，
   并为后续多分类和解析机制积累分扇区样本；
3. 为每个主锚点寻找参数空间中最近的严格平庸绝缘体；
4. 沿单位参数球面上的最短路径（slerp）扫描拓扑点—平庸点之间的相变；
5. 稀疏但严格地复核路径上的 Chern 数，并自动识别 Chern 改变区间；
6. 对直接带隙最小处联合优化 (lambda, kx, ky)，定位真正的闭隙谷；
7. 判断 |C|=1 是否主要由 Gamma/M 高对称谷驱动，|C|=2 是否主要由
   C4/D4 对称相关的普通动量谷共同驱动。

重要原则
--------
- 复用冻结的 Step 01 Hamiltonian、周期规范和非阿贝尔 spin-Chern 标签器；
- 复用 Step 03 的严格多网格标签逻辑；
- 全部物理计算串行，不使用 ProcessPoolExecutor，兼容 Windows/Jupyter；
- 路径密集扫描只计算能隙，Chern 仅在自适应选中的点严格复核；
- 本步骤输出的是机制证据和相边界数据，不把机器学习预测当成拓扑结论。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
import argparse
import io
import json
import math
import time
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

try:
    import TTS_step01_model_and_label_audit_v2 as core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未找到 TTS_step01_model_and_label_audit_v2.py。\n"
        "请把 Step 01 v2 的 py 文件与本脚本放在同一目录。"
    ) from exc

try:
    import TTS_step03_global_sobol_hierarchical_topology_ML as step3
except ImportError:
    try:
        import TTS_step03_global_sobol_hierarchical_topology_ML_final as step3
    except ImportError:
        try:
            import TTS_step03_global_sobol_hierarchical_topology_ML_fixed as step3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "未找到 Step 03 配套 py 文件。支持以下文件名：\n"
                "  TTS_step03_global_sobol_hierarchical_topology_ML.py\n"
                "  TTS_step03_global_sobol_hierarchical_topology_ML_final.py\n"
                "  TTS_step03_global_sobol_hierarchical_topology_ML_fixed.py"
            ) from exc

EXPECTED_STEP01_VERSION = "TTS_STEP01_V2_20260713"
EXPECTED_STEP03_VERSION = "TTS_STEP03_V1_20260713"
if getattr(core, "CODE_VERSION", None) != EXPECTED_STEP01_VERSION:
    raise RuntimeError(
        f"Step 01 核心版本不一致：expected={EXPECTED_STEP01_VERSION}, "
        f"loaded={getattr(core, 'CODE_VERSION', None)}"
    )
if getattr(step3, "CODE_VERSION", None) != EXPECTED_STEP03_VERSION:
    raise RuntimeError(
        f"Step 03 核心版本不一致：expected={EXPECTED_STEP03_VERSION}, "
        f"loaded={getattr(step3, 'CODE_VERSION', None)}"
    )

np.set_printoptions(precision=10, suppress=True)

CODE_VERSION = "TTS_STEP04_V1_20260713"
WORKFLOW_STEP = "step04"
SYSTEM_TAG = "tts8_bns127391"
TASK_TAG = "chern_sector_boundary_valley_tracking"
REDUCED7 = core.REDUCED7.copy()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Step04Config:
    output_dir: Path = Path("outputs_tts_step04_chern_sector_boundary_valley_tracking")
    step3_input: Path | None = None

    # Representative anchors selected independently in each nonzero Chern sector.
    target_sectors: tuple[int, ...] | None = (-2, -1, 1, 2)
    n_anchors_per_sector: int = 2
    anchor_candidate_pool: int = 24
    path_anchors_per_sector: int = 1
    trivial_partner_modes: tuple[str, ...] = ("nearest", "same_t_sign")

    # Local sector-directed Sobol augmentation on the normalized parameter sphere.
    local_angular_radii: tuple[float, ...] = (0.05, 0.10)
    local_sobol_power_per_radius: int = 4  # 16 samples/radius/anchor
    local_seed: int = 20260714

    # Physics verification for local samples and sparse path Chern points.
    local_gap_nk: int = 41
    initial_chern_grids: tuple[int, ...] = (21, 31)
    initial_chern_shifts: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.5))
    strict_gap_grids: tuple[int, ...] = (51, 71)
    strict_gap_shifts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)
    )
    strict_chern_grids: tuple[int, ...] = (31, 41, 51)
    strict_chern_shifts: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.5, 0.5))
    gap_tol: float = 1.0e-3
    direct_gap_skip_chern: float = 3.0e-3
    chern_integer_tol: float = 0.08
    min_link_tol: float = 1.0e-7
    sum_rule_tol: float = 0.15

    # Dense path gap scan and adaptive sparse Chern audit.
    path_n_lambda: int = 121
    path_gap_nk: int = 35
    path_chern_stride: int = 10
    path_extra_gap_minima: int = 6
    max_chern_points_per_path: int = 25
    max_transitions_per_path: int = 4

    # Continuous critical valley refinement.
    critical_k_symmetry_tol: float = 0.10
    critical_optimizer_starts: int = 6
    critical_optimizer_maxiter: int = 350

    # Workflow controls.
    checkpoint_every: int = 8
    force_recalculate_local: bool = False
    force_recalculate_paths: bool = False
    force_recalculate_chern: bool = False

    def normalized(self) -> "Step04Config":
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        if self.step3_input is not None:
            self.step3_input = Path(self.step3_input)
        if self.n_anchors_per_sector < 1:
            raise ValueError("n_anchors_per_sector must be >= 1")
        if self.path_anchors_per_sector < 1:
            raise ValueError("path_anchors_per_sector must be >= 1")
        if self.local_sobol_power_per_radius < 0:
            raise ValueError("local_sobol_power_per_radius must be >= 0")
        if self.path_n_lambda < 11:
            raise ValueError("path_n_lambda must be >= 11")
        if self.path_gap_nk < 9:
            raise ValueError("path_gap_nk must be >= 9")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        return self

    @property
    def n_local_per_radius(self) -> int:
        return 2 ** int(self.local_sobol_power_per_radius)


# =============================================================================
# Safe I/O and Step 03 input discovery
# =============================================================================

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _find_default_step3_input() -> Path:
    candidates = [
        Path.cwd() / "outputs_tts_step03_global_sobol_hierarchical_topology_ml.zip",
        Path.cwd().parent / "outputs_tts_step03_global_sobol_hierarchical_topology_ml.zip",
        Path("/mnt/data") / "outputs_tts_step03_global_sobol_hierarchical_topology_ml.zip",
        Path.cwd() / "outputs_tts_step03_global_sobol_hierarchical_topology_ml",
        Path.cwd().parent / "outputs_tts_step03_global_sobol_hierarchical_topology_ml",
        Path("/mnt/data") / "outputs_tts_step03_global_sobol_hierarchical_topology_ml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到 Step 03 结果。请把结果 ZIP 放在当前目录，或在配置中设置 step3_input。"
    )


def _read_unique_csv_from_zip(zip_path: Path, pattern: str) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(zip_path) as zf:
        matches = [
            name for name in zf.namelist()
            if pattern in Path(name).name and name.lower().endswith(".csv")
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"ZIP 中匹配 {pattern!r} 的 CSV 数量为 {len(matches)}: {matches}"
            )
        member = matches[0]
        with zf.open(member) as fh:
            return pd.read_csv(fh, low_memory=False), f"{zip_path}!/{member}"


def _read_unique_csv_from_directory(root: Path, pattern: str) -> tuple[pd.DataFrame, str]:
    matches = [p for p in root.rglob("*.csv") if pattern in p.name]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"目录中匹配 {pattern!r} 的 CSV 数量为 {len(matches)}: {matches}"
        )
    return pd.read_csv(matches[0], low_memory=False), str(matches[0])


def load_step3_physics(config: Step04Config) -> tuple[pd.DataFrame, str]:
    source = config.step3_input or _find_default_step3_input()
    source = Path(source)
    pattern = "step03_02_physics_labels_final__"
    if source.is_file() and source.suffix.lower() == ".zip":
        return _read_unique_csv_from_zip(source, pattern)
    if source.is_dir():
        return _read_unique_csv_from_directory(source, pattern)
    raise FileNotFoundError(f"无效的 Step 03 输入：{source}")


# =============================================================================
# Parameter geometry on the normalized seven-dimensional sphere
# =============================================================================

def row_vector(row: pd.Series | dict) -> np.ndarray:
    return np.array([float(row[name]) for name in REDUCED7], dtype=float)


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1.0e-14:
        raise ValueError("Cannot normalize zero/nonfinite parameter vector")
    return v / norm


def vector_to_reduced(vector: Sequence[float]) -> dict[str, float]:
    v = normalize_vector(vector)
    return {name: float(value) for name, value in zip(REDUCED7, v)}


def angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa = normalize_vector(a)
    bb = normalize_vector(b)
    return float(np.arccos(np.clip(float(np.dot(aa, bb)), -1.0, 1.0)))


def slerp(a: Sequence[float], b: Sequence[float], lam: float) -> np.ndarray:
    aa = normalize_vector(a)
    bb = normalize_vector(b)
    dot = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    omega = float(np.arccos(dot))
    lam = float(lam)
    if omega < 1.0e-10:
        return aa.copy()
    if abs(math.pi - omega) < 1.0e-7:
        # Nearly antipodal: normalized linear interpolation is the least arbitrary fallback.
        mixed = (1.0 - lam) * aa + lam * bb
        if np.linalg.norm(mixed) < 1.0e-12:
            mixed = aa.copy()
            mixed[0] += 1.0e-6
        return normalize_vector(mixed)
    return normalize_vector(
        math.sin((1.0 - lam) * omega) / math.sin(omega) * aa
        + math.sin(lam * omega) / math.sin(omega) * bb
    )


# =============================================================================
# Anchor and nearest-trivial selection
# =============================================================================

def _strict_topological(physics: pd.DataFrame) -> pd.DataFrame:
    mask = physics["phase_label"].eq("spin_chern_TI_candidate")
    if "strict_chern_verified" in physics:
        mask &= physics["strict_chern_verified"].fillna(0).astype(int).eq(1)
    if "strict_gap_verified" in physics:
        mask &= physics["strict_gap_verified"].fillna(0).astype(int).eq(1)
    return physics.loc[mask].copy()


def _strict_trivial(physics: pd.DataFrame) -> pd.DataFrame:
    mask = physics["phase_label"].eq("trivial_insulator")
    if "is_strict_insulator" in physics:
        mask &= physics["is_strict_insulator"].fillna(0).astype(int).eq(1)
    return physics.loc[mask].copy()


def select_sector_anchors(physics: pd.DataFrame, config: Step04Config) -> pd.DataFrame:
    topo = _strict_topological(physics)
    if topo.empty:
        raise RuntimeError("Step 03 中没有严格 spin-Chern TI 候选。")
    topo["chern_up_int"] = topo["chern_up_int"].astype(int)

    available = sorted(int(x) for x in topo["chern_up_int"].unique() if int(x) != 0)
    requested = available if config.target_sectors is None else [
        int(x) for x in config.target_sectors if int(x) in available
    ]
    missing = [] if config.target_sectors is None else [
        int(x) for x in config.target_sectors if int(x) not in available
    ]
    if missing:
        warnings.warn(f"Step 03 中未找到这些 Chern 扇区，将跳过：{missing}")

    selected_rows: list[dict] = []
    for sector in requested:
        group = topo[topo["chern_up_int"].astype(int) == sector].copy()
        group = group.sort_values("indirect_gap", ascending=False).head(config.anchor_candidate_pool)
        if group.empty:
            continue

        chosen_indices: list[int] = [int(group.index[0])]
        while len(chosen_indices) < min(config.n_anchors_per_sector, len(group)):
            best_idx = None
            best_score = -np.inf
            for idx, row in group.iterrows():
                idx_i = int(idx)
                if idx_i in chosen_indices:
                    continue
                candidate = row_vector(row)
                min_dist = min(
                    angular_distance(candidate, row_vector(physics.loc[j]))
                    for j in chosen_indices
                )
                # Gap is a weak tie-breaker; diversity is primary.
                score = min_dist + 1.0e-3 * float(row["indirect_gap"])
                if score > best_score:
                    best_score = score
                    best_idx = idx_i
            if best_idx is None:
                break
            chosen_indices.append(best_idx)

        for rank, idx in enumerate(chosen_indices, start=1):
            row = physics.loc[idx].to_dict()
            row.update({
                "anchor_id": f"Cup{sector:+d}_anchor{rank:02d}",
                "anchor_sector": int(sector),
                "anchor_rank": int(rank),
                "is_primary_path_anchor": int(rank <= config.path_anchors_per_sector),
            })
            selected_rows.append(row)

    anchors = pd.DataFrame(selected_rows)
    if anchors.empty:
        raise RuntimeError("没有可用的分扇区拓扑锚点。")
    return anchors.reset_index(drop=True)


def select_trivial_partners(
    physics: pd.DataFrame,
    anchors: pd.DataFrame,
    config: Step04Config,
) -> pd.DataFrame:
    trivial = _strict_trivial(physics).copy()
    if trivial.empty:
        raise RuntimeError("Step 03 中没有严格平庸绝缘体。")

    rows: list[dict] = []
    for _, anchor in anchors[anchors["is_primary_path_anchor"].astype(int) == 1].iterrows():
        av = row_vector(anchor)
        anchor_t_sign = int(np.sign(float(anchor["t1"]) * float(anchor["t2"])))
        used_sample_ids: set[str] = set()

        for mode in config.trivial_partner_modes:
            candidates = trivial
            if mode == "same_t_sign":
                t_sign = np.sign(candidates["t1"].astype(float) * candidates["t2"].astype(float)).astype(int)
                candidates = candidates[t_sign == anchor_t_sign]
            elif mode != "nearest":
                raise ValueError(f"Unknown trivial partner mode: {mode}")

            if candidates.empty:
                continue
            scored: list[tuple[float, int]] = []
            for idx, candidate in candidates.iterrows():
                sid = str(candidate["sample_id"])
                if sid in used_sample_ids:
                    continue
                scored.append((angular_distance(av, row_vector(candidate)), int(idx)))
            if not scored:
                continue
            dist, idx = min(scored, key=lambda item: item[0])
            partner = physics.loc[idx]
            used_sample_ids.add(str(partner["sample_id"]))
            rows.append({
                "path_id": f"{anchor['anchor_id']}__{mode}",
                "anchor_id": str(anchor["anchor_id"]),
                "anchor_sample_id": str(anchor["sample_id"]),
                "anchor_sector": int(anchor["anchor_sector"]),
                "partner_mode": mode,
                "trivial_sample_id": str(partner["sample_id"]),
                "angular_distance": float(dist),
                "euclidean_distance": float(np.linalg.norm(normalize_vector(av) - normalize_vector(row_vector(partner)))),
                "anchor_t_product": float(anchor["t1"] * anchor["t2"]),
                "trivial_t_product": float(partner["t1"] * partner["t2"]),
                **{f"anchor_{name}": float(anchor[name]) for name in REDUCED7},
                **{f"trivial_{name}": float(partner[name]) for name in REDUCED7},
                "anchor_indirect_gap": float(anchor["indirect_gap"]),
                "trivial_indirect_gap": float(partner["indirect_gap"]),
            })
    partners = pd.DataFrame(rows)
    if partners.empty:
        raise RuntimeError("无法构造拓扑—平庸路径配对。")
    return partners


# =============================================================================
# Local Sobol augmentation and strict physics labels
# =============================================================================

def generate_local_sobol(anchors: pd.DataFrame, config: Step04Config) -> pd.DataFrame:
    rows: list[dict] = []
    sample_counter = 0
    for anchor_pos, (_, anchor) in enumerate(anchors.iterrows()):
        a = normalize_vector(row_vector(anchor))
        for radius_pos, radius in enumerate(config.local_angular_radii):
            engine = qmc.Sobol(
                d=len(REDUCED7),
                scramble=True,
                seed=int(config.local_seed + 1009 * anchor_pos + 97 * radius_pos),
            )
            cube = 2.0 * engine.random_base2(config.local_sobol_power_per_radius) - 1.0
            for local_idx, raw in enumerate(cube):
                tangent = raw - float(np.dot(raw, a)) * a
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm < 1.0e-12:
                    tangent = np.roll(a, 1) - float(np.dot(np.roll(a, 1), a)) * a
                    tangent_norm = float(np.linalg.norm(tangent))
                tangent /= tangent_norm
                radial_fraction = max(0.10, min(1.0, float(np.linalg.norm(raw) / math.sqrt(len(REDUCED7)))))
                angle = float(radius) * radial_fraction
                candidate = normalize_vector(math.cos(angle) * a + math.sin(angle) * tangent)
                reduced = vector_to_reduced(candidate)
                rows.append({
                    "sample_id": f"local_{sample_counter:06d}",
                    "sample_source": "sector_local_sobol",
                    "is_control": 0,
                    "is_ml_eligible": 1,
                    "parent_anchor_id": str(anchor["anchor_id"]),
                    "parent_sample_id": str(anchor["sample_id"]),
                    "parent_chern_up": int(anchor["anchor_sector"]),
                    "local_radius_max": float(radius),
                    "local_angle_actual": float(angle),
                    "local_index": int(local_idx),
                    **reduced,
                })
                sample_counter += 1
    return pd.DataFrame(rows)


def make_physics_config(config: Step04Config) -> step3.Step03Config:
    return step3.Step03Config(
        output_dir=config.output_dir / "_step03_core_unused",
        train_sobol_power=2,
        external_sobol_power=0,
        gap_nk=config.local_gap_nk,
        initial_chern_grids=config.initial_chern_grids,
        initial_chern_shifts=config.initial_chern_shifts,
        strict_gap_grids=config.strict_gap_grids,
        strict_gap_shifts=config.strict_gap_shifts,
        strict_chern_grids=config.strict_chern_grids,
        strict_chern_shifts=config.strict_chern_shifts,
        direct_gap_skip_chern=config.direct_gap_skip_chern,
        gap_tol=config.gap_tol,
        chern_integer_tol=config.chern_integer_tol,
        min_link_tol=config.min_link_tol,
        sum_rule_tol=config.sum_rule_tol,
        checkpoint_every=config.checkpoint_every,
        ml_n_jobs=1,
    )


def run_local_physics(
    local_params: pd.DataFrame,
    config: Step04Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = config.output_dir
    final_path = out / "step04_02_local_physics_labels_final.csv"
    checkpoint_path = out / "step04_02_local_physics_labels_checkpoint.csv"
    attempts_path = out / "step04_02_local_chern_attempts.csv"
    strict_gap_path = out / "step04_02_local_strict_gap_checks.csv"

    if final_path.exists() and not config.force_recalculate_local:
        final = pd.read_csv(final_path, low_memory=False)
        if set(final["sample_id"].astype(str)) == set(local_params["sample_id"].astype(str)):
            attempts = pd.read_csv(attempts_path, low_memory=False) if attempts_path.exists() else pd.DataFrame()
            gaps = pd.read_csv(strict_gap_path, low_memory=False) if strict_gap_path.exists() else pd.DataFrame()
            return final, attempts, gaps

    if checkpoint_path.exists() and not config.force_recalculate_local:
        final_rows = pd.read_csv(checkpoint_path, low_memory=False).to_dict("records")
        done_ids = {str(row["sample_id"]) for row in final_rows}
    else:
        final_rows, done_ids = [], set()
    attempts_rows = (
        pd.read_csv(attempts_path, low_memory=False).to_dict("records")
        if attempts_path.exists() and not config.force_recalculate_local else []
    )
    strict_gap_rows = (
        pd.read_csv(strict_gap_path, low_memory=False).to_dict("records")
        if strict_gap_path.exists() and not config.force_recalculate_local else []
    )

    pconfig = make_physics_config(config)
    started = time.time()
    processed = 0
    metadata_cols = [
        "parent_anchor_id", "parent_sample_id", "parent_chern_up",
        "local_radius_max", "local_angle_actual", "local_index",
    ]
    for _, row in local_params.iterrows():
        sid = str(row["sample_id"])
        if sid in done_ids:
            continue
        result, attempts, gaps = step3.evaluate_sample_physics(row, pconfig)
        for col in metadata_cols:
            result[col] = row[col]
        final_rows.append(result)
        for attempt in attempts:
            attempt.update({col: row[col] for col in metadata_cols})
        for gap in gaps:
            gap.update({col: row[col] for col in metadata_cols})
        attempts_rows.extend(attempts)
        strict_gap_rows.extend(gaps)
        processed += 1

        if processed % config.checkpoint_every == 0:
            atomic_write_csv(pd.DataFrame(final_rows), checkpoint_path)
            atomic_write_csv(pd.DataFrame(attempts_rows), attempts_path)
            atomic_write_csv(pd.DataFrame(strict_gap_rows), strict_gap_path)
            print(f"Local physics: {len(final_rows)}/{len(local_params)} | elapsed {time.time()-started:.1f} s")

    final = pd.DataFrame(final_rows)
    attempts = pd.DataFrame(attempts_rows)
    gaps = pd.DataFrame(strict_gap_rows)
    atomic_write_csv(final, final_path)
    atomic_write_csv(attempts, attempts_path)
    atomic_write_csv(gaps, strict_gap_path)
    return final, attempts, gaps


def summarize_local(local: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_counts = (
        local.groupby(["parent_chern_up", "parent_anchor_id", "local_radius_max", "phase_label"], dropna=False)
        .size().reset_index(name="count")
    )
    summary_rows: list[dict] = []
    for keys, group in local.groupby(["parent_chern_up", "parent_anchor_id", "local_radius_max"]):
        sector, anchor_id, radius = keys
        same_sector = (
            group["phase_label"].eq("spin_chern_TI_candidate")
            & group["chern_up_int"].fillna(999).astype(int).eq(int(sector))
        )
        summary_rows.append({
            "parent_chern_up": int(sector),
            "parent_anchor_id": str(anchor_id),
            "local_radius_max": float(radius),
            "n_samples": int(len(group)),
            "n_same_sector_TI": int(same_sector.sum()),
            "same_sector_retention_fraction": float(same_sector.mean()),
            "n_other_TI_sector": int((group["phase_label"].eq("spin_chern_TI_candidate") & ~same_sector).sum()),
            "n_trivial_insulator": int(group["phase_label"].eq("trivial_insulator").sum()),
            "n_band_metal": int(group["phase_label"].eq("spin_chern_band_metal").sum()),
            "n_boundary_or_closing": int(group["phase_label"].isin([
                "noninsulating_or_gap_closing", "boundary_or_chern_unreliable"
            ]).sum()),
            "best_indirect_gap_same_sector": float(group.loc[same_sector, "indirect_gap"].max()) if same_sector.any() else np.nan,
        })
    return phase_counts, pd.DataFrame(summary_rows)


# =============================================================================
# Dense path gap scan
# =============================================================================

def fast_gap_scan(params: Dict[str, float], nk: int) -> dict[str, float]:
    kxs, kys = core.bz_grid(int(nk), (0.0, 0.0))
    min_direct = np.inf
    max_valence = -np.inf
    min_conduction = np.inf
    direct_k = (np.nan, np.nan)
    vbm_k = (np.nan, np.nan)
    cbm_k = (np.nan, np.nan)

    for kx in kxs:
        for ky in kys:
            eig = core.eigvals_full(float(kx), float(ky), params)
            direct = float(eig[4] - eig[3])
            if direct < min_direct:
                min_direct = direct
                direct_k = (float(kx), float(ky))
            if float(eig[3]) > max_valence:
                max_valence = float(eig[3])
                vbm_k = (float(kx), float(ky))
            if float(eig[4]) < min_conduction:
                min_conduction = float(eig[4])
                cbm_k = (float(kx), float(ky))
    return {
        "min_direct_gap": float(min_direct),
        "indirect_gap": float(min_conduction - max_valence),
        "direct_gap_kx": direct_k[0],
        "direct_gap_ky": direct_k[1],
        "vbm": float(max_valence),
        "vbm_kx": vbm_k[0],
        "vbm_ky": vbm_k[1],
        "cbm": float(min_conduction),
        "cbm_kx": cbm_k[0],
        "cbm_ky": cbm_k[1],
    }


def generate_path_parameters(partners: pd.DataFrame, config: Step04Config) -> pd.DataFrame:
    rows: list[dict] = []
    lambdas = np.linspace(0.0, 1.0, config.path_n_lambda)
    for _, pair in partners.iterrows():
        a = np.array([float(pair[f"anchor_{name}"]) for name in REDUCED7])
        b = np.array([float(pair[f"trivial_{name}"]) for name in REDUCED7])
        for index, lam in enumerate(lambdas):
            reduced = vector_to_reduced(slerp(a, b, float(lam)))
            rows.append({
                "path_point_id": f"{pair['path_id']}__p{index:04d}",
                "path_id": str(pair["path_id"]),
                "path_index": int(index),
                "lambda": float(lam),
                "anchor_sector": int(pair["anchor_sector"]),
                "partner_mode": str(pair["partner_mode"]),
                "anchor_sample_id": str(pair["anchor_sample_id"]),
                "trivial_sample_id": str(pair["trivial_sample_id"]),
                **reduced,
            })
    return pd.DataFrame(rows)


def run_path_gap_scans(path_params: pd.DataFrame, config: Step04Config) -> pd.DataFrame:
    output = config.output_dir / "step04_04_path_dense_gap_scan.csv"
    checkpoint = config.output_dir / "step04_04_path_dense_gap_scan_checkpoint.csv"
    if output.exists() and not config.force_recalculate_paths:
        df = pd.read_csv(output, low_memory=False)
        if set(df["path_point_id"].astype(str)) == set(path_params["path_point_id"].astype(str)):
            return df

    if checkpoint.exists() and not config.force_recalculate_paths:
        rows = pd.read_csv(checkpoint, low_memory=False).to_dict("records")
        done = {str(row["path_point_id"]) for row in rows}
    else:
        rows, done = [], set()

    started = time.time()
    processed = 0
    for _, row in path_params.iterrows():
        point_id = str(row["path_point_id"])
        if point_id in done:
            continue
        reduced = {name: float(row[name]) for name in REDUCED7}
        params = core.raw8_from_reduced7(reduced, e0=0.0)
        scan = fast_gap_scan(params, config.path_gap_nk)
        rows.append({**row.to_dict(), **scan, "path_gap_nk": int(config.path_gap_nk)})
        processed += 1
        if processed % max(config.checkpoint_every, 10) == 0:
            atomic_write_csv(pd.DataFrame(rows), checkpoint)
            print(f"Path gap: {len(rows)}/{len(path_params)} | elapsed {time.time()-started:.1f} s")

    result = pd.DataFrame(rows).sort_values(["path_id", "path_index"]).reset_index(drop=True)
    atomic_write_csv(result, output)
    return result


def select_adaptive_chern_points(path_gap: pd.DataFrame, config: Step04Config) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    for path_id, group in path_gap.groupby("path_id", sort=False):
        group = group.sort_values("path_index").reset_index(drop=True)
        n = len(group)
        indices: set[int] = set(range(0, n, max(1, config.path_chern_stride)))
        indices.update({0, n - 1})

        # Global and secondary smallest gap points.
        smallest = group.nsmallest(min(config.path_extra_gap_minima, n), "min_direct_gap").index
        for idx in smallest:
            for offset in (-1, 0, 1):
                if 0 <= int(idx) + offset < n:
                    indices.add(int(idx) + offset)

        # All local minima of the discrete path gap.
        gaps = group["min_direct_gap"].to_numpy(float)
        for idx in range(1, n - 1):
            if gaps[idx] <= gaps[idx - 1] and gaps[idx] <= gaps[idx + 1]:
                indices.update({idx - 1, idx, idx + 1})

        if len(indices) > config.max_chern_points_per_path:
            mandatory = {0, n - 1, int(np.argmin(gaps))}
            remaining = sorted(indices - mandatory, key=lambda i: gaps[i])
            keep = mandatory | set(remaining[: max(0, config.max_chern_points_per_path - len(mandatory))])
            indices = keep

        for idx in sorted(indices):
            row = group.iloc[idx].copy()
            row["sample_id"] = str(row["path_point_id"])
            row["sample_source"] = "adaptive_path_chern"
            row["is_control"] = 1  # force strict Chern verification for every selected point
            row["is_ml_eligible"] = 0
            selected_rows.append(row)
    return pd.DataFrame(selected_rows)


def run_path_chern_audit(
    selected: pd.DataFrame,
    config: Step04Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_path = config.output_dir / "step04_04_path_adaptive_chern_labels.csv"
    attempts_path = config.output_dir / "step04_04_path_chern_attempts.csv"
    gaps_path = config.output_dir / "step04_04_path_strict_gap_checks.csv"
    checkpoint = config.output_dir / "step04_04_path_adaptive_chern_checkpoint.csv"

    if final_path.exists() and not config.force_recalculate_chern:
        final = pd.read_csv(final_path, low_memory=False)
        if set(final["sample_id"].astype(str)) == set(selected["sample_id"].astype(str)):
            attempts = pd.read_csv(attempts_path, low_memory=False) if attempts_path.exists() else pd.DataFrame()
            gaps = pd.read_csv(gaps_path, low_memory=False) if gaps_path.exists() else pd.DataFrame()
            return final, attempts, gaps

    if checkpoint.exists() and not config.force_recalculate_chern:
        rows = pd.read_csv(checkpoint, low_memory=False).to_dict("records")
        done = {str(row["sample_id"]) for row in rows}
    else:
        rows, done = [], set()
    attempts_rows = (
        pd.read_csv(attempts_path, low_memory=False).to_dict("records")
        if attempts_path.exists() and not config.force_recalculate_chern else []
    )
    gap_rows = (
        pd.read_csv(gaps_path, low_memory=False).to_dict("records")
        if gaps_path.exists() and not config.force_recalculate_chern else []
    )

    pconfig = make_physics_config(config)
    started = time.time()
    processed = 0
    metadata = [
        "path_id", "path_point_id", "path_index", "lambda", "anchor_sector",
        "partner_mode", "anchor_sample_id", "trivial_sample_id",
    ]
    for _, row in selected.iterrows():
        sid = str(row["sample_id"])
        if sid in done:
            continue
        result, attempts, gaps = step3.evaluate_sample_physics(row, pconfig)
        for col in metadata:
            result[col] = row[col]
        for attempt in attempts:
            attempt.update({col: row[col] for col in metadata})
        for gap in gaps:
            gap.update({col: row[col] for col in metadata})
        rows.append(result)
        attempts_rows.extend(attempts)
        gap_rows.extend(gaps)
        processed += 1
        if processed % config.checkpoint_every == 0:
            atomic_write_csv(pd.DataFrame(rows), checkpoint)
            atomic_write_csv(pd.DataFrame(attempts_rows), attempts_path)
            atomic_write_csv(pd.DataFrame(gap_rows), gaps_path)
            print(f"Path Chern: {len(rows)}/{len(selected)} | elapsed {time.time()-started:.1f} s")

    final = pd.DataFrame(rows).sort_values(["path_id", "lambda"]).reset_index(drop=True)
    attempts = pd.DataFrame(attempts_rows)
    gaps = pd.DataFrame(gap_rows)
    atomic_write_csv(final, final_path)
    atomic_write_csv(attempts, attempts_path)
    atomic_write_csv(gaps, gaps_path)
    return final, attempts, gaps


# =============================================================================
# Transition brackets and continuous critical-valley refinement
# =============================================================================

def wrap_k(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def torus_delta(a: float, b: float) -> float:
    return wrap_k(float(a) - float(b))


def torus_distance(k: tuple[float, float], q: tuple[float, float]) -> float:
    return float(math.hypot(torus_delta(k[0], q[0]), torus_delta(k[1], q[1])))


def classify_k_region(kx: float, ky: float, tol: float) -> dict[str, Any]:
    k = (wrap_k(kx), wrap_k(ky))
    high_symmetry = {
        "Gamma": [(0.0, 0.0)],
        "X": [(math.pi, 0.0), (-math.pi, 0.0)],
        "Y": [(0.0, math.pi), (0.0, -math.pi)],
        "M": [
            (math.pi, math.pi), (math.pi, -math.pi),
            (-math.pi, math.pi), (-math.pi, -math.pi),
        ],
    }
    distances = {
        name: min(torus_distance(k, point) for point in points)
        for name, points in high_symmetry.items()
    }
    nearest_name = min(distances, key=distances.get)
    if distances[nearest_name] <= tol:
        region = nearest_name
        is_high = 1
    else:
        diag_plus = abs(torus_delta(k[0], k[1]))
        diag_minus = abs(torus_delta(k[0], -k[1]))
        axis_x = abs(wrap_k(k[1]))
        axis_y = abs(wrap_k(k[0]))
        if diag_plus <= tol:
            region = "generic_Sigma_kx_eq_ky"
        elif diag_minus <= tol:
            region = "generic_SigmaPrime_kx_eq_minus_ky"
        elif axis_x <= tol or abs(abs(k[1]) - math.pi) <= tol:
            region = "generic_horizontal"
        elif axis_y <= tol or abs(abs(k[0]) - math.pi) <= tol:
            region = "generic_vertical"
        else:
            region = "generic"
        is_high = 0
    return {
        "critical_k_region": region,
        "critical_is_high_symmetry": int(is_high),
        "nearest_high_symmetry": nearest_name,
        "distance_to_nearest_high_symmetry": float(distances[nearest_name]),
        "distance_to_Sigma": float(abs(torus_delta(k[0], k[1]))),
        "distance_to_SigmaPrime": float(abs(torus_delta(k[0], -k[1]))),
    }


def _unique_k_points(points: Iterable[tuple[float, float]], tol: float = 1.0e-7) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in points:
        p = (wrap_k(point[0]), wrap_k(point[1]))
        if not any(torus_distance(p, q) <= tol for q in unique):
            unique.append(p)
    return unique


def c4_orbit(kx: float, ky: float) -> list[tuple[float, float]]:
    return _unique_k_points([
        (kx, ky), (-ky, kx), (-kx, -ky), (ky, -kx),
    ])


def d4_orbit(kx: float, ky: float) -> list[tuple[float, float]]:
    base = c4_orbit(kx, ky)
    mirrors = [(x, -y) for x, y in base]
    return _unique_k_points(base + mirrors)


def detect_transition_brackets(path_chern: pd.DataFrame, path_gap: pd.DataFrame, config: Step04Config) -> pd.DataFrame:
    rows: list[dict] = []
    for path_id, gap_group in path_gap.groupby("path_id", sort=False):
        ch = path_chern[path_chern["path_id"].astype(str) == str(path_id)].copy()
        ch = ch.dropna(subset=["chern_up_int"]).sort_values("lambda")
        valid = ch[ch["chern_exact_consensus"].fillna(0).astype(int) == 1]
        transition_count = 0
        if len(valid) >= 2:
            valid_rows = list(valid.to_dict("records"))
            for left, right in zip(valid_rows[:-1], valid_rows[1:]):
                c_left = int(left["chern_up_int"])
                c_right = int(right["chern_up_int"])
                if c_left == c_right:
                    continue
                rows.append({
                    "path_id": str(path_id),
                    "transition_id": f"{path_id}__transition{transition_count:02d}",
                    "lambda_left": float(left["lambda"]),
                    "lambda_right": float(right["lambda"]),
                    "chern_left": c_left,
                    "chern_right": c_right,
                    "delta_chern_up": int(c_right - c_left),
                    "bracket_source": "strict_chern_change",
                })
                transition_count += 1
                if transition_count >= config.max_transitions_per_path:
                    break

        if transition_count == 0:
            group = gap_group.sort_values("lambda").reset_index(drop=True)
            idx = int(group["min_direct_gap"].astype(float).idxmin())
            # idx above is original index; convert to positional index.
            pos = int(np.argmin(group["min_direct_gap"].to_numpy(float)))
            left_pos = max(0, pos - 1)
            right_pos = min(len(group) - 1, pos + 1)
            rows.append({
                "path_id": str(path_id),
                "transition_id": f"{path_id}__gap_minimum00",
                "lambda_left": float(group.iloc[left_pos]["lambda"]),
                "lambda_right": float(group.iloc[right_pos]["lambda"]),
                "chern_left": np.nan,
                "chern_right": np.nan,
                "delta_chern_up": np.nan,
                "bracket_source": "dense_gap_minimum_fallback",
            })
    return pd.DataFrame(rows)


def direct_gap_at(kx: float, ky: float, reduced: dict[str, float]) -> float:
    params = core.raw8_from_reduced7(reduced, e0=0.0)
    eig = core.eigvals_full(wrap_k(kx), wrap_k(ky), params)
    return float(max(0.0, eig[4] - eig[3]))


def refine_one_transition(
    bracket: pd.Series,
    pair: pd.Series,
    path_gap: pd.DataFrame,
    config: Step04Config,
) -> dict[str, Any]:
    path_id = str(bracket["path_id"])
    a = np.array([float(pair[f"anchor_{name}"]) for name in REDUCED7])
    b = np.array([float(pair[f"trivial_{name}"]) for name in REDUCED7])
    lam_lo = float(min(bracket["lambda_left"], bracket["lambda_right"]))
    lam_hi = float(max(bracket["lambda_left"], bracket["lambda_right"]))
    if lam_hi - lam_lo < 1.0e-7:
        lam_lo = max(0.0, lam_lo - 1.0 / max(10, config.path_n_lambda - 1))
        lam_hi = min(1.0, lam_hi + 1.0 / max(10, config.path_n_lambda - 1))

    local = path_gap[
        (path_gap["path_id"].astype(str) == path_id)
        & (path_gap["lambda"].astype(float) >= lam_lo - 1.0e-12)
        & (path_gap["lambda"].astype(float) <= lam_hi + 1.0e-12)
    ].copy()
    if local.empty:
        local = path_gap[path_gap["path_id"].astype(str) == path_id].nsmallest(
            config.critical_optimizer_starts, "min_direct_gap"
        )
    else:
        local = local.nsmallest(config.critical_optimizer_starts, "min_direct_gap")

    starts: list[np.ndarray] = []
    for _, row in local.iterrows():
        starts.append(np.array([
            float(np.clip(row["lambda"], lam_lo, lam_hi)),
            float(row["direct_gap_kx"]), float(row["direct_gap_ky"]),
        ]))
    if not starts:
        starts = [np.array([(lam_lo + lam_hi) / 2.0, 0.0, 0.0])]

    def objective(x: np.ndarray) -> float:
        lam, kx, ky = float(x[0]), float(x[1]), float(x[2])
        reduced = vector_to_reduced(slerp(a, b, lam))
        return direct_gap_at(kx, ky, reduced)

    best = None
    for start in starts:
        result = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=[(lam_lo, lam_hi), (-math.pi, math.pi), (-math.pi, math.pi)],
            options={"maxiter": int(config.critical_optimizer_maxiter), "ftol": 1.0e-14},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None

    lam = float(best.x[0])
    kx, ky = wrap_k(float(best.x[1])), wrap_k(float(best.x[2]))
    reduced = vector_to_reduced(slerp(a, b, lam))
    c4 = c4_orbit(kx, ky)
    d4 = d4_orbit(kx, ky)
    c4_gaps = [direct_gap_at(x, y, reduced) for x, y in c4]
    d4_gaps = [direct_gap_at(x, y, reduced) for x, y in d4]

    return {
        **bracket.to_dict(),
        "anchor_sector": int(pair["anchor_sector"]),
        "partner_mode": str(pair["partner_mode"]),
        "critical_lambda": lam,
        "critical_direct_gap": float(best.fun),
        "critical_kx": kx,
        "critical_ky": ky,
        "optimizer_success": int(bool(best.success)),
        "optimizer_status": int(best.status),
        "optimizer_message": str(best.message),
        "optimizer_nfev": int(best.nfev),
        "c4_orbit_size": int(len(c4)),
        "d4_orbit_size": int(len(d4)),
        "c4_gap_min": float(min(c4_gaps)),
        "c4_gap_max": float(max(c4_gaps)),
        "c4_gap_spread": float(max(c4_gaps) - min(c4_gaps)),
        "d4_gap_min": float(min(d4_gaps)),
        "d4_gap_max": float(max(d4_gaps)),
        "d4_gap_spread": float(max(d4_gaps) - min(d4_gaps)),
        **classify_k_region(kx, ky, config.critical_k_symmetry_tol),
        **{name: reduced[name] for name in REDUCED7},
    }


def refine_critical_valleys(
    brackets: pd.DataFrame,
    partners: pd.DataFrame,
    path_gap: pd.DataFrame,
    config: Step04Config,
) -> pd.DataFrame:
    rows: list[dict] = []
    partner_map = {str(row["path_id"]): row for _, row in partners.iterrows()}
    started = time.time()
    for count, (_, bracket) in enumerate(brackets.iterrows(), start=1):
        pair = partner_map[str(bracket["path_id"])]
        rows.append(refine_one_transition(bracket, pair, path_gap, config))
        print(f"Critical valley: {count}/{len(brackets)} | elapsed {time.time()-started:.1f} s")
    return pd.DataFrame(rows)


# =============================================================================
# Mechanism summaries and figures
# =============================================================================

def mechanism_summary(critical: pd.DataFrame) -> pd.DataFrame:
    if critical.empty:
        return pd.DataFrame()
    work = critical.copy()
    work["abs_anchor_sector"] = work["anchor_sector"].abs().astype(int)
    return (
        work.groupby(["abs_anchor_sector", "critical_k_region", "critical_is_high_symmetry"], dropna=False)
        .agg(
            n_transitions=("transition_id", "size"),
            median_critical_gap=("critical_direct_gap", "median"),
            min_critical_gap=("critical_direct_gap", "min"),
            median_c4_orbit_size=("c4_orbit_size", "median"),
            median_d4_orbit_size=("d4_orbit_size", "median"),
            median_c4_gap_spread=("c4_gap_spread", "median"),
        )
        .reset_index()
    )


def plot_local_retention(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot_table(
        index="parent_anchor_id",
        columns="local_radius_max",
        values="same_sector_retention_fraction",
        aggfunc="mean",
    )
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("Same-sector TI retention fraction")
    ax.set_xlabel("Parent anchor")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("TTS Step 04: local Chern-sector stability")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_paths(path_gap: pd.DataFrame, critical: pd.DataFrame, figure_dir: Path) -> None:
    for path_id, group in path_gap.groupby("path_id", sort=False):
        group = group.sort_values("lambda")
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(group["lambda"], group["min_direct_gap"], label="direct gap")
        ax.plot(group["lambda"], group["indirect_gap"], label="indirect gap")
        ax.axhline(0.0, linewidth=1.0)
        subset = critical[critical["path_id"].astype(str) == str(path_id)]
        for _, row in subset.iterrows():
            ax.axvline(float(row["critical_lambda"]), linestyle="--", alpha=0.7)
        ax.set_xlabel("Spherical path coordinate λ")
        ax.set_ylabel("Gap")
        ax.set_title(str(path_id))
        ax.legend()
        plt.tight_layout()
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(path_id))
        plt.savefig(figure_dir / f"step04_path_{safe}.png", dpi=180)
        plt.close(fig)


def plot_critical_k(critical: pd.DataFrame, path: Path) -> None:
    if critical.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for sector, group in critical.groupby("anchor_sector"):
        ax.scatter(group["critical_kx"], group["critical_ky"], label=f"C_up={int(sector):+d}")
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xticks([-math.pi, 0.0, math.pi], ["-π", "0", "π"])
    ax.set_yticks([-math.pi, 0.0, math.pi], ["-π", "0", "π"])
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title("Refined gap-closing valleys")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================

def run_step04(config: Step04Config | None = None) -> Dict[str, Any]:
    config = (config or Step04Config()).normalized()
    started = time.time()

    print("[1/8] Load Step 03 strict physics labels")
    physics, physics_source = load_step3_physics(config)
    required = {"sample_id", "phase_label", "chern_up_int", "indirect_gap", *REDUCED7}
    missing = sorted(required - set(physics.columns))
    if missing:
        raise KeyError(f"Step 03 physics CSV 缺少列：{missing}")

    print("[2/8] Select sector anchors and nearest trivial partners")
    anchors = select_sector_anchors(physics, config)
    partners = select_trivial_partners(physics, anchors, config)
    atomic_write_csv(anchors, config.output_dir / "step04_01_selected_chern_sector_anchors.csv")
    atomic_write_csv(partners, config.output_dir / "step04_01_nearest_trivial_path_partners.csv")

    print("[3/8] Generate sector-directed local Sobol samples")
    local_params = generate_local_sobol(anchors, config)
    atomic_write_csv(local_params, config.output_dir / "step04_02_local_sobol_parameters.csv")

    print("[4/8] Strict local physics labels")
    local_physics, local_attempts, local_strict_gaps = run_local_physics(local_params, config)
    local_phase_counts, local_summary = summarize_local(local_physics)
    atomic_write_csv(local_phase_counts, config.output_dir / "step04_03_local_phase_counts.csv")
    atomic_write_csv(local_summary, config.output_dir / "step04_03_local_sector_stability_summary.csv")

    print("[5/8] Dense topology-to-trivial path gap scans")
    path_params = generate_path_parameters(partners, config)
    atomic_write_csv(path_params, config.output_dir / "step04_04_path_parameters.csv")
    path_gap = run_path_gap_scans(path_params, config)

    print("[6/8] Adaptive strict Chern audit on path")
    selected_chern = select_adaptive_chern_points(path_gap, config)
    atomic_write_csv(selected_chern, config.output_dir / "step04_04_path_selected_chern_points.csv")
    path_chern, path_attempts, path_strict_gaps = run_path_chern_audit(selected_chern, config)

    print("[7/8] Detect transition brackets and refine critical valleys")
    brackets = detect_transition_brackets(path_chern, path_gap, config)
    atomic_write_csv(brackets, config.output_dir / "step04_05_transition_brackets.csv")
    critical = refine_critical_valleys(brackets, partners, path_gap, config)
    atomic_write_csv(critical, config.output_dir / "step04_05_refined_critical_valleys.csv")
    mech = mechanism_summary(critical)
    atomic_write_csv(mech, config.output_dir / "step04_05_mechanism_summary.csv")

    print("[8/8] Figures and run summary")
    plot_local_retention(local_summary, config.output_dir / "figures" / "step04_local_sector_retention.png")
    plot_paths(path_gap, critical, config.output_dir / "figures")
    plot_critical_k(critical, config.output_dir / "figures" / "step04_refined_critical_valleys.png")

    sector_counts = {
        str(int(k)): int(v)
        for k, v in anchors["anchor_sector"].value_counts().sort_index().items()
    }
    local_same = int((
        local_physics["phase_label"].eq("spin_chern_TI_candidate")
        & local_physics["chern_up_int"].fillna(999).astype(int).eq(
            local_physics["parent_chern_up"].astype(int)
        )
    ).sum())
    summary = {
        "code_version": CODE_VERSION,
        "step01_version": core.CODE_VERSION,
        "step03_version": step3.CODE_VERSION,
        "step3_physics_source": physics_source,
        "n_step3_rows": int(len(physics)),
        "anchor_counts_by_sector": sector_counts,
        "n_anchors": int(len(anchors)),
        "n_path_pairs": int(len(partners)),
        "n_local_samples": int(len(local_physics)),
        "n_local_same_sector_TI": local_same,
        "local_same_sector_TI_fraction": float(local_same / len(local_physics)) if len(local_physics) else np.nan,
        "n_dense_path_points": int(len(path_gap)),
        "n_adaptive_chern_points": int(len(path_chern)),
        "n_transition_brackets": int(len(brackets)),
        "n_refined_critical_valleys": int(len(critical)),
        "critical_region_counts": {
            str(k): int(v) for k, v in critical["critical_k_region"].value_counts().items()
        } if not critical.empty else {},
        "elapsed_seconds": float(time.time() - started),
        "configuration": asdict(config),
    }
    atomic_write_json(summary, config.output_dir / "step04_06_run_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    return {
        "summary": summary,
        "physics_step3": physics,
        "anchors": anchors,
        "partners": partners,
        "local_parameters": local_params,
        "local_physics": local_physics,
        "local_chern_attempts": local_attempts,
        "local_strict_gap_checks": local_strict_gaps,
        "local_phase_counts": local_phase_counts,
        "local_stability_summary": local_summary,
        "path_parameters": path_params,
        "path_gap": path_gap,
        "path_chern": path_chern,
        "path_chern_attempts": path_attempts,
        "path_strict_gap_checks": path_strict_gaps,
        "transition_brackets": brackets,
        "critical_valleys": critical,
        "mechanism_summary": mech,
    }


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Step04Config.output_dir)
    parser.add_argument("--step3-input", type=Path, default=None)
    parser.add_argument("--anchors-per-sector", type=int, default=2)
    parser.add_argument("--local-power", type=int, default=4)
    parser.add_argument("--path-n-lambda", type=int, default=121)
    parser.add_argument("--path-gap-nk", type=int, default=35)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Step04Config:
    if args.smoke_test:
        return Step04Config(
            output_dir=Path(args.output_dir),
            step3_input=args.step3_input,
            target_sectors=(-2, -1, 1, 2),
            n_anchors_per_sector=1,
            path_anchors_per_sector=1,
            trivial_partner_modes=("nearest",),
            local_angular_radii=(0.06,),
            local_sobol_power_per_radius=1,
            local_gap_nk=17,
            initial_chern_grids=(15, 17),
            strict_gap_grids=(17, 21),
            strict_gap_shifts=((0.0, 0.0), (0.5, 0.5)),
            strict_chern_grids=(17, 21),
            path_n_lambda=21,
            path_gap_nk=15,
            path_chern_stride=7,
            path_extra_gap_minima=2,
            max_chern_points_per_path=7,
            critical_optimizer_starts=2,
            critical_optimizer_maxiter=80,
            checkpoint_every=2,
            force_recalculate_local=args.force,
            force_recalculate_paths=args.force,
            force_recalculate_chern=args.force,
        )
    return Step04Config(
        output_dir=Path(args.output_dir),
        step3_input=args.step3_input,
        n_anchors_per_sector=int(args.anchors_per_sector),
        local_sobol_power_per_radius=int(args.local_power),
        path_n_lambda=int(args.path_n_lambda),
        path_gap_nk=int(args.path_gap_nk),
        force_recalculate_local=args.force,
        force_recalculate_paths=args.force,
        force_recalculate_chern=args.force,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    run_step04(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
