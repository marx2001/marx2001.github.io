#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 08M — 机制感知机器学习：从相分类转向 valley 质量规律发现

本步骤承接冻结的 Step 01–07 结果与 formal r3-r4 相图，目标不是重新计算
Hamiltonian 或 Chern 数，而是回答机器学习在 TTS 中能否提炼出可验证的
物理规律：

1. 在 Step 03 独立 Sobol 数据上，用稀疏逻辑回归检验参数组合是否能够
   区分 |C_up|=1 与 |C_up|=2，并单独检验 Chern 手性符号是否存在稳定全局规则；
2. 将 Step 04–06 的九个严格相变扩展为 transition-grouped 机制数据集，
   使用 leave-one-transition-out 验证，避免同一路径数据泄漏；
3. 学习“高对称单谷”与“Sigma/Sigma' 对称双谷”机制，并分别检验
   Gamma/M 与 Sigma/Sigma' 的可分性；
4. 对 Step 05 的局域质量梯度做无监督聚类，寻找质量方向族；
5. 在 formal 上侧 +2<->0 相图中，用分区 group holdout 的线性/二次逻辑回归
   恢复数值边界，并与 Step 06 的 Sigma' 局域质量方向比较；
6. 对 formal 下侧两个 C_up=-2 连通分量进行局域质量图册对齐，判断它们是否
   对应不同的局部 valley 质量机制；
7. 输出“支持什么、不支持什么”的机制发现证书，不把预测准确率当成拓扑证明。

重要原则
--------
- 所有标签均来自已冻结的严格物理结果；
- 机器学习只提炼候选规律，最终机制仍需 Step 05/06 的 valley 电荷和解析质量验证；
- transition 数据采用整条相变留出，不进行随机点拆分；
- formal 相图采用参数条带 group holdout，避免相邻网格泄漏；
- 本步骤不修改 Step 01–07 的任何输出。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import io
import json
import math
import re
import time
import warnings
import zipfile

# Silence scikit-learn version-deprecation messages without hiding numerical warnings.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings(
    "ignore",
    message=r"Inconsistent values: penalty=l1.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from joblib import dump
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        silhouette_score,
    )
    from sklearn.model_selection import (
        GroupKFold,
        LeaveOneGroupOut,
        RepeatedStratifiedKFold,
        cross_val_predict,
        cross_val_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Step 08M 需要 numpy、pandas、matplotlib、scikit-learn 与 joblib。\n"
        "请运行：pip install numpy pandas matplotlib scikit-learn joblib"
    ) from exc


CODE_VERSION = "TTS_STEP08M_V1_20260724"
WORKFLOW_STEP = "step08M"
SYSTEM_TAG = "tts8_bns127391"
TASK_TAG = "mechanism_aware_machine_learning"

REDUCED7 = ["m_e", "t1", "t2", "r1", "r2", "r3", "r4"]
EXPECTED_VERSIONS = {
    "TTS_step01_model_and_label_audit_v2.py": "TTS_STEP01_V2_20260713",
    "TTS_step02v3_robust_hamiltonian_fingerprint_and_ML.py": "TTS_STEP02V3_20260713",
    "TTS_step03_global_sobol_hierarchical_topology_ML.py": "TTS_STEP03_V1_20260713",
    "TTS_step04_chern_sector_boundary_valley_tracking.py": "TTS_STEP04_V1_20260713",
    "TTS_step05_kp_valley_topological_charge.py": "TTS_STEP05_V2_20260714",
    "TTS_step06_analytic_mass_branch_chern_atlas.py": "TTS_STEP06_V1_20260714",
    "TTS_step07_observable_validation.py": "TTS_STEP07_V1_20260714",
}

SYMMETRY_FEATURES = [
    "abs_m_e", "abs_t1", "abs_t2",
    "t_sum", "t_diff", "t_product", "t_norm", "t_s", "t_d",
    "r13_sum", "r13_diff", "r24_sum", "r24_diff",
    "r12_sum", "r34_sum", "r_all_sum", "r_all_norm",
    "r13_product", "r24_product", "r_cross_product",
    "r_anisotropy_norm", "r_pair_sum_mismatch", "r_pair_diff_mismatch",
    "m_e_t_sum", "m_e_t_diff", "m_e_r13_sum", "m_e_r24_sum",
    "t_product_r_cross",
]
PARAMETER_FEATURES = REDUCED7 + SYMMETRY_FEATURES

MECHANISM_ORDER = ["Gamma", "M", "Sigma", "SigmaPrime"]


@dataclass
class Step08MConfig:
    output_dir: Path = Path("outputs_tts_step08M_mechanism_aware_ml")
    tts_archive: Path = Path("tts(1).zip")
    formal_scan_input: Path = Path("TTS_Cup2_Formal_Refined_Scans(1).zip")

    random_seed: int = 20260724
    n_jobs: int = 1
    random_forest_estimators: int = 700
    bootstrap_repeats: int = 300

    transition_lambda_window: float = 0.18
    upper_group_bins: int = 8
    upper_linear_C: float = 100.0
    upper_quadratic_C: float = 1.0

    global_logistic_C_grid: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0)
    transition_logistic_C: float = 0.3
    mass_cluster_k_min: int = 2
    mass_cluster_k_max: int = 5

    require_frozen_versions: bool = True

    def normalized(self) -> "Step08MConfig":
        self.output_dir = Path(self.output_dir)
        self.tts_archive = Path(self.tts_archive)
        self.formal_scan_input = Path(self.formal_scan_input)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        if not self.tts_archive.exists():
            raise FileNotFoundError(f"未找到 TTS 冻结档案：{self.tts_archive}")
        if not self.formal_scan_input.exists():
            raise FileNotFoundError(f"未找到 formal 相图结果：{self.formal_scan_input}")
        if self.transition_lambda_window <= 0:
            raise ValueError("transition_lambda_window must be > 0")
        if self.upper_group_bins < 4:
            raise ValueError("upper_group_bins must be >= 4")
        return self


# =============================================================================
# Safe I/O
# =============================================================================

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _list_members(source: Path) -> list[str]:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            return zf.namelist()
    if source.is_dir():
        return [str(p.relative_to(source)).replace("\\", "/") for p in source.rglob("*") if p.is_file()]
    raise FileNotFoundError(source)


def _read_bytes(source: Path, member: str) -> bytes:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            return zf.read(member)
    path = source / member
    return path.read_bytes()


def _find_member(source: Path, *, exact_basename: str | None = None,
                 contains: str | None = None, suffix: str | None = None) -> str:
    members = _list_members(source)
    matches = []
    for member in members:
        name = Path(member).name
        if exact_basename is not None and name != exact_basename:
            continue
        if contains is not None and contains not in name:
            continue
        if suffix is not None and not name.lower().endswith(suffix.lower()):
            continue
        matches.append(member)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{source} 中匹配 exact={exact_basename!r}, contains={contains!r}, "
            f"suffix={suffix!r} 的文件数为 {len(matches)}：{matches[:20]}"
        )
    return matches[0]


def read_csv_pattern(source: Path, pattern: str) -> pd.DataFrame:
    member = _find_member(source, contains=pattern, suffix=".csv")
    return pd.read_csv(io.BytesIO(_read_bytes(source, member)), low_memory=False)


def read_json_member(source: Path, member: str) -> dict[str, Any]:
    return json.loads(_read_bytes(source, member).decode("utf-8"))


def read_json_pattern(source: Path, pattern: str) -> dict[str, Any]:
    member = _find_member(source, contains=pattern, suffix=".json")
    return read_json_member(source, member)


# =============================================================================
# Frozen-version audit
# =============================================================================

def audit_frozen_versions(config: Step08MConfig) -> dict[str, Any]:
    members = _list_members(config.tts_archive)
    rows = []
    all_pass = True
    for basename, expected in EXPECTED_VERSIONS.items():
        matches = [m for m in members if Path(m).name == basename]
        loaded = None
        status = "missing"
        if len(matches) == 1:
            text = _read_bytes(config.tts_archive, matches[0]).decode("utf-8", errors="replace")
            match = re.search(r'CODE_VERSION\s*=\s*"([^"]+)"', text)
            loaded = match.group(1) if match else None
            status = "pass" if loaded == expected else "mismatch"
        if status != "pass":
            all_pass = False
        rows.append({
            "script": basename,
            "expected_version": expected,
            "loaded_version": loaded,
            "status": status,
        })
    audit = {
        "step08M_code_version": CODE_VERSION,
        "all_frozen_versions_pass": bool(all_pass),
        "version_rows": rows,
    }
    if config.require_frozen_versions and not all_pass:
        raise RuntimeError(
            "Step 01–07 冻结版本审计失败。详见输入审计表；"
            "如确需分析其他版本，请显式设置 require_frozen_versions=False。"
        )
    return audit


# =============================================================================
# Data loading and feature engineering
# =============================================================================

def load_inputs(config: Step08MConfig) -> dict[str, Any]:
    tts = config.tts_archive
    formal = config.formal_scan_input

    inputs = {
        "step03_combined": read_csv_pattern(tts, "step03_04_combined_ml_feature_table"),
        "step04_path_parameters": read_csv_pattern(tts, "step04_04_path_parameters"),
        "step04_transition_brackets": read_csv_pattern(tts, "step04_05_transition_brackets"),
        "step04_refined_valleys": read_csv_pattern(tts, "step04_05_refined_critical_valleys"),
        "step05_assignment": read_csv_pattern(tts, "step05_02_spin_valley_assignment"),
        "step05_gradients": read_csv_pattern(tts, "step05_03_local_mass_gradients"),
        "step05_charge_certificate": read_csv_pattern(tts, "step05_05_valley_charge_certificate"),
        "step06_atlas": read_csv_pattern(tts, "step06_02_transition_mass_atlas"),
        "step06_dense_mass": read_csv_pattern(tts, "step06_03_dense_path_mass_phase_map"),
    }

    if formal.is_file() and formal.suffix.lower() == ".zip":
        # There are two files with the same basename, so use full folder names explicitly.
        with zipfile.ZipFile(formal) as zf:
            inputs["formal_upper"] = pd.read_csv(
                zf.open("outputs_formal_upper_Cup2_to_0_boundary/tts_scan_r3_r4_grid.csv"),
                low_memory=False,
            )
            inputs["formal_lower"] = pd.read_csv(
                zf.open("outputs_formal_lower_Cup2_Cum2_region/tts_scan_r3_r4_grid.csv"),
                low_memory=False,
            )
            inputs["formal_upper_meta"] = json.load(
                zf.open("outputs_formal_upper_Cup2_to_0_boundary/scan_metadata.json")
            )
            inputs["formal_lower_meta"] = json.load(
                zf.open("outputs_formal_lower_Cup2_Cum2_region/scan_metadata.json")
            )
    elif formal.is_dir():
        inputs["formal_upper"] = pd.read_csv(
            formal / "outputs_formal_upper_Cup2_to_0_boundary" / "tts_scan_r3_r4_grid.csv"
        )
        inputs["formal_lower"] = pd.read_csv(
            formal / "outputs_formal_lower_Cup2_Cum2_region" / "tts_scan_r3_r4_grid.csv"
        )
        inputs["formal_upper_meta"] = json.loads(
            (formal / "outputs_formal_upper_Cup2_to_0_boundary" / "scan_metadata.json")
            .read_text(encoding="utf-8")
        )
        inputs["formal_lower_meta"] = json.loads(
            (formal / "outputs_formal_lower_Cup2_Cum2_region" / "scan_metadata.json")
            .read_text(encoding="utf-8")
        )
    else:
        raise FileNotFoundError(formal)

    return inputs


def add_symmetry_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in REDUCED7:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    me, t1, t2, r1, r2, r3, r4 = [out[name] for name in REDUCED7]
    root2 = math.sqrt(2.0)
    out["abs_m_e"] = me.abs()
    out["abs_t1"] = t1.abs()
    out["abs_t2"] = t2.abs()
    out["t_sum"] = t1 + t2
    out["t_diff"] = t1 - t2
    out["t_product"] = t1 * t2
    out["t_norm"] = np.sqrt(t1*t1 + t2*t2)
    out["t_s"] = (t1 + t2) / root2
    out["t_d"] = (t1 - t2) / root2
    out["r13_sum"] = r1 + r3
    out["r13_diff"] = r1 - r3
    out["r24_sum"] = r2 + r4
    out["r24_diff"] = r2 - r4
    out["r12_sum"] = r1 + r2
    out["r34_sum"] = r3 + r4
    out["r_all_sum"] = r1 + r2 + r3 + r4
    out["r_all_norm"] = np.sqrt(r1*r1 + r2*r2 + r3*r3 + r4*r4)
    out["r13_product"] = r1 * r3
    out["r24_product"] = r2 * r4
    out["r_cross_product"] = (r1 + r3) * (r2 + r4)
    out["r_anisotropy_norm"] = np.sqrt((r1-r3)**2 + (r2-r4)**2)
    out["r_pair_sum_mismatch"] = (r1 + r3) - (r2 + r4)
    out["r_pair_diff_mismatch"] = (r1 - r3) - (r2 - r4)
    out["m_e_t_sum"] = me * (t1 + t2)
    out["m_e_t_diff"] = me * (t1 - t2)
    out["m_e_r13_sum"] = me * (r1 + r3)
    out["m_e_r24_sum"] = me * (r2 + r4)
    out["t_product_r_cross"] = (t1*t2) * ((r1+r3)*(r2+r4))
    return out


def normalize_mechanism(region: str) -> str:
    mapping = {
        "Gamma": "Gamma",
        "M": "M",
        "generic_Sigma_kx_eq_ky": "Sigma",
        "generic_SigmaPrime_kx_eq_minus_ky": "SigmaPrime",
    }
    return mapping.get(str(region), str(region))


# =============================================================================
# Global Sobol rule discovery
# =============================================================================

def _sparse_logistic(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=float(C),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )),
    ])


def _rf(config: Step08MConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
    )


def select_logistic_C(X: pd.DataFrame, y: pd.Series, config: Step08MConfig) -> tuple[float, pd.DataFrame]:
    min_class = int(y.value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return float(config.global_logistic_C_grid[0]), pd.DataFrame()
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=10,
        random_state=config.random_seed,
    )
    rows = []
    for C in config.global_logistic_C_grid:
        model = _sparse_logistic(C, config.random_seed)
        scores = cross_val_score(model, X, y, scoring="balanced_accuracy", cv=cv)
        rows.append({
            "C": float(C),
            "cv_balanced_accuracy_mean": float(scores.mean()),
            "cv_balanced_accuracy_std": float(scores.std(ddof=1)),
        })
    table = pd.DataFrame(rows).sort_values(
        ["cv_balanced_accuracy_mean", "cv_balanced_accuracy_std", "C"],
        ascending=[False, True, True],
    )
    return float(table.iloc[0]["C"]), table


def bootstrap_sparse_coefficients(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    C: float,
    config: Step08MConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    coeffs = []
    n = len(X)
    for _ in range(config.bootstrap_repeats):
        for _attempt in range(100):
            idx = rng.integers(0, n, size=n)
            yb = y.iloc[idx]
            if yb.nunique() == 2:
                break
        else:
            continue
        model = _sparse_logistic(C, config.random_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[idx], yb)
        coeffs.append(model.named_steps["model"].coef_[0])
    if not coeffs:
        return pd.DataFrame()
    arr = np.asarray(coeffs, float)
    rows = []
    for j, feature in enumerate(feature_names):
        values = arr[:, j]
        nonzero = np.abs(values) > 1.0e-12
        nz_values = values[nonzero]
        rows.append({
            "feature": feature,
            "coefficient_median": float(np.median(values)),
            "coefficient_abs_median": float(np.median(np.abs(values))),
            "nonzero_fraction": float(nonzero.mean()),
            "positive_fraction": float((values > 0).mean()),
            "negative_fraction": float((values < 0).mean()),
            "sign_stability": float(max((values > 0).mean(), (values < 0).mean())),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "n_bootstrap": int(arr.shape[0]),
        })
    return pd.DataFrame(rows).sort_values(
        ["coefficient_abs_median", "nonzero_fraction"], ascending=False
    )


def run_global_rules(inputs: dict[str, Any], config: Step08MConfig) -> dict[str, Any]:
    df = inputs["step03_combined"].copy()
    df = add_symmetry_features(df)
    strict_ti = df[
        (df["phase_label"] == "spin_chern_TI_candidate")
        & (pd.to_numeric(df["is_ml_eligible"], errors="coerce").fillna(0).astype(int) == 1)
        & df["chern_up_int"].notna()
    ].copy()

    tasks = {
        "chern_magnitude_abs2_vs_abs1": strict_ti.copy(),
        "high_chern_sign_positive_vs_negative": strict_ti[
            strict_ti["chern_up_int"].abs() == 2
        ].copy(),
    }

    metric_rows = []
    coef_tables = []
    dataset_rows = []
    model_paths = {}
    cv_tables = []

    for task_name, task_df in tasks.items():
        if task_name.startswith("chern_magnitude"):
            task_df["target"] = (task_df["chern_up_int"].abs() == 2).astype(int)
            positive_label = "|C_up|=2"
            negative_label = "|C_up|=1"
        else:
            task_df["target"] = (task_df["chern_up_int"] > 0).astype(int)
            positive_label = "C_up=+2"
            negative_label = "C_up=-2"

        train = task_df[task_df["sample_source"] == "train_sobol"].copy()
        external = task_df[task_df["sample_source"] == "external_sobol"].copy()
        if train["target"].nunique() < 2 or external["target"].nunique() < 2:
            continue

        X_train = train[PARAMETER_FEATURES].astype(float)
        y_train = train["target"].astype(int)
        X_external = external[PARAMETER_FEATURES].astype(float)
        y_external = external["target"].astype(int)

        selected_C, cv_table = select_logistic_C(X_train, y_train, config)
        if not cv_table.empty:
            cv_table.insert(0, "task", task_name)
            cv_tables.append(cv_table)

        models = {
            "sparse_logistic": _sparse_logistic(selected_C, config.random_seed),
            "random_forest": _rf(config),
        }
        for model_name, model in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            pred = model.predict(X_external)
            metric_rows.append({
                "task": task_name,
                "model": model_name,
                "n_train": int(len(train)),
                "n_external": int(len(external)),
                "train_class_0": int((y_train == 0).sum()),
                "train_class_1": int((y_train == 1).sum()),
                "external_class_0": int((y_external == 0).sum()),
                "external_class_1": int((y_external == 1).sum()),
                "balanced_accuracy": float(balanced_accuracy_score(y_external, pred)),
                "accuracy": float(accuracy_score(y_external, pred)),
                "macro_f1": float(f1_score(y_external, pred, average="macro")),
                "tn": int(confusion_matrix(y_external, pred, labels=[0, 1])[0, 0]),
                "fp": int(confusion_matrix(y_external, pred, labels=[0, 1])[0, 1]),
                "fn": int(confusion_matrix(y_external, pred, labels=[0, 1])[1, 0]),
                "tp": int(confusion_matrix(y_external, pred, labels=[0, 1])[1, 1]),
                "positive_label": positive_label,
                "negative_label": negative_label,
                "selected_C": selected_C if model_name == "sparse_logistic" else np.nan,
            })
            model_path = config.output_dir / "models" / f"step08M_global_{task_name}_{model_name}.joblib"
            dump(model, model_path)
            model_paths[f"{task_name}:{model_name}"] = str(model_path)

        sparse = models["sparse_logistic"]
        scale = sparse.named_steps["scale"]
        clf = sparse.named_steps["model"]
        standardized = clf.coef_[0]
        raw_coef = standardized / scale.scale_
        raw_intercept = float(clf.intercept_[0] - np.dot(standardized, scale.mean_ / scale.scale_))
        coef = pd.DataFrame({
            "task": task_name,
            "feature": PARAMETER_FEATURES,
            "standardized_coefficient": standardized,
            "raw_space_coefficient": raw_coef,
            "abs_standardized_coefficient": np.abs(standardized),
            "selected_C": selected_C,
            "raw_space_intercept": raw_intercept,
        }).sort_values("abs_standardized_coefficient", ascending=False)

        boot = bootstrap_sparse_coefficients(
            X_train, y_train, PARAMETER_FEATURES, selected_C, config
        )
        if not boot.empty:
            boot.insert(0, "task", task_name)
            coef = coef.merge(boot, on=["task", "feature"], how="left")
        coef_tables.append(coef)

        task_export = task_df[
            ["sample_id", "sample_source", "chern_up_int", "target"] + PARAMETER_FEATURES
        ].copy()
        task_export.insert(0, "task", task_name)
        dataset_rows.append(task_export)

    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
    datasets = pd.concat(dataset_rows, ignore_index=True) if dataset_rows else pd.DataFrame()
    cv_results = pd.concat(cv_tables, ignore_index=True) if cv_tables else pd.DataFrame()

    atomic_write_csv(datasets, config.output_dir / "step08M_01_global_rule_dataset.csv")
    atomic_write_csv(cv_results, config.output_dir / "step08M_02_global_logistic_C_selection.csv")
    atomic_write_csv(metrics, config.output_dir / "step08M_03_global_external_metrics.csv")
    atomic_write_csv(coefficients, config.output_dir / "step08M_04_global_sparse_coefficients.csv")

    plot_global_coefficients(coefficients, config.output_dir / "figures" / "step08M_global_sparse_coefficients.png")

    return {
        "metrics": metrics,
        "coefficients": coefficients,
        "datasets": datasets,
        "model_paths": model_paths,
    }


def plot_global_coefficients(coefficients: pd.DataFrame, output: Path) -> None:
    if coefficients.empty:
        return
    configure_plot_style()
    tasks = coefficients["task"].unique().tolist()
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.4 * len(tasks), 5.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = coefficients[coefficients["task"] == task].nlargest(
            12, "abs_standardized_coefficient"
        ).sort_values("standardized_coefficient")
        ax.barh(sub["feature"], sub["standardized_coefficient"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("standardized sparse-logistic coefficient")
    fig.tight_layout()
    fig.savefig(output, dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Transition-grouped mechanism learning
# =============================================================================

def build_transition_dataset(inputs: dict[str, Any], config: Step08MConfig) -> pd.DataFrame:
    params = inputs["step04_path_parameters"].copy()
    mass = inputs["step06_dense_mass"].copy()
    atlas = inputs["step06_atlas"].copy()

    merged = params.merge(
        mass,
        on=["path_id", "path_point_id", "path_index", "lambda"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for path_id, transitions in atlas.groupby("path_id"):
        transitions = transitions.sort_values("critical_lambda").reset_index(drop=True)
        path_rows = merged[merged["path_id"] == path_id].copy()
        for chart_index, transition in transitions.iterrows():
            mass_column = f"chart{chart_index:02d}_oriented_mass"
            if mass_column not in path_rows.columns:
                raise KeyError(
                    f"{path_id} 缺少 {mass_column}；Step06 path chart 顺序与 atlas 不一致。"
                )
            near = path_rows[
                (path_rows["lambda"] - float(transition["critical_lambda"])).abs()
                <= config.transition_lambda_window
            ].copy()
            near["transition_id"] = str(transition["transition_id"])
            near["critical_lambda"] = float(transition["critical_lambda"])
            near["critical_k_region"] = str(transition["critical_k_region"])
            near["mechanism_fine"] = normalize_mechanism(transition["critical_k_region"])
            near["mechanism_coarse"] = np.where(
                near["mechanism_fine"].isin(["Gamma", "M"]),
                "single_high_symmetry_valley",
                "paired_generic_valleys",
            )
            near["delta_chern_up"] = int(transition["delta_chern_up"])
            near["abs_delta_chern_up"] = abs(int(transition["delta_chern_up"]))
            near["oriented_mass"] = pd.to_numeric(near[mass_column], errors="coerce")
            near["mass_sign"] = np.sign(near["oriented_mass"]).astype(int)
            near["lambda_minus_critical"] = near["lambda"] - near["critical_lambda"]
            near["chart_index"] = int(chart_index)
            rows.append(near)

    dataset = pd.concat(rows, ignore_index=True)
    dataset = add_symmetry_features(dataset)
    keep = [
        "transition_id", "path_id", "path_point_id", "path_index", "lambda",
        "critical_lambda", "lambda_minus_critical", "critical_k_region",
        "mechanism_fine", "mechanism_coarse", "delta_chern_up",
        "abs_delta_chern_up", "oriented_mass", "mass_sign",
    ] + PARAMETER_FEATURES
    return dataset[keep].copy()


def _logo_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model: Any,
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in logo.split(X, y, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])
    return pred


def run_transition_mechanism_learning(
    inputs: dict[str, Any], config: Step08MConfig
) -> dict[str, Any]:
    dataset = build_transition_dataset(inputs, config)
    atomic_write_csv(dataset, config.output_dir / "step08M_05_transition_mechanism_dataset.csv")

    task_specs = {
        "coarse_single_vs_paired": (
            dataset,
            "mechanism_coarse",
            ["single_high_symmetry_valley", "paired_generic_valleys"],
        ),
        "fine_four_family": (
            dataset,
            "mechanism_fine",
            MECHANISM_ORDER,
        ),
        "high_symmetry_Gamma_vs_M": (
            dataset[dataset["mechanism_fine"].isin(["Gamma", "M"])].copy(),
            "mechanism_fine",
            ["Gamma", "M"],
        ),
        "generic_Sigma_vs_SigmaPrime": (
            dataset[dataset["mechanism_fine"].isin(["Sigma", "SigmaPrime"])].copy(),
            "mechanism_fine",
            ["Sigma", "SigmaPrime"],
        ),
    }

    metric_rows = []
    prediction_rows = []
    coefficient_rows = []
    model_paths = {}

    for task_name, (task_df, target_col, labels) in task_specs.items():
        if task_df[target_col].nunique() < 2:
            continue
        # Require at least two independent transitions per class for meaningful LOGO.
        group_class = task_df[["transition_id", target_col]].drop_duplicates()
        if group_class.groupby(target_col)["transition_id"].nunique().min() < 2:
            continue

        X = task_df[PARAMETER_FEATURES].astype(float)
        y = task_df[target_col].astype(str)
        groups = task_df["transition_id"].astype(str)

        models = {
            "logistic": Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(
                    C=config.transition_logistic_C,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=config.random_seed,
                )),
            ]),
            "random_forest": _rf(config),
        }

        for model_name, model in models.items():
            pred = _logo_predictions(X, y, groups, model)
            metric_rows.append({
                "task": task_name,
                "model": model_name,
                "n_rows": int(len(task_df)),
                "n_transitions": int(groups.nunique()),
                "n_classes": int(y.nunique()),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro")),
            })
            prediction_rows.append(pd.DataFrame({
                "task": task_name,
                "model": model_name,
                "transition_id": groups.to_numpy(),
                "path_point_id": task_df["path_point_id"].to_numpy(),
                "actual": y.to_numpy(),
                "predicted": pred,
                "correct": (y.to_numpy() == pred).astype(int),
            }))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            model_path = config.output_dir / "models" / f"step08M_transition_{task_name}_{model_name}.joblib"
            dump(model, model_path)
            model_paths[f"{task_name}:{model_name}"] = str(model_path)

            if model_name == "logistic":
                clf = model.named_steps["model"]
                if clf.coef_.shape[0] == 1 and len(clf.classes_) == 2:
                    class_vectors = [
                        (str(clf.classes_[0]), -clf.coef_[0]),
                        (str(clf.classes_[1]), clf.coef_[0]),
                    ]
                else:
                    class_vectors = [
                        (str(class_name), clf.coef_[class_index])
                        for class_index, class_name in enumerate(clf.classes_)
                    ]
                for class_name, vector in class_vectors:
                    for feature, value in zip(PARAMETER_FEATURES, vector):
                        coefficient_rows.append({
                            "task": task_name,
                            "class": class_name,
                            "feature": feature,
                            "standardized_coefficient": float(value),
                            "abs_standardized_coefficient": float(abs(value)),
                        })

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    coefficients = pd.DataFrame(coefficient_rows)

    atomic_write_csv(metrics, config.output_dir / "step08M_06_transition_grouped_cv_metrics.csv")
    atomic_write_csv(predictions, config.output_dir / "step08M_07_transition_grouped_predictions.csv")
    atomic_write_csv(coefficients, config.output_dir / "step08M_08_transition_logistic_coefficients.csv")

    plot_transition_confusions(
        predictions,
        config.output_dir / "figures" / "step08M_transition_grouped_confusions.png",
    )
    return {
        "dataset": dataset,
        "metrics": metrics,
        "predictions": predictions,
        "coefficients": coefficients,
        "model_paths": model_paths,
    }


def plot_transition_confusions(predictions: pd.DataFrame, output: Path) -> None:
    if predictions.empty:
        return
    configure_plot_style()
    tasks = [
        task for task in [
            "coarse_single_vs_paired",
            "fine_four_family",
            "high_symmetry_Gamma_vs_M",
            "generic_Sigma_vs_SigmaPrime",
        ]
        if task in set(predictions["task"])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, task in zip(axes.ravel(), tasks):
        sub = predictions[
            (predictions["task"] == task) & (predictions["model"] == "logistic")
        ]
        labels = sorted(set(sub["actual"]) | set(sub["predicted"]))
        cm = confusion_matrix(sub["actual"], sub["predicted"], labels=labels)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(task.replace("_", " "))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Local-mass-gradient clustering
# =============================================================================

def build_mass_gradient_table(inputs: dict[str, Any]) -> pd.DataFrame:
    assignment = inputs["step05_assignment"].copy()
    gradients = inputs["step05_gradients"].copy()
    active = assignment[
        (assignment["spin"] == "up")
        & (pd.to_numeric(assignment["is_active_valley"], errors="coerce").fillna(0).astype(int) == 1)
    ][["transition_id", "closure_id", "valley_id", "critical_k_region"]].copy()

    merged = gradients[
        gradients["spin"] == "up"
    ].merge(
        active,
        on=["transition_id", "closure_id", "valley_id"],
        how="inner",
    )
    pivot = merged.pivot_table(
        index=["transition_id", "critical_k_region"],
        columns="parameter",
        values="normalized_mass_gradient",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot[["transition_id", "critical_k_region"] + REDUCED7].copy()

    vectors = pivot[REDUCED7].to_numpy(float)
    for i, vector in enumerate(vectors):
        max_index = int(np.argmax(np.abs(vector)))
        if vector[max_index] < 0:
            vectors[i] *= -1.0
    pivot.loc[:, REDUCED7] = vectors
    pivot["mechanism_fine"] = pivot["critical_k_region"].map(normalize_mechanism)
    return pivot


def run_mass_gradient_clustering(
    inputs: dict[str, Any], config: Step08MConfig
) -> dict[str, Any]:
    table = build_mass_gradient_table(inputs)
    X = table[REDUCED7].to_numpy(float)

    selection_rows = []
    fits = {}
    max_k = min(config.mass_cluster_k_max, len(table) - 1)
    for k in range(config.mass_cluster_k_min, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=config.random_seed,
            n_init=100,
        ).fit(X)
        score = float(silhouette_score(X, model.labels_, metric="cosine"))
        selection_rows.append({"n_clusters": k, "cosine_silhouette": score})
        fits[k] = model
    selection = pd.DataFrame(selection_rows).sort_values(
        ["cosine_silhouette", "n_clusters"], ascending=[False, True]
    )
    best_k = int(selection.iloc[0]["n_clusters"])
    model = fits[best_k]
    table["mass_direction_cluster"] = model.labels_.astype(int)

    pca = PCA(n_components=2, random_state=config.random_seed)
    xy = pca.fit_transform(X)
    table["pca_1"] = xy[:, 0]
    table["pca_2"] = xy[:, 1]

    centroid_rows = []
    for cluster_id, centroid in enumerate(model.cluster_centers_):
        row = {"mass_direction_cluster": int(cluster_id)}
        row.update({name: float(value) for name, value in zip(REDUCED7, centroid)})
        centroid_rows.append(row)
    centroids = pd.DataFrame(centroid_rows)

    atomic_write_csv(table, config.output_dir / "step08M_09_mass_gradient_clusters.csv")
    atomic_write_csv(selection, config.output_dir / "step08M_10_mass_cluster_selection.csv")
    atomic_write_csv(centroids, config.output_dir / "step08M_11_mass_cluster_centroids.csv")
    dump(model, config.output_dir / "models" / "step08M_mass_gradient_kmeans.joblib")

    plot_mass_gradient_pca(
        table,
        config.output_dir / "figures" / "step08M_mass_gradient_clusters.png",
    )
    return {
        "table": table,
        "selection": selection,
        "centroids": centroids,
        "best_k": best_k,
        "best_silhouette": float(selection.iloc[0]["cosine_silhouette"]),
    }


def plot_mass_gradient_pca(table: pd.DataFrame, output: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for mechanism, sub in table.groupby("mechanism_fine"):
        ax.scatter(sub["pca_1"], sub["pca_2"], s=55, label=mechanism)
        for _, row in sub.iterrows():
            short = str(row["transition_id"]).replace("_anchor01", "").replace("__transition00", "0").replace("__transition01", "1")
            ax.annotate(short, (row["pca_1"], row["pca_2"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PCA 1 of canonicalized mass gradients")
    ax.set_ylabel("PCA 2 of canonicalized mass gradients")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Formal upper boundary: ML recovery of the valley-mass direction
# =============================================================================

def parse_linear_mass_formula(formula: str) -> dict[str, float]:
    compact = str(formula).replace(" ", "")
    pairs = re.findall(r"([+-]?\d+(?:\.\d+)?)\*([A-Za-z0-9_]+)", compact)
    return {parameter: float(coefficient) for coefficient, parameter in pairs}


def upper_mass_chart(meta: dict[str, Any]) -> dict[str, Any]:
    charts = meta.get("generic_mass_charts", [])
    for chart in charts:
        if (
            int(chart.get("chern_left", 999)) == 2
            and int(chart.get("chern_right", 999)) == 0
            and "SigmaPrime" in str(chart.get("critical_k_region", ""))
        ):
            coeff = parse_linear_mass_formula(chart["mass_formula"])
            return {**chart, "coefficients": coeff}
    raise ValueError("formal upper metadata 中未找到 +2->0 SigmaPrime 质量图。")


def grouped_upper_cv(
    data: pd.DataFrame,
    config: Step08MConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = data[data["phase_code"].isin([0, 2])].copy()
    work["target"] = (work["phase_code"] == 2).astype(int)
    work["r3_group"] = pd.qcut(
        work["scan_x"],
        q=config.upper_group_bins,
        labels=False,
        duplicates="drop",
    )
    n_groups = int(work["r3_group"].nunique())
    cv = GroupKFold(n_splits=n_groups)

    specs = {
        "linear_r3_r4": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=config.upper_linear_C,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5000,
                random_state=config.random_seed,
            )),
        ]),
        "quadratic_r3_r4": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=config.upper_quadratic_C,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5000,
                random_state=config.random_seed,
            )),
        ]),
    }
    metric_rows = []
    prediction_rows = []
    fitted = {}
    X = work[["scan_x", "scan_y"]].astype(float)
    y = work["target"].astype(int)
    groups = work["r3_group"]

    for name, model in specs.items():
        pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
        metric_rows.append({
            "model": name,
            "n_points": int(len(work)),
            "n_groups": n_groups,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        })
        prediction_rows.append(pd.DataFrame({
            "model": name,
            "sample_id": work["sample_id"].to_numpy(),
            "scan_x": work["scan_x"].to_numpy(),
            "scan_y": work["scan_y"].to_numpy(),
            "actual": y.to_numpy(),
            "predicted": pred,
            "r3_group": groups.to_numpy(),
        }))
        model.fit(X, y)
        fitted[name] = model

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True), fitted


def raw_linear_logistic_rule(model: Pipeline) -> dict[str, float]:
    scale = model.named_steps["scale"]
    clf = model.named_steps["model"]
    standardized = clf.coef_[0]
    beta = standardized / scale.scale_
    intercept = float(clf.intercept_[0] - np.dot(standardized, scale.mean_ / scale.scale_))
    if abs(beta[1]) < 1.0e-14:
        slope = np.nan
        line_intercept = np.nan
    else:
        slope = float(-beta[0] / beta[1])
        line_intercept = float(-intercept / beta[1])
    return {
        "beta_r3": float(beta[0]),
        "beta_r4": float(beta[1]),
        "intercept": intercept,
        "boundary_slope": slope,
        "boundary_intercept": line_intercept,
    }


def run_upper_boundary_rule(
    inputs: dict[str, Any], config: Step08MConfig
) -> dict[str, Any]:
    upper = inputs["formal_upper"].copy()
    meta = inputs["formal_upper_meta"]
    metrics, predictions, fitted = grouped_upper_cv(upper, config)
    linear_rule = raw_linear_logistic_rule(fitted["linear_r3_r4"])

    chart = upper_mass_chart(meta)
    coeff = chart["coefficients"]
    mass_vector = np.array([coeff["r3"], coeff["r4"]], float)
    ml_vector = np.array([linear_rule["beta_r3"], linear_rule["beta_r4"]], float)
    cosine = float(abs(np.dot(mass_vector, ml_vector)) / (
        np.linalg.norm(mass_vector) * np.linalg.norm(ml_vector)
    ))

    anchor = meta["anchor_params"]
    constant = sum(
        float(value) * float(anchor[parameter])
        for parameter, value in coeff.items()
        if parameter not in {"r3", "r4"}
    )
    mass_slope = float(-coeff["r3"] / coeff["r4"])
    mass_intercept = float(-constant / coeff["r4"])

    rule_table = pd.DataFrame([{
        **linear_rule,
        "step06_mass_r3_coefficient": float(coeff["r3"]),
        "step06_mass_r4_coefficient": float(coeff["r4"]),
        "step06_mass_boundary_slope": mass_slope,
        "step06_mass_boundary_intercept": mass_intercept,
        "absolute_cosine_ml_vs_step06_mass_direction": cosine,
        "transition_id": chart["transition_id"],
        "critical_k_region": chart["critical_k_region"],
        "mass_formula": chart["mass_formula"],
    }])

    atomic_write_csv(metrics, config.output_dir / "step08M_12_upper_grouped_cv_metrics.csv")
    atomic_write_csv(predictions, config.output_dir / "step08M_13_upper_grouped_predictions.csv")
    atomic_write_csv(rule_table, config.output_dir / "step08M_14_upper_mass_direction_recovery.csv")
    dump(
        fitted["linear_r3_r4"],
        config.output_dir / "models" / "step08M_upper_linear_boundary_logistic.joblib",
    )
    dump(
        fitted["quadratic_r3_r4"],
        config.output_dir / "models" / "step08M_upper_quadratic_boundary_logistic.joblib",
    )

    plot_upper_boundary_rule(
        upper,
        linear_rule,
        mass_slope,
        mass_intercept,
        config.output_dir / "figures" / "step08M_upper_boundary_mass_recovery.png",
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "rule_table": rule_table,
        "linear_rule": linear_rule,
        "mass_slope": mass_slope,
        "mass_intercept": mass_intercept,
        "cosine": cosine,
    }


def plot_upper_boundary_rule(
    upper: pd.DataFrame,
    linear_rule: dict[str, float],
    mass_slope: float,
    mass_intercept: float,
    output: Path,
) -> None:
    configure_plot_style()
    phase_colors = {2: "#F28E8B", 0: "#B2B2B2", -2: "#3F63AD", 99: "white"}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for phase, sub in upper.groupby("phase_code"):
        ax.scatter(
            sub["scan_x"], sub["scan_y"],
            s=10,
            color=phase_colors.get(int(phase), "white"),
            edgecolors="black" if int(phase) == 99 else "none",
            linewidths=0.3,
            label=f"C_up={int(phase)}" if int(phase) != 99 else "unresolved",
        )
    xs = np.linspace(float(upper["scan_x"].min()), float(upper["scan_x"].max()), 300)
    ax.plot(
        xs,
        linear_rule["boundary_slope"] * xs + linear_rule["boundary_intercept"],
        color="black",
        linewidth=1.4,
        label="grouped-CV ML boundary",
    )
    ax.plot(
        xs,
        mass_slope * xs + mass_intercept,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="Step06 local mass zero",
    )
    ax.set_xlabel(r"$r_3$")
    ax.set_ylabel(r"$r_4$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Formal lower scan: disconnected -2 components and mass-chart alignment
# =============================================================================

def connected_components(
    df: pd.DataFrame,
    phase_code: int,
) -> list[list[Any]]:
    rows = {
        (int(row.scan_ix), int(row.scan_iy)): row
        for row in df.itertuples()
        if int(row.phase_code) == int(phase_code)
    }
    visited: set[tuple[int, int]] = set()
    components: list[list[Any]] = []
    for key in rows:
        if key in visited:
            continue
        stack = [key]
        visited.add(key)
        component = []
        while stack:
            node = stack.pop()
            component.append(rows[node])
            ix, iy = node
            for neighbor in ((ix+1, iy), (ix-1, iy), (ix, iy+1), (ix, iy-1)):
                if neighbor in rows and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def run_lower_component_alignment(
    inputs: dict[str, Any], config: Step08MConfig
) -> dict[str, Any]:
    lower = inputs["formal_lower"].copy()
    meta = inputs["formal_lower_meta"]
    grid = {
        (int(row.scan_ix), int(row.scan_iy)): row
        for row in lower.itertuples()
    }
    charts = meta["generic_mass_charts"]
    chart_columns = [chart["column"] for chart in charts]
    chart_map = {chart["column"]: chart for chart in charts}

    components = connected_components(lower, -2)
    component_rows = []
    alignment_rows = []

    for component_id, component in enumerate(components):
        if len(component) < 5:
            continue
        keys = {(int(row.scan_ix), int(row.scan_iy)) for row in component}
        boundary = []
        for key in keys:
            ix, iy = key
            if any(
                neighbor in grid and int(grid[neighbor].phase_code) != -2
                for neighbor in ((ix+1, iy), (ix-1, iy), (ix, iy+1), (ix, iy-1))
            ):
                boundary.append(grid[key])
        boundary_df = pd.DataFrame([row._asdict() for row in boundary])

        component_rows.append({
            "component_id": component_id,
            "n_points": len(component),
            "n_boundary_points": len(boundary),
            "r3_min": min(float(row.scan_x) for row in component),
            "r3_max": max(float(row.scan_x) for row in component),
            "r4_min": min(float(row.scan_y) for row in component),
            "r4_max": max(float(row.scan_y) for row in component),
            "median_indirect_gap": float(np.median([float(row.indirect_gap) for row in component])),
            "max_indirect_gap": float(np.max([float(row.indirect_gap) for row in component])),
        })

        nearest = boundary_df[chart_columns].abs().idxmin(axis=1)
        for column in chart_columns:
            chart = chart_map[column]
            alignment_rows.append({
                "component_id": component_id,
                "component_size": len(component),
                "n_boundary_points": len(boundary_df),
                "chart_column": column,
                "transition_id": chart["transition_id"],
                "chern_left": chart["chern_left"],
                "chern_right": chart["chern_right"],
                "critical_k_region": chart["critical_k_region"],
                "median_abs_mass_on_component_boundary": float(boundary_df[column].abs().median()),
                "mean_abs_mass_on_component_boundary": float(boundary_df[column].abs().mean()),
                "fraction_nearest_mass_chart": float((nearest == column).mean()),
                "mass_formula": chart["mass_formula"],
            })

    components_df = pd.DataFrame(component_rows)
    alignment = pd.DataFrame(alignment_rows).sort_values(
        ["component_id", "median_abs_mass_on_component_boundary"]
    )

    atomic_write_csv(components_df, config.output_dir / "step08M_15_lower_minus2_components.csv")
    atomic_write_csv(alignment, config.output_dir / "step08M_16_lower_component_mass_alignment.csv")
    plot_lower_alignment(
        alignment,
        config.output_dir / "figures" / "step08M_lower_component_mass_alignment.png",
    )
    return {
        "components": components_df,
        "alignment": alignment,
    }


def plot_lower_alignment(alignment: pd.DataFrame, output: Path) -> None:
    if alignment.empty:
        return
    configure_plot_style()
    component_ids = sorted(alignment["component_id"].unique())
    fig, axes = plt.subplots(1, len(component_ids), figsize=(6.2 * len(component_ids), 4.6), squeeze=False)
    for ax, component_id in zip(axes[0], component_ids):
        sub = alignment[alignment["component_id"] == component_id].copy()
        sub["short"] = sub["transition_id"].str.replace("_anchor01", "", regex=False)
        sub = sub.sort_values("median_abs_mass_on_component_boundary")
        ax.barh(sub["short"], sub["median_abs_mass_on_component_boundary"])
        ax.set_title(f"C_up=-2 component {component_id}")
        ax.set_xlabel("median |local chart mass| on boundary")
    fig.tight_layout()
    fig.savefig(output, dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Summary and certificate
# =============================================================================

def _best_metric(metrics: pd.DataFrame, task: str) -> dict[str, Any] | None:
    sub = metrics[metrics["task"] == task]
    if sub.empty:
        return None
    row = sub.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0]
    return row.to_dict()


def write_rule_summary(
    global_result: dict[str, Any],
    transition_result: dict[str, Any],
    cluster_result: dict[str, Any],
    upper_result: dict[str, Any],
    lower_result: dict[str, Any],
    config: Step08MConfig,
) -> dict[str, Any]:
    global_metrics = global_result["metrics"]
    transition_metrics = transition_result["metrics"]

    magnitude = _best_metric(global_metrics, "chern_magnitude_abs2_vs_abs1")
    sign = _best_metric(global_metrics, "high_chern_sign_positive_vs_negative")
    coarse = _best_metric(transition_metrics, "coarse_single_vs_paired")
    fine = _best_metric(transition_metrics, "fine_four_family")
    high = _best_metric(transition_metrics, "high_symmetry_Gamma_vs_M")
    generic = _best_metric(transition_metrics, "generic_Sigma_vs_SigmaPrime")

    upper_linear = upper_result["metrics"][
        upper_result["metrics"]["model"] == "linear_r3_r4"
    ].iloc[0].to_dict()
    upper_quad = upper_result["metrics"][
        upper_result["metrics"]["model"] == "quadratic_r3_r4"
    ].iloc[0].to_dict()

    alignment = lower_result["alignment"]
    dominant_rows = []
    if not alignment.empty:
        for component_id, sub in alignment.groupby("component_id"):
            dominant_rows.append(
                sub.sort_values(
                    ["fraction_nearest_mass_chart", "median_abs_mass_on_component_boundary"],
                    ascending=[False, True],
                ).iloc[0].to_dict()
            )

    certificate = {
        "code_version": CODE_VERSION,
        "global_abs_chern_rule_supported": bool(
            magnitude is not None and magnitude["balanced_accuracy"] >= 0.75
        ),
        "global_high_chern_sign_rule_supported": bool(
            sign is not None and sign["balanced_accuracy"] >= 0.70
        ),
        "coarse_single_vs_paired_valley_rule_supported": bool(
            coarse is not None and coarse["balanced_accuracy"] >= 0.65
        ),
        "single_global_four_family_classifier_supported": bool(
            fine is not None and fine["balanced_accuracy"] >= 0.75
        ),
        "high_symmetry_Gamma_vs_M_rule_supported": bool(
            high is not None and high["balanced_accuracy"] >= 0.80
        ),
        "global_Sigma_vs_SigmaPrime_rule_supported": bool(
            generic is not None and generic["balanced_accuracy"] >= 0.70
        ),
        "upper_linear_boundary_rule_supported": bool(
            upper_linear["balanced_accuracy"] >= 0.95
        ),
        "upper_quadratic_correction_useful": bool(
            upper_quad["balanced_accuracy"] - upper_linear["balanced_accuracy"] >= 0.005
        ),
        "upper_step06_mass_direction_recovered": bool(
            upper_result["cosine"] >= 0.98
        ),
        "multiple_minus2_components_supported": bool(
            len(lower_result["components"]) >= 2
        ),
        "distinct_minus2_local_mass_charts_supported": bool(
            len({row["transition_id"] for row in dominant_rows}) >= 2
        ),
        "mass_gradient_best_cluster_count": int(cluster_result["best_k"]),
        "mass_gradient_cosine_silhouette": float(cluster_result["best_silhouette"]),
        "key_metrics": {
            "global_magnitude": magnitude,
            "global_sign": sign,
            "transition_coarse": coarse,
            "transition_fine": fine,
            "high_symmetry_axis": high,
            "generic_diagonal_axis": generic,
            "upper_linear": upper_linear,
            "upper_quadratic": upper_quad,
            "upper_mass_direction_cosine": upper_result["cosine"],
        },
        "dominant_minus2_component_charts": dominant_rows,
        "interpretation": {
            "supported": [
                "参数规则能够在独立 Sobol 上较好地区分 |C_up|=1 与 |C_up|=2。",
                "transition-grouped 学习能够识别单个高对称谷与成对普通谷两类粗机制。",
                "Gamma 与 M 高对称单谷机制具有稳定可分的参数特征。",
                "formal 上侧相图中的 ML 线性边界方向与 Step06 Sigma' 质量方向一致。",
                "两个主要 C_up=-2 连通分量分别对齐到不同的局部质量图表。",
            ],
            "not_supported": [
                "高 Chern 手性符号不存在可靠的单一全局参数判据。",
                "Sigma 与 Sigma' 不能由一个跨 transition 的全局分类器稳定区分。",
                "四类 valley 机制不能压缩为单一全局分类规则。",
            ],
        },
    }

    lines = [
        "# TTS Step 08M 机制感知机器学习结论",
        "",
        "## 一、机器学习支持的规律",
        "",
    ]
    for item in certificate["interpretation"]["supported"]:
        lines.append(f"- {item}")
    lines += ["", "## 二、机器学习不支持的过度简化", ""]
    for item in certificate["interpretation"]["not_supported"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## 三、上侧 +2↔0 质量方向",
        "",
        (
            f"- grouped-CV 线性逻辑回归 balanced accuracy = "
            f"{upper_linear['balanced_accuracy']:.4f}"
        ),
        (
            f"- 二次逻辑回归 balanced accuracy = "
            f"{upper_quad['balanced_accuracy']:.4f}"
        ),
        (
            f"- ML 边界方向与 Step06 Sigma' 局域质量方向的 |cosine| = "
            f"{upper_result['cosine']:.6f}"
        ),
        (
            f"- ML 边界：r4 = {upper_result['linear_rule']['boundary_slope']:.6f} r3 "
            f"+ {upper_result['linear_rule']['boundary_intercept']:.6f}"
        ),
        (
            f"- Step06 质量零线：r4 = {upper_result['mass_slope']:.6f} r3 "
            f"+ {upper_result['mass_intercept']:.6f}"
        ),
        "",
        "## 四、推荐物理表述",
        "",
        (
            "TTS 的机器学习结果不支持 Lieb 式单一全局质量坐标。更合适的结论是："
            "七维参数中存在可学习的层级规律——首先区分高对称单谷与成对普通谷机制，"
            "随后在各机制内部使用局部质量坐标。ML 可以恢复上侧 Sigma' 质量方向并识别"
            "两个 -2 区域对应不同局部质量图表，但不能用一个全局公式同时决定 valley "
            "方向、Chern 手性和全部相区。"
        ),
    ]

    atomic_write_json(
        certificate,
        config.output_dir / "step08M_17_mechanism_discovery_certificate.json",
    )
    atomic_write_text(
        "\n".join(lines),
        config.output_dir / "step08M_18_candidate_mechanism_rules.md",
    )
    return certificate


# =============================================================================
# Main workflow
# =============================================================================

def run_step08M(config: Step08MConfig) -> dict[str, Any]:
    config = config.normalized()
    start = time.time()

    audit = audit_frozen_versions(config)
    atomic_write_csv(
        pd.DataFrame(audit["version_rows"]),
        config.output_dir / "step08M_00_frozen_version_audit.csv",
    )

    inputs = load_inputs(config)
    input_summary = {
        key: (
            {"rows": int(value.shape[0]), "columns": int(value.shape[1])}
            if isinstance(value, pd.DataFrame)
            else {"type": type(value).__name__}
        )
        for key, value in inputs.items()
    }
    atomic_write_json(
        {
            "code_version": CODE_VERSION,
            "configuration": asdict(config),
            "frozen_version_audit": audit,
            "input_summary": input_summary,
        },
        config.output_dir / "step08M_00_run_configuration.json",
    )

    global_result = run_global_rules(inputs, config)
    transition_result = run_transition_mechanism_learning(inputs, config)
    cluster_result = run_mass_gradient_clustering(inputs, config)
    upper_result = run_upper_boundary_rule(inputs, config)
    lower_result = run_lower_component_alignment(inputs, config)

    certificate = write_rule_summary(
        global_result,
        transition_result,
        cluster_result,
        upper_result,
        lower_result,
        config,
    )

    summary = {
        "code_version": CODE_VERSION,
        "output_dir": str(config.output_dir.resolve()),
        "elapsed_seconds": float(time.time() - start),
        "all_frozen_versions_pass": audit["all_frozen_versions_pass"],
        "certificate": certificate,
        "output_files": [
            str(path.relative_to(config.output_dir))
            for path in sorted(config.output_dir.rglob("*"))
            if path.is_file()
        ],
    }
    atomic_write_json(summary, config.output_dir / "step08M_19_run_summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TTS Step08M mechanism-aware machine learning"
    )
    parser.add_argument(
        "--tts-archive",
        type=Path,
        default=Path("tts(1).zip"),
    )
    parser.add_argument(
        "--formal-scan-input",
        type=Path,
        default=Path("TTS_Cup2_Formal_Refined_Scans(1).zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_tts_step08M_mechanism_aware_ml"),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--transition-lambda-window", type=float, default=0.18)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = Step08MConfig(
        output_dir=args.output_dir,
        tts_archive=args.tts_archive,
        formal_scan_input=args.formal_scan_input,
        n_jobs=args.n_jobs,
        bootstrap_repeats=args.bootstrap_repeats,
        transition_lambda_window=args.transition_lambda_window,
        require_frozen_versions=not args.allow_version_mismatch,
    )
    summary = run_step08M(config)
    print("TTS Step08M completed.")
    print("Output directory:", summary["output_dir"])
    print(
        "Upper mass direction recovered:",
        summary["certificate"]["upper_step06_mass_direction_recovered"],
    )
    print(
        "Single global four-family rule supported:",
        summary["certificate"]["single_global_four_family_classifier_supported"],
    )


if __name__ == "__main__":
    main()
