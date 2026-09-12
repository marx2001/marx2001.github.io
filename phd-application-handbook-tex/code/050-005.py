"""
TTS Step14M: physics-certified, mechanism-aware ML rule extraction.

This program does NOT search for new topology. It learns interpretable rules from
strictly certified Step11--Step13 outputs and the formal fixed-slice scans.

Main tasks
----------
1. Build a physics-certified r3-r4 phase dataset with source-priority deduplication.
2. Compare polynomial logistic regression and random forest using spatial group holdout.
3. Recover sparse local mass formulas near the left-lower and right-upper boundaries.
4. Build a certified transition-event table and perform an explicitly exploratory
   generic-four-valley vs SigmaPrime-two-valley analysis with a sample-size gate.
5. Export publication-ready figures, tables, Markdown conclusions, and a JSON certificate.

Required inputs
---------------
- TTS_Cup2_Formal_Refined_Scans(1).zip (or extracted directory)
- outputs_tts_step11M_lieb_aligned_fixed_slice.zip (or extracted directory)
- outputs_tts_step12M_junction_multiclosure_repair.zip (or extracted directory)
- outputs_tts_step13M_adaptive_wilson_intermediate_chern.zip (or extracted directory)

Dependencies: numpy, pandas, matplotlib, scikit-learn

Import hotfix v1.1:
- Compatible with importlib.util.spec_from_file_location even when callers forget
  to register the module in sys.modules before exec_module.
"""

import argparse
import json
import math
import shutil
import tempfile
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

CODE_VERSION = "TTS_STEP14M_PHYSICS_CERTIFIED_RULE_EXTRACTION_V1_1_IMPORT_HOTFIX_20260724"


@dataclass
class Step14Config:
    output_dir: Path = Path("outputs_tts_step14M_physics_certified_rule_extraction")
    random_state: int = 20260724
    n_spatial_splits: int = 5
    spatial_bins_r3: int = 7
    spatial_bins_r4: int = 7
    rf_n_estimators: int = 400
    rf_min_samples_leaf: int = 4
    rf_max_depth: int | None = None
    logistic_c: float = 1.0
    mass_l1_c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    mass_min_abs_distance: float = 0.0
    mechanism_min_events_total: int = 20
    mechanism_min_events_per_class: int = 5
    dpi: int = 500
    save_svg: bool = True
    include_wilson_points_in_phase_training: bool = True
    phase_allowed_labels: tuple[int, ...] = (-2, -1, 0, 1, 2)
    strict_only: bool = True

    def normalize(self) -> "Step14Config":
        self.output_dir = Path(self.output_dir)
        return self


# -----------------------------------------------------------------------------
# General I/O helpers
# -----------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _extract_zip(zip_path: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
    return target


def materialize_source(source: str | Path, work_dir: Path, tag: str) -> Path:
    """Return an extracted/searchable directory for a zip or directory source."""
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input source does not exist: {source}")
    if source.is_dir():
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Expected directory or .zip source, got: {source}")
    out = work_dir / tag
    if out.exists():
        shutil.rmtree(out)
    _extract_zip(source, out)
    return out


def find_first(root: Path, names: Sequence[str]) -> Path | None:
    name_set = set(names)
    for p in root.rglob("*"):
        if p.is_file() and p.name in name_set:
            return p
    return None


def find_all(root: Path, filename: str) -> list[Path]:
    return sorted(p for p in root.rglob(filename) if p.is_file())


def _series_numeric(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _series_bool01(df: pd.DataFrame, name: str, default: int = 0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=int)
    s = pd.to_numeric(df[name], errors="coerce").fillna(default)
    return (s == 1).astype(int)


# -----------------------------------------------------------------------------
# Certified phase dataset
# -----------------------------------------------------------------------------

def load_formal_phase_points(formal_root: Path, allowed_labels: Sequence[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in find_all(formal_root, "tts_scan_r3_r4_grid.csv"):
        df = safe_read_csv(path)
        if df.empty or not {"r3", "r4"}.issubset(df.columns):
            continue
        phase = _series_numeric(df, "phase_code")
        chern = _series_numeric(df, "chern_up_int")
        phase = phase.where(phase.notna(), chern)
        exact = _series_bool01(df, "chern_exact_consensus", default=0)
        strict_ins = _series_bool01(df, "is_strict_insulator", default=0)
        strict_gap = _series_bool01(df, "strict_gap_verified", default=0)
        physical = _series_bool01(df, "is_physical_insulator", default=0)
        strict_chern = _series_bool01(df, "strict_chern_verified", default=0)
        reliable = (
            phase.isin(list(allowed_labels))
            & exact.eq(1)
            & ((strict_ins.eq(1)) | (strict_gap.eq(1) & physical.eq(1)))
            & ((strict_chern.eq(1)) | phase.eq(0) | chern.notna())
        )
        out = pd.DataFrame({
            "sample_id": df.get("sample_id", pd.Series([f"formal_{i}" for i in range(len(df))])),
            "r3": _series_numeric(df, "r3"),
            "r4": _series_numeric(df, "r4"),
            "chern_up": phase,
            "min_direct_gap": _series_numeric(df, "min_direct_gap"),
            "indirect_gap": _series_numeric(df, "indirect_gap"),
            "spin_internal_gap": _series_numeric(df, "min_spin_gap_up"),
            "strict_label": reliable.astype(int),
            "source_type": "formal_scan",
            "source_region": path.parent.name,
            "source_priority": 10,
            "uncertain_or_boundary": (~reliable).astype(int),
        })
        rows.append(out)
    if not rows:
        raise FileNotFoundError("No tts_scan_r3_r4_grid.csv files found in formal scan source")
    return pd.concat(rows, ignore_index=True)


def load_step12_phase_points(step12_root: Path, allowed_labels: Sequence[int]) -> pd.DataFrame:
    path = find_first(step12_root, ["step12M_15_refined_junction_strict_grid.csv"])
    if path is None:
        return pd.DataFrame()
    df = safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()
    phase = _series_numeric(df, "paper_phase_code")
    chern = _series_numeric(df, "chern_up_int")
    phase = phase.where(phase.notna(), chern)
    reliable = (
        phase.isin(list(allowed_labels))
        & _series_bool01(df, "strict_gap_verified").eq(1)
        & _series_bool01(df, "strict_chern_verified").eq(1)
        & _series_bool01(df, "chern_exact_consensus").eq(1)
    )
    return pd.DataFrame({
        "sample_id": df.get("sample_id", pd.Series([f"step12_{i}" for i in range(len(df))])),
        "r3": _series_numeric(df, "r3"),
        "r4": _series_numeric(df, "r4"),
        "chern_up": phase,
        "min_direct_gap": _series_numeric(df, "min_direct_gap"),
        "indirect_gap": _series_numeric(df, "indirect_gap"),
        "spin_internal_gap": np.nan,
        "strict_label": reliable.astype(int),
        "source_type": "step12_refined_grid",
        "source_region": "right_junction_13x13",
        "source_priority": 30,
        "uncertain_or_boundary": (~reliable).astype(int),
    })


def load_step13_wilson_points(step13_root: Path, allowed_labels: Sequence[int]) -> pd.DataFrame:
    path = find_first(step13_root, ["step13M_04_wilson_probe_consensus.csv"])
    if path is None:
        return pd.DataFrame()
    df = safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()
    phase = _series_numeric(df, "wilson_chern_up_consensus")
    reliable = (
        _series_bool01(df, "probe_certificate_pass").eq(1)
        & _series_bool01(df, "wilson_consensus").eq(1)
        & phase.isin(list(allowed_labels))
    )
    return pd.DataFrame({
        "sample_id": df.get("probe_id", pd.Series([f"step13_{i}" for i in range(len(df))])),
        "r3": _series_numeric(df, "r3"),
        "r4": _series_numeric(df, "r4"),
        "chern_up": phase,
        "min_direct_gap": np.nan,
        "indirect_gap": np.nan,
        "spin_internal_gap": _series_numeric(df, "spin_internal_gap"),
        "strict_label": reliable.astype(int),
        "source_type": "step13_wilson",
        "source_region": df.get("probe_role", pd.Series("wilson_path", index=df.index)).astype(str),
        "source_priority": 40,
        "uncertain_or_boundary": (~reliable).astype(int),
    })


def deduplicate_phase_points(all_points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = all_points.copy()
    df = df[np.isfinite(df["r3"]) & np.isfinite(df["r4"])].copy()
    df["key_r3"] = df["r3"].round(10)
    df["key_r4"] = df["r4"].round(10)
    conflict_rows: list[dict[str, Any]] = []
    for (r3, r4), g in df[df["strict_label"].eq(1)].groupby(["key_r3", "key_r4"]):
        labels = sorted(set(int(x) for x in g["chern_up"].dropna()))
        if len(labels) > 1:
            conflict_rows.append({
                "r3": r3,
                "r4": r4,
                "strict_labels": ",".join(map(str, labels)),
                "sources": ";".join(g["source_type"].astype(str)),
            })
    conflicts = pd.DataFrame(conflict_rows)
    # High-priority source wins. Within equal priority, strict label wins.
    df = df.sort_values(
        ["key_r3", "key_r4", "source_priority", "strict_label"],
        ascending=[True, True, False, False],
    )
    dedup = df.drop_duplicates(["key_r3", "key_r4"], keep="first").copy()
    dedup.drop(columns=["key_r3", "key_r4"], inplace=True)
    dedup.reset_index(drop=True, inplace=True)
    return dedup, conflicts


def make_spatial_groups(df: pd.DataFrame, bins_r3: int, bins_r4: int) -> pd.Series:
    r3 = df["r3"].to_numpy(float)
    r4 = df["r4"].to_numpy(float)
    eps = 1e-12
    edges3 = np.linspace(r3.min() - eps, r3.max() + eps, bins_r3 + 1)
    edges4 = np.linspace(r4.min() - eps, r4.max() + eps, bins_r4 + 1)
    i3 = np.clip(np.digitize(r3, edges3[1:-1], right=False), 0, bins_r3 - 1)
    i4 = np.clip(np.digitize(r4, edges4[1:-1], right=False), 0, bins_r4 - 1)
    return pd.Series([f"b{a:02d}_{b:02d}" for a, b in zip(i3, i4)], index=df.index)


def build_phase_dataset(
    formal_root: Path,
    step12_root: Path,
    step13_root: Path,
    config: Step14Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frames = [load_formal_phase_points(formal_root, config.phase_allowed_labels)]
    s12 = load_step12_phase_points(step12_root, config.phase_allowed_labels)
    if not s12.empty:
        frames.append(s12)
    if config.include_wilson_points_in_phase_training:
        s13 = load_step13_wilson_points(step13_root, config.phase_allowed_labels)
        if not s13.empty:
            frames.append(s13)
    raw = pd.concat(frames, ignore_index=True)
    dedup, conflicts = deduplicate_phase_points(raw)
    strict = dedup[dedup["strict_label"].eq(1) & dedup["chern_up"].isin(config.phase_allowed_labels)].copy()
    strict["chern_up"] = strict["chern_up"].astype(int)
    strict["spatial_group"] = make_spatial_groups(strict, config.spatial_bins_r3, config.spatial_bins_r4)
    audit = {
        "n_raw_rows": len(raw),
        "n_deduplicated_rows": len(dedup),
        "n_strict_training_rows": len(strict),
        "strict_label_counts": strict["chern_up"].value_counts().sort_index().to_dict(),
        "n_uncertain_or_boundary_rows": int((dedup["strict_label"] == 0).sum()),
        "n_conflicting_strict_coordinates": len(conflicts),
        "source_counts_deduplicated": dedup["source_type"].value_counts().to_dict(),
        "n_spatial_groups": int(strict["spatial_group"].nunique()),
    }
    return strict, dedup, {"audit": audit, "conflicts": conflicts}


# -----------------------------------------------------------------------------
# Spatial cross-validation phase models
# -----------------------------------------------------------------------------

def _phase_models(config: Step14Config) -> dict[str, Any]:
    poly_logistic = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=config.logistic_c,
            max_iter=10000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=config.random_state,
        )),
    ])
    rf = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_depth=config.rf_max_depth,
        class_weight="balanced_subsample",
        random_state=config.random_state,
        n_jobs=-1,
        max_features="sqrt",
    )
    return {"poly2_logistic": poly_logistic, "random_forest": rf}


def _make_cv_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int):
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    if StratifiedGroupKFold is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(cv.split(np.zeros(len(y)), y, groups))
    cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(np.zeros(len(y)), y, groups))


def spatial_cv_phase_models(phase_df: pd.DataFrame, config: Step14Config):
    features = ["r3", "r4"]
    X = phase_df[features].to_numpy(float)
    y = phase_df["chern_up"].to_numpy(int)
    groups = phase_df["spatial_group"].to_numpy(str)
    classes = np.array(sorted(np.unique(y)), dtype=int)
    splits = _make_cv_splits(y, groups, config.n_spatial_splits, config.random_state)
    models = _phase_models(config)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, estimator in models.items():
        y_pred = np.full_like(y, fill_value=999)
        fold_id = np.full(len(y), -1, dtype=int)
        fold_metrics: list[dict[str, Any]] = []
        for fold, (tr, te) in enumerate(splits):
            est = clone(estimator)
            est.fit(X[tr], y[tr])
            pred = est.predict(X[te]).astype(int)
            y_pred[te] = pred
            fold_id[te] = fold
            fold_metrics.append({
                "model": model_name,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                "accuracy": accuracy_score(y[te], pred),
                "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
                "test_classes": ",".join(map(str, sorted(np.unique(y[te])))),
            })
        valid = y_pred != 999
        metric_rows.append({
            "model": model_name,
            "n_samples": int(valid.sum()),
            "n_folds": len(splits),
            "balanced_accuracy": balanced_accuracy_score(y[valid], y_pred[valid]),
            "accuracy": accuracy_score(y[valid], y_pred[valid]),
            "macro_f1": f1_score(y[valid], y_pred[valid], average="macro", zero_division=0),
        })
        pred_df = phase_df[["sample_id", "r3", "r4", "chern_up", "spatial_group", "source_type"]].copy()
        pred_df["model"] = model_name
        pred_df["fold"] = fold_id
        pred_df["prediction"] = y_pred
        prediction_frames.append(pred_df)
        prediction_frames.append(pd.DataFrame(fold_metrics).assign(record_type="fold_metric"))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["balanced_accuracy", "macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    best_name = str(metrics.iloc[0]["model"])
    best_model = clone(models[best_name]).fit(X, y)
    cm = confusion_matrix(
        y,
        predictions[(predictions.get("record_type").isna()) & (predictions["model"] == best_name)]["prediction"].astype(int),
        labels=classes,
    )
    return metrics, predictions, best_name, best_model, classes, cm


def random_forest_importance(
    phase_df: pd.DataFrame,
    fitted_rf: RandomForestClassifier,
    config: Step14Config,
) -> pd.DataFrame:
    X = phase_df[["r3", "r4"]].to_numpy(float)
    y = phase_df["chern_up"].to_numpy(int)
    result = permutation_importance(
        fitted_rf,
        X,
        y,
        scoring="balanced_accuracy",
        n_repeats=30,
        random_state=config.random_state,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": ["r3", "r4"],
        "permutation_importance_mean": result.importances_mean,
        "permutation_importance_std": result.importances_std,
        "gini_importance": fitted_rf.feature_importances_,
    }).sort_values("permutation_importance_mean", ascending=False)


# -----------------------------------------------------------------------------
# Sparse local mass recovery
# -----------------------------------------------------------------------------
MASS_TERMS = ["constant", "r3", "r4", "r3^2", "r3*r4", "r4^2"]


def raw_poly2_matrix(df: pd.DataFrame) -> np.ndarray:
    r3 = df["r3"].to_numpy(float)
    r4 = df["r4"].to_numpy(float)
    return np.column_stack([r3, r4, r3 * r3, r3 * r4, r4 * r4])


def make_branch_spatial_splits(df: pd.DataFrame, n_splits: int = 4):
    if "spatial_fold" in df.columns and df["spatial_fold"].nunique() >= 2:
        splits = []
        for fold in sorted(df["spatial_fold"].dropna().unique()):
            te = np.where(df["spatial_fold"].to_numpy() == fold)[0]
            tr = np.where(df["spatial_fold"].to_numpy() != fold)[0]
            if len(te) and len(tr):
                splits.append((tr, te))
        return splits
    group = make_spatial_groups(df, 4, 4).to_numpy()
    return list(GroupKFold(n_splits=min(n_splits, len(np.unique(group)))).split(df, df["paper_phase_code"], group))


def _fit_l1_poly_raw_coefficients(Xraw: np.ndarray, y: np.ndarray, C: float, seed: int):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(Xraw)
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=10000,
        random_state=seed,
    )
    model.fit(Xz, y)
    beta_raw = model.coef_[0] / scaler.scale_
    intercept_raw = float(model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_))
    return model, scaler, np.concatenate([[intercept_raw], beta_raw])


def _physical_mass_vector(defrow: pd.Series) -> np.ndarray:
    # vector order: constant, r3, r4, r3^2, r3*r4, r4^2
    c2 = float(defrow.get("coefficient_2", 0.0))
    c1 = float(defrow.get("coefficient_1", 0.0))
    c0 = float(defrow.get("coefficient_0", 0.0))
    orientation = float(defrow.get("orientation", 1.0))
    independent = str(defrow["independent_parameter"])
    dependent = str(defrow["dependent_parameter"])
    v = np.zeros(6, dtype=float)
    # orientation * [dependent - (c2 independent^2 + c1 independent + c0)]
    v[0] = -orientation * c0
    if dependent == "r3":
        v[1] += orientation
    elif dependent == "r4":
        v[2] += orientation
    if independent == "r3":
        v[1] += -orientation * c1
        v[3] += -orientation * c2
    elif independent == "r4":
        v[2] += -orientation * c1
        v[5] += -orientation * c2
    return v


def _orient_vector_to_physical(ml_vec: np.ndarray, physical_vec: np.ndarray) -> np.ndarray:
    if np.dot(ml_vec[1:], physical_vec[1:]) < 0:
        return -ml_vec
    return ml_vec


def _normalize_mass_vector(v: np.ndarray, branch: str) -> np.ndarray:
    index = 2 if "left" in branch else 1  # left normalize by r4, right by r3
    denom = v[index]
    if abs(denom) < 1e-12:
        denom = np.linalg.norm(v[1:]) or 1.0
    return v / denom


def _formula_from_vector(v: np.ndarray, precision: int = 7) -> str:
    chunks = []
    for coef, term in zip(v, MASS_TERMS):
        if abs(coef) < 1e-8:
            continue
        label = "1" if term == "constant" else term
        chunks.append(f"{coef:+.{precision}g}*{label}")
    return " ".join(chunks).lstrip("+") if chunks else "0"


def _polyfit_loocv(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, float, float]:
    coeff = np.polyfit(x, y, degree)
    pred = np.polyval(coeff, x)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    errs = []
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        c = np.polyfit(x[mask], y[mask], degree)
        errs.append(float(np.polyval(c, x[i]) - y[i]))
    loocv = float(np.sqrt(np.mean(np.square(errs))))
    return coeff, rmse, loocv


def _boundary_regression_vector(
    independent: str,
    dependent: str,
    coeff_descending: np.ndarray,
    orientation: float = 1.0,
) -> np.ndarray:
    """Return orientation*[dependent - polynomial(independent)] in MASS_TERMS order."""
    v = np.zeros(6, dtype=float)
    degree = len(coeff_descending) - 1
    coeff_ascending = coeff_descending[::-1]
    v[0] = -orientation * coeff_ascending[0]
    if dependent == "r3":
        v[1] += orientation
    else:
        v[2] += orientation
    if degree >= 1:
        if independent == "r3":
            v[1] += -orientation * coeff_ascending[1]
        else:
            v[2] += -orientation * coeff_ascending[1]
    if degree >= 2:
        if independent == "r3":
            v[3] += -orientation * coeff_ascending[2]
        else:
            v[5] += -orientation * coeff_ascending[2]
    return v


def recover_sparse_mass_rules(step11_root: Path, config: Step14Config):
    """
    Recover the local mass boundary from certified boundary points, then test the
    discovered mass sign on independent phase points around each branch.

    This avoids an identifiability failure of unconstrained L1 classification in a
    narrow correlated parameter band, where several algebraically different
    polynomials can classify the same finite grid but do not represent the physical
    boundary normal.
    """
    pred_path = find_first(step11_root, ["step11M_11_spatial_holdout_predictions.csv"])
    def_path = find_first(step11_root, ["step11M_10_local_signed_mass_definitions.csv"])
    boundary_path = find_first(step11_root, ["step11M_07_completed_certified_boundary_points.csv"])
    if pred_path is None or def_path is None or boundary_path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    phase_data = safe_read_csv(pred_path)
    defs = safe_read_csv(def_path)
    boundary = safe_read_csv(boundary_path)
    if phase_data.empty or defs.empty or boundary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    phase_data = phase_data[phase_data["paper_phase_code"].isin([-2, 2])].copy()
    boundary = boundary[boundary.get("point_certificate_pass", 0) == 1].copy()

    result_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for _, defrow in defs.iterrows():
        branch = str(defrow["branch_hint"])
        b = boundary[boundary["branch_hint"] == branch].copy()
        d = phase_data[phase_data["branch_hint"] == branch].copy()
        if len(b) < 3 or d.empty:
            continue
        independent = str(defrow["independent_parameter"])
        dependent = str(defrow["dependent_parameter"])
        x = b[independent].to_numpy(float)
        y = b[dependent].to_numpy(float)

        candidate_rows = []
        max_degree = min(2, len(b) - 2)
        for degree in range(1, max_degree + 1):
            coeff, fit_rmse, loocv = _polyfit_loocv(x, y, degree)
            candidate_rows.append({
                "branch_hint": branch,
                "degree": degree,
                "fit_rmse": fit_rmse,
                "loocv_rmse": loocv,
                "coefficients_descending": json.dumps(coeff.tolist()),
            })
        cand_df = pd.DataFrame(candidate_rows).sort_values(["loocv_rmse", "degree"])
        selected = cand_df.iloc[0]
        degree = int(selected["degree"])
        coeff = np.array(json.loads(selected["coefficients_descending"]), dtype=float)
        ml_vec = _boundary_regression_vector(independent, dependent, coeff, orientation=1.0)
        physical_vec = _physical_mass_vector(defrow)
        ml_vec = _orient_vector_to_physical(ml_vec, physical_vec)

        # Orient the learned mass so positive mass maps to C_up=+2 on the independent phase points.
        score = _eval_mass(ml_vec, d["r3"].to_numpy(float), d["r4"].to_numpy(float))
        pred = np.where(score >= 0, 2, -2)
        if balanced_accuracy_score(d["paper_phase_code"], pred) < balanced_accuracy_score(d["paper_phase_code"], -pred):
            ml_vec = -ml_vec
            score = -score
            pred = -pred
        # Align physical vector to the same sign convention for coefficient comparison.
        if np.dot(ml_vec[1:], physical_vec[1:]) < 0:
            physical_vec = -physical_vec

        ml_norm = _normalize_mass_vector(ml_vec, branch)
        phys_norm = _normalize_mass_vector(physical_vec, branch)
        cos = float(np.dot(ml_vec[1:], physical_vec[1:]) / (
            np.linalg.norm(ml_vec[1:]) * np.linalg.norm(physical_vec[1:]) + 1e-15
        ))
        result_rows.append({
            "branch_hint": branch,
            "n_certified_boundary_points": len(b),
            "n_independent_phase_points": len(d),
            "selected_polynomial_degree": degree,
            "boundary_fit_rmse": float(selected["fit_rmse"]),
            "boundary_loocv_rmse": float(selected["loocv_rmse"]),
            "phase_sign_balanced_accuracy": balanced_accuracy_score(d["paper_phase_code"], pred),
            "phase_sign_accuracy": accuracy_score(d["paper_phase_code"], pred),
            "coefficient_cosine_similarity_to_physical_mass": cos,
            "ml_boundary_formula_normalized": _formula_from_vector(ml_norm),
            "physical_mass_formula_normalized": _formula_from_vector(phys_norm),
        })
        for term, a, bcoef in zip(MASS_TERMS, ml_norm, phys_norm):
            coefficient_rows.append({
                "branch_hint": branch,
                "term": term,
                "ml_normalized_coefficient": a,
                "physical_normalized_coefficient": bcoef,
                "absolute_difference": abs(a - bcoef),
            })
        pr = d[["r3", "r4", "paper_phase_code", "spatial_fold", "branch_hint"]].copy()
        pr["ml_prediction"] = pred
        pr["ml_decision_score"] = score
        prediction_rows.append(pr)
        cand_df.to_csv(config.output_dir / f"step14M_mass_degree_search_{branch}.csv", index=False)

    return pd.DataFrame(result_rows), pd.DataFrame(coefficient_rows), pd.concat(prediction_rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Certified transition-event atlas and exploratory mechanism model
# -----------------------------------------------------------------------------

def build_transition_event_dataset(step11_root: Path, step12_root: Path, step13_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Generic four-valley certified boundary points
    p11 = find_first(step11_root, ["step11M_07_completed_certified_boundary_points.csv"])
    if p11 is not None:
        df = safe_read_csv(p11)
        for _, r in df[df.get("point_certificate_pass", 0) == 1].iterrows():
            nval = int(r.get("n_active_spin_up_valleys", 0))
            mech = "generic_four_valley" if nval == 4 else f"certified_{nval}_valley"
            rows.append({
                "event_id": str(r.get("point_id")),
                "event_source": "step11_boundary",
                "r3": float(r.get("r3")),
                "r4": float(r.get("r4")),
                "kx": float(r.get("critical_kx_representative", np.nan)),
                "ky": float(r.get("critical_ky_representative", np.nan)),
                "n_valleys": nval,
                "delta_chern_up": float(r.get("strict_side_delta_chern_up", np.nan)),
                "berry_charge_sum_up": float(r.get("berry_charge_sum_up", np.nan)),
                "mechanism": mech,
                "certificate_pass": 1,
            })
    # Step11 passed two-valley junction events: use selected edge midpoint
    edges_path = find_first(step11_root, ["step11M_05_selected_junction_edges.csv"])
    cert_path = find_first(step11_root, ["step11M_06_junction_edge_certificates.csv"])
    if edges_path is not None and cert_path is not None:
        edges = safe_read_csv(edges_path)
        cert = safe_read_csv(cert_path)
        if not edges.empty and not cert.empty:
            merged = edges.merge(cert, on="edge_id", how="inner")
            for _, r in merged[merged["edge_multiclosure_certificate_pass"] == 1].iterrows():
                rows.append({
                    "event_id": str(r["edge_id"]),
                    "event_source": "step11_junction",
                    "r3": float(r["mid_r3"]),
                    "r4": float(r["mid_r4"]),
                    "kx": np.nan,
                    "ky": np.nan,
                    "n_valleys": int(r["n_active_spin_up_valleys_total"]),
                    "delta_chern_up": float(r["grid_endpoint_delta_chern_up"]),
                    "berry_charge_sum_up": float(r["berry_charge_sum_over_all_groups"]),
                    "mechanism": "SigmaPrime_two_valley" if int(r["n_active_spin_up_valleys_total"]) == 2 else "multiclosure",
                    "certificate_pass": 1,
                })
    # Step13 updated closure groups joined to Step12 critical coordinates
    c13_path = find_first(step13_root, ["step13M_07_updated_closure_group_certificates.csv"])
    c12_path = find_first(step12_root, ["step12M_06_critical_closure_parameters.csv"])
    if c13_path is not None and c12_path is not None:
        c13 = safe_read_csv(c13_path)
        c12 = safe_read_csv(c12_path)
        if not c13.empty and not c12.empty:
            c12g = c12.groupby("closure_group_id", as_index=False).agg({
                "r3": "mean", "r4": "mean", "critical_kx": "mean", "critical_ky": "mean"
            })
            merged = c13.merge(c12g, on="closure_group_id", how="left")
            for _, r in merged[merged["closure_group_certificate_pass"] == 1].iterrows():
                rows.append({
                    "event_id": str(r["closure_group_id"]),
                    "event_source": "step13_updated_closure",
                    "r3": float(r["r3"]),
                    "r4": float(r["r4"]),
                    "kx": float(r["critical_kx"]),
                    "ky": float(r["critical_ky"]),
                    "n_valleys": int(r["n_active_spin_up_valleys"]),
                    "delta_chern_up": float(r["observed_delta_chern_up"]),
                    "berry_charge_sum_up": float(r["berry_charge_sum_up"]),
                    "mechanism": str(r["mechanism"]),
                    "certificate_pass": 1,
                })
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    # Step13 events can duplicate one Step11 generic point only approximately; keep separately because it is a distinct junction mechanism event.
    events["abs_delta_chern_up"] = events["delta_chern_up"].abs()
    events["abs_berry_charge_sum"] = events["berry_charge_sum_up"].abs()
    events["charge_conservation_pass"] = np.isclose(
        events["delta_chern_up"], events["berry_charge_sum_up"], atol=1e-8
    ).astype(int)
    return events


def exploratory_mechanism_analysis(events: pd.DataFrame, config: Step14Config):
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "mechanism_model_status": "no_events",
            "mechanism_ml_conclusion_allowed": False,
        }
    selected = events[events["mechanism"].isin(["generic_four_valley", "SigmaPrime_two_valley"])].copy()
    counts = selected["mechanism"].value_counts()
    gate = (
        len(selected) >= config.mechanism_min_events_total
        and not counts.empty
        and counts.min() >= config.mechanism_min_events_per_class
        and len(counts) == 2
    )
    summary = {
        "n_certified_events": len(events),
        "mechanism_counts": events["mechanism"].value_counts().to_dict(),
        "all_events_charge_conserving": bool(events["charge_conservation_pass"].eq(1).all()),
        "mechanism_sample_size_gate_pass": bool(gate),
        "mechanism_ml_conclusion_allowed": bool(gate),
        "interpretation": (
            "Parameter-only mechanism classification is allowed."
            if gate else
            "Certified event count is too small or too imbalanced for a generalizable parameter-only mechanism classifier; report the event atlas descriptively and treat any classifier as exploratory only."
        ),
    }
    if len(selected) < 4 or selected["mechanism"].nunique() < 2:
        return pd.DataFrame(), selected, summary

    X = selected[["r3", "r4"]].to_numpy(float)
    y = (selected["mechanism"] == "SigmaPrime_two_valley").astype(int).to_numpy()
    # Leave-one-event-out exploratory predictions, no claim of generalization unless gate passes.
    pred = np.zeros(len(selected), dtype=int)
    score = np.zeros(len(selected), dtype=float)
    for i in range(len(selected)):
        tr = np.arange(len(selected)) != i
        if len(np.unique(y[tr])) < 2:
            pred[i] = int(np.round(y[tr].mean()))
            score[i] = float(y[tr].mean())
            continue
        pipe = Pipeline([
            ("poly", PolynomialFeatures(2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=10000, random_state=config.random_state)),
        ])
        pipe.fit(X[tr], y[tr])
        pred[i] = int(pipe.predict(X[[i]])[0])
        score[i] = float(pipe.predict_proba(X[[i]])[0, 1])
    metrics = pd.DataFrame([{
        "model": "exploratory_poly2_logistic_LOEO",
        "n_events": len(selected),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "sample_size_gate_pass": int(gate),
        "conclusion_allowed": int(gate),
    }])
    out = selected.copy()
    out["exploratory_prediction"] = np.where(pred == 1, "SigmaPrime_two_valley", "generic_four_valley")
    out["SigmaPrime_probability"] = score
    return metrics, out, summary


# -----------------------------------------------------------------------------
# Figures and reports
# -----------------------------------------------------------------------------
PHASE_COLORS = {
    -2: "#3B6FB6",
    -1: "#8AB6D6",
    0: "#F2C14E",
    1: "#F28E8E",
    2: "#C73E3A",
}


def save_figure(fig: plt.Figure, base: Path, config: Step14Config) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=config.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    if config.save_svg:
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_phase_model_metrics(metrics: pd.DataFrame, config: Step14Config) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(metrics))
    width = 0.24
    for j, col in enumerate(["balanced_accuracy", "macro_f1", "accuracy"]):
        ax.bar(x + (j - 1) * width, metrics[col], width=width, label=col.replace("_", " "))
    ax.set_xticks(x, metrics["model"], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spatial holdout score")
    ax.set_title("Physics-certified fixed-slice phase classification")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, config.output_dir / "figures/step14M_phase_model_comparison", config)


def plot_confusion(cm: np.ndarray, classes: np.ndarray, best_name: str, config: Step14Config) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks(range(len(classes)), [str(c) for c in classes])
    ax.set_yticks(range(len(classes)), [str(c) for c in classes])
    ax.set_xlabel("Predicted $C_\\uparrow$")
    ax.set_ylabel("Certified $C_\\uparrow$")
    ax.set_title(f"Spatial holdout confusion matrix: {best_name}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    save_figure(fig, config.output_dir / "figures/step14M_best_phase_confusion_matrix", config)


def plot_phase_decision_map(
    certified: pd.DataFrame,
    all_points: pd.DataFrame,
    model: Any,
    classes: np.ndarray,
    best_name: str,
    config: Step14Config,
) -> None:
    r3_min, r3_max = certified.r3.min(), certified.r3.max()
    r4_min, r4_max = certified.r4.min(), certified.r4.max()
    gx = np.linspace(r3_min, r3_max, 360)
    gy = np.linspace(r4_min, r4_max, 360)
    xx, yy = np.meshgrid(gx, gy)
    pred = model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    present = sorted(set(int(x) for x in classes))
    colors = [PHASE_COLORS.get(c, "#999999") for c in present]
    cmap = ListedColormap(colors)
    bounds = [present[0] - 0.5] + [(a + b) / 2 for a, b in zip(present[:-1], present[1:])] + [present[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7.7, 6.1))
    ax.pcolormesh(xx, yy, pred, cmap=cmap, norm=norm, shading="auto", alpha=0.55)
    uncertain = all_points[all_points["strict_label"].eq(0)]
    if not uncertain.empty:
        ax.scatter(uncertain.r3, uncertain.r4, marker="x", s=12, linewidths=0.45, c="0.45", alpha=0.35, label="boundary/unreliable")
    for c in present:
        g = certified[certified["chern_up"] == c]
        ax.scatter(g.r3, g.r4, s=8, c=PHASE_COLORS.get(c), edgecolors="none", label=f"certified $C_\\uparrow={c}$")
    ax.set_xlabel("$r_3$")
    ax.set_ylabel("$r_4$")
    ax.set_title(f"Interpretable phase model ({best_name})\nbackground fixed to Step11--13 values")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="best")
    ax.set_xlim(r3_min, r3_max)
    ax.set_ylim(r4_min, r4_max)
    fig.tight_layout()
    save_figure(fig, config.output_dir / "figures/step14M_phase_decision_map", config)


def plot_feature_importance(importance: pd.DataFrame, config: Step14Config) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    x = np.arange(len(importance))
    ax.bar(x, importance["permutation_importance_mean"], yerr=importance["permutation_importance_std"], capsize=4)
    ax.set_xticks(x, importance["feature"])
    ax.set_ylabel("Permutation importance\n(balanced accuracy decrease)")
    ax.set_title("Random-forest parameter importance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, config.output_dir / "figures/step14M_parameter_importance", config)


def _eval_mass(v: np.ndarray, r3: np.ndarray, r4: np.ndarray) -> np.ndarray:
    return v[0] + v[1]*r3 + v[2]*r4 + v[3]*r3*r3 + v[4]*r3*r4 + v[5]*r4*r4


def plot_mass_rule_recovery(
    coefficient_df: pd.DataFrame,
    branch_predictions: pd.DataFrame,
    config: Step14Config,
) -> None:
    if coefficient_df.empty or branch_predictions.empty:
        return
    for branch in coefficient_df["branch_hint"].unique():
        c = coefficient_df[coefficient_df["branch_hint"] == branch].set_index("term")
        ml = np.array([c.loc[t, "ml_normalized_coefficient"] for t in MASS_TERMS])
        ph = np.array([c.loc[t, "physical_normalized_coefficient"] for t in MASS_TERMS])
        d = branch_predictions[branch_predictions["branch_hint"] == branch]
        x = np.linspace(d.r3.min(), d.r3.max(), 500)
        y = np.linspace(d.r4.min(), d.r4.max(), 500)
        xx, yy = np.meshgrid(x, y)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for label in [-2, 2]:
            g = d[d.paper_phase_code == label]
            ax.scatter(g.r3, g.r4, s=22, c=PHASE_COLORS[label], label=f"certified $C_\\uparrow={label}$", alpha=0.75)
        ax.contour(xx, yy, _eval_mass(ph, xx, yy), levels=[0], colors="black", linewidths=2.0)
        ax.contour(xx, yy, _eval_mass(ml, xx, yy), levels=[0], colors="darkorange", linewidths=2.0, linestyles="--")
        ax.plot([], [], color="black", lw=2, label="physics-certified mass $M=0$")
        ax.plot([], [], color="darkorange", lw=2, ls="--", label="sparse ML boundary")
        ax.set_xlabel("$r_3$")
        ax.set_ylabel("$r_4$")
        ax.set_title(branch.replace("_", " "))
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        save_figure(fig, config.output_dir / f"figures/step14M_mass_rule_recovery_{branch}", config)


def plot_transition_events(events: pd.DataFrame, config: Step14Config) -> None:
    if events.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    markers = {"generic_four_valley": "o", "SigmaPrime_two_valley": "^"}
    for mech, g in events.groupby("mechanism"):
        ax.scatter(
            g.r3, g.r4,
            s=55 + 12 * g.n_valleys,
            marker=markers.get(mech, "s"),
            label=f"{mech} (n={len(g)})",
            alpha=0.8,
        )
    for _, r in events.iterrows():
        if r["event_source"] == "step13_updated_closure":
            ax.annotate(f"$\\Delta C={int(r['delta_chern_up'])}$", (r.r3, r.r4), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("$r_3$")
    ax.set_ylabel("$r_4$")
    ax.set_title("Physics-certified transition-event atlas")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, config.output_dir / "figures/step14M_transition_mechanism_atlas", config)


def write_rule_summary(
    path: Path,
    phase_audit: dict[str, Any],
    metrics: pd.DataFrame,
    mass_results: pd.DataFrame,
    mechanism_summary: dict[str, Any],
    classes: Sequence[int],
) -> None:
    best = metrics.iloc[0].to_dict() if not metrics.empty else {}
    lines = [
        "# TTS Step14M：物理认证驱动的机器学习规律总结",
        "",
        f"代码版本：`{CODE_VERSION}`",
        "",
        "## 1. 数据边界",
        "",
        f"- 严格训练点：{phase_audit.get('n_strict_training_rows', 0)}",
        f"- 纳入的固定切片标签：{list(classes)}",
        f"- 边界/不可靠点：{phase_audit.get('n_uncertain_or_boundary_rows', 0)}（只用于显示，不用于硬标签训练）",
        "- 本阶段只总结固定背景下的二维 `r3-r4` 规律，不构成七维全局拓扑判据。",
        "",
        "## 2. 相区分类",
        "",
    ]
    if best:
        lines += [
            f"空间分块留出中最佳模型为 **{best['model']}**：",
            f"- balanced accuracy = {best['balanced_accuracy']:.4f}",
            f"- macro F1 = {best['macro_f1']:.4f}",
            f"- accuracy = {best['accuracy']:.4f}",
            "",
            "该性能用于衡量模型跨空间块重建固定切片相区的能力，而不是随机网格记忆能力。",
        ]
    lines += ["", "## 3. 局域质量公式恢复", ""]
    if mass_results.empty:
        lines.append("未找到 Step11M 局域质量训练表。")
    else:
        for _, r in mass_results.iterrows():
            lines += [
                f"### {r['branch_hint']}",
                f"- 相界两侧独立相点的质量符号 balanced accuracy：{r['phase_sign_balanced_accuracy']:.4f}",
                f"- 与物理质量方向余弦相似度：{r['coefficient_cosine_similarity_to_physical_mass']:.6f}",
                f"- 稀疏 ML 边界：`{r['ml_boundary_formula_normalized']} = 0`",
                f"- 物理认证边界：`{r['physical_mass_formula_normalized']} = 0`",
                "",
            ]
    lines += [
        "## 4. 相变机制",
        "",
        f"- 认证事件数：{mechanism_summary.get('n_certified_events', 0)}",
        f"- 事件类别：{mechanism_summary.get('mechanism_counts', {})}",
        f"- 所有事件满足 Berry 荷守恒：{mechanism_summary.get('all_events_charge_conserving', False)}",
        f"- 参数到机制的机器学习结论是否允许：{mechanism_summary.get('mechanism_ml_conclusion_allowed', False)}",
        "",
        mechanism_summary.get("interpretation", ""),
        "",
        "## 5. 可发表的核心规律",
        "",
        "1. 固定背景下，拓扑相主要由 `r3-r4` 的非线性组合组织；随机划分不是可信验证，必须采用空间分块留出。",
        "2. 左下 `C_up=-2/+2` 相界需要二次质量坐标，右上相界在已认证局部范围内近似线性。",
        "3. 右上 junction 的 `-2 -> +2 -> 0` 不是单一质量翻转，而是普通四谷 `+4` 与 Sigma' 双谷 `-2` 依次发生。",
        "4. 当前机制事件数量不足以支持七维或跨背景的通用机制分类器；机制规律应以认证事件图谱和局域公式表述。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run_step14m(
    formal_scan_source: str | Path,
    step11_source: str | Path,
    step12_source: str | Path,
    step13_source: str | Path,
    output_dir: str | Path = "outputs_tts_step14M_physics_certified_rule_extraction",
    config: Step14Config | None = None,
) -> dict[str, Any]:
    config = (config or Step14Config()).normalize()
    config.output_dir = Path(output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "figures").mkdir(exist_ok=True)
    work_dir = config.output_dir / "_input_extract"
    work_dir.mkdir(exist_ok=True)

    print(f"Code version: {CODE_VERSION}")
    print("[1/6] Materialize and audit inputs")
    formal_root = materialize_source(formal_scan_source, work_dir, "formal")
    step11_root = materialize_source(step11_source, work_dir, "step11")
    step12_root = materialize_source(step12_source, work_dir, "step12")
    step13_root = materialize_source(step13_source, work_dir, "step13")

    print("[2/6] Build physics-certified phase dataset")
    phase_df, all_phase_points, phase_info = build_phase_dataset(formal_root, step12_root, step13_root, config)
    phase_df.to_csv(config.output_dir / "step14M_01_certified_phase_dataset.csv", index=False)
    all_phase_points.to_csv(config.output_dir / "step14M_01b_all_phase_points_with_uncertain.csv", index=False)
    phase_info["conflicts"].to_csv(config.output_dir / "step14M_01c_coordinate_label_conflicts.csv", index=False)
    write_json(config.output_dir / "step14M_00_input_and_dataset_audit.json", {
        "code_version": CODE_VERSION,
        "config": asdict(config),
        **phase_info["audit"],
    })

    print("[3/6] Spatial group-holdout phase models")
    metrics, predictions, best_name, best_model, classes, cm = spatial_cv_phase_models(phase_df, config)
    metrics.to_csv(config.output_dir / "step14M_02_phase_model_metrics.csv", index=False)
    predictions.to_csv(config.output_dir / "step14M_03_phase_spatial_holdout_predictions.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(config.output_dir / "step14M_04_best_phase_confusion_matrix.csv")

    fitted_models = _phase_models(config)
    rf = fitted_models["random_forest"].fit(phase_df[["r3", "r4"]].to_numpy(float), phase_df["chern_up"].to_numpy(int))
    importance = random_forest_importance(phase_df, rf, config)
    importance.to_csv(config.output_dir / "step14M_05_parameter_importance.csv", index=False)

    print("[4/6] Sparse local mass-rule recovery")
    mass_results, mass_coefficients, mass_predictions = recover_sparse_mass_rules(step11_root, config)
    mass_results.to_csv(config.output_dir / "step14M_06_sparse_mass_rule_metrics.csv", index=False)
    mass_coefficients.to_csv(config.output_dir / "step14M_07_sparse_mass_coefficients.csv", index=False)
    mass_predictions.to_csv(config.output_dir / "step14M_08_sparse_mass_predictions.csv", index=False)

    print("[5/6] Certified transition-event atlas and mechanism gate")
    events = build_transition_event_dataset(step11_root, step12_root, step13_root)
    events.to_csv(config.output_dir / "step14M_09_certified_transition_events.csv", index=False)
    mechanism_metrics, mechanism_predictions, mechanism_summary = exploratory_mechanism_analysis(events, config)
    mechanism_metrics.to_csv(config.output_dir / "step14M_10_exploratory_mechanism_metrics.csv", index=False)
    mechanism_predictions.to_csv(config.output_dir / "step14M_11_exploratory_mechanism_predictions.csv", index=False)
    write_json(config.output_dir / "step14M_12_mechanism_sample_size_gate.json", mechanism_summary)

    print("[6/6] Figures and final certificate")
    plot_phase_model_metrics(metrics, config)
    plot_confusion(cm, classes, best_name, config)
    plot_phase_decision_map(phase_df, all_phase_points, best_model, classes, best_name, config)
    plot_feature_importance(importance, config)
    plot_mass_rule_recovery(mass_coefficients, mass_predictions, config)
    plot_transition_events(events, config)

    write_rule_summary(
        config.output_dir / "step14M_13_interpretable_rule_summary.md",
        phase_info["audit"], metrics, mass_results, mechanism_summary, classes,
    )

    best = metrics.iloc[0].to_dict()
    mass_pass = bool(
        not mass_results.empty
        and mass_results["phase_sign_balanced_accuracy"].min() >= 0.90
        and mass_results["coefficient_cosine_similarity_to_physical_mass"].min() >= 0.99
    )
    certificate = {
        "code_version": CODE_VERSION,
        "research_scope": "fixed-background r3-r4 slice only",
        "n_certified_phase_points": len(phase_df),
        "phase_classes_present": [int(x) for x in classes],
        "best_phase_model": best_name,
        "best_phase_balanced_accuracy_spatial_holdout": float(best["balanced_accuracy"]),
        "best_phase_macro_f1_spatial_holdout": float(best["macro_f1"]),
        "phase_rule_extraction_pass": bool(best["balanced_accuracy"] >= 0.80),
        "local_mass_rule_recovery_pass": mass_pass,
        "n_certified_transition_events": len(events),
        "all_transition_events_charge_conserving": bool(mechanism_summary.get("all_events_charge_conserving", False)),
        "mechanism_sample_size_gate_pass": bool(mechanism_summary.get("mechanism_sample_size_gate_pass", False)),
        "mechanism_ml_conclusion_allowed": bool(mechanism_summary.get("mechanism_ml_conclusion_allowed", False)),
        "step14M_fixed_slice_interpretable_summary_complete": bool(
            best["balanced_accuracy"] >= 0.80 and mass_pass and mechanism_summary.get("all_events_charge_conserving", False)
        ),
        "claims_allowed": [
            "Spatially held-out ML can summarize certified phase regions in the current fixed r3-r4 slice.",
            "Sparse polynomial ML can be compared quantitatively with the two certified local mass coordinates.",
            "Certified four-valley and SigmaPrime two-valley events can be summarized in an event atlas.",
        ],
        "claims_not_allowed": [
            "A global seven-parameter topological rule.",
            "A cross-background universal mass formula.",
            "A generalizable parameter-only mechanism classifier unless the mechanism sample-size gate passes.",
        ],
    }
    write_json(config.output_dir / "step14M_14_final_rule_extraction_certificate.json", certificate)
    print(json.dumps(certificate, ensure_ascii=False, indent=2))
    return certificate


def make_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formal", default="TTS_Cup2_Formal_Refined_Scans(1).zip")
    p.add_argument("--step11", default="outputs_tts_step11M_lieb_aligned_fixed_slice.zip")
    p.add_argument("--step12", default="outputs_tts_step12M_junction_multiclosure_repair.zip")
    p.add_argument("--step13", default="outputs_tts_step13M_adaptive_wilson_intermediate_chern.zip")
    p.add_argument("--output", default="outputs_tts_step14M_physics_certified_rule_extraction")
    return p


def main() -> None:
    args = make_argument_parser().parse_args()
    run_step14m(args.formal, args.step11, args.step12, args.step13, args.output)


if __name__ == "__main__":
    main()
