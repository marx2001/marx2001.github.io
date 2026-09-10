#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Step 06 — 解析高对称质量分支、局域对角谷质量图册与加性 Chern 相图

本步骤承接 Step 04/05，目标不是再次计算 Chern，而是把已经严格确认的
9 个 Chern 跳变组织成可解释的“质量分支图册”：

1. 在 Gamma/M 点利用固定反对角宇称算符把 spin-up 4x4 块严格分解为
   两个 2x2 块，给出四条闭式本征值分支；
2. 自动识别每个高对称临界点究竟是哪两条解析分支发生交换，并构造
   精确的有符号质量 m_Gamma 或 m_M；
3. 对 Sigma/Sigma' 普通动量谷，在临界二带子空间中固定 diabatic 基，
   由于 Hamiltonian 对七个参数线性，得到局域但精确线性的质量坐标
       m_valley(p) = sum_j a_j p_j ;
4. 将单谷 Berry 电荷和质量翻转组合成路径级加性公式
       C_up(p) = C_anchor + sum_i DeltaC_i Theta[m_i(p)] ;
5. 用 Step 04 的全部严格路径 Chern 点验证该公式，而不是用 lambda 位置
   直接硬编码相变；
6. 可选地在 Step 03 全局严格绝缘体上审计 (m_Gamma,m_M) 的信息量，
   明确高对称质量为何不足以单独区分 |C|=2 普通谷相。

重要限定
--------
- Gamma/M 分支公式是全参数空间严格解析公式；
- Sigma/Sigma' 质量是围绕各临界 valley 的局部解析图册，不宣称单一全局
  线性质量能够覆盖全部普通动量谷；
- 本步骤给出的是“分支解析 + 局部图册 + 加性相变网络”，若数据表明多个
  质量方向不共线，代码会明确报告不存在单一全局质量坐标。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import argparse
import json
import math
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import TTS_step01_model_and_label_audit_v2 as core
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未找到 TTS_step01_model_and_label_audit_v2.py。请把 Step 01 v2 脚本与本脚本放在同一目录。"
    ) from exc

EXPECTED_STEP01_VERSION = "TTS_STEP01_V2_20260713"
if getattr(core, "CODE_VERSION", None) != EXPECTED_STEP01_VERSION:
    raise RuntimeError(
        f"Step 01 核心版本不一致：expected={EXPECTED_STEP01_VERSION}, "
        f"loaded={getattr(core, 'CODE_VERSION', None)}"
    )

CODE_VERSION = "TTS_STEP06_V1_20260714"
REDUCED7 = core.REDUCED7.copy()
POINTS = {"Gamma": (0.0, 0.0), "M": (math.pi, math.pi)}


@dataclass
class Step06Config:
    output_dir: Path = Path("outputs_tts_step06_analytic_mass_branch_chern_atlas")
    step4_input: Path | None = None
    step5_input: Path | None = None
    step3_input: Path | None = None
    formula_validation_samples: int = 128
    formula_validation_seed: int = 20260714
    finite_lambda_step: float = 1.0e-5
    branch_degeneracy_tol: float = 1.0e-7
    chart_zero_tol: float = 1.0e-8
    chart_sign_tol: float = 1.0e-10
    path_prediction_zero_tol: float = 1.0e-7
    global_audit: bool = True

    def normalized(self) -> "Step06Config":
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        if self.step4_input is not None:
            self.step4_input = Path(self.step4_input)
        if self.step5_input is not None:
            self.step5_input = Path(self.step5_input)
        if self.step3_input is not None:
            self.step3_input = Path(self.step3_input)
        if self.formula_validation_samples < 4:
            raise ValueError("formula_validation_samples must be >= 4")
        return self


# =============================================================================
# I/O helpers
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


def _find_default(name: str) -> Path:
    candidates = [
        Path.cwd() / name,
        Path.cwd().parent / name,
        Path("/mnt/data") / name,
        Path.cwd() / name.removesuffix(".zip"),
        Path.cwd().parent / name.removesuffix(".zip"),
        Path("/mnt/data") / name.removesuffix(".zip"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到输入 {name}。请把结果 ZIP 放在当前目录或显式设置路径。")


def _read_csv(source: Path, basename: str) -> pd.DataFrame:
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            matches = [n for n in zf.namelist() if Path(n).name == basename]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"{source} 中匹配 {basename!r} 的文件数量为 {len(matches)}: {matches}"
                )
            with zf.open(matches[0]) as fh:
                return pd.read_csv(fh, low_memory=False)
    if source.is_dir():
        matches = list(source.rglob(basename))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{source} 中匹配 {basename!r} 的文件数量为 {len(matches)}: {matches}"
            )
        return pd.read_csv(matches[0], low_memory=False)
    raise FileNotFoundError(source)


def _read_csv_pattern(source: Path, pattern: str) -> pd.DataFrame:
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            matches = [
                n for n in zf.namelist()
                if pattern in Path(n).name and Path(n).suffix.lower() == ".csv"
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"{source} 中匹配 {pattern!r} 的 CSV 数量为 {len(matches)}: {matches}"
                )
            with zf.open(matches[0]) as fh:
                return pd.read_csv(fh, low_memory=False)
    if source.is_dir():
        matches = [p for p in source.rglob("*.csv") if pattern in p.name]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{source} 中匹配 {pattern!r} 的 CSV 数量为 {len(matches)}: {matches}"
            )
        return pd.read_csv(matches[0], low_memory=False)
    raise FileNotFoundError(source)


def load_inputs(config: Step06Config) -> dict[str, pd.DataFrame]:
    step4 = config.step4_input or _find_default(
        "outputs_tts_step04_chern_sector_boundary_valley_tracking.zip"
    )
    step5 = config.step5_input or _find_default(
        "outputs_tts_step05_kp_valley_topological_charge.zip"
    )
    config.step4_input = Path(step4)
    config.step5_input = Path(step5)

    tables = {
        "partners": _read_csv(step4, "step04_01_nearest_trivial_path_partners.csv"),
        "dense": _read_csv(step4, "step04_04_path_dense_gap_scan.csv"),
        "adaptive": _read_csv(step4, "step04_04_path_adaptive_chern_labels.csv"),
        "brackets": _read_csv(step4, "step04_05_transition_brackets.csv"),
        "critical": _read_csv(step5, "step05_01_corrected_spin_critical_points.csv"),
        "valleys": _read_csv(step5, "step05_02_spin_valley_assignment.csv"),
        "kp": _read_csv(step5, "step05_03_kp_summary.csv"),
        "gradients": _read_csv(step5, "step05_03_local_mass_gradients.csv"),
        "charges": _read_csv(step5, "step05_04_valley_topological_charges.csv"),
        "certificates": _read_csv(step5, "step05_05_valley_charge_certificate.csv"),
    }

    if config.global_audit:
        try:
            step3 = config.step3_input or _find_default(
                "outputs_tts_step03_global_sobol_hierarchical_topology_ml.zip"
            )
            config.step3_input = Path(step3)
            tables["step3_physics"] = _read_csv_pattern(
                step3, "step03_02_physics_labels_final__"
            )
        except FileNotFoundError:
            tables["step3_physics"] = pd.DataFrame()
    else:
        tables["step3_physics"] = pd.DataFrame()
    return tables


# =============================================================================
# Parameter path geometry
# =============================================================================


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm < 1.0e-14 or not np.isfinite(norm):
        raise ValueError("invalid parameter vector")
    return v / norm


def slerp(a: Sequence[float], b: Sequence[float], lam: float) -> np.ndarray:
    aa = normalize_vector(a)
    bb = normalize_vector(b)
    dot = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    omega = float(np.arccos(dot))
    lam = float(lam)
    if omega < 1.0e-12:
        return aa.copy()
    if abs(math.pi - omega) < 1.0e-7:
        return normalize_vector((1.0 - lam) * aa + lam * bb)
    return normalize_vector(
        math.sin((1.0 - lam) * omega) / math.sin(omega) * aa
        + math.sin(lam * omega) / math.sin(omega) * bb
    )


def pair_vectors(pair: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([float(pair[f"anchor_{name}"]) for name in REDUCED7], dtype=float)
    b = np.array([float(pair[f"trivial_{name}"]) for name in REDUCED7], dtype=float)
    return a, b


def reduced_on_path(pair: pd.Series, lam: float) -> dict[str, float]:
    a, b = pair_vectors(pair)
    vector = slerp(a, b, float(lam))
    return {name: float(value) for name, value in zip(REDUCED7, vector)}


def raw_from_reduced(reduced: dict[str, float]) -> dict[str, float]:
    return core.raw8_from_reduced7(reduced, e0=0.0)


def spin_block_from_reduced(
    kx: float, ky: float, reduced: dict[str, float], spin: str = "up"
) -> np.ndarray:
    return core.h_spin_block_periodic(
        float(kx), float(ky), raw_from_reduced(reduced), spin
    )


# =============================================================================
# Exact Gamma/M parity decomposition
# =============================================================================


PARITY_J = np.fliplr(np.eye(4, dtype=np.complex128))
EYE4 = np.eye(4, dtype=np.complex128)
_e = np.eye(4, dtype=np.complex128)
PARITY_BASIS = np.column_stack([
    (_e[:, 0] + _e[:, 3]) / math.sqrt(2.0),
    (_e[:, 1] + _e[:, 2]) / math.sqrt(2.0),
    (_e[:, 0] - _e[:, 3]) / math.sqrt(2.0),
    (_e[:, 1] - _e[:, 2]) / math.sqrt(2.0),
])


def parity_blocks(point: str, reduced: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    me, t1, t2, r1, r2, r3, r4 = [float(reduced[x]) for x in REDUCED7]
    if point == "Gamma":
        plus = np.array([
            [me + t1, 2.0 * (r3 + r4)],
            [2.0 * (r3 + r4), -me + t2],
        ], dtype=np.complex128)
        minus = np.array([
            [me - t1, 2.0j * (r1 - r2)],
            [-2.0j * (r1 - r2), -me - t2],
        ], dtype=np.complex128)
    elif point == "M":
        plus = np.array([
            [me + t1, 2.0 * (r4 - r3)],
            [2.0 * (r4 - r3), -me + t2],
        ], dtype=np.complex128)
        minus = np.array([
            [me - t1, -2.0j * (r1 + r2)],
            [2.0j * (r1 + r2), -me - t2],
        ], dtype=np.complex128)
    else:
        raise ValueError(point)
    return plus, minus


def exact_branches(point: str, reduced: dict[str, float]) -> dict[str, float]:
    me, t1, t2, r1, r2, r3, r4 = [float(reduced[x]) for x in REDUCED7]
    cp = 0.5 * (t1 + t2)
    cm = -cp
    dp = me + 0.5 * (t1 - t2)
    dm = me - 0.5 * (t1 - t2)
    if point == "Gamma":
        op = 2.0 * (r3 + r4)
        om = 2.0 * (r1 - r2)
    elif point == "M":
        op = 2.0 * (r4 - r3)
        om = 2.0 * (r1 + r2)
    else:
        raise ValueError(point)
    sp = math.sqrt(dp * dp + op * op)
    sm = math.sqrt(dm * dm + om * om)
    return {
        f"{point}_P+_lower": cp - sp,
        f"{point}_P+_upper": cp + sp,
        f"{point}_P-_lower": cm - sm,
        f"{point}_P-_upper": cm + sm,
    }


def branch_formula_strings(point: str) -> dict[str, str]:
    if point == "Gamma":
        rp = "4*(r3+r4)^2"
        rm = "4*(r1-r2)^2"
    elif point == "M":
        rp = "4*(r4-r3)^2"
        rm = "4*(r1+r2)^2"
    else:
        raise ValueError(point)
    return {
        f"{point}_P+_lower": f"(t1+t2)/2 - sqrt((m_e+(t1-t2)/2)^2 + {rp})",
        f"{point}_P+_upper": f"(t1+t2)/2 + sqrt((m_e+(t1-t2)/2)^2 + {rp})",
        f"{point}_P-_lower": f"-(t1+t2)/2 - sqrt((m_e-(t1-t2)/2)^2 + {rm})",
        f"{point}_P-_upper": f"-(t1+t2)/2 + sqrt((m_e-(t1-t2)/2)^2 + {rm})",
    }


def build_formula_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows = [
        {"point": "Gamma", "parity": "+", "matrix": "[[m_e+t1, 2(r3+r4)], [2(r3+r4), -m_e+t2]]"},
        {"point": "Gamma", "parity": "-", "matrix": "[[m_e-t1, 2i(r1-r2)], [-2i(r1-r2), -m_e-t2]]"},
        {"point": "M", "parity": "+", "matrix": "[[m_e+t1, 2(r4-r3)], [2(r4-r3), -m_e+t2]]"},
        {"point": "M", "parity": "-", "matrix": "[[m_e-t1, -2i(r1+r2)], [2i(r1+r2), -m_e-t2]]"},
    ]
    branch_rows = []
    for point in ("Gamma", "M"):
        for label, formula in branch_formula_strings(point).items():
            branch_rows.append({"point": point, "branch_label": label, "formula": formula})
    return pd.DataFrame(block_rows), pd.DataFrame(branch_rows)


def validate_exact_formulas(config: Step06Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.formula_validation_seed)
    rows: list[dict[str, Any]] = []
    for index in range(config.formula_validation_samples):
        vector = normalize_vector(rng.normal(size=7))
        reduced = {name: float(value) for name, value in zip(REDUCED7, vector)}
        for point, (kx, ky) in POINTS.items():
            h = spin_block_from_reduced(kx, ky, reduced, "up")
            numerical = np.linalg.eigvalsh(h)
            analytic = np.sort(np.array(list(exact_branches(point, reduced).values())))
            transformed = PARITY_BASIS.conj().T @ h @ PARITY_BASIS
            offblock = transformed[:2, 2:]
            plus, minus = parity_blocks(point, reduced)
            block_error = max(
                float(np.max(np.abs(transformed[:2, :2] - plus))),
                float(np.max(np.abs(transformed[2:, 2:] - minus))),
            )
            rows.append({
                "sample_index": index,
                "point": point,
                "eigenvalue_max_abs_error": float(np.max(np.abs(numerical - analytic))),
                "parity_commutator_norm": float(np.linalg.norm(PARITY_J @ h - h @ PARITY_J)),
                "off_block_norm": float(np.linalg.norm(offblock)),
                "explicit_block_max_abs_error": block_error,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Diabatic local mass charts
# =============================================================================


def pauli_decompose(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    m = np.asarray(matrix, dtype=np.complex128)
    d0 = 0.5 * float((m[0, 0] + m[1, 1]).real)
    d = np.array([
        float(m[0, 1].real),
        float(-m[0, 1].imag),
        0.5 * float((m[0, 0] - m[1, 1]).real),
    ])
    return d0, d


def crossing_basis(
    pair: pd.Series,
    lam: float,
    kx: float,
    ky: float,
    spin: str,
    config: Step06Config,
) -> tuple[np.ndarray, np.ndarray]:
    reduced = reduced_on_path(pair, lam)
    h0 = spin_block_from_reduced(kx, ky, reduced, spin)
    eig, vec = np.linalg.eigh(h0)
    u = vec[:, 1:3]
    step = config.finite_lambda_step
    hp = spin_block_from_reduced(kx, ky, reduced_on_path(pair, lam + step), spin)
    hm = spin_block_from_reduced(kx, ky, reduced_on_path(pair, lam - step), spin)
    derivative = (hp - hm) / (2.0 * step)
    projected = u.conj().T @ derivative @ u
    _, rotation = np.linalg.eigh(projected)
    u = u @ rotation
    return eig, u


def exact_linear_chart_coefficients(
    u: np.ndarray,
    kx: float,
    ky: float,
    spin: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    mass: dict[str, float] = {}
    d0c: dict[str, float] = {}
    dxc: dict[str, float] = {}
    dyc: dict[str, float] = {}
    for parameter in REDUCED7:
        unit = {name: 0.0 for name in REDUCED7}
        unit[parameter] = 1.0
        h = spin_block_from_reduced(kx, ky, unit, spin)
        projected = u.conj().T @ h @ u
        d0, d = pauli_decompose(projected)
        d0c[parameter] = float(d0)
        dxc[parameter] = float(d[0])
        dyc[parameter] = float(d[1])
        mass[parameter] = float(d[2])
    return mass, d0c, dxc, dyc


def dot_formula(coefficients: dict[str, float], precision: int = 8) -> str:
    terms = []
    for parameter in REDUCED7:
        coefficient = float(coefficients.get(parameter, 0.0))
        terms.append(f"{coefficient:+.{precision}g}*{parameter}")
    return " ".join(terms)


def evaluate_linear(coefficients: dict[str, float], reduced: dict[str, float]) -> float:
    return float(sum(float(coefficients[p]) * float(reduced[p]) for p in REDUCED7))


def identify_high_symmetry_branch_pair(point: str, reduced: dict[str, float]) -> tuple[str, str, float]:
    branches = exact_branches(point, reduced)
    ordered = sorted(branches.items(), key=lambda item: item[1])
    label_a, energy_a = ordered[1]
    label_b, energy_b = ordered[2]
    return label_a, label_b, float(energy_b - energy_a)


def evaluate_branch_difference(
    point: str,
    label_a: str,
    label_b: str,
    reduced: dict[str, float],
) -> float:
    branches = exact_branches(point, reduced)
    return float(branches[label_b] - branches[label_a])


def choose_orientation(left: float, right: float, tol: float) -> tuple[int, int]:
    if left < -tol and right > tol:
        return 1, 1
    if left > tol and right < -tol:
        return -1, 1
    # Keep a deterministic orientation even when a coarse bracket is too close.
    return (1 if right >= left else -1), 0


# =============================================================================
# Build transition mass atlas
# =============================================================================


def build_transition_atlas(
    tables: dict[str, pd.DataFrame], config: Step06Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partners = tables["partners"].set_index("path_id", drop=False)
    critical = tables["critical"].copy()
    kp = tables["kp"].copy()
    gradients = tables["gradients"].copy()
    charges = tables["charges"].copy()
    certificates = tables["certificates"].set_index("transition_id", drop=False)

    atlas_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []

    for _, closure in critical.sort_values(["path_id", "critical_lambda"]).iterrows():
        transition_id = str(closure["transition_id"])
        path_id = str(closure["path_id"])
        pair = partners.loc[path_id]
        lam_c = float(closure["critical_lambda"])
        lam_l = float(closure["lambda_left"])
        lam_r = float(closure["lambda_right"])
        p_c = {name: float(closure[name]) for name in REDUCED7}
        p_l = reduced_on_path(pair, lam_l)
        p_r = reduced_on_path(pair, lam_r)
        p_anchor = reduced_on_path(pair, 0.0)
        region = str(closure["critical_k_region"])
        delta_c = int(closure["delta_chern_up"])
        cert = certificates.loc[transition_id]

        common = {
            "transition_id": transition_id,
            "path_id": path_id,
            "chern_left": int(closure["chern_left"]),
            "chern_right": int(closure["chern_right"]),
            "delta_chern_up": delta_c,
            "critical_lambda": lam_c,
            "critical_kx": float(closure["critical_kx"]),
            "critical_ky": float(closure["critical_ky"]),
            "critical_k_region": region,
            "n_active_spin_up_valleys": int(cert["n_active_spin_up_valleys"]),
            "berry_charge_sum_up": int(round(float(cert["berry_charge_sum_up"]))),
            "mechanism_certificate_pass": int(cert["mechanism_certificate_pass"]),
        }

        if region in ("Gamma", "M"):
            point = region
            label_a, label_b, residual = identify_high_symmetry_branch_pair(point, p_c)
            raw_l = evaluate_branch_difference(point, label_a, label_b, p_l)
            raw_c = evaluate_branch_difference(point, label_a, label_b, p_c)
            raw_r = evaluate_branch_difference(point, label_a, label_b, p_r)
            raw_anchor = evaluate_branch_difference(point, label_a, label_b, p_anchor)
            orientation, sign_flip = choose_orientation(raw_l, raw_r, config.chart_sign_tol)
            formulas = branch_formula_strings(point)
            exact_formula = f"{orientation:+d}*(({formulas[label_b]}) - ({formulas[label_a]}))"
            atlas_rows.append({
                **common,
                "chart_id": f"{transition_id}__exact_{point}",
                "chart_type": "exact_high_symmetry_branch",
                "mass_point": point,
                "mass_branch_a": label_a,
                "mass_branch_b": label_b,
                "mass_orientation": orientation,
                "mass_formula": exact_formula,
                "raw_mass_left": raw_l,
                "raw_mass_critical": raw_c,
                "raw_mass_right": raw_r,
                "oriented_mass_anchor": orientation * raw_anchor,
                "oriented_mass_left": orientation * raw_l,
                "oriented_mass_critical": orientation * raw_c,
                "oriented_mass_right": orientation * raw_r,
                "mass_sign_flip_verified": sign_flip,
                "critical_mass_abs_residual": abs(raw_c),
                "step05_gradient_cosine": np.nan,
                "offdiagonal_norm_at_critical": 0.0,
            })
            branch_rows.append({
                "transition_id": transition_id,
                "point": point,
                "branch_a": label_a,
                "branch_b": label_b,
                "branch_a_energy": exact_branches(point, p_c)[label_a],
                "branch_b_energy": exact_branches(point, p_c)[label_b],
                "branch_gap_abs": residual,
                "branch_a_formula": formulas[label_a],
                "branch_b_formula": formulas[label_b],
            })
        else:
            # One representative active spin-up valley is enough to define the chart;
            # the symmetry partner has the same mass coefficients.
            candidates = kp[(kp["transition_id"].astype(str) == transition_id) & (kp["spin"] == "up")].copy()
            if candidates.empty:
                raise RuntimeError(f"No spin-up k.p row for {transition_id}")
            rep = candidates.sort_values(["critical_kx", "critical_ky"]).iloc[0]
            kx = float(rep["critical_kx"])
            ky = float(rep["critical_ky"])
            eig, u = crossing_basis(pair, lam_c, kx, ky, "up", config)
            mass, d0c, dxc, dyc = exact_linear_chart_coefficients(u, kx, ky, "up")
            raw_l = evaluate_linear(mass, p_l)
            raw_c = evaluate_linear(mass, p_c)
            raw_r = evaluate_linear(mass, p_r)
            raw_anchor = evaluate_linear(mass, p_anchor)
            orientation, sign_flip = choose_orientation(raw_l, raw_r, config.chart_sign_tol)
            oriented = {name: orientation * value for name, value in mass.items()}

            grad = gradients[
                (gradients["transition_id"].astype(str) == transition_id)
                & (gradients["valley_id"].astype(str) == str(rep["valley_id"]))
                & (gradients["spin"] == "up")
            ].set_index("parameter")
            grad_vector = np.array([float(grad.loc[p, "mass_gradient"]) for p in REDUCED7])
            chart_vector = np.array([float(mass[p]) for p in REDUCED7])
            cosine = float(np.dot(grad_vector, chart_vector) / (
                np.linalg.norm(grad_vector) * np.linalg.norm(chart_vector)
            ))

            hcrit = spin_block_from_reduced(kx, ky, p_c, "up")
            projected = u.conj().T @ hcrit @ u
            _, dcrit = pauli_decompose(projected)
            offdiag = float(np.linalg.norm(dcrit[:2]))

            atlas_rows.append({
                **common,
                "chart_id": f"{transition_id}__local_{region}",
                "chart_type": "local_exact_linear_diabatic_mass",
                "mass_point": region,
                "mass_branch_a": "critical_subspace_state_0",
                "mass_branch_b": "critical_subspace_state_1",
                "mass_orientation": orientation,
                "mass_formula": dot_formula(oriented),
                "raw_mass_left": raw_l,
                "raw_mass_critical": raw_c,
                "raw_mass_right": raw_r,
                "oriented_mass_anchor": orientation * raw_anchor,
                "oriented_mass_left": orientation * raw_l,
                "oriented_mass_critical": orientation * raw_c,
                "oriented_mass_right": orientation * raw_r,
                "mass_sign_flip_verified": sign_flip,
                "critical_mass_abs_residual": abs(raw_c),
                "step05_gradient_cosine": cosine,
                "offdiagonal_norm_at_critical": offdiag,
            })
            for parameter in REDUCED7:
                coefficient_rows.append({
                    "transition_id": transition_id,
                    "chart_id": f"{transition_id}__local_{region}",
                    "critical_k_region": region,
                    "critical_kx": kx,
                    "critical_ky": ky,
                    "parameter": parameter,
                    "raw_mass_coefficient": mass[parameter],
                    "oriented_mass_coefficient": oriented[parameter],
                    "d0_coefficient": d0c[parameter],
                    "dx_coefficient": dxc[parameter],
                    "dy_coefficient": dyc[parameter],
                    "step05_mass_gradient": float(grad.loc[parameter, "mass_gradient"]),
                })

    atlas = pd.DataFrame(atlas_rows).sort_values(["path_id", "critical_lambda"]).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_rows)
    branches = pd.DataFrame(branch_rows)
    return atlas, coefficients, branches


# =============================================================================
# Chart evaluators and additive Chern prediction
# =============================================================================


def coefficient_map(coefficients: pd.DataFrame, transition_id: str) -> dict[str, float]:
    subset = coefficients[coefficients["transition_id"].astype(str) == str(transition_id)]
    return {
        str(row["parameter"]): float(row["oriented_mass_coefficient"])
        for _, row in subset.iterrows()
    }


def evaluate_oriented_chart(
    chart: pd.Series,
    reduced: dict[str, float],
    coefficients: pd.DataFrame,
) -> float:
    if chart["chart_type"] == "exact_high_symmetry_branch":
        raw = evaluate_branch_difference(
            str(chart["mass_point"]),
            str(chart["mass_branch_a"]),
            str(chart["mass_branch_b"]),
            reduced,
        )
        return float(chart["mass_orientation"]) * raw
    coeff = coefficient_map(coefficients, str(chart["transition_id"]))
    return evaluate_linear(coeff, reduced)


def predict_path_chern(
    tables: dict[str, pd.DataFrame],
    atlas: pd.DataFrame,
    coefficients: pd.DataFrame,
    config: Step06Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partners = tables["partners"].set_index("path_id", drop=False)
    adaptive = tables["adaptive"].copy()
    dense = tables["dense"].copy()

    adaptive_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    formula_rows: list[dict[str, Any]] = []

    for path_id, charts in atlas.groupby("path_id", sort=False):
        charts = charts.sort_values("critical_lambda").reset_index(drop=True)
        pair = partners.loc[path_id]
        anchor_chern = int(pair["anchor_sector"])
        formula_terms = []
        for _, chart in charts.iterrows():
            formula_terms.append(
                f"({int(chart['delta_chern_up']):+d})*Theta({chart['chart_id']})"
            )
        formula_rows.append({
            "path_id": path_id,
            "anchor_chern": anchor_chern,
            "n_transition_charts": len(charts),
            "additive_formula": f"C_up = {anchor_chern:+d} " + " ".join(formula_terms),
            "transition_ids": ";".join(charts["transition_id"].astype(str)),
        })

        def one_prediction(reduced: dict[str, float]) -> tuple[int, list[float], list[int]]:
            predicted = anchor_chern
            masses: list[float] = []
            crossed_flags: list[int] = []
            for _, chart in charts.iterrows():
                mass = evaluate_oriented_chart(chart, reduced, coefficients)
                masses.append(mass)
                crossed = int(mass > config.path_prediction_zero_tol)
                crossed_flags.append(crossed)
                if crossed:
                    predicted += int(chart["delta_chern_up"])
            return int(predicted), masses, crossed_flags

        a_subset = adaptive[adaptive["path_id"].astype(str) == str(path_id)]
        for _, row in a_subset.iterrows():
            reduced = {name: float(row[name]) for name in REDUCED7}
            pred, masses, flags = one_prediction(reduced)
            actual = int(round(float(row["chern_up_int"]))) if np.isfinite(row["chern_up_int"]) else None
            record = {
                "path_id": path_id,
                "path_point_id": row["path_point_id"],
                "path_index": int(row["path_index"]),
                "lambda": float(row["lambda"]),
                "actual_chern_up": actual,
                "has_actual_chern": int(actual is not None),
                "predicted_chern_up": pred,
                "prediction_match": int(actual == pred) if actual is not None else np.nan,
                "minimum_abs_chart_mass": float(min(abs(x) for x in masses)),
            }
            for j, (mass, flag) in enumerate(zip(masses, flags)):
                record[f"chart{j:02d}_oriented_mass"] = mass
                record[f"chart{j:02d}_crossed"] = flag
            adaptive_rows.append(record)

        d_subset = dense[dense["path_id"].astype(str) == str(path_id)]
        for _, row in d_subset.iterrows():
            reduced = {name: float(row[name]) for name in REDUCED7}
            pred, masses, flags = one_prediction(reduced)
            record = {
                "path_id": path_id,
                "path_point_id": row["path_point_id"],
                "path_index": int(row["path_index"]),
                "lambda": float(row["lambda"]),
                "predicted_chern_up": pred,
                "min_direct_gap": float(row["min_direct_gap"]),
                "indirect_gap": float(row["indirect_gap"]),
                "minimum_abs_chart_mass": float(min(abs(x) for x in masses)),
            }
            for j, (mass, flag) in enumerate(zip(masses, flags)):
                record[f"chart{j:02d}_oriented_mass"] = mass
                record[f"chart{j:02d}_crossed"] = flag
            dense_rows.append(record)

    adaptive_pred = pd.DataFrame(adaptive_rows)
    dense_pred = pd.DataFrame(dense_rows)
    formulas = pd.DataFrame(formula_rows)
    metrics_rows = []
    valid_all = adaptive_pred[adaptive_pred["has_actual_chern"].eq(1)].copy()
    for path_id, group_all in adaptive_pred.groupby("path_id", sort=False):
        group = group_all[group_all["has_actual_chern"].eq(1)]
        metrics_rows.append({
            "path_id": path_id,
            "n_strict_chern_points": len(group),
            "n_unresolved_points_excluded": len(group_all) - len(group),
            "n_correct": int(group["prediction_match"].sum()) if len(group) else 0,
            "accuracy": float(group["prediction_match"].mean()) if len(group) else np.nan,
            "minimum_abs_chart_mass_over_strict_points": float(group["minimum_abs_chart_mass"].min()) if len(group) else np.nan,
        })
    metrics = pd.DataFrame(metrics_rows)
    metrics = pd.concat([
        metrics,
        pd.DataFrame([{
            "path_id": "ALL_PATHS",
            "n_strict_chern_points": len(valid_all),
            "n_unresolved_points_excluded": len(adaptive_pred) - len(valid_all),
            "n_correct": int(valid_all["prediction_match"].sum()),
            "accuracy": float(valid_all["prediction_match"].mean()),
            "minimum_abs_chart_mass_over_strict_points": float(valid_all["minimum_abs_chart_mass"].min()),
        }])
    ], ignore_index=True)
    return adaptive_pred, dense_pred, formulas, metrics


# =============================================================================
# Transition network and global high-symmetry audit
# =============================================================================


def build_transition_network(atlas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in atlas.groupby(
        ["chern_left", "chern_right", "critical_k_region", "delta_chern_up"],
        dropna=False,
    ):
        left, right, region, delta = keys
        rows.append({
            "chern_left": int(left),
            "chern_right": int(right),
            "delta_chern_up": int(delta),
            "critical_k_region": str(region),
            "n_observed_transitions": len(group),
            "transition_ids": ";".join(group["transition_id"].astype(str)),
            "mean_active_spin_up_valleys": float(group["n_active_spin_up_valleys"].mean()),
            "all_mechanism_certificates_pass": int(group["mechanism_certificate_pass"].eq(1).all()),
        })
    return pd.DataFrame(rows).sort_values(["chern_left", "chern_right", "critical_k_region"])


def canonical_high_symmetry_masses(reduced: dict[str, float]) -> tuple[float, float]:
    g = exact_branches("Gamma", reduced)
    m = exact_branches("M", reduced)
    m_gamma = g["Gamma_P+_upper"] - g["Gamma_P-_lower"]
    m_m = m["M_P-_upper"] - m["M_P+_lower"]
    return float(m_gamma), float(m_m)


def global_high_symmetry_audit(physics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if physics.empty:
        return pd.DataFrame(), pd.DataFrame()
    strict = physics.copy()
    if "is_strict_insulator" in strict:
        strict = strict[strict["is_strict_insulator"].fillna(0).astype(int).eq(1)]
    strict = strict[np.isfinite(strict["chern_up_int"])].copy()
    rows = []
    for _, row in strict.iterrows():
        reduced = {name: float(row[name]) for name in REDUCED7}
        mg, mm = canonical_high_symmetry_masses(reduced)
        rows.append({
            "sample_id": row.get("sample_id", ""),
            "sample_source": row.get("sample_source", ""),
            "chern_up_int": int(round(float(row["chern_up_int"]))),
            "phase_label": row.get("phase_label", ""),
            "m_Gamma_exact": mg,
            "m_M_exact": mm,
            "sign_m_Gamma": int(np.sign(mg)),
            "sign_m_M": int(np.sign(mm)),
            **reduced,
        })
    audit = pd.DataFrame(rows)
    ambiguity_rows = []
    for (sg, sm), group in audit.groupby(["sign_m_Gamma", "sign_m_M"]):
        sectors = sorted(group["chern_up_int"].unique().tolist())
        ambiguity_rows.append({
            "sign_m_Gamma": int(sg),
            "sign_m_M": int(sm),
            "n_samples": len(group),
            "chern_sectors": ";".join(str(int(x)) for x in sectors),
            "n_distinct_chern_sectors": len(sectors),
            "high_symmetry_signs_are_unique_classifier": int(len(sectors) == 1),
        })
    return audit, pd.DataFrame(ambiguity_rows)


# =============================================================================
# Figures
# =============================================================================


def plot_formula_validation(validation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data = [
        validation[validation["point"] == point]["eigenvalue_max_abs_error"].to_numpy()
        for point in ("Gamma", "M")
    ]
    ax.boxplot(data, tick_labels=["Gamma", "M"], showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("max |E_exact - E_numeric|")
    ax.set_title("Exact parity-branch validation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mass_charts(dense_pred: pd.DataFrame, atlas: pd.DataFrame, path: Path) -> None:
    paths = list(dense_pred["path_id"].drop_duplicates())
    ncols = 2
    nrows = int(math.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4 * nrows), squeeze=False)
    for ax, path_id in zip(axes.flat, paths):
        group = dense_pred[dense_pred["path_id"] == path_id].sort_values("lambda")
        charts = atlas[atlas["path_id"] == path_id].sort_values("critical_lambda")
        mass_cols = sorted([c for c in group.columns if c.endswith("_oriented_mass") and c.startswith("chart")])
        for index, col in enumerate(mass_cols):
            label = str(charts.iloc[index]["critical_k_region"]) if index < len(charts) else col
            ax.plot(group["lambda"], group[col], label=label)
            if index < len(charts):
                ax.axvline(float(charts.iloc[index]["critical_lambda"]), linestyle="--", linewidth=0.9)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(path_id, fontsize=9)
        ax.set_xlabel("path lambda")
        ax.set_ylabel("oriented mass")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_path_chern_prediction(
    dense_pred: pd.DataFrame,
    adaptive_pred: pd.DataFrame,
    atlas: pd.DataFrame,
    path: Path,
) -> None:
    paths = list(dense_pred["path_id"].drop_duplicates())
    ncols = 2
    nrows = int(math.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    for ax, path_id in zip(axes.flat, paths):
        dense = dense_pred[dense_pred["path_id"] == path_id].sort_values("lambda")
        strict = adaptive_pred[adaptive_pred["path_id"] == path_id].sort_values("lambda")
        charts = atlas[atlas["path_id"] == path_id]
        ax.step(dense["lambda"], dense["predicted_chern_up"], where="mid", label="mass-atlas prediction")
        strict_plot = strict[np.isfinite(strict["actual_chern_up"])].copy()
        ax.scatter(strict_plot["lambda"], strict_plot["actual_chern_up"], marker="o", s=22, label="strict Chern")
        for value in charts["critical_lambda"]:
            ax.axvline(float(value), linestyle="--", linewidth=0.8)
        ax.set_title(path_id, fontsize=9)
        ax.set_xlabel("path lambda")
        ax.set_ylabel("C_up")
        tick_values = dense["predicted_chern_up"].astype(int).tolist()
        tick_values += strict_plot["actual_chern_up"].astype(int).tolist()
        ax.set_yticks(sorted(set(tick_values)))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_transition_network(network: pd.DataFrame, path: Path) -> None:
    nodes = sorted(set(network["chern_left"].astype(int)) | set(network["chern_right"].astype(int)))
    positions = {node: (i, 0.0) for i, node in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for node, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.18, fill=False, linewidth=1.8)
        ax.add_patch(circle)
        ax.text(x, y, f"C={node:+d}", ha="center", va="center")
    offsets: dict[tuple[int, int], int] = {}
    for _, row in network.iterrows():
        left, right = int(row["chern_left"]), int(row["chern_right"])
        key = (left, right)
        count = offsets.get(key, 0)
        offsets[key] = count + 1
        x1, _ = positions[left]
        x2, _ = positions[right]
        rad = 0.25 + 0.16 * count
        ax.annotate(
            "",
            xy=(x2 - 0.18 * np.sign(x2 - x1), 0.0),
            xytext=(x1 + 0.18 * np.sign(x2 - x1), 0.0),
            arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={rad}"),
        )
        ax.text(
            0.5 * (x1 + x2),
            0.32 + 0.18 * count,
            f"{row['critical_k_region']}  ΔC={int(row['delta_chern_up']):+d}",
            ha="center", fontsize=8,
        )
    ax.set_xlim(-0.6, len(nodes) - 0.4)
    ax.set_ylim(-0.6, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("TTS additive Chern transition network")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_global_audit(audit: pd.DataFrame, path: Path) -> None:
    if audit.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for sector, group in audit.groupby("chern_up_int"):
        ax.scatter(group["m_Gamma_exact"], group["m_M_exact"], s=20, alpha=0.75, label=f"C={int(sector):+d}")
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("m_Gamma exact branch difference")
    ax.set_ylabel("m_M exact branch difference")
    ax.set_title("Global strict insulators: high-symmetry mass audit")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================


def run_step06(config: Step06Config) -> dict[str, Any]:
    config = config.normalized()
    started = time.time()
    out = config.output_dir
    figures = out / "figures"

    print("[1/8] Load Step 04/05 strict transition evidence")
    tables = load_inputs(config)
    print(f"      transitions = {len(tables['critical'])}")

    print("[2/8] Derive and validate exact Gamma/M parity branches")
    blocks, branch_formulas = build_formula_tables()
    validation = validate_exact_formulas(config)
    atomic_write_csv(blocks, out / "step06_01_high_symmetry_parity_blocks.csv")
    atomic_write_csv(branch_formulas, out / "step06_01_high_symmetry_branch_formulas.csv")
    atomic_write_csv(validation, out / "step06_01_exact_formula_validation.csv")

    print("[3/8] Build high-symmetry and generic-valley mass charts")
    atlas, coefficients, branch_ids = build_transition_atlas(tables, config)
    atomic_write_csv(atlas, out / "step06_02_transition_mass_atlas.csv")
    atomic_write_csv(coefficients, out / "step06_02_generic_valley_mass_coefficients.csv")
    atomic_write_csv(branch_ids, out / "step06_02_high_symmetry_branch_identification.csv")

    print("[4/8] Construct additive Chern formulas on all Step 04 paths")
    adaptive_pred, dense_pred, path_formulas, metrics = predict_path_chern(
        tables, atlas, coefficients, config
    )
    atomic_write_csv(path_formulas, out / "step06_03_path_additive_chern_formulas.csv")
    atomic_write_csv(adaptive_pred, out / "step06_03_strict_path_chern_prediction.csv")
    atomic_write_csv(dense_pred, out / "step06_03_dense_path_mass_phase_map.csv")
    atomic_write_csv(metrics, out / "step06_03_path_prediction_metrics.csv")

    print("[5/8] Build branch-resolved Chern transition network")
    network = build_transition_network(atlas)
    atomic_write_csv(network, out / "step06_04_additive_chern_transition_network.csv")

    print("[6/8] Audit global information carried by m_Gamma and m_M")
    global_audit, ambiguity = global_high_symmetry_audit(tables["step3_physics"])
    if not global_audit.empty:
        atomic_write_csv(global_audit, out / "step06_05_global_high_symmetry_mass_audit.csv")
        atomic_write_csv(ambiguity, out / "step06_05_high_symmetry_sign_ambiguity.csv")

    print("[7/8] Generate figures")
    plot_formula_validation(validation, figures / "step06_exact_formula_validation.png")
    plot_mass_charts(dense_pred, atlas, figures / "step06_transition_mass_charts.png")
    plot_path_chern_prediction(
        dense_pred, adaptive_pred, atlas, figures / "step06_additive_chern_path_validation.png"
    )
    plot_transition_network(network, figures / "step06_additive_chern_network.png")
    if not global_audit.empty:
        plot_global_audit(global_audit, figures / "step06_global_high_symmetry_mass_audit.png")

    print("[8/8] Save summary and certificates")
    high_sym = atlas[atlas["chart_type"] == "exact_high_symmetry_branch"]
    generic = atlas[atlas["chart_type"] == "local_exact_linear_diabatic_mass"]
    all_accuracy = float(metrics.loc[metrics["path_id"] == "ALL_PATHS", "accuracy"].iloc[0])
    formula_max_error = float(validation["eigenvalue_max_abs_error"].max())
    single_global_mass_supported = False
    if not coefficients.empty:
        vectors = []
        for _, group in coefficients.groupby("transition_id"):
            v = group.set_index("parameter").loc[REDUCED7, "oriented_mass_coefficient"].to_numpy(float)
            vectors.append(v / np.linalg.norm(v))
        if len(vectors) >= 2:
            similarities = [abs(float(np.dot(a, b))) for i, a in enumerate(vectors) for b in vectors[i + 1:]]
            single_global_mass_supported = bool(min(similarities) > 0.95)

    certificate = pd.DataFrame([{
        "exact_high_symmetry_formula_max_error": formula_max_error,
        "all_high_symmetry_branch_crossings_resolved": int(len(high_sym) == 4),
        "all_generic_charts_sign_flip": int(generic["mass_sign_flip_verified"].eq(1).all()),
        "all_critical_mass_residual_below_tol": int(atlas["critical_mass_abs_residual"].lt(config.branch_degeneracy_tol).all()),
        "all_step05_mechanism_certificates_pass": int(atlas["mechanism_certificate_pass"].eq(1).all()),
        "strict_path_additive_chern_accuracy": all_accuracy,
        "all_strict_path_chern_points_matched": int(all_accuracy == 1.0),
        "single_global_generic_mass_coordinate_supported": int(single_global_mass_supported),
        "final_certificate_pass": int(
            formula_max_error < 1.0e-12
            and atlas["mass_sign_flip_verified"].eq(1).all()
            and atlas["mechanism_certificate_pass"].eq(1).all()
            and all_accuracy == 1.0
        ),
    }])
    atomic_write_csv(certificate, out / "step06_06_analytic_mass_chern_certificate.csv")

    summary = {
        "code_version": CODE_VERSION,
        "step01_version": core.CODE_VERSION,
        "step4_input": str(config.step4_input),
        "step5_input": str(config.step5_input),
        "step3_input": str(config.step3_input) if config.step3_input else None,
        "n_transition_charts": len(atlas),
        "n_exact_high_symmetry_charts": len(high_sym),
        "n_generic_valley_charts": len(generic),
        "n_strict_path_chern_points": len(adaptive_pred),
        "strict_path_additive_chern_accuracy": all_accuracy,
        "exact_formula_max_abs_error": formula_max_error,
        "single_global_generic_mass_coordinate_supported": single_global_mass_supported,
        "final_certificate_pass": int(certificate.iloc[0]["final_certificate_pass"]),
        "elapsed_seconds": time.time() - started,
        "configuration": asdict(config),
    }
    atomic_write_json(summary, out / "step06_07_run_summary.json")
    print(f"      certificate pass = {summary['final_certificate_pass']}")
    print(f"      strict path accuracy = {all_accuracy:.6f}")
    print(f"      elapsed = {summary['elapsed_seconds']:.1f} s")
    return {
        "summary": summary,
        "atlas": atlas,
        "certificate": certificate,
        "metrics": metrics,
        "network": network,
    }


# =============================================================================
# CLI
# =============================================================================


def build_config_from_args(args: argparse.Namespace) -> Step06Config:
    return Step06Config(
        output_dir=Path(args.output_dir),
        step4_input=Path(args.step4_input) if args.step4_input else None,
        step5_input=Path(args.step5_input) if args.step5_input else None,
        step3_input=Path(args.step3_input) if args.step3_input else None,
        formula_validation_samples=24 if args.quick else args.formula_validation_samples,
        global_audit=not args.skip_global_audit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs_tts_step06_analytic_mass_branch_chern_atlas",
    )
    parser.add_argument("--step4-input", default=None)
    parser.add_argument("--step5-input", default=None)
    parser.add_argument("--step3-input", default=None)
    parser.add_argument("--formula-validation-samples", type=int, default=128)
    parser.add_argument("--skip-global-audit", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_step06(build_config_from_args(args))
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
