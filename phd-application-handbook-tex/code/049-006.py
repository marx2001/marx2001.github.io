#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FES Step 07 — 周期规范下的解析 Jacobian、局域 Chern 贡献与模型级 no-go

本步骤完成 Step 06 尚缺少的符号代数环节：

1. 从 fes 自旋向上 4×4 周期规范 Hamiltonian 出发；
2. 在 Γ/M 点进行奇偶基变换，显式分离 A 与 B 两个 2×2 分支；
3. 写出 A− 与 B+ 的解析本征矢；
4. 解析计算 <A−|∂kx H|B+> 与 <A−|∂ky H|B+>；
5. 因式分解 Γ/M 的两带 Jacobian；
6. 证明两个谷的局域下占据带 Chern 贡献为

       C_Gamma = sgn(r1*r2*mu_Gamma)/2
       C_M     = sgn(r1*r2*mu_M)/2

   从而

       C_up = sgn(r1*r2)/2 *
              [sgn(mu_Gamma)+sgn(mu_M)].

7. 与 Step 05 的精确条件性 no-go 合并：

       C_up != 0
       => mu_Gamma 与 mu_M 同号
       => B+ 在 Γ/M 间发生 partner switching
       => E_g^ind <= -4|r1^2-r2^2|/(B1+B2) <= 0.

注意
----
这一步给出显式符号推导。全局公式还使用一个已被 Steps 02–06
数值验证的模型事实：在所研究的半填充、平衡自旋、直接开隙扇区内，
拓扑变化由 Γ/M 的 A−–B+ 闭隙控制。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
import argparse
import hashlib
import json
import math
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

import FES_step02_sobol_scan as core
import FES_step05_analytic_no_go as step05

CODE_VERSION = "FES_STEP07_SYMBOLIC_JACOBIAN_V1_20260713"
RESULT_SCHEMA_VERSION = "fes_step07_symbolic_jacobian_schema_v1"

PARAMS5 = ("m_e", "t1", "t2", "r1", "r2")


@dataclass
class Step07Config:
    step05_input: str = "outputs_fes6_step05_no_go.zip"
    step06_input: str = "outputs_fes6_step06_kp.zip"
    output_dir: str = "outputs_fes6_step07_symbolic"
    finite_difference_step: float = 1.0e-6
    symbolic_timeout_note: str = "SymPy expressions are factorized without external CAS."
    overwrite: bool = False
    test_mode: bool = False


# =============================================================================
# 1. I/O
# =============================================================================

def script_sha256() -> str | None:
    if "__file__" not in globals():
        return None
    p = Path(__file__).resolve()
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def prepare_output(config: Step07Config) -> Path:
    out = Path(config.output_dir).expanduser().resolve()
    if out.exists() and config.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _find_root(root: Path, required_name: str) -> Path:
    direct = root / required_name
    if direct.exists():
        return root
    found = list(root.rglob(required_name))
    if not found:
        raise FileNotFoundError(f"Cannot find {required_name} under {root}")
    return sorted((p.parent for p in found), key=lambda x: len(x.parts))[0]


def locate_result(input_path: str | Path, output: Path, required_name: str, tag: str) -> Path:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return _find_root(path, required_name)
    if path.suffix.lower() != ".zip":
        raise ValueError(f"{tag} input must be a directory or ZIP")
    extract = output / f"_{tag}_extracted"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract)
    return _find_root(extract, required_name)


def write_metadata(config: Step07Config, output: Path, step05_dir: Path, step06_dir: Path) -> None:
    data = {
        "code_version": CODE_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": script_sha256(),
        "step02_core_version": getattr(core, "CODE_VERSION", None),
        "step05_core_version": getattr(step05, "CODE_VERSION", None),
        "step05_directory": str(step05_dir),
        "step06_directory": str(step06_dir),
        "config": asdict(config),
    }
    (output / "fes_step07_run_metadata.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =============================================================================
# 2. Symbolic periodic Hamiltonian and parity basis
# =============================================================================

def symbolic_derivation() -> Dict[str, object]:
    """Fast exact derivation using the parity-basis blocks directly."""
    m, t1, t2, r1, r2 = sp.symbols(
        "m_e t_1 t_2 r_1 r_2", real=True
    )
    kx, ky = sp.symbols("k_x k_y", real=True)
    rp = r2 + sp.I*r1
    rm = r2 - sp.I*r1

    # Exact spin-up Hamiltonian in the periodic gauge.
    H = sp.Matrix([
        [m, t1, rp*sp.exp(sp.I*ky), rm],
        [t1, m, rm*sp.exp(-sp.I*kx+sp.I*ky), rp*sp.exp(-sp.I*kx)],
        [rm*sp.exp(-sp.I*ky), rp*sp.exp(sp.I*kx-sp.I*ky), -m, t2],
        [rp, rm*sp.exp(sp.I*kx), t2, -m],
    ])

    ts = (t1+t2)/2
    td = (t1-t2)/2
    u = m+td
    v = m-td
    A1 = sp.sqrt(v**2+4*r1**2)
    A2 = sp.sqrt(v**2+4*r2**2)
    B1 = sp.sqrt(u**2+4*r1**2)
    B2 = sp.sqrt(u**2+4*r2**2)

    # Parity-basis Hamiltonians. Order: (B top, B bottom, A top, A bottom).
    HG = sp.Matrix([
        [m+t1, 2*r2, 0, 0],
        [2*r2, -m+t2, 0, 0],
        [0, 0, m-t1, 2*sp.I*r1],
        [0, 0, -2*sp.I*r1, -m-t2],
    ])
    HM = sp.Matrix([
        [m+t1, -2*sp.I*r1, 0, 0],
        [2*sp.I*r1, -m+t2, 0, 0],
        [0, 0, m-t1, -2*r2],
        [0, 0, -2*r2, -m-t2],
    ])

    # A-to-B derivative blocks <A|dH|B>.
    XG = sp.Matrix([[0, sp.I*r2], [-r1, 0]])
    YG = sp.Matrix([[0, -r1], [-sp.I*r2, 0]])
    XM = sp.Matrix([[0, r1], [sp.I*r2, 0]])
    YM = sp.Matrix([[0, -sp.I*r2], [r1, 0]])

    # Explicit unnormalized eigenvectors in the A and B blocks.
    aG = sp.Matrix([2*sp.I*r1, -(A1+v)])
    bG = sp.Matrix([2*r2, B2-u])
    NAG2 = 4*r1**2+(A1+v)**2
    NBG2 = 4*r2**2+(B2-u)**2

    aM = sp.Matrix([-2*r2, -(A2+v)])
    bM = sp.Matrix([-2*sp.I*r1, B1-u])
    NAM2 = 4*r2**2+(A2+v)**2
    NBM2 = 4*r1**2+(B1-u)**2

    # Direct numerator algebra; all square-root factors are real.
    vxG_num = sp.expand((sp.conjugate(aG).T*XG*bG)[0])
    vyG_num = sp.expand((sp.conjugate(aG).T*YG*bG)[0])
    vxM_num = sp.expand((sp.conjugate(aM).T*XM*bM)[0])
    vyM_num = sp.expand((sp.conjugate(aM).T*YM*bM)[0])

    SG = A1+v+B2-u
    TG = r1**2*(B2-u)+r2**2*(A1+v)
    SM = A2+v+B1-u
    TM = r2**2*(B1-u)+r1**2*(A2+v)

    vxG_expected_num = 2*r1*r2*SG
    vyG_expected_num = 2*sp.I*TG
    vxM_expected_num = -2*r1*r2*SM
    vyM_expected_num = 2*sp.I*TM

    residuals = {
        "vx_Gamma_numerator": sp.expand(vxG_num-vxG_expected_num),
        "vy_Gamma_numerator": sp.expand(vyG_num-vyG_expected_num),
        "vx_M_numerator": sp.expand(vxM_num-vxM_expected_num),
        "vy_M_numerator": sp.expand(vyM_num-vyM_expected_num),
    }

    vxG = vxG_expected_num/sp.sqrt(NAG2*NBG2)
    vyG = vyG_expected_num/sp.sqrt(NAG2*NBG2)
    vxM = vxM_expected_num/sp.sqrt(NAM2*NBM2)
    vyM = vyM_expected_num/sp.sqrt(NAM2*NBM2)
    detG_expected = -4*r1*r2*SG*TG/(NAG2*NBG2)
    detM_expected = 4*r1*r2*SM*TM/(NAM2*NBM2)

    return {
        "m": m, "t1": t1, "t2": t2, "r1": r1, "r2": r2,
        "kx": kx, "ky": ky, "H": H, "HG": HG, "HM": HM,
        "ts": ts, "td": td, "u": u, "v": v,
        "A1": A1, "A2": A2, "B1": B1, "B2": B2,
        "XG": XG, "YG": YG, "XM": XM, "YM": YM,
        "aG": aG, "bG": bG, "NAG2": NAG2, "NBG2": NBG2,
        "SG": SG, "TG": TG, "vxG": vxG, "vyG": vyG,
        "detG_expected": detG_expected,
        "aM": aM, "bM": bM, "NAM2": NAM2, "NBM2": NBM2,
        "SM": SM, "TM": TM, "vxM": vxM, "vyM": vyM,
        "detM_expected": detM_expected,
        "residuals": residuals,
    }


# =============================================================================
# 4. Write symbolic proof
# =============================================================================

def _s(expr: sp.Expr) -> str:
    return sp.sstr(expr)


def _l(expr: sp.Expr) -> str:
    return sp.latex(expr)


def write_symbolic_outputs(output: Path, d: Dict[str, object]) -> None:
    residuals = d["residuals"]
    residual_ok = all(sp.simplify(v) == 0 for v in residuals.values())

    text = f"""FES Step 07 — explicit symbolic Jacobian proof
================================================

Periodic-gauge spin-up Hamiltonian
----------------------------------
H_up(k) =
{sp.pretty(d["H"])}

Parity-basis matrices
---------------------
H_Gamma =
{sp.pretty(d["HG"])}

H_M =
{sp.pretty(d["HM"])}

Definitions
-----------
t_s=(t1+t2)/2
t_d=(t1-t2)/2
u=m_e+t_d
v=m_e-t_d

A1=sqrt(v^2+4 r1^2)
A2=sqrt(v^2+4 r2^2)
B1=sqrt(u^2+4 r1^2)
B2=sqrt(u^2+4 r2^2)

Gamma projected velocities
--------------------------
v_x,Gamma = {_s(d["vxG"])}
v_y,Gamma = {_s(d["vyG"])}

det J_Gamma = {_s(d["detG_expected"])}

Because
  S_Gamma=A1+v+B2-u > 0
  T_Gamma=r1^2(B2-u)+r2^2(A1+v) > 0
away from the singular r1=r2=0 limit,

  sgn(det J_Gamma) = -sgn(r1*r2).

In the ordered basis (A-,B+),
  d_z,Gamma = mu_Gamma/2.

For the lower band of a massive two-band crossing,
  C_Gamma = -sgn(detJ_Gamma*mu_Gamma)/2
          =  sgn(r1*r2*mu_Gamma)/2.

M projected velocities
----------------------
v_x,M = {_s(d["vxM"])}
v_y,M = {_s(d["vyM"])}

det J_M = {_s(d["detM_expected"])}

Because
  S_M=A2+v+B1-u > 0
  T_M=r2^2(B1-u)+r1^2(A2+v) > 0,

  sgn(det J_M) = +sgn(r1*r2).

In the same ordered basis (A-,B+),
  d_z,M = -mu_M/2.

Therefore
  C_M = -sgn(detJ_M*(-mu_M))/2
      =  sgn(r1*r2*mu_M)/2.

Closed formula
--------------
C_up = C_Gamma + C_M
     = sgn(r1*r2)/2 *
       [sgn(mu_Gamma)+sgn(mu_M)].

C_down=-C_up and C_total=0.

Combined model-level no-go
--------------------------
C_up != 0
=> mu_Gamma and mu_M have the same sign
=> the B+ branch exchanges occupied/unoccupied rank between Gamma and M
=> E_g^ind <= -4|r1^2-r2^2|/(B1+B2) <= 0.

Symbolic residuals all zero: {residual_ok}
Residuals:
{json.dumps({k: _s(v) for k, v in residuals.items()}, indent=2)}
"""
    (output / "fes_step07_symbolic_proof.txt").write_text(text, encoding="utf-8")

    tex = rf"""\documentclass{{article}}
\usepackage{{amsmath,amssymb,bm}}
\begin{{document}}
\section*{{Analytic Jacobian and spin-Chern formula for the fes model}}

Define
\[
t_s=\frac{{t_1+t_2}}{{2}},\qquad
t_d=\frac{{t_1-t_2}}{{2}},\qquad
u=m_e+t_d,\qquad v=m_e-t_d,
\]
\[
A_1=\sqrt{{v^2+4r_1^2}},\quad
A_2=\sqrt{{v^2+4r_2^2}},\quad
B_1=\sqrt{{u^2+4r_1^2}},\quad
B_2=\sqrt{{u^2+4r_2^2}}.
\]

At $\Gamma$, in the $(A^-,B^+)$ basis,
\[
\langle A^-|\partial_{{k_x}}H|B^+\rangle
=
{_l(d["vxG"])},
\]
\[
\langle A^-|\partial_{{k_y}}H|B^+\rangle
=
{_l(d["vyG"])}.
\]
Hence
\[
\det J_\Gamma=
{_l(d["detG_expected"])},
\qquad
\operatorname{{sgn}}(\det J_\Gamma)
=-\operatorname{{sgn}}(r_1r_2).
\]
Since $d_{{z,\Gamma}}=\mu_\Gamma/2$,
\[
C_\Gamma=
-\frac12\operatorname{{sgn}}(\det J_\Gamma\mu_\Gamma)
=
\frac12\operatorname{{sgn}}(r_1r_2\mu_\Gamma).
\]

At $M$,
\[
\langle A^-|\partial_{{k_x}}H|B^+\rangle
=
{_l(d["vxM"])},
\]
\[
\langle A^-|\partial_{{k_y}}H|B^+\rangle
=
{_l(d["vyM"])}.
\]
Thus
\[
\det J_M=
{_l(d["detM_expected"])},
\qquad
\operatorname{{sgn}}(\det J_M)
=+\operatorname{{sgn}}(r_1r_2).
\]
Because $d_{{z,M}}=-\mu_M/2$,
\[
C_M=
-\frac12\operatorname{{sgn}}[\det J_M(-\mu_M)]
=
\frac12\operatorname{{sgn}}(r_1r_2\mu_M).
\]

Therefore
\[
\boxed{{
C_\uparrow=
\frac{{\operatorname{{sgn}}(r_1r_2)}}{{2}}
\left[
\operatorname{{sgn}}(\mu_\Gamma)+
\operatorname{{sgn}}(\mu_M)
\right]
}}.
\]
Moreover,
\[
C_\uparrow\neq0
\Longrightarrow
E_g^{{\rm ind}}
\le
-\frac{{4|r_1^2-r_2^2|}}{{B_1+B_2}}
\le0.
\]
\end{{document}}
"""
    (output / "fes_step07_symbolic_proof.tex").write_text(tex, encoding="utf-8")

    json_data = {
        "symbolic_residuals_zero": residual_ok,
        "residuals": {k: _s(v) for k, v in residuals.items()},
        "vx_Gamma": _s(d["vxG"]),
        "vy_Gamma": _s(d["vyG"]),
        "detJ_Gamma": _s(d["detG_expected"]),
        "vx_M": _s(d["vxM"]),
        "vy_M": _s(d["vyM"]),
        "detJ_M": _s(d["detM_expected"]),
        "chern_formula": (
            "C_up=sgn(r1*r2)/2*[sgn(mu_Gamma)+sgn(mu_M)]"
        ),
    }
    (output / "fes_step07_symbolic_proof.json").write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =============================================================================
# 5. Numerical validation of explicit symbolic velocities
# =============================================================================

def reduced_from_row(row: pd.Series) -> Dict[str, float]:
    return {k: float(row[k]) for k in PARAMS5}


def periodic_spin_up(kx: float, ky: float, p: Dict[str, float]) -> np.ndarray:
    raw = core.raw6_from_reduced5(p)
    return core.h_spin_block_periodic(kx, ky, raw, "up")


def explicit_basis_numeric(p: Dict[str, float], valley: str) -> tuple[np.ndarray, np.ndarray]:
    m, t1, t2, r1, r2 = [p[k] for k in PARAMS5]
    ts = 0.5*(t1+t2)
    td = 0.5*(t1-t2)
    u = m + td
    v = m - td

    s2 = math.sqrt(2.0)
    P = np.array([
        [1/s2, 0, 1/s2, 0],
        [1/s2, 0, -1/s2, 0],
        [0, 1/s2, 0, 1/s2],
        [0, 1/s2, 0, -1/s2],
    ], dtype=np.complex128)

    if valley == "Gamma":
        A = math.sqrt(v*v + 4*r1*r1)
        B = math.sqrt(u*u + 4*r2*r2)
        a = np.array([2j*r1, -(A+v)], dtype=np.complex128)
        b = np.array([2*r2, B-u], dtype=np.complex128)
    elif valley == "M":
        A = math.sqrt(v*v + 4*r2*r2)
        B = math.sqrt(u*u + 4*r1*r1)
        a = np.array([-2*r2, -(A+v)], dtype=np.complex128)
        b = np.array([-2j*r1, B-u], dtype=np.complex128)
    else:
        raise ValueError(valley)

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    Avec = P @ np.array([0, 0, a[0], a[1]], dtype=np.complex128)
    Bvec = P @ np.array([b[0], b[1], 0, 0], dtype=np.complex128)
    return Avec, Bvec


def analytic_velocity_numeric(p: Dict[str, float], valley: str) -> tuple[complex, complex, float]:
    m, t1, t2, r1, r2 = [p[k] for k in PARAMS5]
    td = 0.5*(t1-t2)
    u = m + td
    v = m - td

    if valley == "Gamma":
        A = math.sqrt(v*v + 4*r1*r1)
        B = math.sqrt(u*u + 4*r2*r2)
        NA2 = 4*r1*r1 + (A+v)**2
        NB2 = 4*r2*r2 + (B-u)**2
        S = A+v+B-u
        T = r1*r1*(B-u) + r2*r2*(A+v)
        vx = 2*r1*r2*S/math.sqrt(NA2*NB2)
        vy = 2j*T/math.sqrt(NA2*NB2)
        det = -4*r1*r2*S*T/(NA2*NB2)
    else:
        A = math.sqrt(v*v + 4*r2*r2)
        B = math.sqrt(u*u + 4*r1*r1)
        NA2 = 4*r2*r2 + (A+v)**2
        NB2 = 4*r1*r1 + (B-u)**2
        S = A+v+B-u
        T = r2*r2*(B-u) + r1*r1*(A+v)
        vx = -2*r1*r2*S/math.sqrt(NA2*NB2)
        vy = 2j*T/math.sqrt(NA2*NB2)
        det = 4*r1*r2*S*T/(NA2*NB2)
    return complex(vx), complex(vy), float(det)


def validate_velocities(step06_dir: Path, h: float) -> pd.DataFrame:
    roots = pd.read_csv(step06_dir / "fes_step06_critical_mass_roots.csv")
    roots = roots[pd.to_numeric(roots["root_found"], errors="coerce") == 1].copy()
    rows = []

    for _, row in roots.iterrows():
        p = reduced_from_row(row)
        p["m_e"] = float(row["m_critical"])
        valley = str(row["valley"])
        k0 = (0.0, 0.0) if valley == "Gamma" else (math.pi, math.pi)
        A, B = explicit_basis_numeric(p, valley)

        dHdx = (
            periodic_spin_up(k0[0]+h, k0[1], p)
            - periodic_spin_up(k0[0]-h, k0[1], p)
        )/(2*h)
        dHdy = (
            periodic_spin_up(k0[0], k0[1]+h, p)
            - periodic_spin_up(k0[0], k0[1]-h, p)
        )/(2*h)

        vx_num = np.vdot(A, dHdx @ B)
        vy_num = np.vdot(A, dHdy @ B)
        vx_ana, vy_ana, det_ana = analytic_velocity_numeric(p, valley)
        det_num = (
            np.real(vx_num)*(-np.imag(vy_num))
            - np.real(vy_num)*(-np.imag(vx_num))
        )

        rows.append({
            "point_id": row["point_id"],
            "valley": valley,
            **p,
            "vx_numeric_real": np.real(vx_num),
            "vx_numeric_imag": np.imag(vx_num),
            "vx_analytic_real": np.real(vx_ana),
            "vx_analytic_imag": np.imag(vx_ana),
            "vx_abs_error": abs(vx_num-vx_ana),
            "vy_numeric_real": np.real(vy_num),
            "vy_numeric_imag": np.imag(vy_num),
            "vy_analytic_real": np.real(vy_ana),
            "vy_analytic_imag": np.imag(vy_ana),
            "vy_abs_error": abs(vy_num-vy_ana),
            "det_numeric": det_num,
            "det_analytic": det_ana,
            "det_abs_error": abs(det_num-det_ana),
            "det_sign_match": int(np.sign(det_num) == np.sign(det_ana)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. Global formula and no-go audit
# =============================================================================

def formula_from_row(row: pd.Series) -> float:
    rprod = float(row["r1"])*float(row["r2"])
    muG = float(row["mass_Gamma_signed"])
    muM = float(row["mass_M_signed"])
    if abs(rprod) < 1e-14 or abs(muG) < 1e-12 or abs(muM) < 1e-12:
        return np.nan
    return float(
        np.sign(rprod)*0.5*(np.sign(muG)+np.sign(muM))
    )


def audit_strict(step05_dir: Path) -> tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(step05_dir / "fes_step05_strict_analytic_audit.csv")
    reliable = df[
        (pd.to_numeric(df["verified_chern_reliable"], errors="coerce") == 1)
        & (pd.to_numeric(df["verified_is_direct_gapped"], errors="coerce") == 1)
        & (pd.to_numeric(df["verified_is_balanced_spin_sector"], errors="coerce") == 1)
    ].copy()
    reliable["chern_up_symbolic_formula"] = [
        formula_from_row(row) for _, row in reliable.iterrows()
    ]
    reliable["formula_match"] = (
        reliable["chern_up_symbolic_formula"]
        == pd.to_numeric(reliable["verified_chern_up_int"], errors="coerce")
    ).astype(int)
    valid = reliable[np.isfinite(reliable["chern_up_symbolic_formula"])].copy()
    summary = {
        "strict_reliable_rows": int(len(reliable)),
        "formula_evaluated_rows": int(len(valid)),
        "formula_accuracy": float(valid["formula_match"].mean()) if len(valid) else np.nan,
        "formula_mismatch_count": int((valid["formula_match"] == 0).sum()),
    }
    return reliable, summary


def build_theorem_certificate(step05_dir: Path, strict_audit: pd.DataFrame) -> pd.DataFrame:
    cert = pd.read_csv(
        step05_dir / "fes_step05_verified_topological_no_go_certificates.csv"
    )
    if cert.empty:
        return cert

    cols = [
        "point_id", "chern_up_symbolic_formula", "formula_match"
    ]
    merged = cert.merge(
        strict_audit[[c for c in cols if c in strict_audit.columns]],
        on="point_id", how="left"
    )
    merged["symbolic_formula_plus_conditional_nogo"] = (
        (pd.to_numeric(merged["formula_match"], errors="coerce") == 1)
        & (
            pd.to_numeric(
                merged["conditional_no_go_certificate"], errors="coerce"
            ) == 1
        )
    ).astype(int)
    return merged


# =============================================================================
# 7. Plots
# =============================================================================

def plot_velocity_validation(df: pd.DataFrame, output: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["det_analytic"], df["det_numeric"])
    lo = min(df["det_analytic"].min(), df["det_numeric"].min())
    hi = max(df["det_analytic"].max(), df["det_numeric"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("analytic det J")
    ax.set_ylabel("finite-difference det J")
    ax.set_title("Explicit symbolic Jacobian validation")
    fig.tight_layout()
    fig.savefig(output / "fes_step07_jacobian_validation.png", dpi=220)
    plt.close(fig)


# =============================================================================
# 8. Main pipeline
# =============================================================================

def run_pipeline(config: Step07Config) -> Dict[str, str]:
    output = prepare_output(config)
    step05_dir = locate_result(
        config.step05_input, output,
        "fes_step05_strict_analytic_audit.csv", "step05"
    )
    step06_dir = locate_result(
        config.step06_input, output,
        "fes_step06_critical_mass_roots.csv", "step06"
    )
    write_metadata(config, output, step05_dir, step06_dir)

    derivation = symbolic_derivation()
    write_symbolic_outputs(output, derivation)

    velocity = validate_velocities(
        step06_dir, config.finite_difference_step
    )
    velocity.to_csv(
        output / "fes_step07_symbolic_velocity_validation.csv", index=False
    )
    plot_velocity_validation(velocity, output)

    strict, strict_summary = audit_strict(step05_dir)
    strict.to_csv(
        output / "fes_step07_strict_formula_audit.csv", index=False
    )

    theorem = build_theorem_certificate(step05_dir, strict)
    theorem.to_csv(
        output / "fes_step07_model_level_no_go_certificates.csv", index=False
    )

    residual_ok = all(
        sp.simplify(v) == 0 for v in derivation["residuals"].values()
    )
    velocity_ok = bool(
        len(velocity)
        and velocity["det_sign_match"].all()
        and velocity["vx_abs_error"].max() < 1e-8
        and velocity["vy_abs_error"].max() < 1e-8
    )
    theorem_count = int(
        theorem["symbolic_formula_plus_conditional_nogo"].sum()
    ) if len(theorem) else 0

    summary = {
        "code_version": CODE_VERSION,
        "symbolic_residuals_all_zero": bool(residual_ok),
        "velocity_validation_rows": int(len(velocity)),
        "max_vx_abs_error": float(velocity["vx_abs_error"].max()) if len(velocity) else np.nan,
        "max_vy_abs_error": float(velocity["vy_abs_error"].max()) if len(velocity) else np.nan,
        "max_det_abs_error": float(velocity["det_abs_error"].max()) if len(velocity) else np.nan,
        "velocity_validation_passed": velocity_ok,
        "strict_formula_audit": strict_summary,
        "verified_topological_points": int(len(theorem)),
        "model_level_no_go_certificate_count": theorem_count,
        "model_level_no_go_certificate_fraction": (
            float(theorem_count/len(theorem)) if len(theorem) else np.nan
        ),
        "derived_formula": (
            "C_up=sgn(r1*r2)/2*[sgn(mu_Gamma)+sgn(mu_M)]"
        ),
        "derived_no_go": (
            "C_up!=0 => E_g^ind <= "
            "-4|r1^2-r2^2|/(B1+B2) <= 0"
        ),
        "scope_note": (
            "The local Jacobian and valley Chern contributions are symbolic. "
            "The global use of only Gamma/M closures is supported by the "
            "Steps 02–06 phase-boundary and Chern-jump audits."
        ),
    }
    (output / "fes_step07_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""FES Step 07 summary
===================

Symbolic residuals all zero:
  {residual_ok}

Velocity/Jacobian validation:
  rows              = {len(velocity)}
  max |delta vx|    = {summary["max_vx_abs_error"]}
  max |delta vy|    = {summary["max_vy_abs_error"]}
  max |delta det J| = {summary["max_det_abs_error"]}
  passed            = {velocity_ok}

Strict spin-Chern formula:
  evaluated rows = {strict_summary["formula_evaluated_rows"]}
  accuracy       = {strict_summary["formula_accuracy"]}
  mismatches     = {strict_summary["formula_mismatch_count"]}

Model-level no-go certificates:
  {theorem_count}/{len(theorem)}

Derived result
--------------
C_up = sgn(r1*r2)/2 *
       [sgn(mu_Gamma)+sgn(mu_M)].

At Gamma:
  sgn(det J_Gamma) = -sgn(r1*r2),
  d_z = mu_Gamma/2.

At M:
  sgn(det J_M) = +sgn(r1*r2),
  d_z = -mu_M/2.

Therefore each valley contributes
  C_K = sgn(r1*r2*mu_K)/2.

Combining with the exact Step 05 branch-switching identity gives
  C_up != 0
  => E_g^ind <= -4|r1^2-r2^2|/(B1+B2) <= 0.

Interpretation
--------------
Within the half-filled, spin-conserving minimal six-parameter fes
Hamiltonian, the nontrivial spin-Chern phase is analytically forced
to be a spin-Chern band metal rather than a globally insulating
type-II QSH phase, provided the only relevant topological closures
are the Gamma/M A-minus–B-plus crossings verified in Steps 02–06.
"""
    (output / "fes_step07_report.txt").write_text(report, encoding="utf-8")

    return {
        "output_dir": str(output),
        "symbolic_proof_txt": str(output / "fes_step07_symbolic_proof.txt"),
        "symbolic_proof_tex": str(output / "fes_step07_symbolic_proof.tex"),
        "velocity_validation": str(
            output / "fes_step07_symbolic_velocity_validation.csv"
        ),
        "strict_formula_audit": str(
            output / "fes_step07_strict_formula_audit.csv"
        ),
        "no_go_certificates": str(
            output / "fes_step07_model_level_no_go_certificates.csv"
        ),
        "summary": str(output / "fes_step07_summary.json"),
        "report": str(output / "fes_step07_report.txt"),
    }


# =============================================================================
# 9. CLI
# =============================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FES Step 07 symbolic Jacobian and model-level no-go"
    )
    p.add_argument(
        "--step05-input",
        default="outputs_fes6_step05_no_go.zip"
    )
    p.add_argument(
        "--step06-input",
        default="outputs_fes6_step06_kp.zip"
    )
    p.add_argument(
        "--output-dir",
        default="outputs_fes6_step07_symbolic"
    )
    p.add_argument("--test", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    config = Step07Config(
        step05_input=args.step05_input,
        step06_input=args.step06_input,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        test_mode=args.test,
    )

    print("="*78)
    print("FES Step 07 — symbolic Jacobian and model-level no-go")
    print("Version:", CODE_VERSION)
    print("Output :", Path(config.output_dir).resolve())
    print("="*78)

    paths = run_pipeline(config)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
