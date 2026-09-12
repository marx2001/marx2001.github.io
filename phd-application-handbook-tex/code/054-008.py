# -*- coding: utf-8 -*-
"""
Fit symmetry-based TB parameters (e/t/r) against Wannier band reference (wannier90_band.dat)

Stages:
  Stage-1: fit e only
  Stage-2: fit t only (e fixed)
  Stage-3: fit r only (e,t fixed)
  Stage-4: fit e,t,r together (warm start from stage1-3)

Band-matching protection (recommended):
  Stage-4 uses eigenvector-overlap band tracking + Hungarian anchoring at kref_match,
  to mitigate band crossings / index swapping issues.

Windows-ready. Target band set is fixed by reference band indices selected at kref.
"""

import os
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

# Wolfram WXF deserializer (needs: pip install wolframclient)
from wolframclient.deserializers import binary_deserialize
from wolframclient.language.expression import WLFunction, WLSymbol


# ============================================================
# 0) Config (EDIT PATHS HERE)
# ============================================================
TB_FILE = r"E:\马睿骁\组会汇报\Nb2OSSe\pythTB\tb_basis.wxf"
WANNIER_BAND_FILE = r"E:\马睿骁\组会汇报\Nb2OSSe\pythTB\wannier90_band.dat"

# Optional: for cross validation (DFT BAND.dat from VASPKIT)
DFT_BAND_FILE = None  # r"E:\...\BAND.dat"

# DFT Fermi energy baseline: you said "use DFT EF=0"
EF_DFT = 0.0
EF_SHIFT_WANNIER = 0.0

# Select 12 bands: fixed band indices (global columns)
N_TARGET_BANDS = 12
KREF_FOR_PICK = 0  # select target bands at this ref index in reference bands
E_TARGET = 0.0

# Regularization (keeps parameters from exploding)
LAMBDA_E = 1e-8
LAMBDA_T = 1e-8
LAMBDA_R = 1e-8

# Bounds (typical low-energy window)
E_BOUND = 5.0
T_BOUND = 2.5
R_BOUND = 2.5

# Optimizer controls
MAX_NFEV_1 = 2000
MAX_NFEV_2 = 3000
MAX_NFEV_3 = 3000
MAX_NFEV_4 = 6000

DIFF_STEP = 1e-3

# Stage-4 band-matching protection (HIGHLY recommended)
USE_BANDMATCH_STAGE4 = True
KREF_MATCH = KREF_FOR_PICK  # anchor for Hungarian matching + tracking


# ============================================================
# 1) Wolfram expression -> Python number
# ============================================================
def _head_name(h):
    return h.name if isinstance(h, WLSymbol) else str(h)

def wl_to_number(x):
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, (int, float, complex, np.number)):
        return x
    if isinstance(x, (list, tuple)):
        # Some WXF encodes Complex as [re, im]
        if len(x) == 2:
            return complex(float(wl_to_number(x[0])), float(wl_to_number(x[1])))
        if len(x) == 1:
            return wl_to_number(x[0])
    if isinstance(x, WLFunction):
        name = _head_name(x.head)
        args = list(x.args)
        if name == "Complex":
            return complex(wl_to_number(args[0]), wl_to_number(args[1]))
        if name == "Rational":
            return float(wl_to_number(args[0])) / float(wl_to_number(args[1]))
        if name == "Plus":
            return sum(wl_to_number(a) for a in args)
        if name == "Times":
            out = 1
            for a in args:
                out *= wl_to_number(a)
            return out
        if name == "Power":
            return wl_to_number(args[0]) ** wl_to_number(args[1])
        raise TypeError(f"Unsupported WLFunction: {name}")
    # fallback
    return complex(x)

def mat_to_complex_array(mat, nb):
    out = np.empty((nb, nb), dtype=np.complex128)
    for i in range(nb):
        for j in range(nb):
            out[i, j] = wl_to_number(mat[i][j])
    return out


# ============================================================
# 2) Load TB basis from WXF
# ============================================================
def load_tb_basis_wxf(filename, debug=False):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"TB WXF not found: {filename}")

    with open(filename, "rb") as f:
        expr = binary_deserialize(f)

    pars = np.array([str(p) for p in expr["pars"]])
    kpts = np.array(expr["kpts"], dtype=float)  # (Nk,2) or (Nk,3) depending on export
    H0k_raw = expr["H0k"]
    Hbk_raw = expr["Hbk"]

    Nk = len(H0k_raw)
    nb = len(H0k_raw[0])
    Np = len(pars)

    H0k = np.empty((Nk, nb, nb), dtype=np.complex128)
    Hbk = np.empty((Nk, Np, nb, nb), dtype=np.complex128)

    for ik in range(Nk):
        H0k[ik] = mat_to_complex_array(H0k_raw[ik], nb)
        for jp in range(Np):
            Hbk[ik, jp] = mat_to_complex_array(Hbk_raw[ik][jp], nb)

    # Hermitian safety
    H0k = (H0k + H0k.transpose(0, 2, 1).conj()) / 2.0
    Hbk = (Hbk + Hbk.transpose(0, 1, 3, 2).conj()) / 2.0

    if debug:
        print(f"[TB] Nk={Nk}, nb={nb}, Nparams={Np}")
        print("kpts[0:3]:", kpts[:3])

    return pars, kpts, H0k, Hbk


# ============================================================
# 3) Parse wannier90_band.dat
#    Format: two columns (kdist, energy), bands separated by blank lines
# ============================================================
def load_wannier_banddat(filename, ef_shift=0.0, debug=False):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Wannier band file not found: {filename}")

    blocks, buf = [], []
    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                if buf:
                    blocks.append(np.array(buf, dtype=float))
                    buf = []
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            kd = float(parts[0])
            en = float(parts[1]) - ef_shift
            buf.append([kd, en])
    if buf:
        blocks.append(np.array(buf, dtype=float))

    if len(blocks) == 0:
        raise ValueError("No band blocks parsed from wannier90_band.dat")

    Nk = len(blocks[0])
    kdist = blocks[0][:, 0]
    bands = []
    for b in blocks:
        if len(b) != Nk:
            continue
        bands.append(b[:, 1])
    Eall = np.stack(bands, axis=1)  # (Nk, Nb_total)

    if debug:
        print(f"[WANNIER] Nk={Nk}, Nb_total={Eall.shape[1]}")
        print("kdist[0:3]:", kdist[:3])
        print("E range (min,max):", float(Eall.min()), float(Eall.max()))

    return kdist, Eall


# ============================================================
# 4) Optional: Parse VASPKIT BAND.dat (band blocks)
# ============================================================
def load_vaspkit_banddat_auto(filename, ef):
    blocks, buf = [], []
    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                if buf:
                    blocks.append(np.array(buf, dtype=float))
                    buf = []
                continue
            a, b = s.split()[:2]
            buf.append([float(a), float(b) - ef])
    if buf:
        blocks.append(np.array(buf, dtype=float))

    Nk = len(blocks[0])
    kdist = blocks[0][:, 0]
    bands = [b[:, 1] for b in blocks if len(b) == Nk]
    Eall = np.stack(bands, axis=1)  # (Nk, Nb_total)
    return kdist, Eall


# ============================================================
# 5) Fixed band index selection (global 12 columns)
# ============================================================
def pick_fixed_band_indices(Eall, n=12, kref=0, target=0.0):
    if kref < 0 or kref >= Eall.shape[0]:
        raise ValueError("kref out of range")
    e0 = Eall[kref, :]
    order = np.argsort(np.abs(e0 - target))
    chosen = np.sort(order[:n])  # keep stable band order by index
    return chosen

def select_bands_fixed_order(Eall, indices):
    return Eall[:, indices]


# ============================================================
# 6) kdist utilities
# ============================================================
def kdist_from_kpts(kpts):
    kpts = np.asarray(kpts, dtype=float)
    dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    kd = np.concatenate([[0.0], np.cumsum(dk)])
    return kd

def rescale_kdist(kd_tb, kd_ref):
    if kd_tb[-1] == 0:
        return kd_tb.copy()
    return kd_tb * (kd_ref[-1] / kd_tb[-1])

def make_unique_monotonic(x, Y):
    """
    For duplicate x (common at high symmetry joints):
    - average Y for identical x
    - return strictly monotonic x
    """
    x = np.asarray(x)
    Y = np.asarray(Y)
    ux, inv = np.unique(x, return_inverse=True)
    Y2 = np.zeros((len(ux),) + Y.shape[1:], dtype=float)
    cnt = np.zeros(len(ux), dtype=int)
    for i, g in enumerate(inv):
        Y2[g] += Y[i]
        cnt[g] += 1
    if Y2.ndim == 1:
        Y2 = Y2 / cnt
    else:
        Y2 = Y2 / cnt.reshape(-1, *([1] * (Y2.ndim - 1)))
    return ux, Y2


# ============================================================
# 7) TB eigenvalues (simple: energy-sorted at each k)
# ============================================================
def tb_bands_allk(H0k, Hbk, p_full):
    Nk = H0k.shape[0]
    nb = H0k.shape[1]
    E = np.empty((Nk, nb), dtype=float)
    for ik in range(Nk):
        H = H0k[ik] + np.tensordot(p_full, Hbk[ik], axes=(0, 0))
        H = (H + H.conj().T) / 2.0
        E[ik] = np.linalg.eigvalsh(H).real  # ascending
    return E


# ============================================================
# 8) TB eigenvalues with band-matching protection (tracking)
# ============================================================
def tb_bands_allk_tracked(H0k, Hbk, p_full, Eref_k0=None, kref_match=0):
    """
    Compute TB eigenvalues on all k points, with band-order tracking protection.

    Steps:
      A) At kref_match: align TB band order to reference Eref_k0 by energy matching (Hungarian).
      B) Track forward/backward by maximizing eigenvector overlaps.

    Returns:
      E_tracked: (Nk, nb) float, energies in a consistent tracked order.
    """
    Nk = H0k.shape[0]
    nb = H0k.shape[1]

    # solve eigensystem at all k
    evals = np.empty((Nk, nb), dtype=float)
    evecs = np.empty((Nk, nb, nb), dtype=np.complex128)  # columns are eigenvectors
    for ik in range(Nk):
        H = H0k[ik] + np.tensordot(p_full, Hbk[ik], axes=(0, 0))
        H = (H + H.conj().T) / 2.0
        w, v = np.linalg.eigh(H)  # ascending w; v columns are eigenvectors
        evals[ik] = w.real
        evecs[ik] = v

    perm = np.zeros((Nk, nb), dtype=int)

    # anchor matching at kref_match
    if Eref_k0 is None:
        perm[kref_match] = np.arange(nb, dtype=int)
    else:
        tb0 = evals[kref_match]
        ref0 = np.asarray(Eref_k0, dtype=float)
        if ref0.shape[0] != nb:
            raise ValueError(f"Eref_k0 length {ref0.shape[0]} != nb {nb}")

        C = (tb0[:, None] - ref0[None, :]) ** 2
        row_ind, col_ind = linear_sum_assignment(C)  # minimize cost
        # want mapping tracked(ref index) -> tb eigen-index
        ref_to_tb = np.zeros(nb, dtype=int)
        ref_to_tb[col_ind] = row_ind
        perm[kref_match] = ref_to_tb

    def match_by_overlap(v_prev, v_cur, prev_map):
        # v_prev, v_cur: (nb, nb), columns are eigenvectors
        Vp = v_prev[:, prev_map]  # columns in tracked order
        Vc = v_cur
        O = np.abs(Vp.conj().T @ Vc) ** 2  # (nb, nb)
        cost = 1.0 - O  # maximize overlap
        row_ind, col_ind = linear_sum_assignment(cost)
        cur_map = np.zeros(nb, dtype=int)
        cur_map[row_ind] = col_ind
        return cur_map

    # forward
    for ik in range(kref_match + 1, Nk):
        perm[ik] = match_by_overlap(evecs[ik - 1], evecs[ik], perm[ik - 1])

    # backward
    for ik in range(kref_match - 1, -1, -1):
        perm[ik] = match_by_overlap(evecs[ik + 1], evecs[ik], perm[ik + 1])

    E_tracked = np.empty_like(evals)
    for ik in range(Nk):
        E_tracked[ik] = evals[ik, perm[ik]]

    return E_tracked


# ============================================================
# 9) Residual builders (kdist-aligned)
# ============================================================
def residual_kdist_align_fixed_order(p_var, pars, kpts, H0k, Hbk,
                                    kd_ref, Eref,
                                    idx_e, idx_t, idx_r,
                                    e_fixed=None, t_fixed=None, r_fixed=None,
                                    lam=0.0,
                                    fit_mask=("e", "t", "r")):
    """
    Fixed-order residual: assumes energy-sorted eigenvalues at each k already correspond to Eref order.
    """
    Np = len(pars)
    p_full = np.zeros(Np, dtype=float)

    if e_fixed is not None:
        p_full[idx_e] = e_fixed
    if t_fixed is not None:
        p_full[idx_t] = t_fixed
    if r_fixed is not None:
        p_full[idx_r] = r_fixed

    cursor = 0
    if "e" in fit_mask:
        p_full[idx_e] = p_var[cursor: cursor + len(idx_e)]
        cursor += len(idx_e)
    if "t" in fit_mask:
        p_full[idx_t] = p_var[cursor: cursor + len(idx_t)]
        cursor += len(idx_t)
    if "r" in fit_mask:
        p_full[idx_r] = p_var[cursor: cursor + len(idx_r)]
        cursor += len(idx_r)

    E_tb = tb_bands_allk(H0k, Hbk, p_full)
    kd_tb = rescale_kdist(kdist_from_kpts(kpts), kd_ref)

    kd_tb_u, E_tb_u = make_unique_monotonic(kd_tb, E_tb)
    kd_ref_u, Eref_u = make_unique_monotonic(kd_ref, Eref)

    E_tb_on_ref = np.empty_like(Eref_u)
    for n in range(Eref_u.shape[1]):
        E_tb_on_ref[:, n] = np.interp(kd_ref_u, kd_tb_u, E_tb_u[:, n])

    res = (E_tb_on_ref - Eref_u).ravel()

    if lam and lam > 0:
        res = np.concatenate([res, np.sqrt(lam) * p_var])

    return res


def residual_kdist_align_bandmatch(p_var, pars, kpts, H0k, Hbk,
                                  kd_ref, Eref,
                                  idx_e, idx_t, idx_r,
                                  e_fixed=None, t_fixed=None, r_fixed=None,
                                  lam=0.0,
                                  fit_mask=("e", "t", "r"),
                                  kref_match=0):
    """
    Band-matching protected residual:
      - Hungarian energy matching at kref_match to align TB band order to Eref
      - eigenvector-overlap tracking along k to avoid band-index swapping
    """
    Np = len(pars)
    p_full = np.zeros(Np, dtype=float)

    if e_fixed is not None:
        p_full[idx_e] = e_fixed
    if t_fixed is not None:
        p_full[idx_t] = t_fixed
    if r_fixed is not None:
        p_full[idx_r] = r_fixed

    cursor = 0
    if "e" in fit_mask:
        p_full[idx_e] = p_var[cursor: cursor + len(idx_e)]
        cursor += len(idx_e)
    if "t" in fit_mask:
        p_full[idx_t] = p_var[cursor: cursor + len(idx_t)]
        cursor += len(idx_t)
    if "r" in fit_mask:
        p_full[idx_r] = p_var[cursor: cursor + len(idx_r)]
        cursor += len(idx_r)

    # Anchor energies at kref_match from reference bands (assumes consistent k-path ordering)
    Eref_anchor = np.asarray(Eref[kref_match], dtype=float).copy()

    E_tb = tb_bands_allk_tracked(H0k, Hbk, p_full, Eref_k0=Eref_anchor, kref_match=kref_match)

    kd_tb = rescale_kdist(kdist_from_kpts(kpts), kd_ref)

    kd_tb_u, E_tb_u = make_unique_monotonic(kd_tb, E_tb)
    kd_ref_u, Eref_u = make_unique_monotonic(kd_ref, Eref)

    E_tb_on_ref = np.empty_like(Eref_u)
    for n in range(Eref_u.shape[1]):
        E_tb_on_ref[:, n] = np.interp(kd_ref_u, kd_tb_u, E_tb_u[:, n])

    res = (E_tb_on_ref - Eref_u).ravel()

    if lam and lam > 0:
        res = np.concatenate([res, np.sqrt(lam) * p_var])

    return res


# ============================================================
# 10) IO helpers
# ============================================================
def save_params_txt(filename, pars, idx_list, values):
    with open(filename, "w", encoding="utf-8") as f:
        for idx, val in zip(idx_list, values):
            f.write(f"{pars[idx]}  {val:.10f}\n")


# ============================================================
# 11) Main Stage1-4
# ============================================================
def main():
    # ---------- Load TB basis ----------
    pars, kpts, H0k, Hbk = load_tb_basis_wxf(TB_FILE, debug=True)
    nb = H0k.shape[1]
    print(f"[TB] nb = {nb} (your TB model size)")

    # param indices
    idx_e = np.where(np.char.startswith(pars, "e"))[0]
    idx_t = np.where(np.char.startswith(pars, "t"))[0]
    idx_r = np.where(np.char.startswith(pars, "r"))[0]
    print(f"# e params = {len(idx_e)}")
    print(f"# t params = {len(idx_t)}")
    print(f"# r params = {len(idx_r)}")
    if nb != N_TARGET_BANDS:
        print(f"WARNING: TB nb={nb} but N_TARGET_BANDS={N_TARGET_BANDS}. "
              f"Your TB model should be 12-band if you fit 12 bands.")

    # ---------- Load Wannier reference ----------
    kd_w, Eall_w = load_wannier_banddat(WANNIER_BAND_FILE, ef_shift=EF_SHIFT_WANNIER, debug=True)
    print("Eall_w shape:", Eall_w.shape)

    # ---------- Pick fixed 12 band indices ----------
    fixed_indices = pick_fixed_band_indices(Eall_w, n=N_TARGET_BANDS, kref=KREF_FOR_PICK, target=E_TARGET)
    print("Fixed band indices (0-based):", fixed_indices.tolist())
    print("Fixed band indices (1-based):", (fixed_indices + 1).tolist())
    Eref = select_bands_fixed_order(Eall_w, fixed_indices)  # (Nk_ref, 12)

    # ---------- Bounds ----------
    lb_e = -E_BOUND * np.ones(len(idx_e))
    ub_e = +E_BOUND * np.ones(len(idx_e))
    lb_t = -T_BOUND * np.ones(len(idx_t))
    ub_t = +T_BOUND * np.ones(len(idx_t))
    lb_r = -R_BOUND * np.ones(len(idx_r))
    ub_r = +R_BOUND * np.ones(len(idx_r))

    # ---------- Stage-1: fit e only ----------
    print("\n===== Stage-1: fitting e only (Wannier ref) =====")
    p0_e = np.zeros(len(idx_e), dtype=float)
    res1 = least_squares(
        residual_kdist_align_fixed_order,
        p0_e,
        bounds=(lb_e, ub_e),
        args=(pars, kpts, H0k, Hbk, kd_w, Eref, idx_e, idx_t, idx_r,
              None, np.zeros(len(idx_t)), np.zeros(len(idx_r)),
              LAMBDA_E, ("e",)),
        method="trf",
        loss="linear",
        x_scale="jac",
        diff_step=DIFF_STEP,
        verbose=2,
        max_nfev=MAX_NFEV_1
    )
    e_fit = res1.x.copy()
    save_params_txt("fitted_stage1_e_wannier.txt", pars, idx_e, e_fit)
    print("Stage-1 cost =", res1.cost)
    print("Saved: fitted_stage1_e_wannier.txt")

    # ---------- Stage-2: fit t only (e fixed) ----------
    print("\n===== Stage-2: fitting t only (e fixed, Wannier ref) =====")
    p0_t = np.zeros(len(idx_t), dtype=float)
    res2 = least_squares(
        residual_kdist_align_fixed_order,
        p0_t,
        bounds=(lb_t, ub_t),
        args=(pars, kpts, H0k, Hbk, kd_w, Eref, idx_e, idx_t, idx_r,
              e_fit, None, np.zeros(len(idx_r)),
              LAMBDA_T, ("t",)),
        method="trf",
        loss="linear",
        x_scale="jac",
        diff_step=DIFF_STEP,
        verbose=2,
        max_nfev=MAX_NFEV_2
    )
    t_fit = res2.x.copy()
    save_params_txt("fitted_stage2_t_wannier.txt", pars, idx_t, t_fit)
    print("Stage-2 cost =", res2.cost)
    print("Saved: fitted_stage2_t_wannier.txt")

    # ---------- Stage-3: fit r only (e,t fixed) ----------
    print("\n===== Stage-3: fitting r only (e,t fixed, Wannier ref) =====")
    p0_r = np.zeros(len(idx_r), dtype=float)
    res3 = least_squares(
        residual_kdist_align_fixed_order,
        p0_r,
        bounds=(lb_r, ub_r),
        args=(pars, kpts, H0k, Hbk, kd_w, Eref, idx_e, idx_t, idx_r,
              e_fit, t_fit, None,
              LAMBDA_R, ("r",)),
        method="trf",
        loss="linear",
        x_scale="jac",
        diff_step=DIFF_STEP,
        verbose=2,
        max_nfev=MAX_NFEV_3
    )
    r_fit = res3.x.copy()
    save_params_txt("fitted_stage3_r_wannier.txt", pars, idx_r, r_fit)
    print("Stage-3 cost =", res3.cost)
    print("Saved: fitted_stage3_r_wannier.txt")

    # ---------- Stage-4: fit e,t,r together (warm start) ----------
    print("\n===== Stage-4: fitting e,t,r together (warm start, Wannier ref) =====")
    p0_etr = np.concatenate([e_fit, t_fit, r_fit], axis=0)
    lb_etr = np.concatenate([lb_e, lb_t, lb_r], axis=0)
    ub_etr = np.concatenate([ub_e, ub_t, ub_r], axis=0)

    lam_etr = 1e-8

    if USE_BANDMATCH_STAGE4:
        print(f"[Stage-4] Using band-matching protection (kref_match={KREF_MATCH})")
        res4 = least_squares(
            residual_kdist_align_bandmatch,
            p0_etr,
            bounds=(lb_etr, ub_etr),
            args=(pars, kpts, H0k, Hbk, kd_w, Eref, idx_e, idx_t, idx_r,
                  None, None, None,
                  lam_etr, ("e", "t", "r"),
                  KREF_MATCH),
            method="trf",
            loss="linear",
            x_scale="jac",
            diff_step=DIFF_STEP,
            verbose=2,
            max_nfev=MAX_NFEV_4
        )
    else:
        print("[Stage-4] Using fixed-order residual (no band-matching)")
        res4 = least_squares(
            residual_kdist_align_fixed_order,
            p0_etr,
            bounds=(lb_etr, ub_etr),
            args=(pars, kpts, H0k, Hbk, kd_w, Eref, idx_e, idx_t, idx_r,
                  None, None, None,
                  lam_etr, ("e", "t", "r")),
            method="trf",
            loss="linear",
            x_scale="jac",
            diff_step=DIFF_STEP,
            verbose=2,
            max_nfev=MAX_NFEV_4
        )

    etr_fit = res4.x.copy()
    e4 = etr_fit[:len(idx_e)]
    t4 = etr_fit[len(idx_e):len(idx_e) + len(idx_t)]
    r4 = etr_fit[len(idx_e) + len(idx_t):]

    save_params_txt("fitted_stage4_etr_wannier_e.txt", pars, idx_e, e4)
    save_params_txt("fitted_stage4_etr_wannier_t.txt", pars, idx_t, t4)
    save_params_txt("fitted_stage4_etr_wannier_r.txt", pars, idx_r, r4)
    print("Stage-4 cost =", res4.cost)
    print("Saved: fitted_stage4_etr_wannier_[e/t/r].txt")

    # ---------- Optional: Cross validation on DFT BAND.dat ----------
    if DFT_BAND_FILE is not None and os.path.exists(DFT_BAND_FILE):
        print("\n===== Cross validation on DFT BAND.dat (no refit) =====")
        kd_d, Eall_d = load_vaspkit_banddat_auto(DFT_BAND_FILE, EF_DFT)
        dft_indices = pick_fixed_band_indices(Eall_d, n=N_TARGET_BANDS, kref=KREF_FOR_PICK, target=0.0)
        Eref_d = select_bands_fixed_order(Eall_d, dft_indices)

        p_full = np.zeros(len(pars))
        p_full[idx_e] = e4
        p_full[idx_t] = t4
        p_full[idx_r] = r4

        if USE_BANDMATCH_STAGE4:
            # band tracking for evaluation as well (anchor to DFT at kref)
            E_tb = tb_bands_allk_tracked(H0k, Hbk, p_full, Eref_k0=Eref_d[KREF_MATCH], kref_match=KREF_MATCH)
        else:
            E_tb = tb_bands_allk(H0k, Hbk, p_full)

        kd_tb = rescale_kdist(kdist_from_kpts(kpts), kd_d)
        kd_tb_u, E_tb_u = make_unique_monotonic(kd_tb, E_tb)
        kd_d_u, Eref_d_u = make_unique_monotonic(kd_d, Eref_d)

        E_tb_on_d = np.empty_like(Eref_d_u)
        for n in range(Eref_d_u.shape[1]):
            E_tb_on_d[:, n] = np.interp(kd_d_u, kd_tb_u, E_tb_u[:, n])

        rms = np.sqrt(np.mean((E_tb_on_d - Eref_d_u) ** 2))
        print(f"DFT RMS error (eV) = {rms:.6f}")
        print("NOTE: DFT band indices used (1-based):", (dft_indices + 1).tolist())

    print("\nDONE.")


if __name__ == "__main__":
    main()
