#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1) POSCAR parser (VASP5/6)
# ============================================================
def read_poscar(poscar_path: Path):
    lines = poscar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 8:
        raise RuntimeError("POSCAR too short")

    comment = lines[0].strip()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float) * scale

    # VASP5+: line 5 symbols, line 6 counts
    # VASP4: line 5 counts directly
    tokens5 = lines[5].split()
    tokens6 = lines[6].split()

    def is_all_int(toks):
        try:
            for t in toks:
                int(t)
            return True
        except Exception:
            return False

    if is_all_int(tokens5):
        # VASP4 style: no symbols line
        species = [f"X{i+1}" for i in range(len(tokens5))]
        counts = [int(x) for x in tokens5]
        coord_line = 6
    else:
        species = tokens5
        counts = [int(x) for x in tokens6]
        coord_line = 7

    coord_type = lines[coord_line].strip().lower()
    if coord_type.startswith("s"):
        # selective dynamics present
        coord_line += 1
        coord_type = lines[coord_line].strip().lower()

    direct = coord_type.startswith("d")

    n_atoms = sum(counts)
    start = coord_line + 1
    coord_lines = lines[start:start + n_atoms]
    if len(coord_lines) < n_atoms:
        raise RuntimeError("POSCAR missing atomic coordinates")

    frac = []
    for ln in coord_lines:
        toks = ln.split()
        frac.append([float(toks[0]), float(toks[1]), float(toks[2])])
    frac = np.array(frac, dtype=float)

    if not direct:
        # Cartesian -> convert to fractional
        # cart = frac * scale? already included scale in lattice, so treat as Å
        cart = frac.copy()
        frac = cart @ np.linalg.inv(lattice)

    # element per atom list (1-based atom_id)
    elems = []
    for sp, ct in zip(species, counts):
        elems += [sp] * ct

    return {
        "comment": comment,
        "lattice": lattice,   # (3,3) row vectors in Å
        "frac": frac,         # (N,3)
        "elems": elems,       # length N, 1-based id -> elems[id-1]
        "species": species,
        "counts": counts
    }


def frac_to_cart(frac, lattice):
    # lattice row vectors: a,b,c
    return frac @ lattice


def min_image_frac(dfrac):
    # wrap to [-0.5, 0.5)
    return dfrac - np.round(dfrac)


# ============================================================
# 2) hr.dat onsite H(R=0)
# ============================================================
def parse_hr_dat(hr_path: Path):
    lines = hr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise RuntimeError("hr.dat too short")
    num_wann = int(lines[1].strip())
    nrpts = int(lines[2].strip())

    degen = []
    idx = 3
    while len(degen) < nrpts:
        degen.extend([int(t) for t in lines[idx].split()])
        idx += 1
    rec_lines = lines[idx:]
    return num_wann, rec_lines


def build_H0(num_wann, rec_lines):
    H0 = np.zeros((num_wann, num_wann), dtype=np.complex128)
    for ln in rec_lines:
        if not ln.strip():
            continue
        toks = ln.split()
        if len(toks) < 7:
            continue
        Rx, Ry, Rz = int(toks[0]), int(toks[1]), int(toks[2])
        if (Rx, Ry, Rz) != (0, 0, 0):
            continue
        m = int(toks[3]) - 1
        n = int(toks[4]) - 1
        H0[m, n] = float(toks[5]) + 1j * float(toks[6])
    return 0.5 * (H0 + H0.conj().T)


# ============================================================
# 3) edges.csv WF -> (atom_id, elem)
# ============================================================
def read_edges_wf_map(edges_path: Path):
    df = pd.read_csv(edges_path)
    needed = ["m", "n", "atom_m", "atom_n", "elem_m", "elem_n"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"edges.csv missing column: {c}")

    wf_to_pairs = {}
    for _, r in df.iterrows():
        m = int(r["m"]); n = int(r["n"])
        am = int(r["atom_m"]); an = int(r["atom_n"])
        em = str(r["elem_m"]); en = str(r["elem_n"])
        wf_to_pairs.setdefault(m, set()).add((am, em))
        wf_to_pairs.setdefault(n, set()).add((an, en))

    wf_map = {}
    bad = []
    for wf, s in wf_to_pairs.items():
        if len(s) != 1:
            bad.append((wf, sorted(list(s))))
        else:
            wf_map[wf] = next(iter(s))
    if bad:
        msg = ["[ERROR] WF labeling inconsistent in edges.csv (one WF -> multiple atom/elem)."]
        for wf, pairs in bad[:30]:
            msg.append(f"  wf={wf}: {pairs}")
        raise RuntimeError("\n".join(msg))
    return wf_map


def atoms_by_element_from_edges(wf_map, num_wann):
    elem_to_atoms = {}
    for wf in range(1, num_wann + 1):
        aid, elem = wf_map[wf]
        elem_to_atoms.setdefault(elem, set()).add(aid)
    return {e: sorted(list(s)) for e, s in elem_to_atoms.items()}


# ============================================================
# 4) parse win projections => expected WF orbital labels
# ============================================================
def norm_orb(tok: str) -> str:
    t = tok.strip().lower()
    t = t.replace("dx2y2", "dx2-y2").replace("dx2_y2", "dx2-y2")
    t = t.replace("d(z2)", "dz2").replace("d(z^2)", "dz2").replace("d(z**2)", "dz2")
    t = t.replace("d(x2-y2)", "dx2-y2")
    return t


def expand_token(tok: str):
    t = norm_orb(tok)
    if t == "d":
        return ["dxy", "dyz", "dxz", "dz2", "dx2-y2"]
    if t == "p":
        return ["px", "py", "pz"]
    if t == "s":
        return ["s"]
    return [t]


def parse_win_projections(win_path: Path):
    lines = win_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_proj = False
    proj_lines = []
    for ln in lines:
        s = ln.strip()
        low = s.lower()
        if low.startswith("begin projections"):
            in_proj = True
            continue
        if low.startswith("end projections"):
            in_proj = False
            continue
        if not in_proj:
            continue
        if (not s) or s.startswith("#") or s.startswith("!"):
            continue
        proj_lines.append(s)

    out = []
    for raw in proj_lines:
        if ":" not in raw:
            continue
        elem, rhs = raw.split(":", 1)
        elem = elem.strip()
        rhs = rhs.replace(",", " ").replace(";", " ")
        toks = [t for t in rhs.split() if t.strip()]
        orbs = []
        for t in toks:
            orbs += expand_token(t)
        out.append((elem, orbs))
    if not out:
        raise RuntimeError("[ERROR] No projections found in wannier90.win.")
    return out


def build_expected_labels(proj_spec, elem_to_atoms, num_wann):
    spatial = []
    for elem, orbs in proj_spec:
        if elem not in elem_to_atoms:
            raise RuntimeError(f"[ERROR] elem '{elem}' in win not found in edges-derived elements: {list(elem_to_atoms.keys())}")
        for aid in elem_to_atoms[elem]:
            for orb in orbs:
                spatial.append((aid, elem, orb))

    if len(spatial) == num_wann:
        return spatial, False
    if 2 * len(spatial) == num_wann:
        spinor = []
        for x in spatial:
            spinor.append(x); spinor.append(x)
        return spinor, True

    raise RuntimeError(f"[ERROR] Projection count mismatch: spatial={len(spatial)}, num_wann={num_wann}")


# ============================================================
# 5) Build Tc 5x5 onsite from spinor-duplicated 10x10
#    by averaging over the two duplicates per orbital label.
# ============================================================
D_ORBS = ["dxy", "dyz", "dxz", "dz2", "dx2-y2"]

def build_orbital_indices_for_atom(atom_id, expected_labels, wf_map, target_orbs):
    """
    Returns: dict orb -> list of wf indices (1-based) belonging to atom_id with that orbital label
    Requires WF ordering to match expected_labels (checked).
    """
    orb2wfs = {o: [] for o in target_orbs}
    for wf in range(1, len(expected_labels) + 1):
        aid_true, elem_true = wf_map[wf]
        aid_e, elem_e, orb_e = expected_labels[wf - 1]
        if (aid_true != aid_e) or (elem_true != elem_e):
            # ordering mismatch
            return None
        if aid_true == atom_id and orb_e in orb2wfs:
            orb2wfs[orb_e].append(wf)
    return orb2wfs


def average_subblock(H0, wfs_a, wfs_b):
    # average over all pairs (a,b)
    A = np.array([w - 1 for w in wfs_a], dtype=int)
    B = np.array([w - 1 for w in wfs_b], dtype=int)
    sub = H0[np.ix_(A, B)]
    return np.mean(sub)


def build_H5_for_atom(H0, orb2wfs):
    # 5x5 in orbital-label space
    H5 = np.zeros((5, 5), dtype=np.complex128)
    for i, oi in enumerate(D_ORBS):
        for j, oj in enumerate(D_ORBS):
            wi = orb2wfs.get(oi, [])
            wj = orb2wfs.get(oj, [])
            if len(wi) == 0 or len(wj) == 0:
                raise RuntimeError(f"Missing WF indices for orb {oi} or {oj}. Check win projections.")
            H5[i, j] = average_subblock(H0, wi, wj)
    return 0.5 * (H5 + H5.conj().T)


# ============================================================
# 6) Local axis from octahedron (PCA on neighbor vectors)
# ============================================================
def pca_local_axes(vectors, weight_mode="inv_r2"):
    """
    vectors: (6,3) Cartesian vectors from center to neighbors (Å)
    weight_mode: 'none' | 'inv_r2'
    Return R (3,3) with columns = local x',y',z' in global coords (right-handed).
    """
    V = np.array(vectors, dtype=float)
    r = np.linalg.norm(V, axis=1)
    if weight_mode == "inv_r2":
        w = 1.0 / np.maximum(r*r, 1e-12)
    else:
        w = np.ones_like(r)

    M = np.zeros((3, 3), dtype=float)
    for vi, wi in zip(V, w):
        M += wi * np.outer(vi, vi)

    evals, evecs = np.linalg.eigh(M)  # ascending
    # choose axis order: largest eigenvalue => "principal" axis (often distortion axis)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    # enforce right-handed
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0

    return evecs, evals[order]


# ============================================================
# 7) Rotate real d-orbital basis using quadratic-form trick
#    f(x)=x^T Q x. Under x = R x', Q' = R^T Q R.
# ============================================================
def d_quadratic_matrices():
    """
    Real cubic harmonics basis as traceless symmetric matrices Q:
      dxy ~ xy
      dyz ~ yz
      dxz ~ xz
      dx2-y2 ~ x^2 - y^2
      dz2 ~ 2 z^2 - x^2 - y^2  (proportional to 3z^2-r^2)
    """
    Q = {}
    Q["dxy"] = np.array([[0, 0.5, 0],
                         [0.5, 0, 0],
                         [0, 0, 0]], float)
    Q["dyz"] = np.array([[0, 0, 0],
                         [0, 0, 0.5],
                         [0, 0.5, 0]], float)
    Q["dxz"] = np.array([[0, 0, 0.5],
                         [0, 0, 0],
                         [0.5, 0, 0]], float)
    Q["dx2-y2"] = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 0]], float)
    Q["dz2"] = np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 2]], float)
    return Q


def vec_sym_traceless(Qm):
    """
    Map symmetric matrix to 5D vector in our basis space using inner products.
    We'll solve coefficients by least squares against basis matrices.
    """
    # flatten symmetric components (xx,yy,zz,xy,xz,yz) for robustness
    return np.array([Qm[0, 0], Qm[1, 1], Qm[2, 2], Qm[0, 1], Qm[0, 2], Qm[1, 2]], float)


def build_d_rotation_matrix(R):
    """
    Given 3x3 rotation R (columns are local axes in global coords),
    construct 5x5 matrix C such that coefficients transform as:
      v_local = C v_global   (in the D_ORBS order)
    Using Q'_a = R^T Q_a R and project onto basis.
    """
    Q = d_quadratic_matrices()
    B = np.stack([vec_sym_traceless(Q[o]) for o in D_ORBS], axis=1)  # (6,5)
    # solve for each rotated basis matrix
    C = np.zeros((5, 5), float)
    for a, orb in enumerate(D_ORBS):
        Qp = R.T @ Q[orb] @ R   # expressed in local coords
        vp = vec_sym_traceless(Qp)  # (6,)
        # least squares: B c = vp
        c, *_ = np.linalg.lstsq(B, vp, rcond=None)
        C[:, a] = c
    # Orthonormalize C (numerical)
    # Use QR on C^T then transpose back
    Qt, Rt = np.linalg.qr(C.T)
    C_ortho = Qt.T
    return C_ortho


# ============================================================
# 8) Nearest-neighbor search Tc->6 Se (PBC)
# ============================================================
def get_cart_vectors_to_neighbors(frac, lattice, center_idx0, neighbor_indices0):
    """
    center_idx0: 0-based index
    neighbor_indices0: list of 0-based indices
    return vectors (Nn,3) in Cartesian using minimum-image in fractional space
    """
    fc = frac[center_idx0]
    out = []
    for j0 in neighbor_indices0:
        df = frac[j0] - fc
        df = min_image_frac(df)
        out.append(df @ lattice)
    return np.array(out, float)


def find_k_nearest_neighbors(frac, lattice, center_idx0, candidate_idx0, k=6):
    fc = frac[center_idx0]
    dists = []
    for j0 in candidate_idx0:
        df = frac[j0] - fc
        df = min_image_frac(df)
        v = df @ lattice
        d = float(np.linalg.norm(v))
        dists.append((d, j0))
    dists.sort(key=lambda x: x[0])
    return dists[:k]


# ============================================================
# 9) Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", default="POSCAR")
    ap.add_argument("--hr", default="wannier90_hr.dat")
    ap.add_argument("--edges", default="edges.csv")
    ap.add_argument("--win", default="wannier90.win")
    ap.add_argument("--center", default="Tc")
    ap.add_argument("--neighbor", default="Se")
    ap.add_argument("--center_atom_id", type=int, default=None, help="If set, analyze only this POSCAR atom id (1-based)")
    ap.add_argument("--Ef", type=float, default=0.0, help="Shift energies as E_rel = E - Ef")
    ap.add_argument("--weight_mode", default="inv_r2", choices=["inv_r2", "none"])
    ap.add_argument("--out_prefix", default="TcCF")
    args = ap.parse_args()

    poscar_path = Path(args.poscar)
    hr_path = Path(args.hr)
    edges_path = Path(args.edges)
    win_path = Path(args.win)
    for p in [poscar_path, hr_path, edges_path, win_path]:
        if not p.exists():
            raise FileNotFoundError(p.resolve())

    # POSCAR
    S = read_poscar(poscar_path)
    frac = S["frac"]
    lattice = S["lattice"]
    elems = S["elems"]

    # center/neighbor indices in POSCAR (0-based)
    center_idx0 = [i for i, e in enumerate(elems) if e == args.center]
    neigh_idx0 = [i for i, e in enumerate(elems) if e == args.neighbor]
    if not center_idx0:
        raise RuntimeError(f"No center element '{args.center}' found in POSCAR element list {set(elems)}")
    if not neigh_idx0:
        raise RuntimeError(f"No neighbor element '{args.neighbor}' found in POSCAR element list {set(elems)}")

    if args.center_atom_id is not None:
        i0 = args.center_atom_id - 1
        if i0 < 0 or i0 >= len(elems):
            raise RuntimeError("center_atom_id out of range")
        if elems[i0] != args.center:
            raise RuntimeError(f"center_atom_id={args.center_atom_id} is element {elems[i0]}, not {args.center}")
        center_idx0 = [i0]

    # hr onsite
    num_wann, rec_lines = parse_hr_dat(hr_path)
    H0 = build_H0(num_wann, rec_lines)

    # WF mapping
    wf_map = read_edges_wf_map(edges_path)
    elem_to_atoms_edges = atoms_by_element_from_edges(wf_map, num_wann)
    proj_spec = parse_win_projections(win_path)
    expected_labels, spinor_dup = build_expected_labels(proj_spec, elem_to_atoms_edges, num_wann)

    # Output rows
    rows_axes = []
    rows_levels = []
    rows_neighbors = []

    for c0 in center_idx0:
        atom_id = c0 + 1  # POSCAR 1-based

        # find 6 nearest Se around this Tc
        nn = find_k_nearest_neighbors(frac, lattice, c0, neigh_idx0, k=6)
        nn_ids0 = [j0 for _, j0 in nn]
        nn_d = [d for d, _ in nn]

        # vectors center->Se
        vecs = get_cart_vectors_to_neighbors(frac, lattice, c0, nn_ids0)

        # PCA local axes
        Rloc, evals = pca_local_axes(vecs, weight_mode=args.weight_mode)  # columns: x',y',z' in global coords

        # Rotation in d space (global->local coefficients)
        C = build_d_rotation_matrix(Rloc)  # 5x5 real
        # NOTE: coefficients v_local = C v_global

        # Build Tc H5 (global d basis) from wf_map/expected_labels using atom_id in edges atom indexing.
        # Important: edges atom_id indexing must match POSCAR atom_id (you already established this mapping earlier).
        orb2wfs = build_orbital_indices_for_atom(atom_id, expected_labels, wf_map, D_ORBS)
        if orb2wfs is None:
            raise RuntimeError("WF ordering mismatch vs win expansion; cannot build orbital indices safely.")

        H5g = build_H5_for_atom(H0, orb2wfs)  # 5x5 complex hermitian
        # Rotate to local basis (treat C as real orthonormal): H_local = C H_global C^T
        H5l = C @ H5g @ C.T

        # Diagonalize local 5x5 (energies; eigenvectors in local orbital basis)
        evals_d, evecs_d = np.linalg.eigh(H5l)
        order = np.argsort(np.real(evals_d))
        evals_d = np.real(evals_d[order]) - args.Ef
        evecs_d = evecs_d[:, order]

        # record neighbor table
        for rank, (d, j0) in enumerate(nn, start=1):
            rows_neighbors.append({
                "center_atom_id": atom_id,
                "neighbor_rank": rank,
                "neighbor_atom_id": j0 + 1,
                "neighbor_elem": elems[j0],
                "distance_A": float(d)
            })

        # record axes
        rows_axes.append({
            "center_atom_id": atom_id,
            "center_elem": args.center,
            "pca_eval1": float(evals[0]),
            "pca_eval2": float(evals[1]),
            "pca_eval3": float(evals[2]),
            "xprime_global": ",".join([f"{x:.6f}" for x in Rloc[:, 0]]),
            "yprime_global": ",".join([f"{x:.6f}" for x in Rloc[:, 1]]),
            "zprime_global": ",".join([f"{x:.6f}" for x in Rloc[:, 2]]),
            "nn6_distances_A": ",".join([f"{x:.4f}" for x in nn_d]),
            "nn6_mean_A": float(np.mean(nn_d)),
            "nn6_std_A": float(np.std(nn_d)),
            "nn6_min_A": float(np.min(nn_d)),
            "nn6_max_A": float(np.max(nn_d)),
            "spinor_dup": bool(spinor_dup)
        })

        # record local CF levels and orbital characters in local basis
        for i in range(5):
            v = evecs_d[:, i]
            w = np.abs(v)**2
            dom = int(np.argmax(w))
            rows_levels.append({
                "center_atom_id": atom_id,
                "level_index": i + 1,
                "E_rel_eV": float(evals_d[i]),
                "dom_orb_local": D_ORBS[dom] + "'",
                "w_dxy_p": float(w[0]),
                "w_dyz_p": float(w[1]),
                "w_dxz_p": float(w[2]),
                "w_dz2_p": float(w[3]),
                "w_dx2y2_p": float(w[4]),
            })

    # write outputs
    out_axes = f"{args.out_prefix}_Tc_local_axes.csv"
    out_nn = f"{args.out_prefix}_Tc_Se6_neighbors.csv"
    out_lv = f"{args.out_prefix}_Tc_local_d_levels.csv"

    pd.DataFrame(rows_axes).to_csv(out_axes, index=False)
    pd.DataFrame(rows_neighbors).to_csv(out_nn, index=False)
    pd.DataFrame(rows_levels).to_csv(out_lv, index=False)

    print("[OK] wrote:")
    print(" ", out_axes)
    print(" ", out_nn)
    print(" ", out_lv)
    print("[NOTE]")
    print(" - Energies are from Tc onsite 5x5 effective d Hamiltonian (R=0), rotated into local octahedron axes by PCA.")
    print(" - dom_orb_local is defined in the local axes (x',y',z'), hence meaningful for distorted octahedron CF discussion.")


if __name__ == "__main__":
    main()
