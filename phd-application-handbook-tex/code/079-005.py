#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba


def set_global_font_times():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    # Make mathtext (legend labels like $d_{xy}$) use Times New Roman too
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


def read_poscar_elements_counts(poscar_path: str) -> Tuple[List[str], List[int]]:
    with open(poscar_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 8:
        raise RuntimeError(f"POSCAR too short: {poscar_path}")
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    return elements, counts


def element_to_atom_indices_1based(elements: List[str], counts: List[int], pick: Optional[List[str]]) -> np.ndarray:
    total = sum(counts)
    if not pick:
        return np.arange(1, total + 1, dtype=int)
    pick_set = set(pick)
    out = []
    s = 1
    for el, n in zip(elements, counts):
        if el in pick_set:
            out.extend(range(s, s + n))
        s += n
    if not out:
        raise RuntimeError(f"Selected elements {pick} not found in POSCAR elements={elements}")
    return np.array(out, dtype=int)


def read_fermi_from_outcar(outcar_path: str) -> Optional[float]:
    if not os.path.exists(outcar_path):
        return None
    pat = re.compile(r"E-fermi\s*:\s*([-\d\.]+)")
    fermi = None
    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = pat.search(ln)
            if m:
                fermi = float(m.group(1))
    return fermi


# ---- plain non-italic HS labels (Times) ----
def plain_label(lbl: str) -> str:
    s = lbl.strip()
    if s.upper() == "GAMMA":
        return "Γ"
    if s in ("K'", "K’", "Kp", "KP"):
        return "K′"
    return s


def parse_vaskit_kpoints_nodes(kpoints_path: str) -> List[str]:
    coord_pat = re.compile(
        r"^\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([A-Za-z0-9\-\+'’]+)\s*$"
    )
    labels_all = []
    with open(kpoints_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = coord_pat.match(ln.strip())
            if m:
                labels_all.append(m.group(4).strip())
    if len(labels_all) < 2:
        raise RuntimeError("Failed to parse high-symmetry labels from KPOINTS.")

    compressed = []
    for lab in labels_all:
        if not compressed or lab != compressed[-1]:
            compressed.append(lab)

    cleaned = []
    i = 0
    while i < len(compressed):
        cleaned.append(compressed[i])
        if i + 2 < len(compressed) and compressed[i + 1] == compressed[i + 2]:
            i += 2
        else:
            i += 1
    return cleaned


def make_kticks_knames(nodes: List[str], nk: int) -> Tuple[List[str], List[int]]:
    nseg = len(nodes) - 1
    kticks = [int(round(i * (nk - 1) / nseg)) for i in range(len(nodes))]
    kticks[0] = 0
    kticks[-1] = nk - 1
    knames = [plain_label(x) for x in nodes]
    return knames, kticks


ORBITAL_INDEX = {
    "s": 0,
    "py": 1, "pz": 2, "px": 3,
    "dxy": 4, "dyz": 5, "dz2": 6, "dxz": 7, "x2-y2": 8,
}


def normalize_orb(s: str) -> str:
    t = s.strip().lower().replace(" ", "").replace("^", "").replace("_", "")
    if t in ("dx2-y2", "dx2y2", "x2y2", "x2-y2"):
        return "x2-y2"
    return t


def expand_orbitals(orb_str: str) -> List[str]:
    raw = [x for x in re.split(r"[,\+;\s]+", orb_str) if x.strip()]
    out = []
    for tok in raw:
        k = normalize_orb(tok)
        if k == "s":
            out.append("s")
        elif k == "p":
            out += ["px", "py", "pz"]
        elif k == "d":
            out += ["dxy", "dyz", "dz2", "dxz", "x2-y2"]
        else:
            if k not in ORBITAL_INDEX:
                raise RuntimeError(
                    f"Illegal orbital token '{tok}'. Only s/p/d are allowed."
                )
            out.append(k)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def latex_label(orb: str) -> str:
    # 图例仍用 mathtext（显示 d_{xy}），但字体我们强制 Times New Roman
    if orb == "s":
        return r"$s$"
    if orb in ("px", "py", "pz"):
        return rf"$p_{{{orb[-1]}}}$"
    if orb == "dz2":
        return r"$d_{z^2}$"
    if orb == "dxy":
        return r"$d_{xy}$"
    if orb == "dyz":
        return r"$d_{yz}$"
    if orb == "dxz":
        return r"$d_{xz}$"
    if orb == "x2-y2":
        return r"$d_{x^2-y^2}$"
    return rf"${orb}$"


def parse_procar_lm_decomposed_autospin(procar_path: str, force_nspin: Optional[int] = None):
    with open(procar_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    m = re.search(
        r"#\s*of\s*k-points:\s*([0-9]+)\s*#\s*of\s*bands:\s*([0-9]+)\s*#\s*of\s*ions:\s*([0-9]+)",
        "".join(lines)
    )
    if not m:
        raise RuntimeError("Failed to parse PROCAR header.")

    nk = int(m.group(1)); nb = int(m.group(2)); nion = int(m.group(3))
    norb = 9

    header_text = "".join(lines[:80]).lower()
    kpt1_pat = re.compile(r"^\s*k-point\s+1\s*:", re.IGNORECASE)
    kpt1_count = sum(1 for ln in lines if kpt1_pat.match(ln))

    if force_nspin is not None:
        nspin = int(force_nspin)
    else:
        if "spin component" in header_text:
            nspin = 2
        else:
            nspin = 2 if kpt1_count >= 2 else 1

    kpoints = np.zeros((nk, 3), dtype=float)
    energies = np.full((nk, nb, nspin), np.nan, dtype=float)
    weights  = np.zeros((nk, nb, nion, norb, nspin), dtype=float)

    spin_pat = re.compile(r"spin\s+component\s+([12])", re.IGNORECASE)
    kpt_pat  = re.compile(r"^\s*k-point\s+(\d+)\s*:\s*([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)")
    band_pat = re.compile(r"^\s*band\s+(\d+)\s*#\s*energy\s+([-\d\.Ee+]+)")
    ion_header_pat = re.compile(
        r"^\s*ion\s+s\s+py\s+pz\s+px\s+dxy\s+dyz\s+dz2\s+dxz\s+x2-y2\s+tot",
        re.IGNORECASE
    )

    cur_spin = 0
    cur_k = -1
    cur_b = -1

    seen_k_cycle = 0
    last_k_index = None

    i = 0
    while i < len(lines):
        ln = lines[i]

        ms = spin_pat.search(ln)
        if ms:
            cur_spin = int(ms.group(1)) - 1
            i += 1
            continue

        mk = kpt_pat.match(ln)
        if mk:
            k_index_1b = int(mk.group(1))

            if force_nspin is None and ("spin component" not in header_text):
                if last_k_index is not None and k_index_1b == 1 and last_k_index != 1:
                    seen_k_cycle += 1
                cur_spin = min(seen_k_cycle, nspin - 1)

            last_k_index = k_index_1b
            cur_k = k_index_1b - 1

            if cur_spin == 0:
                kpoints[cur_k, :] = [float(mk.group(2)), float(mk.group(3)), float(mk.group(4))]
            i += 1
            continue

        mb = band_pat.match(ln)
        if mb:
            cur_b = int(mb.group(1)) - 1
            energies[cur_k, cur_b, cur_spin] = float(mb.group(2))
            i += 1
            continue

        if ion_header_pat.match(ln):
            for j in range(nion):
                row = lines[i + 1 + j].split()
                ion_id = int(row[0]) - 1
                vals = [float(x) for x in row[1:1+norb]]
                weights[cur_k, cur_b, ion_id, :, cur_spin] = vals
            i = i + 1 + nion + 1
            continue

        i += 1

    if np.isnan(energies).any():
        nan_pos = np.argwhere(np.isnan(energies))
        raise RuntimeError(f"PROCAR parse incomplete: energies has NaNs, e.g. {nan_pos[:8].tolist()}")

    return kpoints, energies, weights, nk, nb, nion, nspin, kpt1_count


def parse_fixed_colors(s: Optional[str]) -> Dict[str, str]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise RuntimeError(f"Bad --fixed_colors token '{part}', expect key=#RRGGBB")
        k, v = part.split("=", 1)
        k = normalize_orb(k)
        v = v.strip()
        to_rgba(v)
        out[k] = v
    return out


def parse_colors_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    cols = [c.strip() for c in s.split(",") if c.strip()]
    if not cols:
        return None
    for c in cols:
        to_rgba(c)
    return cols


def auto_color_list(n: int) -> List[str]:
    cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def assign_colors(orbs: List[str], fixed: Dict[str, str], colors_list: Optional[List[str]]) -> List:
    auto = auto_color_list(len(orbs))
    out = []
    for i, o in enumerate(orbs):
        if colors_list:
            out.append(fixed.get(o, colors_list[i % len(colors_list)]))
        else:
            out.append(fixed.get(o, auto[i]))
    return out


def bold_ticks(ax):
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
        t.set_fontfamily("Times New Roman")
        t.set_fontstyle("normal")
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_fontfamily("Times New Roman")
        t.set_fontstyle("normal")


def main():
    set_global_font_times()

    ap = argparse.ArgumentParser("Fatband (s/p/d only).")
    ap.add_argument("--dirname", default=".")
    ap.add_argument("--elements", default=None)
    ap.add_argument("--orbitals", required=True)
    ap.add_argument("--fixed_colors", default=None)
    ap.add_argument("--colors", default=None)

    ap.add_argument("--fermi", type=float, default=None)
    ap.add_argument("--emin", type=float, default=-1.2)
    ap.add_argument("--emax", type=float, default=1.2)
    ap.add_argument("--out", default="fatband.png")
    ap.add_argument("--dpi", type=int, default=400)

    ap.add_argument("--plain_color", default="0.82")
    ap.add_argument("--plain_lw", type=float, default=1.1)
    ap.add_argument("--bubble_scale", type=float, default=220.0)
    ap.add_argument("--bubble_pow", type=float, default=1.0)
    ap.add_argument("--bubble_clip", type=float, default=1.0)
    ap.add_argument("--width_scale", type=float, default=1.0)

    ap.add_argument("--force_nspin", type=int, default=None)

    ap.add_argument("--legend_y", type=float, default=1.14, help="legend bbox_to_anchor y (closer to top)")
    ap.add_argument("--legend_fontsize", type=float, default=20)
    ap.add_argument("--legend_markersize", type=float, default=10.0)

    args = ap.parse_args()
    d = os.path.abspath(args.dirname)

    procar = os.path.join(d, "PROCAR")
    poscar = os.path.join(d, "POSCAR")
    kpoints = os.path.join(d, "KPOINTS")
    outcar = os.path.join(d, "OUTCAR")
    for p in [procar, poscar, kpoints]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    orbs = expand_orbitals(args.orbitals)
    if not orbs:
        raise RuntimeError("No valid orbitals parsed from --orbitals")

    elements, counts = read_poscar_elements_counts(poscar)
    pick = None
    if args.elements:
        pick = [x.strip() for x in args.elements.split(",") if x.strip()]
    sel_ions_0b = element_to_atom_indices_1based(elements, counts, pick) - 1

    fermi = args.fermi
    if fermi is None:
        f0 = read_fermi_from_outcar(outcar)
        fermi = f0 if f0 is not None else 0.0

    _, E, W, nk, nb, _, nspin, _ = parse_procar_lm_decomposed_autospin(procar, force_nspin=args.force_nspin)

    nodes = parse_vaskit_kpoints_nodes(kpoints)
    knames, kticks = make_kticks_knames(nodes, nk)

    fixed = parse_fixed_colors(args.fixed_colors)
    colors_list = parse_colors_list(args.colors)
    colors = assign_colors(orbs, fixed, colors_list)

    def bubble_size(w):
        ww = np.clip(w, 0.0, args.bubble_clip)
        ww = np.power(ww, args.bubble_pow)
        return args.width_scale * args.bubble_scale * ww

    x = np.arange(nk, dtype=float)
    XX = np.repeat(x[:, None], nb, axis=1).ravel()

    sizes_list = []
    for o in orbs:
        oi = ORBITAL_INDEX[o]
        w = W[:, :, sel_ions_0b, oi, :].sum(axis=2)
        sizes_list.append(bubble_size(w))

    scale = 0.7

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=140)

    for sp in range(nspin):
        ax.plot(x, (E[:, :, sp] - fermi), color=args.plain_color, lw=args.plain_lw, zorder=1)

    for i, o in enumerate(orbs):
        S = sizes_list[i]
        col = colors[i]
        z = 3 + i
        for sp in range(nspin):
            YY = (E[:, :, sp] - fermi).ravel()
            SS = S[:, :, sp].ravel()
            ax.scatter(XX, YY, s=SS, c=[col], marker="o", linewidths=0, alpha=1.0, zorder=z)

    ax.axhline(0.0, color="0.55", ls="--", lw=1.2, zorder=0)
    for t in kticks[1:-1]:
        ax.axvline(t, color="0.55", ls="--", lw=1.2, zorder=0)

    ax.set_xlim(0, nk - 1)
    ax.set_ylim(args.emin, args.emax)
    ax.set_yticks([-1.2, -0.6, 0.0, 0.6, 1.2])

    ax.set_ylabel("Energy (eV)", fontsize=34 * scale, fontweight="bold", fontfamily="Times New Roman")
    ax.set_xticks(kticks)
    ax.set_xticklabels(knames, fontsize=32 * scale, fontweight="bold", fontfamily="Times New Roman")

    # [OK] 只保留下/左刻度线，朝内，并把刻度标签与轴线距离设为 pad=14
    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    ax.tick_params(axis="y", pad=10, labelsize=25 * scale, width=3.0, length=8)
    ax.tick_params(axis="x", pad=10, width=3.0, length=8)

    bold_ticks(ax)

    for spn in ax.spines.values():
        spn.set_linewidth(3.0)
    ax.grid(False)

    proxies = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=colors[i], markeredgecolor="none",
               markersize=args.legend_markersize)
        for i in range(len(orbs))
    ]
    labels = [latex_label(o) for o in orbs]
    n_items = len(orbs)
    ncol = min(n_items, 5)
    nrows = int(np.ceil(n_items / ncol))

    legend_y = args.legend_y + 0.03 * (nrows - 1)

    leg = ax.legend(
        proxies, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.9,
        fontsize=args.legend_fontsize,
    )

    for txt in leg.get_texts():
        txt.set_fontweight("normal")
        txt.set_fontfamily("Times New Roman")

    top_margin = 0.88 - 0.06 * (nrows - 1)
    top_margin = max(0.70, top_margin)
    plt.subplots_adjust(top=top_margin)

    plt.tight_layout(pad=0.6)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")

    print("[OK] saved:", args.out)


if __name__ == "__main__":
    main()
