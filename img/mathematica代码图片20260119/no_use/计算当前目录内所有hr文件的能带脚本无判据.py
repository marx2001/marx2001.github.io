from pythtb import tb_model
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==================================================
# 工作目录与输入文件
# ==================================================
workdir = Path(r"E:\马睿骁\组会汇报\Nb2OSSe\pythTB\workflow")
dat_files = sorted(workdir.glob("wannier90_*.dat"))

# ==================================================
# Hermitian hopping 的“规范 R”判据
# ==================================================
def canonical_R(R):
    """
    返回 True 表示该 R 是 Hermitian 共轭对中的规范代表
    规则：R 在字典序上大于 -R
    """
    R = tuple(R)
    return R > tuple(-r for r in R)

# ==================================================
# 仅用于可视化的 mid-gap 计算
# ==================================================
def find_midgap(E_all, search_window=2.0, gap_min=1e-3):
    """
    若在能量窗口内找到可靠带隙，则返回 mid-gap
    否则返回 None（如金属或拓扑相界）
    """
    E = np.asarray(E_all, dtype=float)

    mask = (E >= -search_window) & (E <= search_window)
    Ew = E[mask]
    if Ew.size < 10:
        return None

    Es = np.unique(np.sort(Ew))
    if Es.size < 10:
        return None

    gaps = Es[1:] - Es[:-1]
    igap = int(np.argmax(gaps))
    gap = gaps[igap]

    if gap < gap_min:
        return None

    VBM = Es[igap]
    CBM = Es[igap + 1]
    return 0.5 * (VBM + CBM)

# ==================================================
# 高对称路径：Γ–X–M–Y–Γ
# ==================================================
path = [
    [0.0, 0.0],
    [0.5, 0.0],
    [0.5, 0.5],
    [0.0, 0.5],
    [0.0, 0.0],
]
label = (r"$\Gamma$", r"$X$", r"$M$", r"$Y$", r"$\Gamma$")
nk = 301

# ==================================================
# 主循环：无筛选，生成所有体能带
# ==================================================
for dat in dat_files:
    name = dat.stem
    calc_dir = workdir / name
    calc_dir.mkdir(exist_ok=True)

    print(f"\nProcessing {dat.name}")

    # --------------------------------------------------
    # 1) 读取 Wannier hr
    # --------------------------------------------------
    num_wann, onsite, hoppings = extract_hoppings_from_hr(dat, dim=2)

    if num_wann != len(orb_vecs):
        print(f"  [SKIP] Orbital mismatch: {num_wann} vs {len(orb_vecs)}")
        continue

    # --------------------------------------------------
    # 2) 构造 TB 模型
    # --------------------------------------------------
    model = tb_model(
        dim_k=2,
        dim_r=2,
        lat=lat_vecs,
        orb=orb_vecs,
    )

    # 2.1 onsite（不做任何能量平移）
    model.set_onsite(onsite.real.tolist())

    # 2.2 hopping（Hermitian 严格去重）
    nhop_used = 0
    for R, i, j, t in hoppings:
        R = tuple(R)
        if (i > j) or (i == j and not canonical_R(R)):
            continue
        model.set_hop(t, i, j, list(R))
        nhop_used += 1

    print(f"  [INFO] Used hoppings: {nhop_used}")

    # --------------------------------------------------
    # 3) 体能带计算（原始本征值）
    # --------------------------------------------------
    k_vec, k_dist, k_node = model.k_path(path, nk, report=False)
    evals = model.solve_ham(k_pts=k_vec).T  # (nband, nk)

    # --------------------------------------------------
    # 3.5) 仅用于可视化：mid-gap 对齐
    # --------------------------------------------------
    midgap = find_midgap(evals.ravel(), search_window=2.0)

    if midgap is not None:
        evals_plot = evals - midgap
        title_suffix = " (mid-gap aligned)"
        print(f"  [INFO] mid-gap = {midgap:+.6f} eV → shifted to 0")
    else:
        evals_plot = evals
        title_suffix = " (no clear gap)"
        print("  [INFO] No clear gap found (possible metal or phase boundary)")

    # --------------------------------------------------
    # 4) 绘图
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(k_node[0], k_node[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)

    for x in k_node:
        ax.axvline(x=x, lw=0.5, color="k")

    for band in evals_plot:
        ax.plot(k_dist, band, color="black", lw=1)

    ax.axhline(0.0, color="red", ls="--", lw=0.8)
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_title(f"{name}{title_suffix}")

    plt.tight_layout()

    # --------------------------------------------------
    # 5) 保存结果
    # --------------------------------------------------
    fig.savefig(calc_dir / "bandstructure.png", dpi=300)
    fig.savefig(calc_dir / "bandstructure.pdf")
    plt.close(fig)

    print(f"  [OK] Saved to {calc_dir.name}/")
