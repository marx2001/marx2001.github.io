from pythtb import tb_model
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==================================================
# 工作目录
# ==================================================
workdir = Path(r"E:\马睿骁\组会汇报\Nb2OSSe\pythTB\workflow")
dat_files = sorted(workdir.glob("wannier90_*.dat"))

# ==================================================
# Hermitian hopping 的规范 R 判据
# ==================================================
def canonical_R(R):
    R = tuple(R)
    return R > tuple(-r for r in R)

# ==================================================
# 从整套能带数据中找到"最大能隙"的上下边界（VBM/CBM）
# - 对绝缘体/半导体：最大能隙通常就是价带-导带之间的基本带隙
# - 对金属：最大能隙往往出现在高能处，不可靠；因此增加限制：
#   只在"能量窗口附近"寻找最大 gap（可控、更物理）
# ==================================================
def find_vbm_cbm_from_gap(E_all, search_window=2.0, gap_min=1e-3):
    """
    E_all: 1D array，包含所有 (band, k) 的能量
    search_window: 仅在 [-search_window, +search_window] eV 内寻找带隙（避免把高能处的大空洞当成带隙）
    gap_min: 最小带隙阈值（eV），小于此认为无带隙（金属/数值噪声）

    return:
        (VBM, CBM, gap) 或 (None, None, 0.0)
    """
    E = np.asarray(E_all, dtype=float)
    # 限制到窗口内以避免误判
    mask = (E >= -search_window) & (E <= search_window)
    Ew = E[mask]
    if Ew.size < 10:
        return None, None, 0.0

    # 排序并去重（去重有助于避免同一能级大量重复影响 gap 判断）
    Es = np.unique(np.sort(Ew))

    if Es.size < 10:
        return None, None, 0.0

    gaps = Es[1:] - Es[:-1]
    igap = int(np.argmax(gaps))
    gap = float(gaps[igap])

    if gap < gap_min:
        return None, None, 0.0

    VBM = float(Es[igap])     # gap 下边的上沿
    CBM = float(Es[igap + 1]) # gap 上边的下沿
    return VBM, CBM, gap

# ==================================================
# 高对称路径：Γ–X–M–Y–Γ
# ==================================================
path = [[0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [0.0, 0.0]]
label = (r"$\Gamma$", r"$X$", r"$M$", r"$Y$", r"$\Gamma$")
nk = 301

# ==================================================
# 判据参数（相对于 EF_mid = 0）
# ==================================================
E_vbm_win = 0.10     # 你的要求：VBM 靠近 EF 0.1 eV 内（即 VBM ∈ [-0.1, 0]）
search_window = 2.0  # 用于找带隙的能量搜索范围（eV），可按你体系带宽调整
gap_min = 1e-3       # 最小带隙阈值（eV），小于认为"无带隙/不可靠"

# ==================================================
# 批量处理
# ==================================================
for dat in dat_files:
    name = dat.stem
    print(f"\nProcessing {dat.name}")

    # --------------------------------------------------
    # 文件有效性检查
    # --------------------------------------------------
    if dat.stat().st_size == 0:
        print("  [SKIP] Empty file")
        continue

    text = dat.read_text(errors="ignore")
    if text.strip() == "":
        print("  [SKIP] Blank file")
        continue

    # --------------------------------------------------
    # 1) 读取 hr（需要先定义 extract_hoppings_from_hr 函数）
    # --------------------------------------------------
    # 注意：这里需要您先定义 extract_hoppings_from_hr 函数
    # 假设它返回 num_wann, onsite_raw, hoppings
    num_wann, onsite_raw, hoppings = extract_hoppings_from_hr(dat, dim=2)

    # 假设 orb_vecs 和 lat_vecs 已定义
    if num_wann != len(orb_vecs):
        print(f"  [SKIP] Orbital mismatch: {num_wann} vs {len(orb_vecs)}")
        continue

    # --------------------------------------------------
    # 2) 拆分 onsite（不做任何 EF 平移）
    # --------------------------------------------------
    onsite = onsite_raw.real.copy()
    clean_hoppings = []

    for R, i, j, t in hoppings:
        R = tuple(R)
        if R == (0, 0) and i == j:
            onsite[i] += t.real
        else:
            clean_hoppings.append((R, i, j, t))

    # --------------------------------------------------
    # 3) 构造 TB 模型
    # --------------------------------------------------
    model = tb_model(
        dim_k=2,
        dim_r=2,
        lat=lat_vecs,
        orb=orb_vecs,
    )
    model.set_onsite(onsite.tolist())

    nhop_used = 0
    for R, i, j, t in clean_hoppings:
        if (i > j) or (i == j and not canonical_R(R)):
            continue
        model.set_hop(t, i, j, list(R))
        nhop_used += 1
    print(f"  [INFO] Used hoppings: {nhop_used}")

    # --------------------------------------------------
    # 4) 能带计算（原始本征值）
    # --------------------------------------------------
    k_vec, k_dist, k_node = model.k_path(path, nk, report=False)
    evals_raw = model.solve_ham(k_pts=k_vec).T  # (nband, nk)

    # --------------------------------------------------
    # 5) 用 VBM/CBM 中点定义 EF_mid（同一套 TB 数据）
    # --------------------------------------------------
    E_all = evals_raw.ravel()
    VBM_raw, CBM_raw, gap = find_vbm_cbm_from_gap(
        E_all, search_window=search_window, gap_min=gap_min
    )

    if VBM_raw is None:
        print(f"  [SKIP] No reliable gap found within ±{search_window} eV (metal/semimetal or noisy).")
        continue

    # 计算 VBM 和 CBM 的中点作为费米能级
    EF_mid = 0.5 * (VBM_raw + CBM_raw)

    # 全局能带平移：E -> E - EF_mid
    evals = evals_raw - EF_mid

    # 平移后的 VBM/CBM（用于判据/输出）
    VBM = VBM_raw - EF_mid  # 应该是 -gap/2
    CBM = CBM_raw - EF_mid  # 应该是 +gap/2

    print(f"  [INFO] Gap = {gap:.6f} eV  (VBM_raw={VBM_raw:+.6f}, CBM_raw={CBM_raw:+.6f})")
    print(f"  [INFO] EF_mid = {EF_mid:+.6f} eV  => shifted VBM={VBM:+.6f}, CBM={CBM:+.6f}")

    # --------------------------------------------------
    # 6) 判据：VBM 靠近 EF（0.1 eV 内）
    #     平移后 EF=0，因此要求：VBM ∈ [-0.1, 0]
    # --------------------------------------------------
    if not (-E_vbm_win <= VBM <= 0.0 + 1e-12):
        print(f"  [SKIP] VBM={VBM:+.4f} eV not within {E_vbm_win:.2f} eV of EF")
        continue

    print(f"  [PASS] VBM within {E_vbm_win:.2f} eV of EF")

    # --------------------------------------------------
    # 7) 绘图（完全基于 EF_mid=0）
    # --------------------------------------------------
    calc_dir = workdir / name
    calc_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(k_node[0], k_node[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)

    for x in k_node:
        ax.axvline(x=x, lw=0.5, color="k")

    for band in evals:
        ax.plot(k_dist, band, color="black", lw=1)

    ax.axhline(0.0, color="red", ls="--", lw=1)  # EF_mid = 0
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_title(f"{name}  (EF = (VBM+CBM)/2 shifted to 0)")

    fig.savefig(calc_dir / "bandstructure.png", dpi=300)
    fig.savefig(calc_dir / "bandstructure.pdf")
    plt.close(fig)

    print(f"  [OK] Saved to {calc_dir.name}/")