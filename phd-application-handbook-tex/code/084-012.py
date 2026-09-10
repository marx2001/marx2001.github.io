
import numpy as np
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================
HR_FILE = "wannier90_hr.dat"

# 手动设置费米能级（单位 eV，必须与 hr/TB 的能量参考一致）
EF = -1.4862

# 0K 阶跃占据：KBT 取很小；若想平滑（模拟有限温度）可取 0.01 eV 等
KBT = 1e-6  # eV

# “first three conduction bands”：从 CB_START_1BASED 开始取 3 条
# 你体系 VBM=43 时通常 CBM=44
CB_START_1BASED = 44
N_CB = 3

# 数值参数
N_PER_SEG = 3
KZ_FRAC = 0.0
DK_CART = 1e-4
OUT_PREFIX = "hr_kubo_EFmanual_VBsum_onecurve"

# reciprocal vectors from your wannier90.wout (Ang^-1), x,y components
b1 = np.array([0.955608, 0.551736], float)   # 1/Å
b2 = np.array([0.000011, 1.103456], float)   # 1/Å

# =========================
# Read wannier90_hr.dat
# =========================
def read_hr(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    nwan = int(lines[1].split()[0])
    nrpt = int(lines[2].split()[0])

    deg = []
    idx = 3
    while len(deg) < nrpt:
        deg += [int(x) for x in lines[idx].split()]
        idx += 1
    deg = np.array(deg[:nrpt], dtype=int)

    data = []
    for L in lines[idx:]:
        sp = L.split()
        if len(sp) < 7:
            continue
        R1,R2,R3 = map(int, sp[0:3])
        i = int(sp[3]) - 1
        j = int(sp[4]) - 1
        re = float(sp[5]); im = float(sp[6])
        data.append((R1,R2,R3,i,j,re+1j*im))
    return nwan, nrpt, deg, data

nwan, nrpt, deg, data = read_hr(HR_FILE)

# CB band indices (0-based)
cb0 = CB_START_1BASED - 1
CBANDS = list(range(cb0, cb0 + N_CB))
if cb0 < 0 or cb0 + N_CB > nwan:
    raise ValueError(f"CB_START_1BASED/N_CB out of range. Valid bands: 1..{nwan}")

# unique R points in order
R_list = [(int(r[0]), int(r[1]), int(r[2])) for r in data]
seen=set(); Runiq=[]
for R in R_list:
    if R not in seen:
        Runiq.append(R); seen.add(R)
Runiq = Runiq[:nrpt]

HR = {R: np.zeros((nwan,nwan), complex) for R in Runiq}
for (R1,R2,R3,i,j,amp) in data:
    R=(int(R1),int(R2),int(R3))
    HR[R][int(i),int(j)] = complex(amp)

for idxR,R in enumerate(Runiq):
    HR[R] = HR[R] / deg[idxR]

# =========================
# reduced <-> cart mapping
# =========================
M = np.stack([b1,b2], axis=1)     # 2x2
Minv = np.linalg.inv(M)

def red_to_cart2(kred2):  return M @ np.asarray(kred2, float)
def cart2_to_red(kcart2): return Minv @ np.asarray(kcart2, float)

# =========================
# H(k) from hr (Wannier90)
# H(k)= Σ_R H(R)/deg(R) * exp(i 2π k·R), k in reduced coords
# =========================
def H_of_kred(kred):
    kred = np.asarray(kred, float)
    if kred.size < 3:
        kred = np.array([kred[0], kred[1], 0.0], float)
    kred = kred.copy()
    kred[2] = KZ_FRAC
    k1,k2,k3 = kred
    Hk = np.zeros((nwan,nwan), complex)
    for (R1,R2,R3), HRmat in HR.items():
        phase = np.exp(2j*np.pi*(k1*R1 + k2*R2 + k3*R3))
        Hk += HRmat * phase
    return 0.5*(Hk + Hk.conj().T)

# =========================
# Fermi-Dirac (for occupied sum)
# =========================
def fermi_dirac(E):
    x = (E - EF) / max(KBT, 1e-12)
    x = np.clip(x, -200, 200)
    return 1.0 / (1.0 + np.exp(x))

# =========================
# One-k evaluation: eigen + velocities + all-band Omega_n
# Omega_n = -2 Im Σ_{m≠n} <n|vx|m><m|vy|n> / (Em-En)^2
# =========================
def omega_all_bands_at_k(kred):
    kred = np.asarray(kred, float)
    if kred.size < 3:
        kred = np.array([kred[0], kred[1], 0.0], float)
    kred = kred.copy(); kred[2] = KZ_FRAC

    H0 = H_of_kred(kred)
    E, U = np.linalg.eigh(H0)

    # velocities by Cartesian finite diff
    kcart2 = red_to_cart2(kred[:2])

    kx_p = kcart2 + np.array([DK_CART, 0.0])
    kx_m = kcart2 - np.array([DK_CART, 0.0])
    kred_p = kred.copy(); kred_p[:2] = cart2_to_red(kx_p)
    kred_m = kred.copy(); kred_m[:2] = cart2_to_red(kx_m)
    vx = (H_of_kred(kred_p) - H_of_kred(kred_m)) / (2.0 * DK_CART)

    ky_p = kcart2 + np.array([0.0, DK_CART])
    ky_m = kcart2 - np.array([0.0, DK_CART])
    kred_p = kred.copy(); kred_p[:2] = cart2_to_red(ky_p)
    kred_m = kred.copy(); kred_m[:2] = cart2_to_red(ky_m)
    vy = (H_of_kred(kred_p) - H_of_kred(kred_m)) / (2.0 * DK_CART)

    # eigenbasis
    Udag = U.conj().T
    Vx = Udag @ vx @ U
    Vy = Udag @ vy @ U

    dE = (E[None,:] - E[:,None])
    denom = dE*dE
    np.fill_diagonal(denom, np.inf)

    P = Vx * Vy.T
    omega = -2.0 * np.imag(np.sum(P / denom, axis=1))
    omega = np.real(omega)
    return E, omega

# =========================
# k-path: Γ–K'–M–K–Γ (reduced)
# =========================
G  = np.array([0.0,0.0,0.0])
Kp = np.array([2/3,-1/3,0.0])     # K'
Mpt= np.array([1/2,0.0,0.0])
K  = np.array([1/3,1/3,0.0])

nodes=[("G",G),("Kp",Kp),("M",Mpt),("K",K),("G",G)]

def interp(k0,k1,npt):
    t=np.linspace(0,1,npt)
    return (1-t)[:,None]*k0[None,:] + t[:,None]*k1[None,:]

k_list=[]; tick=[0]; lab=[nodes[0][0]]; pos=0
for i in range(len(nodes)-1):
    _,p0=nodes[i]
    name1,p1=nodes[i+1]
    seg=interp(p0,p1,N_PER_SEG)
    if i>0: seg=seg[1:]
    k_list.append(seg)
    pos += len(seg)
    tick.append(pos-1); lab.append(name1)

k_path=np.vstack(k_list)
x=np.arange(len(k_path))

# =========================
# Compute band-group Omega along path
# =========================
Omega_VB = np.zeros(len(k_path), float)     # occupied sum (valence bands)
Omega_CB3 = np.zeros(len(k_path), float)   # sum of first three CB (optional)

for i, kred in enumerate(k_path):
    E, omega_all = omega_all_bands_at_k(kred)

    f = fermi_dirac(E)               # occupied weights
    Omega_VB[i] = np.sum(f * omega_all)
    Omega_CB3[i] = np.sum(omega_all[CBANDS])

    if i % 50 == 0:
        print(f"[INFO] {i}/{len(k_path)}  Omega_VB={Omega_VB[i]: .6f}")

# =========================
# Save
# =========================
out_dat = OUT_PREFIX + ".dat"
header = "idx k1 k2 EF(eV) Omega_VB(occ_sum) Omega_CB3(sum_first3CB)"
np.savetxt(out_dat,
           np.column_stack([x, k_path[:,0], k_path[:,1], np.full_like(x, EF, float), Omega_VB, Omega_CB3]),
           fmt="%.10f", header=header)
print("[INFO] saved", out_dat)

# =========================
# Plot: ONE curve (literature-style single line)
# Here: Valence bands (occupied sum)
# =========================
plt.figure(figsize=(8.0,3.6), dpi=220)
plt.plot(x, Omega_VB, lw=2.0)
plt.axhline(0, lw=0.8)
for tp in tick:
    plt.axvline(tp, lw=0.6)
plt.xticks(tick, lab)
plt.xlabel("k-path: G–K'–M–K–G")
plt.ylabel(r"$\Omega_{xy}$ (valence bands, occupied sum)")
plt.tight_layout()
out_png = OUT_PREFIX + ".png"
plt.savefig(out_png, bbox_inches="tight")
plt.show()
print("[INFO] saved", out_png)
