from pythtb import W90
from pythtb import TBModel, Lattice, WFArray, Mesh
import matplotlib.pyplot as plt
import numpy as np

nso = W90(".", "wannier90")

# hard coded fermi level in eV
fermi_ev = -1.4862 

# all pair distances between the orbitals
print("Shells:\n", nso.shells())

# plot hopping terms as a function of distance on a log scale
(dist, ham) = nso.dist_hop()
fig, ax = plt.subplots()
ax.scatter(dist, np.abs(ham))
ax.hlines(
    5e-6, xmin=0, xmax=max(dist), colors="r", linestyles="dashed", label="Cutoff = 1meV"
)
ax.legend()
ax.set_xlabel("Distance (A)")
ax.set_ylabel(r"$H$ (eV)")
ax.set_yscale("log")
