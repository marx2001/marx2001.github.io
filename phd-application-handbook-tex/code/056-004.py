#新模型，边缘态能谱的计算

fin_model = my_model.make_finite(periodic_dirs=[0], num_cells=[26])

#k_nodes = [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5], [0, 0.0]]  # G X M Y G
k_nodes = [[0, 0], [0.5, 0], [0, 0]]
k_labels = [
    r"$\bar{\Gamma}$",
    r"$\bar{X}$",
    r"$\bar{\Gamma}$",
]

fig, ax = plt.subplots(figsize=(8, 6))
fin_model.plot_bands(
    k_nodes=k_nodes, k_node_labels=k_labels, lw=1, nk=500, fig=fig, ax=ax
)

#ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
#        label=r'$E_F$ (Fermi level)')
ax.set_ylim(-0.5, 0.5)

plt.show()
fig.savefig(
    "edge_band.png",
    dpi=300,
    bbox_inches="tight"
)
