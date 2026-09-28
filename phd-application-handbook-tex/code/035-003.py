
k_vec, k_dist, k_node_dist = my_model.k_path(w90_k_nodes, nk=500, report=False)

int_evals = my_model.solve_ham(w90_kpt)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(w90_k_dist, w90_evals[:, 0] - fermi_ev, "k-", zorder=0, label="Wannier90")
ax.plot(w90_k_dist, w90_evals[:, 1:] - fermi_ev, "k-", zorder=0)
ax.plot(w90_k_dist, int_evals[:, 0], "r--", zorder=1, label="TBModel")
ax.plot(w90_k_dist, int_evals[:, 1:], "r--", zorder=1)

# set x-ticks at k-point nodes
ax.set_xticks(k_node_dist)
for n in range(len(w90_k_nodes)):
    ax.axvline(x=k_node_dist[n], linewidth=0.5, color="k", zorder=1)
ax.set_xticklabels(w90_k_labels, size=12)
ax.set_xlim(k_node_dist[0], k_node_dist[-1])
ax.set_ylabel("Band energy (eV)")
plt.show()
