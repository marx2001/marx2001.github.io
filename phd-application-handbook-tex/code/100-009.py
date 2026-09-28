
# get tb model in which some small terms are ignored
my_model = nso.model(
    zero_energy=fermi_ev,
    min_hopping_norm=5e-6,
)

(w90_kpt, w90_evals, w90_k_dist, w90_k_nodes, w90_k_labels) = nso.bands_w90(
    return_k_dist=True, return_k_nodes=True
)

print("k-point labels:", w90_k_labels)
print("k-point nodes (fractional):\n", w90_k_nodes)
