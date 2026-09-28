H_k = my_model.hamiltonian(k_pts)

print("Hamiltonian shape:", H_k.shape)
print("Hamiltonian at first k-point:\n", H_k[0])
print("Hamiltonian at second k-point:\n", H_k[1])
