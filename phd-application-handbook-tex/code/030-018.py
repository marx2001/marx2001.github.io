# Once frozen, the model no longer expects 't'
H_frozen = tb.hamiltonian(k_mesh)
print("Hamiltonian shape after freezing parameters:")
print(H_frozen.shape)
