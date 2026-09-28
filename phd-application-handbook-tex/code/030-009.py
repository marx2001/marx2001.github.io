k_mesh = tb.k_uniform_mesh([20, 20])

# Single value (scalar)
H_single = tb.hamiltonian(k_mesh, t=0.5)

# Sweep over five scalar values (1-D array of scalars)
t_values = np.linspace(-1.0, 1.0, 5)
H_sweep = tb.hamiltonian(k_mesh, t=t_values)

print("Hamiltonian shape with t=0.5:")
print(H_single.shape)
print("Hamiltonian shape sweeping over 5 t values:")
print(H_sweep.shape)
