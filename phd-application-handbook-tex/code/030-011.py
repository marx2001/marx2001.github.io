
evals_single = tb.solve_ham(k_mesh, t=0.5)  # passing a single scalar value
evals_sweep = tb.solve_ham(k_mesh, t=t_values)  # passing a 1-D array of scalars

print("Energies shape with t=0.5:")
print(evals_single.shape)
print("Energies shape sweeping over 5 t values:")
print(evals_sweep.shape)
