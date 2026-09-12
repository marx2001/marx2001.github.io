velocity_single = tb.velocity(k_mesh, t=0.5)
velocity_sweep = tb.velocity(k_mesh, t=t_values)

print("Velocity shape with t=0.5:")
print(velocity_single.shape)
print("Velocity shape sweeping over 5 t values:")
print(velocity_sweep.shape)
