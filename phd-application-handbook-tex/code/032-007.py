# Wind the first k-space dimension
mesh.loop(axis_idx=0, component_idx=0, winds_bz=True, closed=False)

# Now wind the second k-space dimension
mesh.loop(axis_idx=0, component_idx=1, winds_bz=True, closed=False)
print(mesh)
