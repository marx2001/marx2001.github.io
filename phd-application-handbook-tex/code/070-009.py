
mesh = Mesh(dim_k=2, axis_types=["k"])

# Path from (-0.5,-0.5) to (0.5, 0.5) including endpoints (endpoint=True)
points = np.linspace([-0.5, -0.5], [0.5, 0.5], 10, endpoint=True)

mesh.build_custom(points)
print(mesh)
