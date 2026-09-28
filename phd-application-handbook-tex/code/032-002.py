mesh = Mesh(["k"], dim_k=2)
points = np.linspace(
    [0, 0], [1, 1], 10, endpoint=False
)  # path from (0, 0) to (0.5, 0.5) to (1, 1)
mesh.build_custom(points)
