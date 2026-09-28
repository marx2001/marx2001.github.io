
t = -1.3
delta = 2.0

model = TBModel(lattice=lattice)

# nearest-neighbour hoppings (last hop wraps to the next cell)
model.set_hop(t, 0, 1, [0])
model.set_hop(t, 1, 2, [0])
model.set_hop(t, 2, 0, [1])

# Per-orbital onsite as lambdas of lmbda
onsite = [
    lambda lmbda: delta * -np.cos(2 * np.pi * (lmbda - 0 / 3)),
    lambda lmbda: delta * -np.cos(2 * np.pi * (lmbda - 1 / 3)),
    lambda lmbda: delta * -np.cos(2 * np.pi * (lmbda - 2 / 3)),
]

model.set_onsite(onsite)
