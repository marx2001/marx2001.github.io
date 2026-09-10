
delta = 0
t1 = -1
t2 = 0.15
phi = np.pi / 2

model.set_onsite([-delta, delta], mode="set")

for lvec in ([0, 0], [-1, 0], [0, -1]):
    model.set_hop(t1, 0, 1, lvec, mode="set")

for lvec in ([1, 0], [-1, 1], [0, -1]):
    model.set_hop(t2 * np.exp(1j * phi), 0, 0, lvec, mode="set")
for lvec in ([-1, 0], [1, -1], [0, 1]):
    model.set_hop(t2 * np.exp(1j * phi), 1, 1, lvec, mode="set")
