tb_symbolic = TBModel(lattice)
tb_symbolic.set_onsite([0.0, 0.0])
tb_symbolic.set_hop("t", 0, 1, [0, 0])
tb_symbolic.set_hop(0.3, 1, 0, [1, 0])
tb_symbolic.set_hop(0.1, 1, 0, [0, 1])

tb_frozen = tb_symbolic.with_parameters(t=0.3)

# tb_symbolic still demands a parameter, tb_frozen does not
try:
    tb_symbolic.hamiltonian(k_mesh)
except ValueError as exc:
    print("symbolic:", exc)

H_frozen = tb_frozen.hamiltonian(k_mesh)
print("Frozen Hamiltonian shape:")
print(H_frozen.shape)
