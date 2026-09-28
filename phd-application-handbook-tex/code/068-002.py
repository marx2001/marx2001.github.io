
lat_vecs = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
orb_vecs = [[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]

lattice = Lattice(lat_vecs, orb_vecs, periodic_dirs=...)
tb = TBModel(lattice)

# Fixed on-site energies
tb.set_onsite([0.0, 0.0])

# Symbolic nearest-neighbour hopping: string 't'
tb.set_hop("t", 0, 1, [0, 0])
tb.set_hop(0.3, 1, 0, [1, 0])
tb.set_hop(0.1, 1, 0, [0, 1])
