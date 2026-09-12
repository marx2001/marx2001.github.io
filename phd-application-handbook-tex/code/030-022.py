lat_vecs = [[1, 0], [0, 1]]
orb_vecs = [[0, 0], [1 / 2, 1 / 2]]
lat = Lattice(lat_vecs=lat_vecs, orb_vecs=orb_vecs, periodic_dirs=...)
model = TBModel(lattice=lat, spinful=False)

model.set_onsite("m", ind_i=0)
model.set_onsite("m", ind_i=1)
model.set_hop("t1", 0, 1, [0, 0])
model.set_hop("t2", 1, 0, [0, 1])
print(model)
