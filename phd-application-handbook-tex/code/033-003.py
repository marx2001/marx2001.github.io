lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]
lat = Lattice(lat_vecs=lat_vecs, orb_vecs=orb_vecs, periodic_dirs=[0, 1])
model = TBModel(lattice=lat, spinful=False)
