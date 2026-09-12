overlap_mat = lat.lat_vecs @ lat.recip_lat_vecs.T
print(overlap_mat / (2 * np.pi))
