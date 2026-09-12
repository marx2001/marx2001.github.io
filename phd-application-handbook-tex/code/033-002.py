# v1.8 code
from pythtb import tb_model 

lat_vecs = [[1, 0], [1/2, np.sqrt(3) / 2]]
orb_vecs = [[1/3, 1/3], [2/3, 2/3]]
model = tb_model(dim_r = 2, dim_k = 2, lat=lat_vecs, orb=orb_vecs, per=[0,1], nspin=1)
