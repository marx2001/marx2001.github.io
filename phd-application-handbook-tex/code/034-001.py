# v1.x
from pythtb import tb_model, wf_array
model = tb_model(1, 1, lat=[[1.0, 0.0], [0.0, 1.0]], orb=[[0, 1/3, 2/3]])
wfa = wf_array(model, [20])
wfa.solve_on_grid(0.0)
