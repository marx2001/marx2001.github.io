#有限体系下的边缘态波函数图
#fin_model = my_model.make_finite(
#    periodic_dirs=[0, 1], num_cells=[10, 10], glue_edges=[False, False]
#)
fin_model = my_model.make_finite(
    periodic_dirs=[0, 1], num_cells=[3, 3], glue_edges=[True, False]
)
(evals, evecs) = fin_model.solve_ham(return_eigvecs=True)
print("Number of states:", len(evals))
