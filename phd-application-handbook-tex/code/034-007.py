
model_fixed = model.with_parameters(lmbda=0.25)
wfa.solve_model(model_fixed)
