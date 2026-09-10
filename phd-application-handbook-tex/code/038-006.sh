conda activate pybinding_env
conda install -n pybinding_env ipykernel --update-deps --force-reinstall
python -m ipykernel install --user --name pybinding_env --display-name "Python (pybinding_env)"
