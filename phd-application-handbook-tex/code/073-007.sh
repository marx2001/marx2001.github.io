conda create -n pybinding python=3.10
conda activate pybinding
conda install numpy matplotlib scipy
python -m pip install -U "pip<25.3"
pip install pybinding-dev==1.0.6 --no-use-pep517 --no-cache-dir
pip show pybinding-dev
conda install -n pybinding ipykernel --update-deps --force-reinstall
