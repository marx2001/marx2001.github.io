    python 1step.py --win wannier90.1.win --centres wannier90.1_centres.xyz --hr wannier90.1_hr.dat --tol 0.10 --min_absH 1e-3 --topk 30 --out_summary 1step_up.csv --pairs ALL --skip_same_atom_R0 --out_edges 2step_up.csv

    python 1step.py --win wannier90.2.win --centres wannier90.2_centres.xyz --hr wannier90.2_hr.dat --tol 0.10 --min_absH 1e-3 --topk 30 --out_summary 1step_dw.csv --pairs ALL --skip_same_atom_R0 --out_edges 2step_dw.csv
