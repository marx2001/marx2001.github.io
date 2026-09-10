    python 2step.py --edges 2step_dw.csv --out dw.csv --mediators Ir --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.2_hr.dat

    python 2step.py --edges 2step_up.csv --out up.csv --mediators Ir --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.1_hr.dat

    python 2step.py --edges 2step_dw.csv --out dw.csv --mediators Ge --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.2_hr.dat

    python 2step.py --edges 2step_up.csv --out up.csv --mediators Ge --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.1_hr.dat
