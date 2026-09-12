Reading vasprun.xml file from
    `/public/home/cssong/song/1mrx/17_AL_stack/26_Nb2OSe2_better/U=1.5/monolayer/new_wannier/1_k=11_wannier/1_static_ncl/选取合适的轨道/2_bandsoc/vasprun.xml` 
for DOS analysis...
    Fermi level : -1.06222 eV
    DOS Gap     : 0.00000 eV

Calculated DOS Energy Range: -3.06222313, 0.93777687


   species  structure_id  orb_id orb_name key_string       dos
14      Nb             1       5      dyz   Nb_1_dyz  1.000000
7       Nb             0       7      dxz   Nb_0_dxz  0.999150
15      Nb             1       6      dz2   Nb_1_dz2  0.957923
6       Nb             0       6      dz2   Nb_0_dz2  0.957750
13      Nb             1       4      dxy   Nb_1_dxy  0.490933
4       Nb             0       4      dxy   Nb_0_dxy  0.490911
17      Nb             1       8      dx2   Nb_1_dx2  0.303780
8       Nb             0       8      dx2   Nb_0_dx2  0.303752
16      Nb             1       7      dxz   Nb_1_dxz  0.287098
5       Nb             0       5      dyz   Nb_0_dyz  0.287097
38       O             4       2       pz     O_4_pz  0.216118
30      Se             3       3       px    Se_3_px  0.165473
28      Se             3       1       py    Se_3_py  0.165409
19      Se             2       1       py    Se_2_py  0.148639
21      Se             2       3       px    Se_2_px  0.148499
29      Se             3       2       pz    Se_3_pz  0.098211
37       O             4       1       py     O_4_py  0.081851
39       O             4       3       px     O_4_px  0.081809
20      Se             2       2       pz    Se_2_pz  0.074695
0       Nb             0       0        s     Nb_0_s  0.032390
9       Nb             1       0        s     Nb_1_s  0.032390
26      Se             2       8      dx2   Se_2_dx2  0.021172
35      Se             3       8      dx2   Se_3_dx2  0.018524
12      Nb             1       3       px    Nb_1_px  0.012137
1       Nb             0       1       py    Nb_0_py  0.012137
18      Se             2       0        s     Se_2_s  0.011819
27      Se             3       0        s     Se_3_s  0.010019
22      Se             2       4      dxy   Se_2_dxy  0.010018
3       Nb             0       3       px    Nb_0_px  0.009842
10      Nb             1       1       py    Nb_1_py  0.009842
24      Se             2       6      dz2   Se_2_dz2  0.008653
31      Se             3       4      dxy   Se_3_dxy  0.006800
33      Se             3       6      dz2   Se_3_dz2  0.006723
25      Se             2       7      dxz   Se_2_dxz  0.006332
23      Se             2       5      dyz   Se_2_dyz  0.006332
32      Se             3       5      dyz   Se_3_dyz  0.005652
34      Se             3       7      dxz   Se_3_dxz  0.005652
36       O             4       0        s      O_4_s  0.005392
2       Nb             0       2       pz    Nb_0_pz  0.002285
11      Nb             1       2       pz    Nb_1_pz  0.002281
40       O             4       4      dxy    O_4_dxy  0.000000
41       O             4       5      dyz    O_4_dyz  0.000000
42       O             4       6      dz2    O_4_dz2  0.000000
43       O             4       7      dxz    O_4_dxz  0.000000
44       O             4       8      dx2    O_4_dx2  0.000000

Based on your input, set the lower selection bound to 0.1 and 15 orbitals are selected.
Number of WFs selected: 15 (with degeneracy 1)

Orbitals Selected: 
  species site       orb
0      Nb   -1       [d]
1      Se   -1  [py, px]
2       O   -1      [pz]

Wannier90 Projection:
Nb:l=2
Se:py;px
O:pz

pyw90 --extra input:
Nb,0-1,4-8;Se,2-3,1|3;O,4,2

Plotted with selected orbitals in `brown` and non-selected orbitals in `orange`.
Plotted with a total of 28 orbitals, 15 of which are selected.
Figure should be stored at /public/home/cssong/song/1mrx/17_AL_stack/26_Nb2OSe2_better/U=1.5/monolayer/new_wannier/1_k=11_wannier/1_static_ncl/选取合适的轨道/2_bandsoc/dos_analysis.pdf
