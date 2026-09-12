usage: plot_fatbands.py [-h] [-b VASPRUN_FILE_BANDS] [-K KPOINTS_FILE] [-c POSCAR_FILE] [-O PROCAR_FILE] [-w POTCAR_FILE] [-d VASPRUN_FILE_DOS] [-p PROJECT [PROJECT ...]]
                        [-n {all,selection}] [-l {1,2,3}] [-e EMIN] [-E EMAX] [-s SCALE] [-H HEIGHT] [-W WIDTH] [-r RATIO] [-f FONT_SIZE] [-o OUTPUT_FILE] [--format {pdf,png}]
                        [--element ELEMENT] [--orbital-type {d,p,f}] [--bubble-min BUBBLE_MIN] [--bubble-max BUBBLE_MAX] [--bubble-scale BUBBLE_SCALE] [--bubble-density BUBBLE_DENSITY]
                        [--bubble-threshold BUBBLE_THRESHOLD]

Plot projected band structure (fatbands) from a VASP calculation.
Author: Marco Cappelletti. Heavily inspired by Kevin Waters (kwaters4.github.io) and sumo-bandplot.
By default is assumes that:
	the current directory contains KPOINTS, vasprun.xml from the band calculations
	the directory ../dos contains vasprun.xml from the dos calculation
	the parent directory (../) contains the POSCAR file
	the parent parent directory (../../) contains the POTCAR file

optional arguments:
  -h, --help            show this help message and exit
  -b VASPRUN_FILE_BANDS, --vasprun-file-bands VASPRUN_FILE_BANDS
                        Path of the vasprun.xml file of the band calculation (default: vasprun.xml)
  -K KPOINTS_FILE, --KPOINTS-file KPOINTS_FILE
                        Path of the KPOINTS file with the band path (default: KPOINTS)
  -c POSCAR_FILE, --POSCAR-file POSCAR_FILE
                        Path of the POSCAR file (default: ../POSCAR)
  -O PROCAR_FILE, --PROCAR-file PROCAR_FILE
                        Path of the PROCAR file from the band calculation (default: PROCAR)
  -w POTCAR_FILE, --POTCAR-file POTCAR_FILE
                        Path of the POTCAR file (default: ../../POTCAR)
  -d VASPRUN_FILE_DOS, --vasprun-file-dos VASPRUN_FILE_DOS
                        Path of the vasprun.xml file of the dos calculation (default: ../dos/vasprun.xml)
  -p PROJECT [PROJECT ...], --project PROJECT [PROJECT ...]
                        Band projection to rgb. Either 2 (red and green) or 3 arguments (red, green, blue). Nomenclature:
                        	- E: all orbitals of element E (H, C, N, O, ...)
                        	- E.o: o-orbital of element E (s, px, py, pz, dxy, ...)
                        	- E.s.pz: s+pz orbitals of element E
                        	- X.s: s orbitals of all elements
                        	- O.s.pz+N.pz: sum of O(s,pz) and N(pz)
                         (default: ['X.px', 'X.py', 'X.s.pz'])
  -n {all,selection}, --normalization {all,selection}
                        Normalization of the projection.
                        	-'all': with respect to all contributions.
                        	-'selection': with respect to selection only
                         (default: selection)
  -l {1,2,3}, --max-l {1,2,3}
                        Maximum value of l (angular momemtum) for the projection. Increases computational costs, so increase it only if necessary. (default: 1)
  -e EMIN, --emin EMIN  Minimum of energy in the plot. If none, it chooses the lower limit (default: None)
  -E EMAX, --emax EMAX  Maximum of energy in the plot. If none, it chooses the upper limit (default: None)
  -s SCALE, --scale SCALE
                        DOS scale factor (default: 2.0)
  -H HEIGHT, --height HEIGHT
                        Height of the plot in inches (default: 3.5)
  -W WIDTH, --width WIDTH
                        Width of the plot in inches (default: 3.3)
  -r RATIO, --ratio RATIO
                        Bandplot - dosplot width ratio (default: 3.0)
  -f FONT_SIZE, --font-size FONT_SIZE
                        Fontsize (default: 8)
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Path and name of the output file, excluding the format (default: fatbands)
  --format {pdf,png}    Output file format (default: pdf)
  --element ELEMENT     Element for orbital bubble plot (e.g., Fe) (default: None)
  --orbital-type {d,p,f}
                        Type of orbital to plot (d, p or f) (default: d)
  --bubble-min BUBBLE_MIN
                        Minimum bubble size (default: 0.05)
  --bubble-max BUBBLE_MAX
                        Maximum bubble size (default: 0.3)
  --bubble-scale BUBBLE_SCALE
                        Bubble size scaling factor (default: 100)
  --bubble-density BUBBLE_DENSITY
                        Bubble density (1: every point, 2: every second point, etc.) (default: 5)
  --bubble-threshold BUBBLE_THRESHOLD
                        Minimum weight to plot bubble (default: 0.05)
