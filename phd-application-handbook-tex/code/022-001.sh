
# File location
#runfile: w90.script
vasp_band_file: bnd.dat
seedname: wannier90

# system
jobname: w90
username: xpwu

# run at the local machine not via SLURM
# ! YOU HAVE TO KNOWN WHAT YOU ARE DOING
# 
# If you have any question, please ask the administrator of your server. DO AT YOUR OWN RISK
# If you choose to run locally, you don't need to give `runfile` value
# how to run your job locally. Only used when `local` is `True`.
local: True

# ! IMPORTANT: Make sure you have added `wannier90.x` to your `$PATH` if you want to run as following
# single core
localrun: wannier90.x wannier90
