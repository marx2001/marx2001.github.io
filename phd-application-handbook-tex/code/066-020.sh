
#!/bin/bash
#PBS -l nodes=node15:ppn=64
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q sugon
#PBS -N w90   

cd $PBS_O_WORKDIR

echo "------------------------------------"
echo "Wannier90 job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Working directory: $(pwd)"
echo "------------------------------------"

# 模块加载
source /public/software/profile.d/compiler_intel-compiler-2021.3.0.sh 
source /public/software/profile.d/mpi_intelmpi-2021.3.0.sh 

# Conda 环境激活（请确认路径）
source /public/software/profile.d/apps_anaconda3-2021.05.sh
conda activate pyw90_env

# 确认环境加载正确
echo "Python path: $(which python)"
python -c "import pyw90; print('Pyw90 path:', pyw90.__file__)"

# 运行 W90
EXE=/public/home/xpwu/pack/wannier90-3.1.0/bin/wannier90.x
SYSTEM="wannier90" 

NP=$(cat $PBS_NODEFILE | wc -l)
cat $PBS_NODEFILE | uniq -c | awk '{ printf("%s:%s\n", $2, $1); }' > $PBS_JOBID-$PBS_JOBCOOKIE.hosts

echo "Running Wannier90 on $NP cores..."
mpirun -np $NP $EXE $SYSTEM > wannier90.log

echo "------------------------------------"
echo "Wannier90 job finished at: $(date)"
echo "------------------------------------"

rm -f $PBS_JOBID-$PBS_JOBCOOKIE.hosts
