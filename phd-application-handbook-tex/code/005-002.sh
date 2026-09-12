#!/bin/bash 
#PBS -l nodes=1:ppn=64
#PBS -l walltime=640:01:00 
#PBS -j oe
#PBS -q sugon
#PBS -N D_MP

cd $PBS_O_WORKDIR

echo job id is $PBS_JOBID | tee  pbslog
echo run nodes is following: | tee -a pbslog
cat $PBS_NODEFILE | tee  -a pbslog

echo begin time is `date` | tee -a  pbslog
id=`echo $PBS_JOBID|awk -F. '{print $1}' `
NP=`cat $PBS_NODEFILE|wc -l`


source /public/software/profile.d/compiler_intel-compiler-2021.3.0.sh 
source  /public/software/profile.d/mpi_intelmpi-2021.3.0.sh 
RUN_1=/public/home/cssong/app/vasp/intel_2021.3.0/5.4.4/vasp_std
RUN_2=/public/home/cssong/app/vasp/intel_2021.3.0/5.4.4/vasp_ncl
cat $PBS_NODEFILE | uniq -c | awk '{ printf("%s:%s\n", $2, $1); }' >> $PBS_JOBID-$PBS_JOBCOOKIE.hosts


for i in {cw,acw}
do
mkdir $i
cd $i
cp ../incar.$i ./INCAR
cp ../POTCAR .
cp ../KPOINTS .
cp ../POSCAR .
mpirun  -np $NP  ${RUN_2}>&log
rm CHG  
rm WAVECAR

cd ..
done
