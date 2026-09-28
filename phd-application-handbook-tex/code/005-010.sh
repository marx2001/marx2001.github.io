
#!/bin/bash 
#PBS -l nodes=1:ppn=64
#PBS -l walltime=640:01:00 
#PBS -j oe
#PBS -q sugon
#PBS -N test

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

	#mkdir static
	#cd static
	#cp ../POSCAR .
	#cp ../POTCAR .
	#cp ../KPOINTS .
	#cp ../incar.static ./INCAR
	#mpirun  -np $NP  ${RUN_2}>&log
	#cd ..

	for k in {1,2,3,4}
	do
	mkdir $k
	cd $k
	cp ../POSCAR .
	cp ../POTCAR .
	cp ../KPOINTS .
	cp ../INCAR$k ./INCAR
	#cp ../static/CHGCAR .
	mpirun  -np $NP  ${RUN_2}>&log

	cd ..
	done




rm -rf $PBS_JOBID-$PBS_JOBCOOKIE.hosts
