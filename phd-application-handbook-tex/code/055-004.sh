#!/bin/bash 
#PBS -l nodes=node19:ppn=64
#PBS -l walltime=640:01:00 
#PBS -j oe
#PBS -q sugon
#PBS -N 35materials_geo2

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

# 正确定义材料列表数组
materials=(
  VScGe2S6 VGaGe2S6 VCdGe2Te6 CrScGe2Se6 CrGaGe2S6
  MnVGe2S6 MnNiGe2Te6 MnAgGe2Se6 MnAgGe2Te6 MnCuGe2Se6
  MnGaGe2Se6 MnHgGe2Te6 MnIrGe2S6 MnIrGe2Se6 MnScGe2Se6
  MnCoGe2Te6 FeZnGe2S6 FeZnGe2Se6 CoVGe2S6 CoVGe2Se6
  CoVGe2Te6 CoMoGe2Se6 CoMoGe2Te6 CoRuGe2Te6 NbCdGe2Te6
  MoAgGe2S6 MoAgGe2Te6 MoCdGe2Se6 MoHgGe2S6 MoPbGe2Se6
  MoSnGe2Te6 MoBiGe2Se6 TcAgGe2S6 TcCdGe2S6 ReScGe2Se6
)

# 循环遍历数组中的每个材料
for i in "${materials[@]}"
do
  echo "正在处理材料: $i"
  target_dir="$i"
  
  # 1. 进入材料目录
  cd "$target_dir" || { echo "无法进入目录 $target_dir"; exit 1; }
  
  # 2. 在材料目录内创建geo_2子目录
  mkdir -p "geo_2"
  
  # 3. 进入geo_2目录
  cd "geo_2" || { echo "无法进入geo_2目录"; exit 1; }
  
  # 4. 从上层目录复制CONTCAR作为POSCAR
  cp ../CONTCAR POSCAR || { echo "复制CONTCAR失败"; exit 1; }
  
  # 5. 运行vaspkit生成输入文件
  echo -e "102\n1\n0.03" | vaspkit > vaspkit_dos.log 2>&1
  vaspkit_exit_code=$?
  if [ $vaspkit_exit_code -ne 0 ]; then
    echo "错误：在 $target_dir/geo_2 中执行 vaspkit 失败（错误码 $vaspkit_exit_code）"
    cd ../..  # 返回原始工作目录
    continue
  fi
  
  # 6. 从上上层目录复制INCAR
  cp ../../INCAR . || { echo "复制INCAR失败"; exit 1; }
  
  # 7. 运行VASP计算
  mpirun -np $NP ${RUN_1} > log 2>&1
  
  # 8. 计算完成后，清理大文件
  rm -f CHG CHGCAR WAVECAR
  
  # 9. 返回原始工作目录（两次cd..）
  cd ../..
done

echo "所有材料处理完毕！"
echo end time is `date` | tee -a pbslog
