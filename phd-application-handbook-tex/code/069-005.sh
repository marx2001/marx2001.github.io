#!/bin/bash 
#PBS -l nodes=1:ppn=64
#PBS -l walltime=640:01:00 
#PBS -j oe
#PBS -q sugon
#PBS -N 35materials_geo

cd $PBS_O_WORKDIR

echo job id is $PBS_JOBID | tee pbslog
echo run nodes is following: | tee -a pbslog
cat $PBS_NODEFILE | tee -a pbslog

echo begin time is `date` | tee -a pbslog
id=`echo $PBS_JOBID|awk -F. '{print $1}'`
NP=`cat $PBS_NODEFILE|wc -l`

source /public/software/profile.d/compiler_intel-compiler-2021.3.0.sh 
source /public/software/profile.d/mpi_intelmpi-2021.3.0.sh 
RUN_1=/public/home/cssong/app/vasp/intel_2021.3.0/5.4.4/vasp_std
RUN_2=/public/home/cssong/app/vasp/intel_2021.3.0/5.4.4/vasp_ncl
cat $PBS_NODEFILE | uniq -c | awk '{ printf("%s:%s\n", $2, $1); }' >> $PBS_JOBID-$PBS_JOBCOOKIE.hosts

# 正确定义材料列表数组
#materials=(
#VScGe2S6 VGaGe2S6 VCdGe2Te6 CrScGe2Se极 CrGaGe2S6
#MnVGe2S6 MnNiGe2Te6 MnAgGe2Se6 MnAgGe2Te6 
#MnGaGe2Se6 MnHgGe2Te6 MnIrGe2S6 MnIrGe2Se6 MnScGe2Se6
#MnCoGe2Te6 FeZnGe2S6 FeZnGe2Se6 CoV极2S6 CoVGe2Se6
#CoVGe2Te6 CoMoGe2Se6 CoMoGe2Te6 CoRuGe2Te6 NbCdGe2Te6
#MoAgGe2S6 MoAgGe2Te6 MoCdGe2Se6 MoHgGe2S6 MoPbGe2Se6
#MoSnGe2Te6 MoBiGe2Se6 TcAgGe2S6 TcCdGe2S6 ReScGe2Se6
#)

materials=(
MnGaGe2Se6 MnHgGe2Te6 MnIrGe2S6 MnIrGe2Se6 MnScGe2Se6
MnCoGe2Te6 FeZnGe2S6 FeZnGe2Se6 CoV极2S6 CoVGe2Se6
CoVGe2Te6 CoMoGe2Se6 CoMoGe2Te6 CoRuGe2Te6 NbCdGe2Te6
MoAgGe2S6 MoAgGe2Te6 MoCdGe2Se6 MoHgGe2S6 MoPbGe2Se6
MoSnGe2Te6 MoBiGe2Se6 TcAgGe2S6 TcCdGe2S6 ReScGe2Se6
)

# 1. 创建3_geo文件夹
echo "创建3_geo文件夹..."
mkdir -p 3_geo
if [ $? -eq 0 ]; then
    echo "成功创建3_geo文件夹"
else
    echo "错误：无法创建3_geo文件夹"
    exit 1
fi

# 2. 循环遍历数组中的每个材料
echo "开始复制CONTCAR文件..."
for i in "${materials[@]}"
do
  echo "正在处理材料: $i"
  
  # 2.1 检查geo_2文件夹是否存在
  if [ ! -d "$i/geo_2" ]; then
    echo "警告：目录 $i/geo_2 不存在，跳过此材料"
    continue
  fi
  
  # 2.2 进入材料目录的geo_2文件夹
  echo "进入目录 $i/geo_2"
  cd "$i/geo_2" || { echo "错误：无法进入目录 $i/geo_2"; continue; }
  
  # 2.3 检查CONTCAR文件是否存在
  if [ ! -f "CONTCAR" ]; then
    echo "警告：CONTCAR文件不存在于 $i/geo_2，跳过此材料"
    cd ../..
    continue
  fi
  
  # 2.4 复制CONTCAR
  echo "复制CONTCAR到 ../../3_geo/${i}_CONTCAR"
  cp CONTCAR ../../3_geo/"$i"_CONTCAR
  if [ $? -eq 0 ]; then
    echo "成功复制CONTCAR"
  else
    echo "错误：复制CONTCAR失败"
    cd ../..
    continue
  fi
  
  # 2.5 返回两次到脚本所在目录
  cd ../..
  echo "返回脚本所在目录"
done

echo "已完成CONTCAR复制，共处理了 $(ls 3_geo/*_CONTCAR 2>/dev/null | wc -l) 个文件"

# 3. 在3_geo文件夹中为每种材料创建子文件夹
echo "开始在3_geo中创建材料子文件夹..."
for i in "${materials[@]}"
do
  # 3.1 检查CONTCAR文件是否存在
  if [ ! -f "3_geo/${i}_CONTCAR" ]; then
    echo "警告：CONTCAR文件 3_geo/${i}_CONTCAR 不存在，跳过此材料"
    continue
  fi
  
  # 3.2 在3_geo中创建材料文件夹
  echo "创建目录 3_geo/$i"
  mkdir -p "3_geo/$i"
  if [ $? -eq 0 ]; then
    echo "成功创建目录 3_geo/$i"
  else
    echo "错误：无法创建目录 3_geo/$i"
    continue
  fi
  
  # 3.3 将复制的CONTCAR移动到新文件夹并重命名为POSCAR
  echo "移动文件 3_geo/${i}_CONTCAR 到 3_geo/$i/POSCAR"
  mv "3_geo/${i}_CONTCAR" "3_geo/$i/POSCAR"
  if [ $? -eq 0 ]; then
    echo "成功移动文件"
  else
    echo "错误：移动文件失败"
    continue
  fi
  
  # 3.4 进入新创建的材料文件夹
  echo "进入目录 3_geo/$i"
  cd "3_geo/$i" || { echo "错误：无法进入目录 3_geo/$i"; continue; }
  
  # 3.5 运行vaspkit生成输入文件
  echo "运行vaspkit生成输入文件..."
  echo -e "102\n1\n0.03" | vaspkit > vaspkit_dos.log 2>&1
  vaspkit_exit_code=$?
  if [ $vaspkit_exit_code -eq 0 ]; then
    echo "vaspkit运行成功"
  else
    echo "错误：在 3_geo/$i 中执行 vaspkit 失败（错误码 $vaspkit_exit_code）"
    cd ../..  # 返回原始工作目录
    continue
  fi
  
  # 3.6 从脚本所在目录复制INCAR到当前目录
  echo "复制INCAR文件..."
  cp $PBS_O_WORKDIR/INCAR .
  if [ $? -eq 0 ]; then
    echo "成功复制INCAR"
  else
    echo "错误：复制INCAR失败"
    cd ../..
    continue
  fi
  
  # 3.7 运行VASP计算
  echo "开始VASP计算..."
  mpirun -np $NP ${RUN_1} > log 2>&1
  vasp_exit_code=$?
  if [ $vasp_exit_code -eq 0 ]; then
    echo "VASP计算完成"
  else
    echo "警告：VASP计算可能出错（退出码 $vasp_exit_code）"
  fi
  
  # 3.8 计算完成后，清理大文件
  echo "清理大文件..."
  rm -f CHG CHGCAR WAVECAR
  echo "清理完成"
  
  # 3.9 返回原始工作目录
  cd ../..
  echo "返回脚本所在目录"
done

echo "所有材料处理完毕！"
echo end time is `date` | tee -a pbslog
