#!/bin/bash

echo "开始处理材料..."

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
  mkdir -p "$target_dir"
  cd "$target_dir" || exit 1

  # 复制POSCAR文件
  cp ../POSCAR .

  cd ..
done

echo "所有材料处理完毕！"
