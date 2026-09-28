#!/usr/bin/env python
import os
import subprocess
import re

# 材料列表
materials = [
    "VScGe2S6", "VGaGe2S6", "VCdGe2Te6", "CrScGe2Se6", "CrGaGe2S6",
    "MnVGe2S6", "MnNiGe2Te6", "MnAgGe2Se6", "MnAgGe2Te6", "MnCuGe2Se6",
    "MnGaGe2Se6", "MnHgGe2Te6", "MnIrGe2S6", "MnIrGe2Se6", "MnScGe2Se6",
    "MnCoGe2Te6", "FeZnGe2S6", "FeZnGe2Se6", "CoVGe2S6", "CoVGe2Se6",
    "CoVGe2Te6", "CoMoGe2Se6", "CoMoGe2Te6", "CoRuGe2Te6", "NbCdGe2Te6",
    "MoAgGe2S6", "MoAgGe2Te6", "MoCdGe2Se6", "MoHgGe2S6", "MoPbGe2Se6",
    "MoSnGe2Te6", "MoBiGe2Se6", "TcAgGe2S6", "TcCdGe2S6", "ReScGe2Se6"
]

# 存储结果
results = {}

# 检查4_DMI文件夹是否存在
if not os.path.exists("4_DMI"):
    print("错误: 4_DMI文件夹不存在")
    exit(1)

# 切换到4_DMI文件夹
os.chdir("4_DMI")
print(f"当前工作目录: {os.getcwd()}")

for material in materials:
    print(f"\n正在处理材料: {material}")
    
    # 检查材料文件夹是否存在
    if not os.path.exists(material):
        print(f"警告: 文件夹 {material} 不存在，跳过")
        results[material] = "文件夹不存在"
        continue
        
    # 进入材料文件夹
    os.chdir(material)
    print(f"  进入文件夹: {os.getcwd()}")
    
    # 在材料文件夹中运行qvasp -e命令
    try:
        # 运行qvasp -e命令获取所有能量信息
        result = subprocess.run(["qvasp", "-e"], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        print(f"  qvasp -e输出:\n{output}")
        
        # 从输出中提取cw和acw的能量值
        cw_energy = None
        acw_energy = None
        
        # 使用正则表达式匹配总结部分
        lines = output.split('\n')
        for line in lines:
            # 匹配cw行的能量值
            if 'folder:  cw' in line:
                match = re.search(r'energy:\s*([-]?\d+\.\d+)', line)
                if match:
                    cw_energy = float(match.group(1))
                    print(f"  提取到cw能量: {cw_energy} eV")
            
            # 匹配acw行的能量值
            elif 'folder:  acw' in line:
                match = re.search(r'energy:\s*([-]?\d+\.\d+)', line)
                if match:
                    acw_energy = float(match.group(1))
                    print(f"  提取到acw能量: {acw_energy} eV")
        
        # 计算最终结果 - 先乘1000再除以12
        if cw_energy is not None and acw_energy is not None:
            calculation = (cw_energy - acw_energy) * 1000 / 12
            results[material] = calculation
            print(f"  计算结果: {calculation:.6f} meV")
        else:
            error_msg = ""
            if cw_energy is None:
                error_msg += "cw能量获取失败 "
            if acw_energy is None:
                error_msg += "acw能量获取失败"
            print(f"  警告: {error_msg}")
            results[material] = error_msg
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  运行qvasp -e命令时出错: {e}")
        results[material] = f"命令执行错误: {e}"
    except ValueError as e:
        print(f"  转换能量值时出错: {e}")
        results[material] = f"数据转换错误: {e}"
        
    # 返回4_DMI文件夹
    os.chdir("..")
    print(f"  返回目录: {os.getcwd()}")

# 输出所有结果
print("\n\n最终结果:")
for material, value in results.items():
    if isinstance(value, float):
        print(f"{material}: {value:.6f} meV")
    else:
        print(f"{material}: {value}")

# 将结果保存到文件
with open("DMI_results.txt", "w") as f:
    f.write("材料\tDMI能量(meV)\n")
    for material, value in results.items():
        if isinstance(value, float):
            f.write(f"{material}\t{value:.6f}\n")
        else:
            f.write(f"{material}\t{value}\n")

print("\n所有材料处理完毕！结果已保存到DMI_results.txt")
