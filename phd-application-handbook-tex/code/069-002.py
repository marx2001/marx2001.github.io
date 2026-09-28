#!/usr/bin/env python
import os
import re
from ase.io import read, write

# 获取当前目录下的所有文件夹
folders = [f for f in os.listdir('.') if os.path.isdir(f)]

# 元素符号正则表达式模式
element_pattern = re.compile(r'[A-Z][a-z]*')

# 处理每个文件夹
for folder in folders:
    # 从文件夹名称中提取元素符号
    folder_elements = element_pattern.findall(folder)
    
    if not folder_elements:
        print(f"跳过文件夹 '{folder}'：未提取到元素符号")
        continue
    
    print(f"处理文件夹 '{folder}'，提取的元素: {folder_elements}")
    
    # 构建POSCAR文件路径
    poscar_path = os.path.join(folder, 'POSCAR')
    
    if not os.path.exists(poscar_path):
        print(f"  警告: 文件夹中未找到POSCAR文件")
        continue
    
    # 读取POSCAR文件
    atoms = read(poscar_path, format='vasp')
    
    # 获取当前POSCAR中的元素符号
    current_symbols = atoms.get_chemical_symbols()
    
    # 获取当前POSCAR中所有唯一的元素（按出现顺序）
    current_unique_elements = []
    for symbol in current_symbols:
        if symbol not in current_unique_elements:
            current_unique_elements.append(symbol)
    
    # 创建一个映射：将当前元素映射到文件夹名称中的元素（从左到右）
    element_mapping = {}
    for i, elem in enumerate(folder_elements):
        if i < len(current_unique_elements):
            element_mapping[current_unique_elements[i]] = elem
    
    # 应用映射到所有原子
    final_symbols = [
        element_mapping.get(symbol, symbol) 
        for symbol in current_symbols
    ]
    
    # 设置新的化学符号
    atoms.set_chemical_symbols(final_symbols)
    
    # 备份原文件
    backup_path = os.path.join(folder, 'POSCAR.bak')
    os.rename(poscar_path, backup_path)
    print(f"  已备份原文件为: {backup_path}")
    
    # 保存新文件
    write(poscar_path, atoms, format='vasp', direct=True, vasp5=True)
    print(f"  已更新文件: {poscar_path}")
    
    # 获取最终的元素顺序
    final_unique_elements = []
    for symbol in final_symbols:
        if symbol not in final_unique_elements:
            final_unique_elements.append(symbol)
    
    print(f"  最终元素顺序: {final_unique_elements}")
    print()

print("所有文件夹处理完成！")
