import numpy as np

def filter_csv_by_xyz(file_path, x_range, y_range, z_range, output_path=None):
    """
    根据给定的 x, y, z 范围筛选 CSV 文件中的点，并保存更新后的数据，
    同时打印筛选前后 xyz 的范围。
    """
    data = np.loadtxt(file_path, delimiter=",")

    # 筛选前范围
    print("筛选前范围：")
    print(f"x: [{data[:,0].min():.6e}, {data[:,0].max():.6e}]")
    print(f"y: [{data[:,1].min():.6e}, {data[:,1].max():.6e}]")
    print(f"z: [{data[:,2].min():.6e}, {data[:,2].max():.6e}]")

    # 筛选
    mask = (
        (data[:, 0] >= x_range[0]) & (data[:, 0] <= x_range[1]) &
        (data[:, 1] >= y_range[0]) & (data[:, 1] <= y_range[1]) &
        (data[:, 2] >= z_range[0]) & (data[:, 2] <= z_range[1])
    )
    filtered_data = data[mask]

    # 筛选后范围
    if filtered_data.size == 0:
        print("警告：筛选后没有点满足条件！")
    else:
        print("筛选后范围：")
        print(f"x: [{filtered_data[:,0].min():.6e}, {filtered_data[:,0].max():.6e}]")
        print(f"y: [{filtered_data[:,1].min():.6e}, {filtered_data[:,1].max():.6e}]")
        print(f"z: [{filtered_data[:,2].min():.6e}, {filtered_data[:,2].max():.6e}]")

    # 默认覆盖原文件
    if output_path is None:
        output_path = file_path

    np.savetxt(output_path, filtered_data, delimiter=",", fmt="%.18e")
    print(f"已筛选并保存至 {output_path}，剩余 {filtered_data.shape[0]} 个点。")

    return filtered_data.shape[0]

filter_csv_by_xyz(
    "2026-03-05_08-36-18_Image-00_Spins_600000.csv",   ## 源文件名
    x_range=(2995.7427548963037, 3255.24496400857),
    y_range=(1506.399353766607, 1699.6781734408135),
    z_range=(4.623772823642497, 7.254555104183036),
    output_path="data_filtered.csv"  ## 新文件名
)
