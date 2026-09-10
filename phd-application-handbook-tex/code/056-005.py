
# 创建有限模型：x方向周期性，y方向开边界（26个原胞）
fin_model = my_model.make_finite(periodic_dirs=[0], num_cells=[26])

# 定义x方向的k路径：Γ → X → Γ
# 由于y方向是开边界，所以ky分量设为0
k_points = [
    [0, 0],     # Γ点
    [0.5, 0],   # X点
    [0, 0]      # 返回Γ点
]

# 在路径上采样
nk_segment = 150  # 每段路径的采样点数
nk_total = 2 * nk_segment  # 总点数（两段）

# 生成路径上的k点
k_path = []
k_labels = []  # 用于标记高对称点
segment_lengths = []  # 每段路径的长度

# 第一段：Γ → X
for i in range(nk_segment):
    # 线性插值
    t = i / (nk_segment - 1)
    kx = (1 - t) * k_points[0][0] + t * k_points[1][0]
    ky = (1 - t) * k_points[0][1] + t * k_points[1][1]
    k_path.append([kx, ky])

# 第二段：X → Γ
for i in range(nk_segment):
    t = i / (nk_segment - 1)
    kx = (1 - t) * k_points[1][0] + t * k_points[2][0]
    ky = (1 - t) * k_points[1][1] + t * k_points[2][1]
    k_path.append([kx, ky])

# 计算k点路径的距离坐标（用于绘图x轴）
k_path_array = np.array(k_path)
k_dist = [0]  # 距离坐标从0开始

# 计算累积距离
for i in range(1, len(k_path)):
    # 两点之间的欧几里得距离
    dk = k_path_array[i] - k_path_array[i-1]
    dist = np.sqrt(np.sum(dk**2))
    k_dist.append(k_dist[-1] + dist)

k_dist = np.array(k_dist)

# 计算能带
bands = []
for k in k_path:
    # 注意：y方向是开边界，所以ky分量应该设为0
    # 但为了通用性，我们保留ky分量
    evals = fin_model.solve_ham(k)
    bands.append(evals)

bands = np.array(bands)  # shape: (nk_total, n_bands)

# 高对称点在距离坐标中的位置
k_node_positions = []
for k in k_points:
    # 找到路径上最近的k点位置
    k_array = np.array(k)
    distances = np.sqrt(np.sum((k_path_array - k_array)**2, axis=1))
    idx = np.argmin(distances)
    k_node_positions.append(k_dist[idx])

# 高对称点标签
k_node_labels = [
    r"$\bar{\Gamma}$",
    r"$\bar{X}$",
    r"$\bar{\Gamma}$"
]

# 绘制能带
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制所有能带
for n in range(bands.shape[1]):
    ax.plot(k_dist, bands[:, n], color="k", lw=0.5, alpha=0.7)

# 标记高对称点
for pos, label in zip(k_node_positions, k_node_labels):
    ax.axvline(x=pos, color="red", linestyle="--", alpha=0.5, lw=1)

# 设置x轴
ax.set_xticks(k_node_positions)
ax.set_xticklabels([r"$\bar{\Gamma}$", r"$\bar{X}$", r"$\bar{\Gamma}$"])

# 设置y轴
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(k_dist[0], k_dist[-1])
ax.set_xlabel(r"$k_\parallel$ (沿x方向)")
ax.set_ylabel("Energy (eV)")

plt.tight_layout()
plt.show()

# 可选：保存数据
# np.save("edge_bands_gamma_X_gamma.npy", bands)
# np.save("k_dist_gamma_X_gamma.npy", k_dist)
# np.save("k_path_gamma_X_gamma.npy", k_path_array)
