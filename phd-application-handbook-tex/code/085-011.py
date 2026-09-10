import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator

input_csv = "data_filtered.csv"
output_dir = "./result_fig"
output_name = "data_filtered_sz.png"

discrete_num = 500
save_dpi = 300
cmap = "bwr"

os.makedirs(output_dir, exist_ok=True)

data = np.loadtxt(input_csv, delimiter=",")
x = data[:, 0]
y = data[:, 1]
sz = data[:, 5]

xi = np.linspace(x.min(), x.max(), discrete_num)
yi = np.linspace(y.min(), y.max(), discrete_num)
XI, YI = np.meshgrid(xi, yi)

interp_sz = CloughTocher2DInterpolator(np.column_stack([x, y]), sz)
SZ = interp_sz(XI, YI)
SZ = np.clip(SZ, -1, 1)

plt.figure(figsize=(6, 6))
plt.imshow(
    SZ,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    aspect="equal"
)
plt.axis("off")
plt.savefig(os.path.join(output_dir, output_name), dpi=save_dpi,
            bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

print("完成")
