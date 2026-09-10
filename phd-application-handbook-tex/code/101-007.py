import cv2
import numpy as np

png_file = "B15-T500-final.png"
csv_file = "2026-03-05_08-36-18_Image-00_Spins_600000.csv"

# 你的 ROI
x, y, w, h = 912, 602, 79, 68

# 读图片尺寸
img = cv2.imread(png_file)
if img is None:
    raise ValueError(f"无法读取图片: {png_file}")
H, W = img.shape[:2]

# 读 csv
data = np.loadtxt(csv_file, delimiter=',', skiprows=1)

xmin, xmax = data[:, 0].min(), data[:, 0].max()
ymin, ymax = data[:, 1].min(), data[:, 1].max()
zmin, zmax = data[:, 2].min(), data[:, 2].max()

# 像素 -> 实际坐标
x_range = (
    float(xmin + (x / W) * (xmax - xmin)),
    float(xmin + ((x + w) / W) * (xmax - xmin))
)

y_range = (
    float(ymin + ((H - (y + h)) / H) * (ymax - ymin)),
    float(ymin + ((H - y) / H) * (ymax - ymin))
)

z_range = (
    float(zmin - 1.0),
    float(zmax + 1.0)
)

print("可直接填写到 下方cell 中：")
print(f"x_range={x_range},")
print(f"y_range={y_range},")
print(f"z_range={z_range},")
