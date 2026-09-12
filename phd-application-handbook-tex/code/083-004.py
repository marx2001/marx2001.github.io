import cv2
import os

# =========================
# 路径设置
# =========================
path = './'   # 原图文件夹
savepath = './resize/'

os.makedirs(savepath, exist_ok=True)

# =========================
# 找到所有 png
# =========================
png_files = sorted([f for f in os.listdir(path) if f.endswith('.png')])

if len(png_files) == 0:
    raise FileNotFoundError("当前文件夹下没有找到 .png 文件")

# =========================
# 用第一张图手动框选 ROI
# =========================
first_file = png_files[0]
first_path = os.path.join(path, first_file)

img0 = cv2.imread(first_path)
if img0 is None:
    raise ValueError(f"无法读取图片: {first_path}")

print(f"请在弹出的窗口中框选需要保留的区域：{first_file}")
print("操作说明：")
print("1. 鼠标左键拖动框选")
print("2. 按 Enter 或 Space 确认")
print("3. 按 c 取消重选")

x, y, w, h = cv2.selectROI("Select ROI", img0, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise ValueError("未选中有效区域，程序终止。")

print(f"选中的 ROI: x={x}, y={y}, w={w}, h={h}")
print(f"对应裁剪写法: img[{y}:{y+h}, {x}:{x+w}]")

# =========================
# 批量裁剪
# =========================
for file in png_files:
    png_name = os.path.join(path, file)
    img = cv2.imread(png_name)

    if img is None:
        print(f"跳过无法读取的文件: {file}")
        continue

    # 防止不同图片尺寸略有不同导致越界
    H, W = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)

    cropped = img[y0:y1, x0:x1]

    savepng_name = os.path.join(savepath, file)
    cv2.imwrite(savepng_name, cropped)
    print(f"已保存: {savepng_name}")

print('结束')
