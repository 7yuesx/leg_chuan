import numpy as np

nrow, ncol = 10, 10
size_xy = 10.0
x = np.linspace(0, size_xy, ncol)
y = np.linspace(0, size_xy, nrow)
xv, yv = np.meshgrid(x, y)
elev = 0.5 * (np.sin(2*np.pi*1.5*xv/size_xy) * np.sin(2*np.pi*2.0*yv/size_xy) + 1)
elev_str = ' '.join(f'{v:.4f}' for v in elev.ravel())
print(elev_str)  # 复制输出到 XML 的 elevation 中
elev_flat = elev_str.split()
print("生成的数字个数:", len(elev_flat))