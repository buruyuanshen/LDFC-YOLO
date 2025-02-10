import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())


# 打印可用的GPU设备数量
print(torch.cuda.device_count())

# 打印每个设备的名称
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")


#打开python 终端获取路径

import matplotlib    
print(matplotlib.matplotlib_fname())

#/usr/local/lib/python3.8/dist-packages/matplotlib/mpl-data/matplotlibrc
# python 终端获取缓存路径
import matplotlib
print(matplotlib.get_cachedir())
# /root/.cache/matplotlib
# 删除缓冲目录

