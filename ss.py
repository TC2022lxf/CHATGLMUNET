# 获取 other_module.py 所在的绝对路径
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

other_module_path = os.path.abspath(os.path.join(current_dir, "..", "unet-pytorch-main"))
print(other_module_path)
sys.path.append(other_module_path)