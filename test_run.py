# import subprocess
#
# result = subprocess.check_output(['/media/ubuntu/fosu2/anaconda/envs/unet/bin/python','/media/ubuntu/fosu2/UNET/unet-pytorch-main/runrun_predict.py','/media/ubuntu/fosu2/UNET/unet-pytorch-main/cell_datasets/Images/0.png'])
# result = result.strip().decode('utf-8')
# print(result)
import sys
sys.path.append('/media/ubuntu/fosu2/GLM/ChatGLM-6B/cell_unet/')
sys.path.append('/media/ubuntu/fosu2/GLM/ChatGLM-6B/cell_unet/Utils')

from cell_unet.unet import Unet
from cell_unet.data_process import count_cell, merge
img=r'cell_unet/Medical_Datasets/Images/0.png'
Unet = Unet()
r_string, r_image = Unet.detect_image(img=img)