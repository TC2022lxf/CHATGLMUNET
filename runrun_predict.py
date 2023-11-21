import subprocess
import sys

path = str(sys.argv[1])
result = subprocess.check_output(['/media/ubuntu/fosu2/anaconda/envs/unet/bin/python','/media/ubuntu/fosu2/UNET/unet-pytorch-main/run_predict.py',path])
result = result.strip().decode('utf-8')
print(result)