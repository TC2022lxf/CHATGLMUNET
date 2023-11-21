import sys
import time

import cv2
import numpy as np
from PIL import Image
import random
from unet import Unet_ONNX, Unet

def run_predict(img):
    count = False
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]

    unet = Unet()
    r_string, r_image = unet.detect_image(img, count=count, name_classes=name_classes)
    path = '/media/ubuntu/fosu2/GLM/ChatGLM-6B/img/'+str(random.randint(10000,99999))+'.png'

    cv2.imwrite(path,r_image)
    print(r_string)
    return r_string,path
if __name__ == '__main__':
    img = '/media/ubuntu/fosu2/UNET/unet-pytorch-main/cell_datasets/Images/0.png'
    print(img)
    r_string,path = run_predict(img)
    print(r_string+'---'+path)
