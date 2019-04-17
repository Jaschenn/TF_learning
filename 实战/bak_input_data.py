'''
读取本地文件模块
'''
import cv2
import numpy as np
import os
import matplotlib.pylab as plt
input_dir = "/Users/jaschen/Desktop/data/train/"


def next_data(indx):
    list = os.listdir(input_dir)
    path = os.path.join(input_dir, list[indx])
    name = list[indx][0]  # 取得代表图片数字的值
    if os.path.isfile(path):
        # 返回一张图像 和 标签
        img = cv2.imread(path, 0)  # 灰度方式读取
        return img, name  # 返回的img是一个二维的数组

def read_datasets():

    list = os.listdir(input_dir)
    for i in range(0, len(list)):
        image = []
        label = []
        img, name = next_data(i)
        if name != ".":
            img = np.array(img).reshape(-1, 1334)
            image.extend(img)
            index = int(name)
            name = np.zeros((1, 10))
            name[0][index] = 1
            label.extend(name)  # int 不可以进行进行extend 直接append  extend是扩展列表
    image = np.array(image)
    label = np.array(label)
    return image, label
