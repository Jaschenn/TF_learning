import cv2
import numpy as np
import matplotlib.pylab as plt
'''
todo:读取一张图片，切分称为四部分，文件名字表示标签。然后存储到文件夹中。
注意 文件命名格式。{num}_{str}.png size = 24 * 30 

'''
def cut_pic(img):
    num1 = img[0:46, 0:30]
    num2 = img[0:46, 31:60]
    num3 = img[0:46, 61:90]
    num4 = img[0:46, 91:120]
    return list(num1, num2, num3, num4)
def img2gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    img = cv2.GaussianBlur(img, (1, 1))
    # todo:去除躁点，增加亮度，锐化。膨胀处理
    img = cv2.cvtColor(img, -1)
    return img


