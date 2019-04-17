import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import random

'''
todo:读取一张图片，切分称为四部分，文件名字表示标签。然后存储到文件夹中。
注意 文件命名格式。{num}_{str}.png size = 29 * 46

'''
name = "0000.png"

def cut_pic(img):
    num1 = img[0:46, 0:30]
    num2 = img[0:46, 31:60]
    num3 = img[0:46, 61:90]
    num4 = img[0:46, 91:120]
    return [num1, num2, num3, num4]


def img2gray(i_img):
    i_img = cv2.GaussianBlur(i_img, (1, 1), 1)
    # todo:去除躁点，增加亮度，锐化。膨胀处理
    # todo:将单个的照片进行灰度处理+边缘检测+膨胀+腐蚀，然后保存
    i_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)
    i_img = cv2.adaptiveThreshold(i_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C, 17, 8)  # 二值化
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))  # 定义核 然后进行闭运算
    # cv2.morphologyEx(i_img, cv2.MORPH_CLOSE, kernel)
    i_img = cv2.resize(i_img, (29, 46), interpolation=cv2.INTER_LINEAR)
    return i_img

def save_img(img,lable):
    if int(lable) == 0:
        cv2.imwrite(os.path.join(output_dir, "0/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 1:
        cv2.imwrite(os.path.join(output_dir, "1/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 2:
        cv2.imwrite(os.path.join(output_dir, "2/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 3:
        cv2.imwrite(os.path.join(output_dir, "3/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 4:
        cv2.imwrite(os.path.join(output_dir, "4/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 5:
        cv2.imwrite(os.path.join(output_dir, "5/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 6:
        cv2.imwrite(os.path.join(output_dir, "6/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 7:
        cv2.imwrite(os.path.join(output_dir, "7/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 8:
        cv2.imwrite(os.path.join(output_dir, "8/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)
    if int(lable) == 9:
        cv2.imwrite(os.path.join(output_dir, "9/" + lable + "_" + str(random.randint(0, 9999999)) + ".png"), img)



root_dir = "/Users/jaschen/Desktop/data/images/"
output_dir = "/Users/jaschen/Desktop/data/train/"
list = os.listdir(root_dir)
for i in range(0, len(list)):
    path = os.path.join(root_dir, list[i])
    name = path[::-1]
    name = name[7:11]
    name = name[::-1]
    if os.path.isfile(path):
        image = cv2.imread(path)  # 读取文件，然后调用切割
        list_images = cut_pic(image)  # 首先切割图片，取得所有的列表。
        for j in range(0, len(list_images)):
            img = img2gray(list_images[j])
            if name[j] != "_":
                save_img(img, name[j])


