#!usr/bin/python
from matplotlib import pyplot as plt
import numpy as np
import cv2

rawimg = cv2.imread(r"/Users/jaschen/Desktop/1.png")
# rawimg = cv2.imread(r"D:\LearningFiles\splitCharactersOfPicture(python+cv2)\6E57.jpg")
fig = plt.figure(figsize=(10, 15))
fig.add_subplot(2, 3, 1)
plt.title("raw image")
plt.imshow(rawimg)

fig.add_subplot(2, 3, 2)
plt.title("grey scale image")
# 在处理之前，我们首先应该将图像去RGB，即在它对应的灰度图像上进行处理。
# 我们可以使用opencv python库中的cvtColor函数来实现到灰度图像的转换
rawimg = cv2.GaussianBlur(rawimg,(3,3),0) # 高斯率波 去除躁点
grayscaleimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2GRAY)
grayscaleimg = cv2.erode(grayscaleimg,None)# 腐蚀
grayscaleimg = cv2.dilate(grayscaleimg,None)# 膨胀
# todo 需要将图片进行处理，比如说锐化或者是增强对比度
def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols = img1.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return dst


grayscaleimg = contrast_img(grayscaleimg,2,5)# 第一个为对比度，第二个为亮度

plt.imshow(grayscaleimg, cmap='gray')

# 然后，我们需要对图片进行归一化，这样可以减少最后分割出的数字中的噪声
# 这里我们采取了对每个像素减去图像总像素的平均数，并设置阈值50以下的像素归零来实现归一化
# 这样基本上背景像素就变成0了
grayscaleimg = grayscaleimg - int(np.mean(grayscaleimg))
grayscaleimg[grayscaleimg < 127] = 0


# 寻找轮廓
contours, hierarchy = cv2.findContours(grayscaleimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 声明画布 拷贝自img
canvas = np.copy(rawimg)

for cidx,cnt in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(cnt)
    print('RECT: x={}, y={}, w={}, h={}'.format(x, y, w, h))
    # 原图绘制圆形
    cv2.rectangle(canvas, pt1=(x, y), pt2=(x+w, y+h),color=(255, 0, 0), thickness=3)
    # 截取ROI图像
    cv2.imwrite("number_boudingrect_cidx_{}.png".format(cidx), grayscaleimg[y:y+h, x:x+w])

cv2.imwrite("number_boundingrect_canvas.png", canvas)
cv2.imwrite("number_boudingrect_cidx_{}.png".format(cidx), grayscaleimg[y:y+h, x:x+w])

# counting non-zero value by row , axis y
# 可以得到字符高的边界
row_nz = []
for row in grayscaleimg.tolist():
    row_nz.append(len(row) - row.count(0))
fig.add_subplot(2, 3, 3)
plt.title("non-zero values on y (by row)")
plt.plot(row_nz)

# counting non-zero value by column, x axis
# 可以得到字符宽的边界，波形的波谷即间隔
col_nz = []
for col in grayscaleimg.T.tolist():
    col_nz.append(len(col) -col.count(0))
fig.add_subplot(2, 3, 4)
plt.title("zero values on y (by col)")
plt.plot(col_nz)

##### start split
# first find upper and lower boundary of y (row)
fig.add_subplot(2, 3, 5)
plt.title("y boudary deleted")
upper_y = 0
# 遇到行不为0，即有数字时，记录行数
for i, x in enumerate(row_nz):
    if x != 0:
        upper_y = i
        break
lower_y = 0
for i, x in enumerate(row_nz[::-1]):
    if x != 0:
        lower_y = len(row_nz) - i
        break
sliced_y_img = grayscaleimg[upper_y:lower_y, :]
plt.imshow(sliced_y_img)

# then we find left and right boundary of every digital (x, on column)
column_boundary_list = []
record = False
# list[:-1],slice all the list without the last one
trigger  = np.mean(col_nz)
for i, x in enumerate(col_nz[:-1]):
    # 寻找边界i
    if (col_nz[i]==0 and  col_nz[i + 1]!=0 ) or col_nz[i] !=0 and col_nz[i + 1]==0:
        column_boundary_list.append(i + 1)
img_list = []
# i是所有左边界，[i:i+2]切片得到每个字符的左右边界
xl = [column_boundary_list[i:i + 2] for i in range(0, len(column_boundary_list), 2)]
for x in xl:
    img_list.append(sliced_y_img[:, x[0]:x[1]])

# del invalid image
# 删去宽度不大于5像素的错误图片
img_list = [x for x in img_list if x.shape[1] > 10]

# show image
fig = plt.figure()
plt.title("x boudary deleted")
for i, img in enumerate(img_list):
    fig.add_subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.imsave(r"/Users/jaschen/Desktop/%s.jpg" % i, img)

plt.show()
