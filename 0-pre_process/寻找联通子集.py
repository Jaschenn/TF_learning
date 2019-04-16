import numpy as np
import cv2
def draw_edge(imagename):
    image = cv2.imread(imagename)  # 灰度方式读取
    heigth = len(image)
    width = len(image[0])
    cv2.imshow('原图', image)  # 显示原图像
    image = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯率波 去除躁点
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.erode(image, None)  # 腐蚀
    image = cv2.dilate(image, None)  # 膨胀

    # todo 需要将图片进行处理，比如说锐化或者是增强对比度
    def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
        rows, cols = img1.shape
        # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
        blank = np.zeros([rows, cols], img1.dtype)
        dst = cv2.addWeighted(img1, c, blank, 1 - c, b)
        return dst

    image = contrast_img(image, 2, 5)  # 第一个为对比度，第二个为亮度


    # 然后，我们需要对图片进行归一化，这样可以减少最后分割出的数字中的噪声
    # 这里我们采取了对每个像素减去图像总像素的平均数，并设置阈值50以下的像素归零来实现归一化
    # 这样基本上背景像素就变成0了
    grayscaleimg = image - int(np.mean(image))
    grayscaleimg[grayscaleimg < 127] = 0

    # 为方便标记联通域，转为bgr，实际上我不需要用到它
    thorg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    G = [[(0, 0)]]
    for x in range(0, heigth):
        print(x)
        for y in range(0, width):
            if thorg[x, y][0] == 1 and thorg[x, y][1] == 1 and thorg[x, y][2] == 1:
                # 找黑色的联通域
                count = [0, 0]  # 统计和之前几个区域相连
                for index, g in enumerate(G):
                    for i in g[::-1]:  # 倒序遍历计算量更少
                        if abs(i[0] - x) > 1:  # 相隔超过一行的点不需要看了
                            continue
                        if (abs(i[0] - x) + abs(i[1] - y)) == 1:  # 说明和之前发现的联通域相连
                            if count[0] != 1:
                                # 一个新的像素可能和之前两个联通域相连，那么他们实际上是同一个联通域，合并
                                G[count[1]] += G[index]
                                G.pop(index)  # 合并两个联通域
                                break
                            else:  # 在此联通域上增加新的像素
                                G[index].append((x, y))
                                count[0] = 1
                                count[1] = index
                                break
                if count[0] == 1:  # 新的联通域
                    G.append([(x, y)])

    print(len(G))  # 共有多少个联通域

    for i in G:
        for j in i:
            x = j[0]
            y = j[1]
            thorg[x, y][0] = 255  # 全部联通域设为浅蓝色
            thorg[x, y][1] = 255
    cv2.imshow('thorg', thorg)  #
    cv2.waitKey(0)


if __name__ == "__main__":
	path="/Users/jaschen/Desktop/"
	imagename=path+'34.png'
	draw_edge(imagename)
