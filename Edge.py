# 边缘检测算子

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys


# 高斯平滑滤波
def smooth(img, sigma, length):
    # 定义高斯核函数
    k = length // 2
    gas = np.zeros([length, length])

    for i in range(length):
        for j in range(length):
            gas[i, j] = np.exp(-((i-k) ** 2 + (j-k) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    gas = gas / np.sum(gas)

    # 进行卷积
    h, w = img.shape
    new_img = np.zeros([h, w])

    for i in range(h - k * 2):
        for j in range(w - k * 2):
            new_img[i+1, j+1] = np.sum(img[i:i+length, j:j+length] * gas)
    new_img = np.uint8(new_img)
    return new_img

# 定义sobel算子
def sobel(img):
    h, w = img.shape
    new_img = np.zeros([h, w])
    x_img = np.zeros(img.shape)     # x梯度
    y_img = np.zeros(img.shape)     # y梯度
    img_grad = np.zeros(img.shape)  # 梯度角度
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    for i in range(h-2):
        for j in range(w-2):
            x_img[i+1, j+1] = np.sum(img[i:i+3, j:j+3] * sobel_x)
            y_img[i+1, j+1] = np.sum(img[i:i+3, j:j+3] * sobel_y)
            new_img[i+1, j+1] = np.sqrt(np.square(x_img[i+1, j+1]) + np.square(y_img[i+1, j+1]))
            img_grad[i+1, j+1] = np.arctan2(y_img[i+1, j+1], x_img[i+1, j+1])
            if img_grad[i+1, j+1] < 0:
                img_grad[i+1, j+1] = 2 * np.pi + img_grad[i+1, j+1]
    img_sobel = np.uint8(new_img)
    return img_sobel, img_grad, x_img, y_img

# 非极大值抑制
def NMS(img, grad):
    h, w = img.shape
    nms = np.copy(img)

    for i in range(1, h-1):
        for j in range(1, w-1):
            theta = grad[i, j]
            k = np.tan(theta)  # k = dy/dx

            # 插值
            if (theta <= np.pi / 4) or (np.pi < theta <= 5 * np.pi / 4):
                d1 = img[i, j-1] * (1-k) + img[i+1, j-1] * k
                d2 = img[i, j+1] * (1-k) + img[i-1, j+1] * k
            elif (np.pi / 4 < theta <= np.pi / 2) or (5 * np.pi / 4 < theta < 3 * np.pi / 2):
                k = 1 / k
                d1 = img[i-1, j] * (1-k) + img[i-1, j+1] * k
                d2 = img[i+1, j] * (1-k) + img[i+1, j-1] * k
            elif (np.pi / 2 < theta <= 3 * np.pi / 4) or (3 * np.pi / 2 < theta < 7 * np.pi / 4):
                k = -1 / k
                d1 = img[i-1, j] * (1-k) + img[i-1, j-1] * k
                d2 = img[i+1, j] * (1-k) + img[i+1, j+1] * k
            else:
                k *= -1
                d1 = img[i, j-1] * (1-k) + img[i-1, j-1] * k
                d2 = img[i, j+1] * (1-k) + img[i+1, j+1] * k

            if d1 > img[i, j] or d2 > img[i, j]:
                nms[i, j] = 0

    return nms

# 阈值化边缘
def thresholding(img, min, max):
    h, w = img.shape
    edge = np.zeros([h, w])
    vis = np.zeros([h, w])  # 记录这个点是否能改变

    # 检查强边缘点附近是否有中边缘点
    def check(i, j):
        # 增大可递归深度
        sys.setrecursionlimit(10000)
        if i >= h or i < 0 or j >= w or j < 0 or vis[i, j] == 1:
            return
        vis[i, j] = 1
        if img[i, j] >= min:
            edge[i, j] = 255
            check(i - 1, j - 1)
            check(i, j - 1)
            check(i - 1, j)
            check(i - 1, j + 1)
            check(i, j + 1)
            check(i + 1, j - 1)
            check(i + 1, j)
            check(i + 1, j + 1)

    for i in range(h):
        for j in range(w):
            if vis[i, j] == 1:
                continue
            elif img[i, j] <= min:
                vis[i, j] = 1
            elif img[i, j] >= max:
                vis[i, j] = 1
                edge[i, j] = 255
                check(i - 1, j - 1)
                check(i    , j - 1)
                check(i - 1, j    )
                check(i - 1, j + 1)
                check(i    , j + 1)
                check(i + 1, j - 1)
                check(i + 1, j    )
                check(i + 1, j + 1)

    return edge

# 定义canny算子
def canny(img):
    smooth_img = smooth(img, 50, 5)
    sobel_img, grad, x, y = sobel(smooth_img)
    nms = NMS(sobel_img, grad)
    edge = thresholding(nms, 20, 60)
    return edge

