# encoding = utf-8
# SDS-PAGE图像分析，根据条带灰度值计算纯度

import numpy as np
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import matplotlib.pyplot as plt


def pre_cut(img:str, cut_bg:bool=True) -> np.ndarray:
    '''图片预处理，裁剪，灰度化，背景减除'''
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # 滚球法减除背景
    if cut_bg:
        img, _ = subtract_background_rolling_ball(img, 30)

    # 根据阈值二值化
    _, img_b = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    row = np.array(img_b)

    # 统计横向平均灰度值，确定胶孔边界
    row_mean = row.mean(axis=1)
    top = 0
    for i in range(len(row_mean)):
        if row_mean[i] < 5:
            top = i
            break

    # 裁剪胶孔
    img = img[top+5:,:]

    return np.array(img)


def gel_crop(img:np.ndarray) ->list:
    '''
    根据泳道裁剪图片
    img: 图片像素矩阵
    return: 返回切片列表
    '''
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    col = np.array(img)
    col_mean = col.mean(axis=0)

    edge = []
    left = 0
    right = 0
    for i in range(len(col_mean)-1):
        if col_mean[i]>0.1 and col_mean[i+1]<=0.1:
            left = i
        elif col_mean[i]<=0.1 and col_mean[i+1]>0.1:
            right = i
            if left and right:
                edge.append((left,right))
                left = 0
                right = 0
    lines = [0]
    lines += [int((i+j)/2) for (i,j) in edge]
    lines.append(len(img[0]))
    print(lines)
    crops = [img[:,lines[i]:lines[i+1]] for i in range(len(lines)-1)]
    return crops


def show(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

img = pre_cut("1.png", cut_bg=False)
show(img)