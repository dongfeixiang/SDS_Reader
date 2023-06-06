# encoding = utf-8
# SDS-PAGE图像分析，根据条带灰度值计算纯度

import numpy as np
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences,peak_widths


def pre_cut(img:str, cut_bg:bool=True) -> np.ndarray:
    '''图片预处理，裁剪，灰度化，背景减除'''
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # 滚球法减除背景
    if cut_bg:
        img, _ = subtract_background_rolling_ball(img, 30)

    # 根据阈值二值化
    _, img_b = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    row = np.array(img_b)

    # 统计纵向平均灰度值，确定胶孔边界
    row_mean = row.mean(axis=1)
    top = 0
    for i in range(len(row_mean)):
        if row_mean[i] < 5:
            top = i
            break

    # 裁剪胶孔
    img = img[top+5:,:]
    # 灰度反转
    img = np.ones(img.shape, dtype=np.uint8)*255-np.array(img)

    return img


def gel_crop(img:np.ndarray) ->list:
    '''
    根据泳道裁剪图片
    img: 图片像素矩阵
    return: 返回切片列表
    '''
    # 迭代合适的阈值，合理分割泳道
    thresh = 255
    while True:
        # 二值化
        _, img_b = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

        # 统计列平均灰度曲线
        col = np.array(img_b)
        col_mean = col.mean(axis=0)

        # 识别极小值点作为分割边界
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

        # 边界列表
        lines = [0] + [int((i+j)/2) for (i,j) in edge] + [len(img[0])]
        crops = []
        for i in range(len(lines)-1):
            if lines[i+1]-lines[i]>50:
                crops.append(img[:,lines[i]:lines[i+1]])

        # 分割15泳道即循环结束
        if len(crops) >= 15 or thresh <= 50:
            break
        else:
            thresh -= 1
    return crops


def normalize(marker_img:np.ndarray, std:list):
    '''
    根据标准Marker归一化条带大小
    marker_img: Marker胶图
    std: 标准Marker参考列表
    return: 
    '''
    # 平均灰度曲线
    intensity_profile = np.mean(marker_img, axis=1)

    # 查找峰顶坐标，即Marker对应坐标
    peaks,_ = find_peaks(intensity_profile, height=40, distance=20)

    # 线性回归：ln(Mr) = a*X + b, Mr分子量, X坐标
    p = np.poly1d(np.polyfit(peaks, np.log(std[:len(peaks)]), 1))

    # 返回可调用函数
    def func(x):
        return np.exp(p(x))
    return func


def intensiy_integrate(lane:np.ndarray):
    '''
    条带灰度积分
    lane: 泳道灰度图
    return: 返回[(MW,Area)]
    '''
    # 累积灰度曲线
    intensity_profile = np.sum(lane, axis=1)

    # 查找峰值，计算峰高，峰宽，左右边界
    peaks,_ = find_peaks(intensity_profile, height=2000)
    h = peak_prominences(intensity_profile, peaks, wlen=100)[0]
    w,_, left, right = peak_widths(intensity_profile, peaks, rel_height=1, wlen=100)

    # 峰面积积分
    area = []
    for i in range(len(peaks)):
        pi = intensity_profile[peaks[i]]-h[i]
        peak_int = intensity_profile[int(left[i]):int(right[i])] - pi
        area.append(np.sum(peak_int))
    
    plt.plot(np.arange(len(intensity_profile)), intensity_profile)
    plt.plot(peaks, intensity_profile[peaks], "x")
    plt.hlines(intensity_profile[peaks]-h, left, right)
    plt.show()

    # 返回峰值，面积
    return (peaks, area)


def show(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

img = pre_cut("1.png", cut_bg=False)
lanes = gel_crop(img)
# for i in range(len(lanes)):
#     plt.subplot(1,len(lanes),i+1)
#     plt.imshow(lanes[i])
# plt.show()
mw = normalize(lanes[7], [185,115,80,65,50,30,25,15,10])
p,a = intensiy_integrate(lanes[8])
print(mw(p),a)