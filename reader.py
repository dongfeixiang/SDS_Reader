# encoding = utf-8
# SDS-PAGE图像分析，根据条带灰度值计算纯度

import numpy as np
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences,peak_widths
import time


def pre_cut(img:str, cut_bg:bool=False) -> np.ndarray:
    '''图片预处理，裁剪，灰度化，背景减除'''
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
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
    img = img[top+10:,:]

    # 滑动抛物面算法减除背景
    if cut_bg:
        t1 = time.time()
        img, _ = subtract_background_rolling_ball(img, 30, use_paraboloid=True)
        print(time.time()-t1)
    
    # 灰度反转
    img = np.ones(img.shape, dtype=np.uint8)*255-np.array(img)

    return img


def gel_crop(img:np.ndarray) ->list:
    '''
    根据泳道裁剪图片
    img: 图片像素矩阵
    return: 返回切片列表
    '''
    # 平均校正算法
    # 1.均分，根据宽度15均分
    width = len(img[0])
    edges = [int((i+1)*width/15) for i in range(14)]
    # 2.校正1：横向灰度校正
    edges = gray_check(img, edges)

    plt.imshow(img, cmap="gray")
    plt.vlines(edges, 0, len(img), colors="red")
    plt.show()

    # 边界列表
    lines = [0] + edges + [len(img[0])]

    return [img[:,lines[i]:lines[i+1]] for i in range(len(lines)-1)]


def gray_check(img:np.ndarray, edges:list):
    '''横向灰度极值校正'''
    # 统计列平均灰度曲线
    img_inv = np.ones(img.shape, dtype=np.uint8)*255-np.array(img)
    col = np.array(img_inv)
    col_mean = col.mean(axis=0)

    # 查找峰顶，即分割界限
    peaks,_ = find_peaks(col_mean, height=250, distance=10, prominence=2)

    for c in col:
        plt.plot(np.arange(len(col_mean)), c)
    # plt.plot(peaks, col_mean[peaks], "x")
    plt.show()

    # 查找最近边界并校正
    for i,e in enumerate(edges):
        dis = [abs(e-p) for p in peaks]
        if min(dis) < len(img[0])/15:
            edges[i] = peaks[dis.index(min(dis))]
    return edges


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
    peaks,_ = find_peaks(intensity_profile, distance=20, prominence=200)
    h = peak_prominences(intensity_profile, peaks, wlen=100)[0]
    w,_, left, right = peak_widths(intensity_profile, peaks, rel_height=1, wlen=100)

    # 峰面积积分
    area = []
    for i in range(len(peaks)):
        pi = intensity_profile[peaks[i]]-h[i]
        peak_int = intensity_profile[int(left[i]):int(right[i])] - pi
        area.append(np.sum(peak_int))
    
    # plt.plot(np.arange(len(intensity_profile)), intensity_profile)
    # plt.plot(peaks, intensity_profile[peaks], "x")
    # plt.hlines(intensity_profile[peaks]-h, left, right)
    # plt.show()

    # 返回峰值，面积
    return (peaks, area)

def band_purity(area, band_num:int):
    '''估计条带纯度，根据主条带数'''
    sorted_area = sorted(area, reverse=True)
    return sum(sorted_area[:band_num])/sum(area)


def show(img):
    cv2.imshow("", img)
    cv2.waitKey(0)