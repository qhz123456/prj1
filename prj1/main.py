# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths
from scipy import signal

image = cv2.imread('..\\img\\road\\r7.jpg', flags=1)
imgCrop = image[1000:4608,:]  #图像剪裁
imgResize = cv2.resize(imgCrop,(round(3608/7),round(3456/7))) #改变图像大小

Gauss = cv2.GaussianBlur(imgResize, (5, 5), 1)

# BGR图转为HSV
hsv = cv2.cvtColor(Gauss, cv2.COLOR_BGR2HSV)
# 提取hsv中H通道数据
# h = hsv[:, :, 0]
# s = hsv[:, :, 1]
# v = hsv[:, :, 2]
hists = cv2.calcHist([hsv], [1], None, [180], [0, 180])
histv = cv2.calcHist([hsv], [2], None, [255], [0, 255])
# histsf = savgol_filter(hists, 101, 1, mode= 'nearest')
#滤波
b, a = signal.butter(8, 0.1, 'lowpass')
filtedhists = signal.filtfilt(b, a, hists.flatten())       #data为要过滤的信号

hists_peaks, hists_properties = find_peaks(filtedhists, rel_height=0.8,width=4,distance=20,height=2000)#prominence=1,
hists_results_half = peak_widths(filtedhists.flatten(), hists_peaks, rel_height=0.8)
h_threshold_min=round(hists_results_half[3][0])

histv_peaks, histv_properties = find_peaks(histv.flatten(), rel_height=0.8,width=10,distance=255,height=2500)#prominence=1,
histv_results_half = peak_widths(histv.flatten(), histv_peaks, rel_height=0.8)


plt.plot(hists, color="r")
plt.plot(histv, color="g")
plt.plot(filtedhists, color="b")

plt.plot(hists_peaks, filtedhists[hists_peaks], "x")
plt.hlines(*hists_results_half[1:], color="C2")

plt.plot(histv_peaks, histv[histv_peaks], "x")
plt.hlines(*histv_results_half[1:], color="C3")

print("shape " ,len(hists_results_half[1]))
# 直方图显示
# plt.subplot(221),plt.hist(h, 180),plt.title("h")
# plt.subplot(222),plt.hist(s, 255),plt.title("s")
# plt.subplot(223),plt.hist(v, 255),plt.title("v")#, [0, 180]
plt.grid()
plt.show()

# gray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',hsv[:, :, 1])
# cv2.waitKey(0)

# 滑动条的回调函数，获取滑动条位置处的值
def empty(a):
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    return h_min, h_max, s_min, s_max, v_min, v_max


path = 'Resources/11.jpg'
# 创建一个窗口，放置6个滑动条
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 12, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 20, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", h_threshold_min, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", round(histv_results_half[3][0]), 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = imgResize
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 调用回调函数，获取滑动条的值
    h_min, h_max, s_min, s_max, v_min, v_max = empty(0)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    # 获得指定颜色范围内的掩码
    mask = cv2.inRange(imgHSV, lower, upper)
    # 对原图图像进行按位与的操作，掩码区域保留
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)

    cv2.waitKey(1)


size = image.shape
print(size)
print(imgCrop.shape)
print(imgResize.shape)
cv2.imshow('Example',imgResize)
# cv2.imshow('Example',imgResize)
cv2.waitKey(0)

