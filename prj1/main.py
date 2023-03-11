# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

cv2.namedWindow('Example', 0)
image = cv2.imread('..\\img\\road\\r6.jpg', flags=1)
imgCrop = image[1000:4608,:]  #图像剪裁
imgResize = cv2.resize(imgCrop,(round(3608/7),round(3456/7))) #改变图像大小
# BGR图转为HSV
hsv = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)
# 提取hsv中H通道数据
h = hsv[:, :, 0].ravel()
s = hsv[:, :, 1].ravel()
v = hsv[:, :, 2].ravel()
# 直方图显示
plt.subplot(221),plt.hist(h, 180)
plt.subplot(222),plt.hist(s, 180)
plt.subplot(223),plt.hist(v, 180)#, [0, 180]
plt.show()

gray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',hsv[:, :, 1])
cv2.waitKey(0)

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
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
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

