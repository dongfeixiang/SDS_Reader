# encoding = utf8
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("4.jpg", cv2.IMREAD_GRAYSCALE)
img, _ = subtract_background_rolling_ball(img, 30)
_, img_b = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
# cv2.imshow("0",img)
# cv2.waitKey(0)

row = np.array(img_b)
row_mean = row.mean(axis=1)
# plt.plot(np.arange(len(col_sum)),col_sum)
# plt.show()

top = 0
for i in range(len(row_mean)):
    if row_mean[i] < 5:
        top = i
        break
img = img[top+5:,:]
# cv2.imshow("0",img)
# cv2.waitKey(0)

_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("0",img)
# cv2.waitKey(0)
col = np.array(img)
col_mean = col.mean(axis=0)
# plt.plot(np.arange(len(col_mean)),col_mean)
# plt.show()

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
# cv2.imshow("0",crops[10])
# cv2.waitKey(0)
for i in range(len(crops)):
    plt.subplot(1,len(crops),i+1)
    plt.imshow(crops[i])
plt.show()