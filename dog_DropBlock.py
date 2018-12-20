#  coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import cv2

rows, cols = 224, 224
root = '/media/dhao/系统/05-weiwei/FR/dog.jpg'
img = cv2.imread(root)
img = cv2.resize(img, (rows,cols))
raw = img
block_size = 10
keep_prob = random.uniform(0.75, 0.95)
feat_size = rows
mask = np.ones((feat_size,feat_size), dtype=np.float32)

gamma = (1-keep_prob)/(block_size**2)*(feat_size**2)/(feat_size-block_size+1)**2

for i in range(block_size//2,rows-block_size//2):
    for j in range(block_size//2, cols-block_size//2):
        if random.random()<gamma:
            mask[i][j] = 0
        else:
            mask[i][j] = 1

center = np.where(mask == 0)
act= zip(center[0], center[1])

for item in act:
    mask[item[0]-block_size//2:item[0]+block_size//2, item[1]-block_size//2:item[1]+block_size//2] = 0

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.imshow('mask', mask)
cv2.waitKey(0)
mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
result = mask*raw
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imwrite('dog_dropblock.jpg',result)
cv2.imshow('result', result/255.)
cv2.waitKey(0)
cv2.destroyAllWindows()
