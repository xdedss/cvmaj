

# 剪裁1280x720

import os
import cv2
import numpy as np
import numpy.linalg as npl

def colorSim(c1, c2, thr):
    return npl.norm(c1[:3] - c2[:3]) < thr


def isVertBlack(img, x):
    for y in range(10, img.shape[0] - 10, 10):
        pixel = img[y, x]
        if (not colorSim(pixel, (0, 0, 0), 30)):
            return False
    return True

def findHorBorder(img):
    interval = 10
    startX = None
    l = None
    r = None
    for x in range(0, 500, interval):
        if (not isVertBlack(img, x)):
            startX = x - interval
            break
    for x in range(startX, startX + interval + 1):
        if (not isVertBlack(img, x)):
            l = x
            break
    for x in range(img.shape[1]-1, img.shape[1]-1 - 500, -interval):
        if (not isVertBlack(img, x)):
            startX = x + interval
            break
    for x in range(startX, startX - interval - 1, -1):
        if (not isVertBlack(img, x)):
            r = x + 1
            break
    return (l, r)

print(os.listdir())
for fname in os.listdir():
    img = cv2.imread(fname)
    if (np.any(img == None)):
        continue
    h = img.shape[0]
    w = img.shape[1]
    if (h != 720 or w != 1280):
        print(fname)
        l, r = findHorBorder(img)
        print(l, r)
        res = cv2.resize(img[1:h-1, l:r], (1280, 720))
        cv2.imwrite(fname, res)

