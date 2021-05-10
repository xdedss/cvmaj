


# (╯‵□′)╯︵┻━┻

import cv2
import numpy as np
import numpy.linalg as npl
import time

# m
import scrutils

tileBackColor = (220, 223, 223)
tileBackColorInf = (215, 218, 218)
tileSup = (225, 228, 228)

horScanY = 650

def ptint(pt):
    return (int(pt[0]), int(pt[1]))

def colorSim(c1, c2, thr):
    return npl.norm(c1[:3] - c2[:3]) < thr

def colorLess(c1, c2):
    return (c1[0] < c2[0] and c1[1] < c2[1] and c1[2] < c2[2])

# 牌之间的缝
def isTileGap(img, x, u, d):
    for y in range(u, d, 3):
        pixel = img[y, x]
        if (isTileBack(pixel)):
            return False
    return True

# 牌底色
def isTileBack(c):
    return colorSim(c, tileBackColor, 10) or colorLess(tileSup, c)

# 定位最左牌
def findLeftMargin(img):
    res = None
    for x in range(140, 180):
        pixel = img[horScanY, x]
        if (colorSim(pixel, tileBackColor, 10)):
            res = x
            break
    if (x == None):
        print('failed to find left margin')
        return 151
    return res

# 定位整张牌左右边界
def findTileHor(img, startX, u, d):
    l = None
    r = None
    # find left border
    hasBeenIn = False
    for x in range(startX + 30, startX - 10, -1):
        pixel = img[horScanY, x]
        if ((not isTileBack(pixel)) and isTileGap(img, x, u, d)):
            if (hasBeenIn):
                l = x + 1
                break
        else:
            hasBeenIn = True
    # find right border
    hasBeenIn = False
    if (l != None):
        for x in range(l + 55, l + 75, 1):
            pixel = img[horScanY, x]
            if ((not isTileBack(pixel)) and isTileGap(img, x, u, d)):
                if (hasBeenIn):
                    r = x
                    break
            else:
                hasBeenIn = True
    if (l == None or r == None):
        print('failed to find l or r (%s, %s)' % (l, r))
        return None
    return (l, r)

# 定位牌上下边界
def findVerticalBorder(img, x):
    u = None
    d = None
    for y in range(620, 640):
        pixel = img[y, x]
        if (colorSim(pixel, tileBackColor, 15)):
            u = y
            break
    for y in range(719, 700, -1):
        pixel = img[y, x]
        print(pixel)
        if (colorSim(pixel, tileBackColor, 15)):
            d = y
            break
    if (u == None or d == None):
        print('failed to find u or d (%s, %s)' % (u, d))
        return None
    return (u, d)



if __name__ == '__main__':
    
    testImg = ['05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    testImgPath = ['samples/%s.png' % w for w in testImg]
    
    # debugOverride = cv2.imread(testImgPath[0])
    
    agent = scrutils.ScreenAgent(override=cv2.imread('samples/05.png'))
    
    agent.calibrateMultiple()
        
        
    for fpath in testImgPath:
        start = time.time()
        # start timing
        agent.override = cv2.imread(fpath)
        agent.capture()
        canvas = agent.current.copy()
        
        r = findLeftMargin(agent.current)
        u, d = findVerticalBorder(agent.current, r + 30)
        print('left=%s u=%s d=%s' % (r, u, d))
        
        while (True):
            lr = findTileHor(agent.current, r, u, d)
            
            if (lr == None):
                break
            
            l, r = lr
            print('new lr = %s %s', l, r)
            cv2.line(canvas, [l, u], [l, d], (0, 255, 0))
            cv2.line(canvas, [r, u], [r, d], (0, 0, 255))
            
            
            cv2.imshow('v', cv2.resize(agent.current[u:d, l:r], (32, 32)))
            cv2.waitKey(100)
            
            
        
        # end timing
        print(time.time() - start)
        
        #cv2.imshow('v', canvas)
        #cv2.waitKey(10)
    
    #match = agent.matchPatch(cv2.imread('patch1.png'))
    #print(match)
    #agent.click((match[0] + match[1]) / 2)
    
