

# (╯‵□′)╯︵┻━┻

# 画面分割

import math
import cv2
import numpy as np
import numpy.linalg as npl

tileBackColor = (220, 223, 223)
tileBackColorInf = (215, 218, 218)
tileSup = (225, 228, 228)

horScanY = 640


# verbose print
verbose = False
def vprint(*o):
    if (verbose):
        print(*o)

def ptint(pt):
    return (int(pt[0]), int(pt[1]))

def v(*args):
    return np.array(args, dtype=float)

def vint(*args):
    return np.array(args, dtype=int)

def v32(*args):
    return np.array(args, dtype=np.float32)

# https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def calcIou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def colorSim(c1, c2, thr):
    return npl.norm(c1[:3] - c2[:3]) < thr

def colorLess(c1, c2):
    return (c1[0] < c2[0] and c1[1] < c2[1] and c1[2] < c2[2])

# --------------------------------手牌--------------------------------------


# 牌之间的缝
def isTileGap(img, x, u, d, tol=10):
    for y in range(u, d, 3):
        pixel = img[y, x]
        if (isTileBack(pixel, tol)):
            return False
    return True

# 牌底色
def isTileBack(c, tol=10):
    return colorSim(c, tileBackColor, 10) or colorLess(tileSup, c)

# 定位最左牌
def findLeftMargin(img):
    res = None
    for x in range(140, 180):
        pixel = img[horScanY, x]
        if (colorSim(pixel, tileBackColor, 10)):
            res = x
            break
    if (res == None):
        print('failed to find left margin')
        raise Exception('can not find left margin')
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
        vprint('failed to find l or r (%s, %s)' % (l, r))
        return None
    return (l, r)

# 定位牌上下边界
def findVerticalBorder(img, x):
    u = None
    d = None
    for y in range(570, 640):
        pixel = img[y, x]
        if (isTileBack(pixel)):
            u = y
            break
    for y in range(719, 650, -1):
        pixel = img[y, x]
        if (isTileBack(pixel)):
            d = y
            break
    if (u == None or d == None):
        print('failed to find u or d (%s, %s)' % (u, d))
        raise Exception('can not find vertical border')
    return (u, d)

# 从全图提取自己的牌
def extractTilesImg(img):
    r = findLeftMargin(img)
    u, d = findVerticalBorder(img, r + 30)
    vprint('[extractTiles] left=%s u=%s d=%s' % (r, u, d))
    res = []
    while (True):
        lr = findTileHor(img, r, u, d)
        if (lr == None):
            vprint('[extractTiles] No more tiles')
            break
        
        l, r = lr
        u, d = findVerticalBorder(img, int((l + r) / 2))
        vprint('[extractTiles] new tile %s ~ %s  %s ~ %s' % (l, r, u, d))
        
        res.append((vint(l, u), vint(r, d), img[u:d, l:r]))
    if (len(res) == 0):
        raise Exception('no tile found!')
    return res

def wrapPers(m, xy):
    uv = np.matmul(m, [xy[0], xy[1], 1])
    uv /= uv[2]
    return uv[:2]



# --------------------------------牌河-------------------------------------


# 中间区域透视校正
def extractCenterRegeon(img):
    ulc = [349, 81]
    urc = [932, 81]
    blc = [237, 564]
    brc = [1043, 564]
    size = 800
    mat = cv2.getPerspectiveTransform(v32(ulc, urc, blc, brc), 
        v32([0, 0], [size, 0], [0, size], [size, size]))
    res = cv2.warpPerspective(img, mat, (size, size))
    return res

# 底色分割
def tileBackRange(img):
    tol = 5
    #print(img.shape, (tileBackColor[0] - tol, tileBackColor[1] - tol, tileBackColor[2] - tol), (tileBackColor[0] + tol, tileBackColor[1] + tol, tileBackColor[2] + tol))
    seg = cv2.inRange(img, 
        (tileBackColor[0] - tol, tileBackColor[1] - tol, tileBackColor[2] - tol), 
        (tileBackColor[0] + tol, tileBackColor[1] + tol, tileBackColor[2] + tol))
    seg2 = cv2.inRange(img, tileSup, (255, 255, 255))
    res = cv2.bitwise_or(seg, seg2)
    #res = cv2.Laplacian(res, cv2.CV_8U, ksize=3)
    #res = cv2.GaussianBlur(res, (5, 5), 3)
    return res

## 双向不等价的角度比较
#def angleSim(a1, a2, thr = 10 / 180 * np.pi):
#    return abs((a1 - a2 + np.pi) % (2 * np.pi) - np.pi) < thr
#
## 双向等价的角度比较
#def angleSim2(a1, a2, thr = 10 / 180.0 * np.pi):
#    return abs((a1 - a2 + np.pi/2) % np.pi - np.pi/2) < thr
#
## check vertical edge at x from ys to ye
#def checkVerticalEdge(img_mag, img_dir, x, ys, ye):
#    for y in range(ys, ye+1, 2):
#        px_mag = img_mag[y, x]
#        px_dir = img_dir[y, x]
#        if (not angleSim2(px_dir, 0)):
#            return False
#    return True
#
## check if edge exists
#def checkEdge(img_mag, img_dir, pstart, pend):
#    pstart = np.array(pstart, dtype=float)
#    pend = np.array(pend, dtype=float)
#    edge_dir = (pend - pstart)
#    edge_len = npl.norm(edge_dir)
#    edge_dir /= edge_len
#    edge_normdir = math.atan2(-edge_dir[0], edge_dir[1])
#    for i in range(0, math.ceil(edge_len), 2):
#        x, y = ptint(pstart + i * edge_dir)
#        px_dir = img_dir[y, x]
#        px_mag = img_mag[y, x]
#        if ((angleSim2(px_dir, edge_normdir)) or (px_mag < 50)):
#            pass # safe
#        else:
#            return False
#    return True

# get match box with best iou   return box,iou
def getBestMatchBox(box, boxlist):
    bestBox = None
    bestIou = 0
    for box1 in boxlist:
        iou = calcIou([box[0][0], box[0][1], box[1][0], box[1][1]],
            [box1[0][0], box1[0][1], box1[1][0], box1[1][1]])
        if (iou > bestIou):
            bestIou = iou
            bestBox = box1
    return bestBox, bestIou # could be None

# check shape
def isInGoodShape(box, p_width, p_height):
    p1, p2 = box
    w = p2[0] - p1[0]
    h = p2[1] - p1[1]
    area = w * h
    p_area = p_width * p_height
    if (max(p_area/area, area/p_area) > 1.1):
        return False
    if (max(w/p_width, p_width/w) > 1.05):
        return False
    if (max(h/p_height, p_height/h) > 1.05):
        return False
    return True

# 一家的打出的牌
def extractPartTilesImg(img):
    img_seg = tileBackRange(img)
    edged = cv2.Canny(img_seg, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vprint('[extractPartTilesImg] found %s contours' % len(contours))
    contours_outer = []
    for i in range(len(contours)):
        if (hierarchy[0][i][3] == -1):
            contours_outer.append(contours[i])
    
    # 粗筛
    #canvas = img.copy()
    rect_candidates = []
    for contour in contours_outer:
        x,y,w,h = cv2.boundingRect(contour)
        if (w * h >= 600 and max(w/h, h/w) < 4):
            rect_candidates.append(([x, y], [x+w, y+h]))
            #cv2.rectangle(canvas, [x, y], [x+w, y+h], (0, 255, 0), 1)
        else:
            #cv2.rectangle(canvas, [x, y], [x+w, y+h], (0, 255, 255), 1)
            pass
    vprint('[extractPartTilesImg] %s candidates' % len(rect_candidates))
    
    # prior params
    p_width = 43 # 牌宽
    p_height = 56 # 牌高
    p_winterval = 46 # 竖直摆放的水平间隔
    p_wwinterval = 61 # 水平摆放的水平间隔
    p_hinterval = 62 # 竖直摆放的竖直间隔
    p_horsink = 6 # 水平摆放的高度差
    p_start = [10, 10] 
    p_maxX = 260 # maximum line length
    p_maxLine = 3 # maximum line num
    # match window
    cur = p_start
    curLine = 0
    rects = []
    while (True):
        estBox= [cur, [cur[0]+p_width, cur[1]+p_height]]
        estBoxHor = [[cur[0], cur[1]+p_horsink], [cur[0]+p_height, cur[1]+p_width]]
        vprint('[extractPartTilesImg] check est box %s %s' % (estBox, estBoxHor))
        bestBox, iou = getBestMatchBox(estBox, rect_candidates)
        bestBoxHor, iouHor = getBestMatchBox(estBoxHor, rect_candidates)
        vprint('[extractPartTilesImg] matching box %s(%s) %s(%s)' % (bestBox, iou, bestBoxHor, iouHor))
        if (bestBox == None):
            if (cur[0] > p_maxX):
                # try next line
                vprint('[extractPartTilesImg] line break')
                curLine += 1
                if (curLine >= p_maxLine):
                    #oh no
                    break
                cur = [p_start[0], p_start[1] + p_hinterval * curLine]
            else:
                # maybe try next
                vprint('[extractPartTilesImg] no luck, skipping tile')
                cur = [cur[0] + p_winterval, cur[1]]
        else:
            # something is here
            if (iou > (iouHor-0.1)): # more likely to be vertical
                #it might be vertical
                vprint('[extractPartTilesImg] found vertical tile')
                if (isInGoodShape(bestBox, p_width, p_height)):
                    rects.append((bestBox, iou))
                    cur = [bestBox[0][0] + p_winterval, bestBox[0][1]]
                else:
                    rects.append((estBox, iou))
                    cur = [estBox[0][0] + p_winterval, estBox[0][1]]
            else:
                #it might be horizontal
                vprint('[extractPartTilesImg] found horizontal tile')
                if (isInGoodShape(bestBoxHor, p_height, p_width)):
                    rects.append((bestBoxHor, iouHor))
                    cur = [bestBox[0][0] + p_wwinterval, bestBox[0][1] - p_horsink]
                else:
                    rects.append((estBoxHor, iouHor))
                    cur = [estBoxHor[0][0] + p_wwinterval, estBoxHor[0][1] - p_horsink]
        
#    for p1, p2 in rects:
#        cv2.rectangle(canvas, p1, p2, (0, 0, 255), 1)
#    cv2.imshow('canvas', canvas)
#    cv2.waitKey(0)
    
    res = []
    for (p1, p2), iou in rects:
        w = p2[0] - p1[0]
        h = p2[1] - p1[1]
        if (w > h):
            res.append((vint(*p1), vint(*p2), cv2.rotate(img[p1[1]:p2[1], p1[0]:p2[0]], cv2.ROTATE_90_COUNTERCLOCKWISE)))
        else:
            res.append((vint(*p1), vint(*p2), img[p1[1]:p2[1], p1[0]:p2[0]]))
#        cv2.imshow('prev', res[-1])
#        cv2.waitKey()
    
    return res

# 从全图提取
def extractCenterTilesImg(img):
    # extract center region
    center = extractCenterRegeon(img)
    blcorner = (252, 523)
    brcorner = (542, 525)
    urcorner = (545, 241)
    ulcorner = (256, 231)
    secH = 210
    secH = [210, 210, 210, 210]
    secW = [290, 290, 290, 290]
    parts = [None]*4
    parts[0] = center[blcorner[1]:blcorner[1]+secH[0], blcorner[0]:blcorner[0]+secW[0]].copy()
    parts[1] = cv2.rotate(center[brcorner[1]-secW[1]:brcorner[1], brcorner[0]:brcorner[0]+secH[1]], cv2.ROTATE_90_CLOCKWISE)
    parts[2] = cv2.rotate(center[urcorner[1]-secH[2]:urcorner[1], urcorner[0]-secW[2]:urcorner[0]], cv2.ROTATE_180)
    parts[3] = cv2.rotate(center[ulcorner[1]:ulcorner[1]+secW[3], ulcorner[0]-secH[3]:ulcorner[0]], cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    tiles = []
    for i, part in enumerate(parts):
        vprint('[extractCenterTilesImg] part %s' % i)
        tiles.append(extractPartTilesImg(part))
    return tiles
    
    

# --------------------------------其他杂项--------------------------------------





def testRange(img):
    img = img[353:410, 434:567]
        
        
    num_pixel = img.shape[0] * img.shape[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((5,5),np.uint8)
    chii_range = cv2.morphologyEx(cv2.inRange(hsv, (75, 50, 50), (83, 255, 255)), cv2.MORPH_OPEN, kernel)
    pon_range = cv2.morphologyEx(cv2.inRange(hsv, (97, 50, 50), (100, 255, 255)), cv2.MORPH_OPEN, kernel)
    kan_range = cv2.morphologyEx(cv2.inRange(hsv, (150, 50, 50), (155, 255, 255)), cv2.MORPH_OPEN, kernel)
    ron_range = cv2.morphologyEx(cv2.bitwise_or(
        cv2.inRange(hsv, (0, 50, 50), (2, 255, 255)),
        cv2.inRange(hsv, (178, 50, 50), (180, 255, 255))  ), cv2.MORPH_OPEN, kernel)
    riichi_range = cv2.morphologyEx(cv2.inRange(hsv, (10, 50, 50), (14, 255, 255)), cv2.MORPH_OPEN, kernel)
    
    chii_ratio = np.sum(chii_range) / 255.0 / num_pixel
    pon_ratio = np.sum(pon_range) / 255.0 / num_pixel
    kan_ratio = np.sum(kan_range) / 255.0 / num_pixel
    ron_ratio = np.sum(ron_range) / 255.0 / num_pixel
    riichi_ratio = np.sum(riichi_range) / 255.0 / num_pixel
    
    
    cv2.imshow('chii', chii_range)
    cv2.imshow('pon', pon_range)
    cv2.imshow('kan', kan_range)
    cv2.imshow('ron', ron_range)
    cv2.imshow('riichi', riichi_range)
    
    print(chii_ratio, pon_ratio, kan_ratio, ron_ratio, riichi_ratio)
    
    cv2.waitKey()


# 碰吃杠立直和
def extractActions(img):
    actionROI = img[527:527+58, 614:614+164]
    
    num_pixel = actionROI.shape[0] * actionROI.shape[1]
    hsv = cv2.cvtColor(actionROI, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((5,5),np.uint8)
    chii_range = cv2.morphologyEx(cv2.inRange(hsv, (75, 50, 50), (83, 255, 255)), cv2.MORPH_OPEN, kernel)
    pon_range = cv2.morphologyEx(cv2.inRange(hsv, (97, 50, 50), (100, 255, 255)), cv2.MORPH_OPEN, kernel)
    kan_range = cv2.morphologyEx(cv2.inRange(hsv, (150, 50, 50), (155, 255, 255)), cv2.MORPH_OPEN, kernel)
    ron_range = cv2.morphologyEx(cv2.bitwise_or(
        cv2.inRange(hsv, (0, 50, 50), (2, 255, 255)),
        cv2.inRange(hsv, (178, 50, 50), (180, 255, 255))  ), cv2.MORPH_OPEN, kernel)
    riichi_range = cv2.morphologyEx(cv2.inRange(hsv, (10, 50, 50), (14, 255, 255)), cv2.MORPH_OPEN, kernel)
    tsumo_range = cv2.morphologyEx(cv2.inRange(hsv, (124, 50, 50), (127, 255, 255)), cv2.MORPH_OPEN, kernel)
    
    action_names = ['chii', 'pon', 'kan', 'ron', 'riichi', 'tsumo']
    action_ratio = [
        np.sum(chii_range) / 255.0 / num_pixel,
        np.sum(pon_range) / 255.0 / num_pixel,
        np.sum(kan_range) / 255.0 / num_pixel,
        np.sum(ron_range) / 255.0 / num_pixel,
        np.sum(riichi_range) / 255.0 / num_pixel,
        np.sum(tsumo_range) / 255.0 / num_pixel
    ]
    print(action_ratio)
    action_idx = np.argmax(action_ratio)
    if (action_ratio[action_idx] < 0.06):
        return None
    return action_names[action_idx]
    

