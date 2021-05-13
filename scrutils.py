


# (╯‵□′)╯︵┻━┻
# 图像匹配相关功能

#import pyscreeze
import cv2
import numpy as np
import numpy.linalg as npl
import win32gui, win32ui, win32con, win32api
import os, time
import ctypes


# tell windows i'm dpi aware
PROCESS_PER_MONITOR_DPI_AWARE = 2
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)


g_orb = cv2.ORB_create()
g_sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#index_params= dict(algorithm = FLANN_INDEX_LSH,
#                   table_number = 6, # 12
#                   key_size = 12,     # 20
#                   multi_probe_level = 1) #2
search_params = dict(checks=50)
g_flann = cv2.FlannBasedMatcher(index_params, search_params)

# find window hwnd by name
def findMajWindow():
    foundHwnd = []
    
    def winEnumHandler( hwnd, ctx ):
        if win32gui.IsWindowVisible( hwnd ):
            windowTitle = win32gui.GetWindowText( hwnd )
            if ('雀魂' in windowTitle):
                # print (hex(hwnd), windowTitle)
                foundHwnd.append(hwnd)
    win32gui.EnumWindows( winEnumHandler, None )
    
    if (len(foundHwnd) > 0):
        return foundHwnd[0]
    return None

## screen shot browser window if found ; else capture full screen
#def scrshotMaj():
#    hwnd = findMajWindow()
#    if (hwnd == None):
#        print('can not find majsoul window')
#        return screenshot()
#    else:
#        return background_screenshot(hwnd)
#
## capture full screen
#def screenshot():
#    img = pyscreeze.screenshot()
#    return np.asarray(img)[:, :, ::-1].copy()

# capture specific window
def background_screenshot(hwnd):
    #left, top, right, bot = win32gui.GetClientRect(hwnd)
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    print('window location', left, top, right, bot)
    w = right - left
    h = bot - top
#    # in case there's DPI scaling
#    w *= 2
#    h *= 2

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    #result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
    result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
    #print(result)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    #print(bmpinfo)
    img = np.fromstring(bmpstr, dtype='uint8')
    img.shape = (h, w, 4)
    #print(img.shape)
    return img[:, :, :3]

# fit matrix xy -> uv  Scale & Translation
def fitST(xy, uv):
    assert len(xy) == len(uv), 'xy and uv have different length'
    A = np.zeros((len(xy) * 2, 4), dtype=float)
    for i in range(len(xy)):
        A[2*i,   0] = xy[i][0]
        A[2*i+1, 1] = xy[i][1]
        A[2*i,   2] = 1
        A[2*i+1, 3] = 1
    B = np.concatenate(uv)
    M = np.matmul(np.linalg.pinv(A), B)
    
    res = np.zeros((3, 3), dtype=float)
    res[0, 0] = M[0]
    res[1, 1] = M[1]
    res[0, 2] = M[2]
    res[1, 2] = M[3]
    res[2, 2] = 1
    return res

# constraint: scalex = scaley
def fitST_uni(xy, uv):
    assert len(xy) == len(uv), 'xy and uv have different length'
    A = np.zeros((len(xy) * 2, 3), dtype=float)
    for i in range(len(xy)):
        A[2*i,   0] = xy[i][0]
        A[2*i+1, 0] = xy[i][1]
        A[2*i,   1] = 1
        A[2*i+1, 2] = 1
    B = np.concatenate(uv)
    M = np.matmul(np.linalg.pinv(A), B)
    
    res = np.zeros((3, 3), dtype=float)
    res[0, 0] = M[0]
    res[1, 1] = M[0]
    res[0, 2] = M[1]
    res[1, 2] = M[2]
    res[2, 2] = 1
    return res

# fit matrix xy -> uv  Scale & Translation
def imgFitST(imgxy, imguv):
    # extract
    kp1, des1 = g_sift.detectAndCompute(imgxy, None)
    kp2, des2 = g_sift.detectAndCompute(imguv, None)
    kp1, des1 = filterKp(kp1, des1)
    kp2, des2 = filterKp(kp2, des2)
    # match
    matches = getGoodMatches(des1, des2)
    
    print('[imgFitST] kp1: %s kp2: %s best matches: %s' % (len(kp1), len(kp2), len(matches)))
    
    xy = []
    uv = []
    for id1, id2, score in matches:
        xy.append(kp1[id1].pt)
        uv.append(kp2[id2].pt)
    
    res = fitST_uni(xy, uv)
    return res
    
# remove big keypoints
def filterKp(kp, des):
    kpr = []
    desr = []
    for i in range(len(kp)):
        if (kp[i].size < 30):
            kpr.append(kp[i])
            desr.append(des[i])
    return np.array(kpr), np.array(desr)


def getGoodMatches(des1, des2, thr = 0.6):
    res = []
    flip = len(des1) > len(des2)
    if (flip):
        matches = g_flann.knnMatch(des2, des1, k=2)
    else:
        matches = g_flann.knnMatch(des1, des2, k=2)
    
    for match1, match2 in matches:
        score = match1.distance / match2.distance # 越小越好
        if (score < thr):
            if (flip):
                res.append([match1.trainIdx, match1.queryIdx, score])
            else:
                res.append([match1.queryIdx, match1.trainIdx, score])
    
    res.sort(key=lambda item:item[2])
        
    return res

# match and delete bad points in loop
def adaptiveMatchFitST_uni(kp1, des1, kp2, des2):
    matches = g_flann.knnMatch(des1, des2, k=2)
    matches.sort(key=lambda m:m[0].distance)
    binSize = 500
    binSizeMin = 400
    
    while(True):
        matches_near = matches[:min(len(matches), binSize)]
        print('[adaptiveMatchFitST_uni] total %s, pick %s' % (len(matches), len(matches_near)))
        if (len(matches_near) < binSizeMin):
            print('[adaptiveMatchFitST_uni] failed. too few remaining points before converge')
            return None
        
        # fit
        xy = []
        uv = []
        reperr = []
        for i in range(len(matches_near)):
            match1 = matches_near[i][0]
            match2 = matches_near[i][1]
            xy.append(kp1[match1.queryIdx].pt)
            uv.append(kp2[match1.trainIdx].pt)
        s2n = fitST_uni(xy, uv)
        # reprojection
        for i in range(len(matches_near)):
            rep = np.matmul(s2n, [xy[i][0], xy[i][1], 1])[:2]
            reperr.append(npl.norm(uv[i] -  rep))
        mean = np.mean(reperr)
        std = np.std(reperr)
        print('[adaptiveMatchFitST_uni] mean %s std %s' % (mean, std))
        if (mean < 1):
            return s2n
        # remove anomaly
        matches_to_remove = []
        for i in range(len(matches_near)):
            matches_to_remove.append((matches_near[i], reperr[i]))
        matches_to_remove.sort(key=lambda m:m[1], reverse=True)
        rem_count = 0
        for i in range(len(matches_near)):
            if (matches_to_remove[i][1] > (mean * 3)):
                matches.remove(matches_to_remove[i][0])
                rem_count += 1
        print('[adaptiveMatchFitST_uni] removed %s anomalies' % (rem_count))
        if (rem_count == 0):
            for i in range(30):
                matches.remove(matches_to_remove[i][0])
                rem_count += 1
            print('[adaptiveMatchFitST_uni] removed %s extra points' % (rem_count))
    

# click without focus
def clickInto(hwnd, x, y):
    lParam = win32api.MAKELONG(x, y)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
    print('ldown')
    time.sleep(0.1)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, None, lParam)
    print('lup')

def moveInto(hwnd, x, y):
    lParam = win32api.MAKELONG(x, y)
    win32api.PostMessage(hwnd, win32con.WM_MOUSEMOVE, None, lParam)
    


def ptint(pt):
    return (int(pt[0]), int(pt[1]))

def v(*args):
    return np.array(args, dtype=float)

# normalized size
nh = 720
nw = 1280

# screen recognition
class ScreenAgent:
    
    def __init__(self, override = None):
        self.s2n = None
        self.n2s = None
        self.hwnd = findMajWindow()
        self.needUpdate = True
        self.override = override
        self.lastMousePos = v(0,0)
        if (override is None):
            assert self.hwnd != None, 'can not find majsoul window!'
        
        # templates
        print('[ScreenAgent] building template keypoints')
        kp = []
        des = []
        for fname in os.listdir('templates/multiple'):
            fpath = 'templates/multiple/' + fname
            template = cv2.imread(fpath)
            if (np.any(template == None)):
                continue
            print(fpath)
            kp_, des_ = g_sift.detectAndCompute(template, None)
            kp_, des_ = filterKp(kp_, des_)
            kp.append(kp_)
            des.append(des_)
        kp = np.concatenate(kp)
        des = np.concatenate(des)
        self.kp_template = kp
        self.des_template = des
    
    # calibrate using main menu
    def calibrateMain(self):
        template = cv2.imread('templates/mainTemplate.png')
        template = cv2.resize(template, (nw, nh))
        if (np.all(self.override != None)):
            scr = self.override
        else:
            scr = background_screenshot(self.hwnd)
        
        self.s2n = imgFitST(scr, template)
        self.n2s = np.linalg.inv(self.s2n)
        
        print(self.s2n)
    
    # 多场景匹配
    def calibrateMultiple(self):
        # screen
        if (np.all(self.override != None)):
            scr = self.override
        else:
            scr = background_screenshot(self.hwnd)
        kp_scr, des_scr = g_sift.detectAndCompute(scr, None)
        kp_scr, des_scr = filterKp(kp_scr, des_scr)
        # match & fit
        self.s2n = adaptiveMatchFitST_uni(kp_scr, des_scr, self.kp_template, self.des_template)
        self.n2s = np.linalg.inv(self.s2n)
        print(self.s2n)
    
    # capture and save to self.current
    def capture(self):
        if (np.all(self.override != None)):
            scr = self.override
        else:
            scr = background_screenshot(self.hwnd)
        self.current = cv2.warpAffine(scr, self.s2n[:2], (nw, nh))
        self.raw = scr
        self.needUpdate = True
    
    # find patch in current capture
    def matchPatch(self, patch):
        kp2, des2 = g_sift.detectAndCompute(patch, None)
        kp2, des2 = filterKp(kp2, des2)
        return self.matchPatchKp(kp2, des2, patch.shape)
    
    # find patch in current capture(with keypoints)
    def matchPatchKp(self, kp, des, shape):
        if (self.needUpdate):
            self.kp, self.des = g_sift.detectAndCompute(self.current, None)
        matches = getGoodMatches(self.des, des)
        print('[matchPatchKp] kp1: %s kp2: %s best matches: %s' % (len(self.kp), len(kp), len(matches)))
        xy = []
        uv = []
        for id1, id2, score in matches:
            xy.append(self.kp[id1].pt)
            uv.append(kp[id2].pt)
        p2n = fitST_uni(uv, xy)
        print(p2n)
        ulcorner = np.matmul(p2n, [0, 0, 1])
        brcorner = np.matmul(p2n, [shape[1], shape[0], 1])
        return (ulcorner[:2], brcorner[:2])
    
    # click(normalized coordinates)
    def click(self, uv):
        xy = np.matmul(self.n2s, [uv[0], uv[1], 1])[:2]
        clickInto(self.hwnd, int(xy[0]), int(xy[1]))
        self.lastMousePos = xy
    
    # move(normalized coordinates)
    def moveSync(self, uv, steps=10, total_time=0.3):
        interval = total_time / steps
        xy = np.matmul(self.n2s, [uv[0], uv[1], 1])[:2]
        for i in range(steps):
            t = i / steps
            lerped = self.lastMousePos * (1 - t) + xy * t
            moveInto(self.hwnd, int(lerped[0]), int(lerped[1]))
            time.sleep(interval)
        moveInto(self.hwnd, int(xy[0]), int(xy[1]))
        self.lastMousePos = xy



