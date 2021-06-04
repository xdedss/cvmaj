


# (╯‵□′)╯︵┻━┻

import cv2
import numpy as np
import numpy.linalg as npl
import time, os, sys
import traceback

# m
import scrutils
import segutils
import visionutils


def mkdirIfNotExists(dirs):
    for d in dirs:
        if (not os.path.exists(d)):
            os.makedirs(d)

class MajInfo:
    def __init__(self):
        self.hand = None # 手牌
        self.discard = None # 打出的牌
        


def scrapData():
    agent = scrutils.ScreenAgent()
    agent.calibrateMultiple()
    
    while (True):
        input()
        agent.capture()
        
        try:
            
            table = segutils.extractCenterTilesImg(agent.current)
            for part in table:
                for start, end, img in part:
                    for i in range(10):
                        outpath = 'samples/tiles/unlabeled/small/%.3f_%d.png' % (time.time(), i)
                        if (not os.path.exists(outpath)):
                            cv2.imwrite(outpath, img)
                            break
            
            tiles = segutils.extractTilesImg(agent.current)
            for start, end, img in tiles:
                for i in range(10):
                    outpath = 'samples/tiles/unlabeled/big/%.3f_%d.png' % (time.time(), i)
                    if (not os.path.exists(outpath)):
                        cv2.imwrite(outpath, img)
                        break
            
        except Exception as e:
            outpath = 'samples/error/%.3f.png' % (time.time())
            outpathscr = 'samples/error/%.3f_scr.png' % (time.time())
            cv2.imwrite(outpath, agent.current)
            cv2.imwrite(outpathscr, agent.raw)
            traceback.print_exc()

def offlineTest():
    testImg = ['05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    testImgPath = ['samples/scr/%s.png' % w for w in testImg]
    classifier = visionutils.loadModel('pretrained.tar')
    agent = scrutils.ScreenAgent(cv2.imread(testImgPath[0]))
    agent.calibrateMultiple()
    for path in testImgPath:
        agent.override = cv2.imread(path)
        agent.capture()
        
        try:
            print('牌河')
            table = segutils.extractCenterTilesImg(agent.current)
            for part in table:
                partPred = classifier([img for start, end, img in part])
                print(partPred)
#                for start, end, img in part:
#                    cv2.imshow('debug', img)
#                    cv2.waitKey()
            
            tiles = segutils.extractTilesImg(agent.current)
            pred = classifier([img for start, end, img in tiles])
            print('手牌识别')
            print(pred)
#            for start, end, img in tiles:
#                cv2.imshow('debug', img)
#                cv2.waitKey()
            
            cv2.imshow('scene', agent.current)
            k = cv2.waitKey()
            if (k == 120):
                # x
                raise Exception('You think there\'s something wrong')
            
        except Exception as e:
            outpath = 'samples/error/%.3f.png' % (time.time())
            outpathscr = 'samples/error/%.3f_scr.png' % (time.time())
            cv2.imwrite(outpath, agent.current)
            cv2.imwrite(outpathscr, agent.raw)
            print('error')
            traceback.print_exc()


def clickTest():
    agent = scrutils.ScreenAgent()
    agent.calibrateMultiple()
    agent.capture()
#    cv2.imshow('test', agent.current)
#    cv2.waitKey()
    while (True):
        x = int(input())
        y = int(input())
        print('clicking %s, %s' % (x, y))
        
        center = (x, y)
        agent.moveSync(center)
        time.sleep(0.3)
        agent.click(center)
        time.sleep(0.3)
        agent.moveSync([0, 0])
    

def onlineTest():
    classifier = visionutils.loadModel('pretrained.tar')
    agent = scrutils.ScreenAgent()
    agent.calibrateMultiple()
    input('...')
    while (True):
        agent.capture()
        
        try:
            print('牌河')
            table = segutils.extractCenterTilesImg(agent.current)
            for part in table:
                partPred = classifier([img for start, end, img in part])
                print(partPred)
            
            tiles = segutils.extractTilesImg(agent.current)
            pred = classifier([img for start, end, img in tiles])
            print('手牌识别')
            print(pred)
            
            #cv2.imshow('scene', agent.current)
            k = input('...')
            if (k == 'x'):
                # x
                raise Exception('You think there\'s something wrong')
            
        except Exception as e:
            outpath = 'samples/error/%.3f.png' % (time.time())
            outpathscr = 'samples/error/%.3f_scr.png' % (time.time())
            cv2.imwrite(outpath, agent.current)
            cv2.imwrite(outpathscr, agent.raw)
            print('error')
            traceback.print_exc()


def runAI(moduleName):
    m = __import__(moduleName, fromlist=[''])    
    if (not('discard' in dir(m) and 'action' in dir(m))):
        raise Exception('missing  implementation')
    
    classifier = visionutils.loadModel('pretrained.tar')
    agent = scrutils.ScreenAgent()
    agent.calibrateMultiple()
    
    input('按回车开始运行...')
    
    while (True):
        time.sleep(1)
        agent.capture()
        
        try:
            print('牌河')
            table = segutils.extractCenterTilesImg(agent.current)
            tablePred = []
            for part in table:
                partPred = classifier([img for start, end, img in part])
                tablePred.append(partPred)
                print(partPred)
            
            tiles = segutils.extractTilesImg(agent.current)
            pred = classifier([img for start, end, img in tiles])
            print('手牌识别')
            print(pred)
            
            
            info = MajInfo()
            info.hand = pred
            info.discard = tablePred
            if (len(pred) % 3 == 2):
                action = segutils.extractActions(agent.current)
                if (action is None):
                    print('no action')
                    # 该出牌了
                    print('Consulting AI for discard')
                    discardIndex = m.discard(info)
                    if (discardIndex is None):
                        print('discard nothing')
                    else:
                        discardIndex = int(discardIndex) 
                        print('discard %d' % discardIndex)
                        assert discardIndex >= 0 and discardIndex < len(pred), 'discard index out of range'
                        start, end, img = tiles[discardIndex]
                        center = (start + end) // 2
                        agent.moveSync(center)
                        time.sleep(0.1)
                        agent.click(center)
                        time.sleep(0.1)
                        agent.moveSync([0, 0])
                else:
                    print('action: %s' % action)
                    # 立直/和
                    print('Consulting AI for action')
                    actionResult = m.action(info, action)
                    if (actionResult is None):
                        print('do nothing')
                    elif (actionResult):
                        print('YES')
                        agent.click([694, 558])
                        time.sleep(0.1)
                        agent.moveSync([0, 0])
                    else:
                        print('NOOOO')
                        agent.click([873, 558])
                        time.sleep(0.1)
                        agent.moveSync([0, 0])
                
            else:
                action = segutils.extractActions(agent.current)
                if (action is None):
                    print('no action')
                else:
                    print('action: %s' % action)
                    # 碰吃杠
                    print('Consulting AI for action')
                    actionResult = m.action(info, action)
                    if (actionResult is None):
                        print('do nothing')
                    elif (actionResult):
                        print('YES')
                        agent.click([694, 558])
                        time.sleep(0.1)
                        agent.moveSync([0, 0])
                    else:
                        print('NOOOO')
                        agent.click([873, 558])
                        time.sleep(0.1)
                        agent.moveSync([0, 0])
                
            
        except Exception as e:
            outpath = 'samples/error/%.3f.png' % (time.time())
            outpathscr = 'samples/error/%.3f_scr.png' % (time.time())
            cv2.imwrite(outpath, agent.current)
            cv2.imwrite(outpathscr, agent.raw)
            print('error')
            traceback.print_exc()
    
    


if __name__ == '__main__':
    
    mkdirIfNotExists([
    'samples/error'
    ])
    
    if (len(sys.argv) > 1):
        runAI('ai.' + sys.argv[1])
    else:
        #onlineTest()
        #offlineTest()
        
        #clickTest()
        pass
        
    
