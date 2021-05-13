


# (╯‵□′)╯︵┻━┻

import cv2
import numpy as np
import numpy.linalg as npl
import time, os
import traceback

# m
import scrutils
import segutils
import visionutils


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

if __name__ == '__main__':
    
    onlineTest()
    
