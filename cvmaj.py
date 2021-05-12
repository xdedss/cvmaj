


# (╯‵□′)╯︵┻━┻

import cv2
import numpy as np
import numpy.linalg as npl
import time, os
import traceback

# m
import scrutils
import segutils

if __name__ == '__main__':
    
    testImg = ['05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    testImgPath = ['samples/scr/%s.png' % w for w in testImg]
    
    # debugOverride = cv2.imread(testImgPath[0])
    
    agent = scrutils.ScreenAgent()
    #agent.override = cv2.imread('samples/scr/e2.png')
    agent.calibrateMultiple()
    
#    for fpath in testImgPath:
#        # start timing
    while (True):
        input()
        #agent.override = cv2.imread('samples/scr/e3.png')
        agent.capture()
        #canvas = agent.current.copy()
        
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
            print('error')
            traceback.print_exc()
        
        #break
        #cv2.imshow('v', center_seg)
        #cv2.waitKey()
    
    #match = agent.matchPatch(cv2.imread('patch1.png'))
    #print(match)
    #agent.click((match[0] + match[1]) / 2)
    
