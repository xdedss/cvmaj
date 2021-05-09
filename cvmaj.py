


# (╯‵□′)╯︵┻━┻

import cv2
import numpy as np

# m
import scrutils



def ptint(pt):
    return (int(pt[0]), int(pt[1]))

if __name__ == '__main__':
    
    agent = scrutils.ScreenAgent()
    agent.calibrateMain()
    
    
    agent.capture()
    cv2.imshow('n', agent.current)
    cv2.waitKey()
    
    match = agent.matchPatch(cv2.imread('patch1.png'))
    print(match)
    agent.click((match[0] + match[1]) / 2)
    
