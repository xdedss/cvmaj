


# 数据集封装

import torch
import torch.nn as nn
import cv2
import numpy as np


import os, hashlib, shutil
import random

i2s = []
s2i = {}
for i in range(1, 10):
    i2s.append('%dm' % i)
for i in range(1, 10):
    i2s.append('%dp' % i)
for i in range(1, 10):
    i2s.append('%ds' % i)
for i in range(1, 8):
    i2s.append('%dz' % i)
for i, s in enumerate(i2s):
    s2i[s] = i

def name2index(name):
    if name in s2i:
        return s2i[name]
    raise Exception('no such name: %s' % name)

def index2name(index):
    if index > 0 and index < len(i2s):
        return i2s[index]
    raise Exception('index out of range: %d' % index)


def strhash(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

class MajData(torch.utils.data.Dataset):

    def __init__(self, hashCriteria, size=32):
        self.nameList = []
        self.size = 32
        rootPath = 'samples/tiles/labeled/label'
        for label in i2s:
            labelPath = os.path.join(rootPath, label)
            if (os.path.exists(labelPath)):
                for sample in os.listdir(labelPath):
                    fpath = os.path.join(labelPath, sample)
                    if (sample.endswith('.png') and hashCriteria(strhash(fpath))):
                        self.nameList.append((fpath, label))
        print(len(self.nameList))
        

    def __getitem__(self, index):
        fpath, label = self.nameList[index]
        img = cv2.imread(fpath)
        img = cv2.resize(img, (self.size, self.size))
        # draw random things
        if (random.random() > 0.4):
            # insert noise in some images
            cv2.line(img, (random.randint(0, self.size), random.randint(0, self.size)), (random.randint(0, self.size), random.randint(0, self.size)), (255, 255, 255), 5)
        # random rotation
        if (random.random() > 0.2):
            img = cv2.warpAffine(img, np.matrix([[1,0,random.gauss(0, 1)],[0,1,random.gauss(0, 1)]]), (self.size, self.size), None,
                cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (128, 128, 128))
        
        
        #cv2.imshow('img', img)
        #cv2.waitKey(10)
        #print(name2index(label))
        return (np.transpose(img.astype(np.float32), axes=[2, 0, 1]) / 255, name2index(label))

    def __len__(self):
        return len(self.nameList)
