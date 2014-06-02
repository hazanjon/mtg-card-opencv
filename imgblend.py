#!/usr/bin/env python

'''


'''

import cv2
import sys
import numpy as np
import glob

try: fn = sys.argv[1]
except: fn = 0

def nothing(*arg):
    pass

#cv2.namedWindow('output')

cv2.namedWindow('edge')
count = 0
files = glob.glob('C:\mtg\set2\*\*.jpg')

first = cv2.imread(files[0])
height, width, depth = first.shape
new = np.zeros((height,width,3), np.uint8)
newgray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

alpha = 1.0
beta = 0.0

for filename in files:
    print filename
    
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edge = cv2.Canny(gray, 0, 500, apertureSize=5)
    # vis = img.copy()
    # vis /= 2
    # vis[edge != 0] = (0, 255, 0)
    count = count + 1
    
    alpha = 1.0 / count
        
    beta = 1.0 - alpha
    
    newgray = cv2.addWeighted(newgray,0.99,gray,0.01,0)
    new = cv2.addWeighted(new,beta,img,alpha,0)
    
    #cv2.imshow('edge', new)
    cv2.imshow('edge2', newgray)

    
    ch = cv2.waitKey(1)
    if ch == 27:
        break
            

    flag, newy = cv2.threshold(newgray, 50, 255, cv2.THRESH_TRUNC)
    #cv2.imshow('edge3', newy)
    flag, newg = cv2.threshold(newgray, 50, 255, cv2.THRESH_BINARY)
    #cv2.imshow('edge4', newg)
    

newg = ~newg
newg = cv2.copyMakeBorder(newg,2,2,2,2,cv2.BORDER_CONSTANT)
newg = ~newg
newgray = ~newgray
newgray = cv2.copyMakeBorder(newgray,2,2,2,2,cv2.BORDER_CONSTANT)
newgray = ~newgray
cv2.imwrite('template.png',newg)
cv2.imwrite('template2.png',newgray)


newgray = cv2.GaussianBlur(newgray,(5,5),1000)
edge = cv2.Canny(newgray, 0, 500, apertureSize=5)
# vis = img.copy()
# vis /= 2
# vis[edge != 0] = (0, 255, 0)
cv2.imshow('edge4', edge)

ch = cv2.waitKey(0)
cv2.destroyAllWindows()
