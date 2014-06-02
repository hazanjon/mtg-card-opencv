#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

import cv2
import math
import sys
import numpy as np
from pytesser.pytesser import *
import glob


if __name__ == '__main__':
    print __doc__

    try: fn = sys.argv[1]
    except: fn = 0

    def nothing(*arg):
        pass

    #cv2.namedWindow('thresh')
    #cv2.namedWindow('edge')
    # cv2.namedWindow('full')
    # cv2.namedWindow('cut')

    # cv2.namedWindow('output')
    
    cardthreshold = 90
        
    #nW = img.shape[1] / 4
    #nH = img.shape[0] / 4
    
    cardsize = 630,880
    carddimensions = np.array([ [0,0],[cardsize[0],0],[0,cardsize[1]],[cardsize[0],cardsize[1]] ],np.float32)
    
    def rectify(h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew
  
    def getCard(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),1000)
        flag, threshold = cv2.threshold(gray, cardthreshold, 255, cv2.THRESH_BINARY_INV)
        
        # grayinv = ~gray
        # flag, grayinv = cv2.threshold(grayinv, 220, 255, cv2.THRESH_TRUNC)
        # flag, trun = cv2.threshold(gray, 30, 0, cv2.THRESH_TOZERO)
        # #flag, trun = cv2.threshold(grayinv, 230, 255, cv2.THRESH_BINARY)
        # trun = ~trun
        
        #threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 0)
        
           
        # nW = gray.shape[1] / 4
        # nH = gray.shape[0] / 4
        # smaller = cv2.resize(gray,(nW, nH))
        # cv2.imshow('ori', smaller)
        
        # nW = trun.shape[1] / 4
        # nH = trun.shape[0] / 4
        # smaller = cv2.resize(trun,(nW, nH))
        # cv2.imshow('thr', smaller)
        
        # nW = threshold.shape[1] / 4
        # nH = threshold.shape[0] / 4
        # smaller = cv2.resize(threshold,(nW, nH))
        # cv2.imshow('thre', smaller)
        
        
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)
       
        card = contours[0]
        
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.01*peri,True)
        
        test = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        #test = img.copy()
        cv2.drawContours(test,contours,0,(255, 255, 255),2)
        
        # nW = test.shape[1] / 4
        # nH = test.shape[0] / 4
        # smaller = cv2.resize(test,(nW, nH))
        # cv2.imshow('fer', smaller)
        
        minLineLength = 1000
        maxLineGap = 20
        lines = cv2.HoughLinesP(test,1,np.pi/180,80,minLineLength,maxLineGap)
        test2 = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        
        ver = []
        ver1 = []
        ver2 = []
        hor = []
        hor1 = []
        hor2 = []
        hor1 = []
        hor2 = []
        for line in lines[0]:
            x1,y1,x2,y2 = line
            xdiff = abs(x1 - x2)
            ydiff = abs(y1 - y2)
            
            if(xdiff < ydiff):
                ver.append(line)
            else:
                hor.append(line)
        
        hortotal = 0
        for line in hor:
            x1,y1,x2,y2 = line
            hortotal += y1 + y2
        
        horavg = hortotal / (len(hor) * 2)
        
        print horavg
        
        for line in hor:
            x1,y1,x2,y2 = line
            
            if(((y1 + y2) / 2) < horavg):
                hor1.append(line)
            else:
                hor2.append(line)
                
        vertotal = 0
        for line in ver:
            x1,y1,x2,y2 = line
            vertotal += x1 + x2
        
        veravg = vertotal / (len(ver) * 2)
        
        print veravg
        
        for line in ver:
            x1,y1,x2,y2 = line
            
            if(((x1 + x2) / 2) < veravg):
                ver1.append(line)
            else:
                ver2.append(line)
        
        def extendedLine(lines, length):
            
            avg = [0,0,0,0]
            for line in lines:    
                x1,y1,x2,y2 = line
                avg[0] += line[0]
                avg[1] += line[1]
                avg[2] += line[2]
                avg[3] += line[3]
            
            avg[0] /= len(lines)
            avg[1] /= len(lines)
            avg[2] /= len(lines)
            avg[3] /= len(lines)
            
            seglen = math.sqrt((avg[0] - avg[2])**2 + (avg[1] - avg[3])**2)
            Cx = avg[2] + (avg[2] - avg[0]) / seglen * length;
            Cy = avg[3] + (avg[3] - avg[1]) / seglen * length;
            Dx = avg[2] + (avg[2] - avg[0]) / seglen * -length;
            Dy = avg[3] + (avg[3] - avg[1]) / seglen * -length;

            return Cx, Cy, Dx, Dy
        
        def displayline(ln):
            cv2.line(test2,(int(ln[0]), int(ln[1])),(int(ln[2]), int(ln[3])),(255,255,255),2)
            
        hor1line = extendedLine(hor1, max(img.shape[0:1]))
        displayline(hor1line)
        
        hor2line = extendedLine(hor2, max(img.shape[0:1]))
        displayline(hor2line)
        
        ver1line = extendedLine(ver1, max(img.shape[0:1]))
        displayline(ver1line)
        
        ver2line = extendedLine(ver2, max(img.shape[0:1]))
        displayline(ver2line)
                
        def makeline(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x,y
            else:
                return False
                
        H1 = makeline([hor1line[0], hor1line[1]], [hor1line[2], hor1line[3]])
        H2 = makeline([hor2line[0], hor2line[1]], [hor2line[2], hor2line[3]])
        V1 = makeline([ver1line[0], ver1line[1]], [ver1line[2], ver1line[3]])
        V2 = makeline([ver2line[0], ver2line[1]], [ver2line[2], ver2line[3]])

        H1V1 = intersection(H1, V1)
        H1V2 = intersection(H1, V2)
        H2V1 = intersection(H2, V1)
        H2V2 = intersection(H2, V2)
        
        cv2.circle(test2, (int(H1V1[0]),int(H1V1[1])), 20, (255,0,255), -1) #top left
        cv2.circle(test2, (int(H1V2[0]),int(H1V2[1])), 20, (0,255,0), -1) #top right
        cv2.circle(test2, (int(H2V1[0]),int(H2V1[1])), 20, (0,0,255), -1) #bot left
        cv2.circle(test2, (int(H2V2[0]),int(H2V2[1])), 20, (0,255,255), -1) #bot right
        
        target = np.array([ [int(H1V1[0]),int(H1V1[1])],[int(H1V2[0]),int(H1V2[1])],[int(H2V1[0]),int(H2V1[1])],[int(H2V2[0]),int(H2V2[1])] ],np.float32)
        transform = cv2.getPerspectiveTransform(target,carddimensions)
        warp = cv2.warpPerspective(img,transform,(630,880))

        # nW = test2.shape[1] / 4
        # nH = test2.shape[0] / 4
        # smaller = cv2.resize(test2,(nW, nH))
        # cv2.imshow('test', smaller)
        # cv2.imshow('warp', warp)
        
        '''
        cv2.drawContours(test,[card],0,(255, 0, 0),2)
        cv2.drawContours(test,[approx],0,(0,0,255),2)
        nW = test.shape[1] / 4
        nH = test.shape[0] / 4
        smaller = cv2.resize(test,(nW, nH))
        cv2.imshow('test', smaller)
        '''
               
        return warp
   
    def getCardName(img):
        #cardsize = 976,1364
        secx1 = int(cardsize[0]*0.06)
        secx2 = int(cardsize[0]*0.7)
        secy1 = int(cardsize[1]*0.04)
        secy2 = int(cardsize[1]*0.12)
        sech = secy2-secy1
        secw = secx2-secx1
        crop_img = img[secy1:secy2, secx1:secx2]
        gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        gray = ~gray
        flag, gray = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
        
        #@TODO Look at finding the largest contour here which should be the area behind the text
        h, w = gray.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        mask[:] = 0
        lo = 20
        hi = 20
        flags = 4
        flags |= cv2.FLOODFILL_FIXED_RANGE
        
        cv2.line(gray, (0, 0), (0, sech-1), (0,0,0))
        cv2.line(gray, (0, 0), (secw-1, 0), (0,0,0))
        cv2.line(gray, (secw-1, 0), (secw-1, sech-1), (0,0,0))
        cv2.line(gray, (0, sech-1), (secw-1, sech-1), (0,0,0))
        cv2.floodFill(gray, mask, (0,0), (255, 255, 255), (lo,)*3, (hi,)*3, flags)
        
        #cv2.imshow('full', test)
        return gray
        
    def getCardSet(img):
        #cardsize = 976,1364
        secx1 = int(cardsize[0]*0.8)
        secx2 = int(cardsize[0]*0.95)
        secy1 = int(cardsize[1]*0.56)
        secy2 = int(cardsize[1]*0.63)
        sech = secy2-secy1
        secw = secx2-secx1
        crop_img = img[secy1:secy2, secx1:secx2]
        gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        # gray = ~gray
        # flag, gray = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow('set', gray)
        # h, w = gray.shape[:2]
        # mask = np.zeros((h+2, w+2), np.uint8)
        # mask[:] = 0
        # lo = 20
        # hi = 20
        # flags = 4
        # flags |= cv2.FLOODFILL_FIXED_RANGE
        
        # cv2.line(gray, (0, 0), (0, sech-1), (0,0,0))
        # cv2.line(gray, (0, 0), (secw-1, 0), (0,0,0))
        # cv2.line(gray, (secw-1, 0), (secw-1, sech-1), (0,0,0))
        # cv2.line(gray, (0, sech-1), (secw-1, sech-1), (0,0,0))
        # cv2.floodFill(gray, mask, (0,0), (255, 255, 255), (lo,)*3, (hi,)*3, flags)
        
        #cv2.imshow('full', test)
        return gray
        
    def getCardCost(img):
        #cardsize = 976,1364
        secx1 = int(cardsize[0]*0.7)
        secx2 = int(cardsize[0]*0.95)
        secy1 = int(cardsize[1]*0.04)
        secy2 = int(cardsize[1]*0.12)
        sech = secy2-secy1
        secw = secx2-secx1
        crop_img = img[secy1:secy2, secx1:secx2]
        gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        #gray = ~gray
        #flag, gray = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

        #cv2.imshow('cost', gray)
        # h, w = gray.shape[:2]
        # mask = np.zeros((h+2, w+2), np.uint8)
        # mask[:] = 0
        # lo = 20
        # hi = 20
        # flags = 4
        # flags |= cv2.FLOODFILL_FIXED_RANGE
        
        # cv2.line(gray, (0, 0), (0, sech-1), (0,0,0))
        # cv2.line(gray, (0, 0), (secw-1, 0), (0,0,0))
        # cv2.line(gray, (secw-1, 0), (secw-1, sech-1), (0,0,0))
        # cv2.line(gray, (0, sech-1), (secw-1, sech-1), (0,0,0))
        # cv2.floodFill(gray, mask, (0,0), (255, 255, 255), (lo,)*3, (hi,)*3, flags)
        
        cv2.imshow('cost', gray)
        return gray
    
    def ocrImg(img):
        cv2.imwrite('temp.png',img)
        text = image_file_to_string('temp.png')
        print text
        
    
    def parseImg(img):
        #cv2.imshow('full', img)
        
        found = getCard(img)
        
        '''
        gray = cv2.cvtColor(found,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),1000)
        gray = ~gray
        
        h, w = gray.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        mask[:] = 0
        lo = 20
        hi = 20
        flags = 4
        flags |= cv2.FLOODFILL_FIXED_RANGE
        #cv2.floodFill(gray, mask, (0,0), (255, 255, 255), (lo,)*3, (hi,)*3, flags)
        
        flag, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 0)
        cv2.imshow('cut3', gray)
        '''
        '''
        smaller = found.copy()
        #con = ~con
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)
       
        card = contours[0]
        
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.02*peri,True)
        
        cv2.drawContours(smaller,[approx],0,(0, 255, 0),2)
        
        minLineLength = 100
        maxLineGap = 200
        lines = cv2.HoughLinesP(gray,1,np.pi/180,80,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(smaller,(x1,y1),(x2,y2),(0,255,0),2)
        
        nW = smaller.shape[1] / 2
        nH = smaller.shape[0] / 2
        smaller = cv2.resize(smaller,(nW, nH))
        cv2.imshow('cut2', smaller)
        '''
        '''
        approx = rectify(approx)
        transform = cv2.getPerspectiveTransform(approx,carddimensions)
        warp = cv2.warpPerspective(found,transform,(cardsize))
        cv2.imshow('cut', warp)
        '''
        
        nW = found.shape[1] / 2
        nH = found.shape[0] / 2
        smaller = cv2.resize(found,(nW, nH))
        cv2.imshow('found', smaller)
        cardNameImg = getCardName(found)
        
        cv2.imshow('output', cardNameImg)
        ocrImg(cardNameImg)
        
        cardCostImg = getCardCost(found)
        cardSetImg = getCardSet(found)
        
        
    for filename in glob.glob('c:\mtg\cards\*.jpg'):
        print filename
        ch = cv2.waitKey(0)
        if ch == 27:
            break
            
        #if ch == 32:
        img = cv2.imread(filename)
        parseImg(img)
                

    cv2.destroyAllWindows()
