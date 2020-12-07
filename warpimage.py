import  numpy as np
import cv2

def getwarp(img,biggest):
    biggest=reorder(biggest)
    print('shape',biggest.shape)
    pt1=np.float32(biggest)
    pt2=np.float32([[0,0],[framewidth,0], [0,frameheight], [framewidth,frameheight]])
    matrix=cv2.getPerspectiveTransform(pt1,pt2)
    print('matrix',matrix)
    outputimg=cv2.warpPerspective(img,matrix,(framewidth,frameheight))
    return outputimg


def reorder(points):
    points = points.reshape((4, 2))
    newpoints=np.zeros((4,1,2),np.int32)
    add= points.sum(axis=1)
    newpoints[0]=(points[np.argmin(add)])
    newpoints[3]=(points[np.argmax(add)])
    diff=np.diff(points,axis=1)
    newpoints[1]=points[np.argmin(diff)]
    newpoints[2]=points[np.argmax(diff)]
    print(newpoints)
    return newpoints

def preprocessing1(img):
    kernel=np.ones((5,5))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),1)
    canny=cv2.Canny(blur,200,200)
    dilated=cv2.dilate(canny,kernel,iterations=2)
    threshold=cv2.erode(dilated,kernel,iterations=1)
    return threshold

def getboundry(img):
    maxarea=0
    biggest=np.array([])
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>3000:
            peri=cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx)==4 and area >maxarea:
                biggest=approx
                maxarea=area
    cv2.drawContours(image, biggest, -1, (255, 0, 0), 10)
    return biggest

framewidth=200
frameheight=200
img=cv2.imread('download1.jpg')
image=img.copy()
threshold=preprocessing1(img)
biggest=getboundry(threshold)
warpped=getwarp(img,biggest)
threshold1=preprocessing1(warpped)
cv2.imshow("warpped",warpped)
cv2.waitKey(0)