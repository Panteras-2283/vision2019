#!/usr/bin/python3.5
# coding: utf8
import numpy as np
import pandas as pd
import cv2
import datetime
import collections
from datetime import timedelta
from tkinter import *

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640

cap = cv2.VideoCapture(0)
cap.set(4, IMAGE_HEIGHT)
cap.set(3, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -15.0)

# Global variables
canny_thresh = 100
min_hull_area = 500

H_LOW = 60
H_HIGH = 90
S_LOW = 100#30#80
S_HIGH = 255
V_LOW = 42#70#80
V_HIGH = 255

MORPH_KERNEL = None #np.ones((3, 3), np.uint8)
MORPH_ANCHOR = (-1, -1)
MORPH_ITERATIONS = 3
MORPH_BORDER_TYPE = cv2.BORDER_CONSTANT
MORPH_BORDER_VALUE = (-1)


def smoothImage(src):
    dst = cv2.blur(src, (3,3))
    return dst

def hsvThreshold(src):
    dst = cv2.inRange(src, (H_LOW, S_LOW, V_LOW), (H_HIGH, S_HIGH, V_HIGH))
    return dst

# Dilate and then erode to eliminate holes
def closeFrame(src):
    dst = cv2.dilate(src, MORPH_KERNEL, MORPH_ANCHOR, iterations=MORPH_ITERATIONS, borderType = MORPH_BORDER_TYPE, borderValue = MORPH_BORDER_VALUE)
    dst = cv2.erode(dst, MORPH_KERNEL, MORPH_ANCHOR, iterations=MORPH_ITERATIONS, borderType = MORPH_BORDER_TYPE, borderValue = MORPH_BORDER_VALUE)
    return dst

# Erode and then dilate to eliminate noise
def openFrame(src):
    dst = cv2.erode(src, MORPH_KERNEL, MORPH_ANCHOR, iterations=MORPH_ITERATIONS, borderType = MORPH_BORDER_TYPE, borderValue = MORPH_BORDER_VALUE)
    dst = cv2.dilate(dst, MORPH_KERNEL, MORPH_ANCHOR, iterations=MORPH_ITERATIONS, borderType = MORPH_BORDER_TYPE, borderValue = MORPH_BORDER_VALUE)
    return dst

def findHulls(src):
    # Find edges
    dst = cv2.Canny(src, canny_thresh, canny_thresh*2)
    # Find countours
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get convex hull for each contour
    hulls = []
    for i in range(len(contours)):
        if len(contours) > 0:
            hull = cv2.convexHull(contours[i])
            hulls.append(hull)
    return hulls


def filterHulls(hulls):
    return filter(lambda hull: cv2.contourArea(hull) > min_hull_area, hulls)

#def findBoxes(hulls):
#    boxes = []
#    for hull in hulls:
#        target_poly = cv2.approxPolyDP(hull, 3, True)
#        target_box = cv2.minAreaRect(target_poly)
#        boxes.append(target_box)
#    return boxes

# Find the minimun area rectangle (often rotated) for each hull
def findBox(hull):
    poly = cv2.approxPolyDP(hull, 3, True)
    box = cv2.minAreaRect(poly)
    return box

# Fit a line that goes through the hull length-wise and return its angle in degrees
def findFittedLineAngle(hull, drawing):
    rows,cols = drawing.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(hull, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(drawing,(cols-1,righty),(0,lefty),(0,255,0),2)

    angle = np.arctan(vy/vx)
    angle = np.rad2deg(angle)
    return angle
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #text = 'angle = {}deg'.format(angle)
    #cv2.putText(drawing,text,(10,400), font, 1,(255,255,255),2,cv2.LINE_AA)


def findBoxAngle(box):
    points = cv2.boxPoints(box)
    d1 = cv2.norm(points[0] - points[1])
    d2 = cv2.norm(points[1] - points[2])
    angle = np.arctan((points[0][1] - points[3][1])/(points[0][0] - points[3][0]))
    angle = angle + (90 if d2 > d1 else 0)
    return angle - 90


# For each hull return a pair of its minAreaRect and respective angle
def findBoxesAndAngles(hulls, drawing):
    angledBoxes = []
    for i in range(len(hulls)):
        hull = hulls[i]
        box = findBox(hull)

        color = (200, 0, 0) 
        dbox = cv2.boxPoints(box)
        dbox = np.int0(dbox)
        cv2.drawContours(drawing,[dbox],0,color,2)

        #angle = findFittedLineAngle(hull, drawing)
        angle = findBoxAngle(box)

        angledBox = (box, angle)
        angledBoxes.append(angledBox)

    return angledBoxes

# Analyze rects and angle pairs to determine valid targets
def findTargets(angledBoxes):
    targets = []
    if len(angledBoxes) < 2:
        return targets
        
    # Sort boxes by x position to find pairs that are actually next to each other in the image    
    angledBoxes = sorted(angledBoxes, key=lambda box: cv2.boxPoints(box[0])[0][0], reverse=True)
    #print(len(angledBoxes))

    for i in range(len(angledBoxes)-1):
        leftBoxAngle = angledBoxes[i][1]
        rightBoxAngle = angledBoxes[i+1][1]

        #print(leftBoxAngle)

        #if leftBoxAngle > 0 and rightBoxAngle < 0:
        if rightBoxAngle - leftBoxAngle > 0:
            targets.append((angledBoxes[i], angledBoxes[i+1]))
    return targets

# Get bounding box from target pair
def sorroundTargetWithBox(target):
    points = cv2.boxPoints(target[0][0])
    points = np.append(points, cv2.boxPoints(target[1][0]))
    minX, minY, maxX, maxY = 10000000, 1000000, 0, 0
    for i in range(0, len(points), 2):
        x, y = points[i], points[i+1] 
        minX = min(x, minX)
        minY = min(y, minY)
        maxX = max(x, maxX)
        maxY = max(y, maxY)
    return (minX, minY, maxX, maxY)

# Point order from https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect
def sorroundTargetPoly(target):
    leftRectPoints = cv2.boxPoints(target[0][0])
    rightRectPoints = cv2.boxPoints(target[1][0])
    #print(leftRectPoints)

                                        # Target seen at an angle
    minX = leftRectPoints[3][0]         #       maxYL
    maxX = rightRectPoints[1][0]        #         /-------/              maxYR
    minYL = leftRectPoints[0][1]        #        /       /          \------\               |¯¯¯---____
    maxYL = leftRectPoints[2][1]        #       /       /            \      \        =>    |          |
    minYR = rightRectPoints[0][1]       #      /       /              \      \             |          |
    maxYR = rightRectPoints[2][1]       #     /       /                \______\            |____---¯¯¯
                                        #    /_______/               minYR     maxX
                                        #  minX     minYL       

    #return ((minX, minYL), (minX, maxYL), (maxX, maxYR), (maxX, minYR))
    return np.array([[minX, minYL], [minX, maxYL], [maxX, maxYR], [maxX, minYR]], np.int32)

# Drawing utilities
def drawContours(img, contours):
    cv2.drawContours(img, contours, -1, (255, 0,0), 3)

class AveragingBuffer(object):
    def __init__(self, maxlen):
        assert( maxlen>1)
        self.q=collections.deque(maxlen=maxlen)
        self.xbar=0.0
    def append(self, x):
        if len(self.q)==self.q.maxlen:
            # remove first item, update running average
            d=self.q.popleft()
            self.xbar=self.xbar+(self.xbar-d)/float(len(self.q))
        # append new item, update running average
        self.q.append(x)
        self.xbar=self.xbar+(x-self.xbar)/float(len(self.q))



# main
ab = AveragingBuffer(10)

# pandas
columns = ['yDistance', 'xDistance', 'theta', 'width', 'height', 'centerX', 'centerY', 'aspectRatio', 'heightRatio', 'yDiff']
df = pd.DataFrame(columns=columns)
currentIndex = 0

# setup tkinter
window = Tk()
window.geometry("200x200")
window.title("Data Gathering") 
thetaLbl = Label(window, text="Angle")
thetaLbl.grid(column=0, row=0)
thetaSpin = Spinbox(window, from_=0, to=100)
thetaSpin.grid(column=1,row=0)

yDistLbl = Label(window, text="Y Distance")
yDistLbl.grid(column=0, row=1)
yDistSpin = Spinbox(window, from_=0, to=100)
yDistSpin.grid(column=1,row=1)

xDistLbl = Label(window, text="X Distance")
xDistLbl.grid(column=0, row=2)
xDistSpin = Spinbox(window, from_=0, to=100)
xDistSpin.grid(column=1,row=2)

bboxX = Label(window, text="BBox Center X")
bboxX.grid(column=0, row=3)
bboxH = Label(window, text="BBox Height")
bboxH.grid(column=0, row=4)
bboxAR = Label(window, text="BBox Aspect R.")
bboxAR.grid(column=0, row=5)
bboxHR = Label(window, text="BBox Height R.")
bboxHR.grid(column=0, row=6)

def visionLoop():
    global currentIndex

    # Get timestamp
    tStart = datetime.datetime.now()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Image preprocessing and thresholding
    dst = smoothImage(frame)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    dst = hsvThreshold(dst)
    dst = closeFrame(dst)
    dst = openFrame(dst)

    cv2.imshow('mid', dst)

    #cv2.imshow("bruh", dst)

    # Find convex hulls over thresholded image
    hulls = findHulls(dst)
    hulls = filterHulls(hulls)

    # Sort hulls by area, biggest area first
    hulls = sorted(hulls, key=lambda cnt: cv2.contourArea(cnt), reverse=True)

    #print(hulls)

    # Initialize mat for target box drawing
    #drawing = np.zeros((dst.shape[0], dst.shape[1], 3), dtype=np.uint8)

    if len(hulls) > 0:
        # Find rotated rect and its respective angle from each hull (an angled box is a pair of a rect and its angle(float))
        angledBoxes = findBoxesAndAngles(hulls, frame)
        # Analyze boxes and return valid targets (a target in this case is a pair of two angled boxes)
        targets = findTargets(angledBoxes)

        #drawContours(frame, hulls)

        if len(targets) > 0:
            # Arbitrairly choose the first target
            target = targets[0]

            # Get bounding box sourounding target
            targetBox = sorroundTargetWithBox(target)

            # Get polygon sourounding target
            targetPoly = sorroundTargetPoly(target)

            # Draw target box
            color = (0, 200, 0)
            cv2.rectangle(frame, (targetBox[0], targetBox[1]), (targetBox[2], targetBox[3]), color, 3) 
            color = (0, 0, 200)
            pts = targetPoly.reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, color, thickness=2)

            # box properties
            centerX = ((targetBox[0] + targetBox[2])/2) - IMAGE_WIDTH/2
            centerY = ((targetBox[1] + targetBox[3])/2) - (IMAGE_HEIGHT/2)
            width = targetBox[2] - targetBox[0]
            height = targetBox[3] - targetBox[1]
            aspectRatio = width/height

            # polygon properties
            heightL = targetPoly[1][1] - targetPoly[0][1]
            heightR = targetPoly[2][1] - targetPoly[3][1]
            heightRatio = heightL/heightR
            yDiff = targetPoly[1][1] - targetPoly[2][1]


            bboxX.config(text="BBox Center X: {0:.2f}".format(centerX))
            bboxH.config(text="BBox Height: {0:.2f}".format(height))
            bboxAR.config(text="BBox Aspect R.: {0:.2f}".format(aspectRatio))
            bboxHR.config(text="BBox Height R.: {0:.2f}".format(heightRatio))

            if cv2.waitKey(1) & 0xFF == ord('c'):
                yDistance = yDistSpin.get()
                xDistance = xDistSpin.get()
                theta = thetaSpin.get()

                df.loc[currentIndex] = [yDistance, xDistance, theta, width, height, centerX, centerY, aspectRatio, heightRatio, yDiff]
                print(df)
                df.to_csv("data/dataBuffer.csv")
                currentIndex += 1


            #table.putNumber("rpi/center X", centerX)
            #table.putNumber("rpi/center Y", centerY)
            #table.putNumber("rpi/width", width)
            #table.putNumber("rpi/height", height)
            #table.putNumber("rpi/aspect ratio", aspectRatio)
            #table.putNumber("rpi/height ratio", heightRatio)


    # Get timestamp and calculate time difference
    tEnd = datetime.datetime.now()
    tElapsed = (tEnd - tStart).microseconds / 1000

    # Average time difference over the last 10 frames
    ab.append(tElapsed)
    framerate = 1000/ab.xbar
    #print("framerate :{}".format(framerate))
    cv2.putText(frame, 'Framerate: {:f}'.format(framerate), (10,450), cv2.FONT_HERSHEY_SIMPLEX, .75,(255,255,255),2, cv2.LINE_AA)


    # Display original frame with target box on top
    cv2.imshow('result', frame)

    # Send data to dashboard
    #table.putBoolean("DB/LED 0", True)
    #table.putNumber("rpi/framrate", framerate)


    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    window.after(10, visionLoop)


window.after(10, visionLoop)
window.mainloop()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H;%M;%S")
df.to_csv("data/data-{}.csv".format(timestamp))
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

