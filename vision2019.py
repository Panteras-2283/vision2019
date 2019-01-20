import numpy as np
import cv2
import datetime
from datetime import timedelta
import collections

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -10.0)

# Global variables
canny_thresh = 100
min_hull_area = 1000

H_LOW = 40
H_HIGH = 80
S_LOW = 80
S_HIGH = 255
V_LOW = 80
V_HIGH = 255


def smoothImage(src):
    dst = cv2.blur(src, (3,3))
    return dst

def hsvThreshold(src):
    dst = cv2.inRange(src, (H_LOW, S_LOW, V_LOW), (H_HIGH, S_HIGH, V_HIGH))
    return dst

# Dilate and then erode to eliminate holes
def closeFrame(src):
    kernel = np.ones((5, 5), np.uint8)
    dst = cv2.dilate(src, kernel, iterations=3)
    dst = cv2.erode(dst, kernel, iterations=3)
    return dst

# Erode and then dilate to eliminate noise
def openFrame(src):
    kernel = np.ones((5, 5), np.uint8)
    dst = cv2.erode(src, kernel, iterations=3)
    dst = cv2.dilate(dst, kernel, iterations=3)
    return dst

def findHulls(src):
    # Find edges
    dst = cv2.Canny(src, canny_thresh, canny_thresh*2)
    # Find countours
    contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get convex hull for each contour
    hulls = []
    for i in range(len(contours)):
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
    [vx,vy,x,y] = cv2.fitLine(hull, cv2.DIST_L2,0,0.01,0.01)
    #lefty = int((-x*vy/vx) + y)
    #righty = int(((cols-x)*vy/vx)+y)
    #cv2.line(drawing,(cols-1,righty),(0,lefty),(0,255,0),2)

    angle = np.arctan(vy/vx)
    angle = np.rad2deg(angle)
    return angle
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #text = 'angle = {}deg'.format(angle)
    #cv2.putText(drawing,text,(10,400), font, 1,(255,255,255),2,cv2.LINE_AA)


# For each hull return a pair of its minAreaRect and respective angle
def findBoxesAndAngles(hulls, drawing):
    angledBoxes = []
    for i in range(len(hulls)):
        hull = hulls[i]
        box = findBox(hull)
        angle = findFittedLineAngle(hull, drawing)

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

    for i in range(len(angledBoxes)-1):
        leftBoxAngle = angledBoxes[i][1]
        rightBoxAngle = angledBoxes[i+1][1]

        if leftBoxAngle > 0 and rightBoxAngle < 0:
            targets.append((angledBoxes[i], angledBoxes[i+1]))
    return targets

# Get bounding box from target pair
def sorroundTarget(target):
    points = cv2.boxPoints(target[0][0])
    points = np.append(points, cv2.boxPoints(target[1][0]))
    minX, minY, maxX, maxY = 10000000, 1000000, 0, 0
    for i in range(0, len(points), 2):
        x, y = points[i], points[i+1] 
        minX = x if x < minX else minX
        minY = y if y < minY else minY
        maxX = x if x > maxX else maxX
        maxY = y if y > maxY else maxY
    return (minX, minY, maxX, maxY)





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


ab = AveragingBuffer(10)
while(True):
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

    # Find convex hulls over thresholded image
    hulls = findHulls(dst)
    hulls = filterHulls(hulls)

    # Sort hulls by area, biggest area first
    hulls = sorted(hulls, key=lambda cnt: cv2.contourArea(cnt), reverse=True)

    # Initialize mat for target box drawing
    drawing = np.zeros((dst.shape[0], dst.shape[1], 3), dtype=np.uint8)

    if len(hulls) > 0:
        # Find rotated rect and its respective angle from each hull (an angled box is a pair of a rect and its angle(float))
        angledBoxes = findBoxesAndAngles(hulls, drawing)
        # Analyze boxes and return valid targets (a target in this case is a pair of two angled boxes)
        targets = findTargets(angledBoxes)

        if len(targets) > 0:
            # Arbitrairly choose the first target
            target = targets[0]
            # Get bounding box sourounding target
            targetBox = sorroundTarget(target)

            # Draw target box
            color = (0, 200, 0)
            cv2.rectangle(drawing, (targetBox[0], targetBox[1]), (targetBox[2], targetBox[3]), color, 3) 


    # Get timestamp and calculate time difference
    tEnd = datetime.datetime.now()
    tElapsed = (tEnd - tStart).microseconds / 1000

    # Average time difference over the last 10 frames
    ab.append(tElapsed)

    cv2.putText(frame, 'Framerate: {:f}'.format(1000/ab.xbar), (10,450), cv2.FONT_HERSHEY_SIMPLEX, .75,(255,255,255),2, cv2.LINE_AA)


    # Display original frame with target box on top
    result = frame + drawing
    cv2.imshow('result', result)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

