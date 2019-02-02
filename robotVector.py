import tkinter as tk
from tkinter import Tk, Canvas
import math


TARGET_TOP_HEIGHT = 31 * 2.54 # cm
TARGET_HEIGHT = 6 * 2.54 # cm
TARGET_WIDTH = 14.5 * 2.54 # cm
CAMERA_HEIGHT = 23.75 * 2.54 # cm

CAMERA_X_ANGLE = 70.42 * math.pi/180 # radians
CAMERA_Y_ANGLE = 43.3 * math.pi/180 # radians
FOCAL_LENGTH = 0.367#SENSOR_WIDTH/(2*math.tan(CAMERA_X_ANGLE/2)) # cm (probably ~0.3245)
SENSOR_WIDTH = FOCAL_LENGTH * 2 * math.tan(CAMERA_X_ANGLE/2) # FOCAL_LENGTH / (2*math.tan(CAMERA_X_ANGLE/2)) #0.39 # cm
SENSOR_HEIGHT = FOCAL_LENGTH * 2 * math.tan(CAMERA_Y_ANGLE/2) # FOCAL_LENGTH / (2*math.tan(CAMERA_Y_ANGLE/2)) # 0.22 # cm

Y_ANGLE_OFFSET = 0#15 * math.pi/180 # radians

PICTURE_WIDTH = 640 # pixels
PICTURE_HEIGHT = 360 # pixels probably should be 180 to keep 16:9 aspect ratio
PICTURE_CENTER_U = PICTURE_WIDTH/2
PICTURE_CENTER_V = PICTURE_HEIGHT/2

root = Tk()
canvas = Canvas(root, width=400, height=400)
canvas.pack()
l1 = tk.Label(root)
l1.pack()
l2 = tk.Label(root)
l2.pack()
l3 = tk.Label(root)
l3.pack()
l4 = tk.Label(root)
l4.pack()
l5 = tk.Label(root)
l5.pack()
l6 = tk.Label(root)
l6.pack()
l7 = tk.Label(root)
l7.pack()
l8 = tk.Label(root)
l8.pack()
l9 = tk.Label(root)
l9.pack()

def getAngleForPixelV(v):
	v = ((v - PICTURE_CENTER_V) * SENSOR_HEIGHT)/PICTURE_HEIGHT
	return math.atan(v/FOCAL_LENGTH) + Y_ANGLE_OFFSET

def getAngleForPixelU(u):
	u = ((u - PICTURE_CENTER_U) * SENSOR_WIDTH)/PICTURE_WIDTH
	return -math.atan(u/FOCAL_LENGTH)

def getDistanceForYAngle(theta):
	return (TARGET_TOP_HEIGHT - CAMERA_HEIGHT)/math.tan(theta)

def rotatePointByAngle(point, angle):
	x, y = point
	newX = x * math.cos(angle) + y * math.cos(math.pi/2 + angle)
	newY = x * math.sin(angle) + y * math.sin(math.pi/2 + angle)
	return (newX, newY)


def calculateRobotVector(targetMinU, targetMaxU, targetMaxVL, targetMaxVR):
	#targetPixelHeightL = 50
	#targetpixelHeightR = 50
	#targetPixelWidth = 100
	#targetCenterU = PICTURE_CENTER_U
	#targetCenterV = PICTURE_CENTER_V - 9.1 # empirical, must corroborate with calculations
	
	#targetMaxU = targetCenterU + targetPixelWidth/2
	#targetMinU = targetCenterU - targetPixelWidth/2

	#targetMaxVL = targetCenterV - targetPixelHeightL/2
	#targetMaxVR = targetCenterV - targetpixelHeightR/2
	try:

		targetMaxVL = PICTURE_HEIGHT - targetMaxVL
		targetMaxVR = PICTURE_HEIGHT - targetMaxVR
	
		targetCenterU = (targetMaxU + targetMinU)/2
		
		targetAngleL = getAngleForPixelV(targetMaxVL)
		targetAngleR = getAngleForPixelV(targetMaxVR)
		
		distanceL = getDistanceForYAngle(targetAngleL)
		distanceR = getDistanceForYAngle(targetAngleR)
	
	
		print("distanceR: {}".format(distanceR))
		print("distanceL: {}".format(distanceL))
		l1.config(text="distanceR: {}".format(distanceR))
		l2.config(text="distanceL: {}".format(distanceL))
	
		#print("angleMinU: {}".format(getAngleForPixelU(targetMinU) *180/math.pi))
		#print("angleMaxU: {}".format(getAngleForPixelU(targetMaxU) *180/math.pi))
		
		
		theta2 = abs(getAngleForPixelU(targetMinU) - getAngleForPixelU(targetMaxU))
		theta = math.acos(((distanceL**2) + (distanceR**2) - (TARGET_WIDTH**2))/(2 * distanceL * distanceR))
		l8.config(text="theta: {}".format(theta *180/math.pi))
		l9.config(text="theta2: {}".format(theta2 *180/math.pi))
		#print("theta: {}".format(theta *180/math.pi))
		#print("theta2: {}".format(theta2 *180/math.pi))
	
		phi = getAngleForPixelU(targetMaxU)
		#w2 = math.asin(distanceL * math.sin(theta)/TARGET_WIDTH)
		w = math.acos(((TARGET_WIDTH**2) + (distanceR**2) - (distanceL**2))/(2 * TARGET_WIDTH * distanceR))
		d = math.pi/2 + phi
		angle = d - w
	
	
		print("phi: {}".format(phi *180/math.pi))
		print("w: {}".format(w *180/math.pi))
		#print("w2: {}".format(w2 *180/math.pi))
		print("d: {}".format(d *180/math.pi))
		l3.config(text="phi: {}".format(phi *180/math.pi))
		l4.config(text="w: {}".format(w *180/math.pi))
		l5.config(text="d: {}".format(d *180/math.pi))
		
		
		distanceToTarget = (distanceL + distanceR)/2 # Not true. Just an approximation
		targetX = distanceToTarget * math.cos(math.pi/2 - getAngleForPixelU(targetCenterU))
		targetZ = distanceToTarget * math.sin(math.pi/2 - getAngleForPixelU(targetCenterU))
		
		targetPoint = (targetX, targetZ)

		#vr = math.hypot(targetX, targetZ)
		#vphi = math.atan2(targetZ, targetX)
	
		# Convert to target oriented coordinate system
		targetPoint = rotatePointByAngle(targetPoint, angle)
		print("Angle: {}".format(angle *180/math.pi))
		print("Target point: {}".format(targetPoint))
		l6.config(text="Angle: {}".format(angle *180/math.pi))
		l7.config(text="Target point: {}".format(targetPoint))
		draw(targetPoint, angle)
	
		#return ((vr, vphi), angle)
		
		# Convert to target oriented coordinate system
		# targetPoint = rotatePointByAngle(targetPoint, angle)

	except Exception as e:
		print(e)
	

def draw(targetPoint, angle):
	targetX, targetY = targetPoint
	#angle = angle * math.pi/180

	canvas.delete("all")
	canvas.create_line(180, 50, 220, 50, fill="#00ff00")

	robot = (200 - targetX, 50 + targetY)
	lineLength = 50
	lineAngle = math.pi/2 + angle
	lineEnd = (robot[0] + lineLength*math.cos(lineAngle), robot[1] + lineLength*math.sin(lineAngle))


	canvas.create_line(robot[0], robot[1], lineEnd[0], lineEnd[1], fill="#ff0000")

	canvas.pack()
	#root.geometry("400x400")
	root.update()

#calculateRobotVector(PICTURE_CENTER_U - 50, PICTURE_CENTER_U + 50, PICTURE_CENTER_V - 9.1 - 25, PICTURE_CENTER_V - 9.1 - 25)