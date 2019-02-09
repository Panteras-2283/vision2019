import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import tkinter as tk
from tkinter import Tk, Canvas
import math


root = Tk()
canvas = Canvas(root, width=400, height=400)
canvas.pack()


theta_gbr = None
x_gbr = None
y_mlp = None

def loadModels():
	global theta_gbr, x_gbr, y_mlp
	theta_gbr = joblib.load("thetaGBR.joblib.dat")
	x_gbr = joblib.load("xGBR.joblib.dat")
	y_mlp = joblib.load("yMLP.joblib.dat")


def calculateRobotVector(data):
	global theta_gbr, x_gbr, y_mlp
	width, height, centerX, centerY, aspectRatio, heightRatio, yDiff = data
	df = pd.DataFrame(data=np.array([[width, centerX, centerY, aspectRatio, heightRatio, yDiff]]), axis=0)
	df.columns = ['width', 'height', 'center_x', 'center_y', 'aspect_ratio', 'height_ratio', 'y_diff']

	# load model from file
	theta_gbr = joblib.load("thetaGBR.joblib.dat")
	x_gbr = joblib.load("xGBR.joblib.dat")
	y_MLP = joblib.load("yMLP.joblib.dat")

	#make predictions
	theta = theta_gbr.predict(df)
	x = x_gbr.predict(df)
	y = y_MLP.predict(df)

	dist = sqrt(x**2 + y**2)
	robotPoint = (x, y)
	#phi = arctan(y/x)

	draw(robotPoint, theta)

	return (robotPoint, theta)


def draw(targetPoint, angle):
	x, y = targetPoint
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