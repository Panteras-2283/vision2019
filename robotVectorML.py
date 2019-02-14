import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import tkinter as tk
from tkinter import Tk, Canvas
import math
import collections
import pykalman
from pykalman import KalmanFilter
import time


root = Tk()
canvas = Canvas(root, width=400, height=400)
canvas.pack()


theta_gbr = None
x_gbr = None
y_mlp = None

# kalman filter parameters
kf = None
state_mean = None
state_covariance = [[50, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 50, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 10]]


observation_matrix = [[1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1]]

transition_matrix = [[1, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 1, 1, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]]


observation_covariance = [[50, 0, 0],
                          [0, 50, 0],
                          [0, 0,  10]]




def loadModels():
    global theta_gbr, x_gbr, y_mlp
    theta_gbr = joblib.load("models/rasp/thetaMLP.joblib.dat")
    x_gbr = joblib.load("models/rasp/xMLP.joblib.dat")
    y_mlp = joblib.load("models/rasp/yMLP.joblib.dat")


def initKalman(initial_state_mean):
    global kf, state_mean
    state_mean = initial_state_mean
    kf = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean,
                      initial_state_covariance = state_covariance, 
                      observation_covariance = observation_covariance)

tLast = 0
def calculateRobotVector(data, resetKalman):
    global theta_gbr, x_gbr, y_mlp, state_mean, state_covariance, kf, tLast
    width, height, centerX, centerY, aspectRatio, heightRatio, yDiff = data
    df = pd.DataFrame(data=np.array(
        [[width, height, centerX, centerY, aspectRatio, heightRatio, yDiff]]))
    df.columns = ['width', 'height', 'center_x', 'center_y',
                  'aspect_ratio', 'height_ratio', 'y_diff']

    # make predictions
    theta = theta_gbr.predict(df)[0]
    x = -x_gbr.predict(df)[0]
    y = y_mlp.predict(df)[0]

    if resetKalman or kf == None:
        initKalman([x, 0, y, 0, theta])
        print("Kalman reset!")

        tLast = time.time()
    else:
        # Calculate timestep in seconds
        dt = time.time() - tLast
        tLast = time.time()

        observation = [x, y, theta]

        transition_matrix = np.array([[1, 1, 0, 0, 0],
                     		 [0, 1, 0, 0, 0],
                     		 [0, 0, 1, 1, 0],
                     		 [0, 0, 0, 1, 0],
                     		 [0, 0, 0, 0, 1]])

        state_mean, state_covariance = kf.filter_update(transition_matrix=transition_matrix,
        											  filtered_state_mean=state_mean,
                                                      filtered_state_covariance=state_covariance,
                                                      observation=observation)

    x_new = state_mean[0]
    y_new = state_mean[2]
    theta_new = state_mean[4]

    robotPoint = (x_new, y_new)

    #print("point: {}".format(robotPoint))
    #print("theta: {}".format(theta_new))
    #phi = arctan(y/x)

    # convert to rad
    theta_new = theta_new * math.pi / 180

    draw(robotPoint, theta_new)

    return (robotPoint, theta_new)


def draw(targetPoint, angle):
    x, y = targetPoint

    # Scale values
    x = 3 * x
    y = 3 * y

    canvas.delete("all")
    canvas.create_line(160, 50, 240, 50, fill="#00ff00")

    robot = (200 - x, 50 + y)
    lineLength = 50
    lineAngle = math.pi / 2 + angle
    lineEnd = (robot[0] + lineLength * math.cos(lineAngle),
               robot[1] - lineLength * math.sin(lineAngle))

    canvas.create_line(robot[0], robot[1], lineEnd[0],
                       lineEnd[1], fill="#ff0000")

    canvas.pack()
    # root.geometry("400x400")
    root.update()


