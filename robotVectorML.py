import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib


def calculateRobotVector(data):
	df = pd.DataFrame(data=data)