import csv
import pandas as pd
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import cv2
import numpy as np
import matplotlib.transforms as mtransforms
import time
from sklearn.cluster import KMeans
from numpy import arccos, array
from numpy.linalg import norm
from math import atan
import math

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def plot_bounds(corners, phase4_lower_line_with_tolerance, phase4_upper_line_with_tolerance, phase2_left_line_with_tolerance, phase2_right_line_with_tolerance):
    """Plot the outer boundaries of the crosswalks"""
    plt.xlim([-40, 46])
    plt.ylim([-40, 46])
    plt.gca().set_aspect('equal', adjustable='box')
    corner_xs = [corner[0] for corner in corners]
    corner_ys = [corner[1] for corner in corners]
    plt.scatter(corner_xs, corner_ys, color='b')
    a1, b1 = phase4_lower_line_with_tolerance
    a2, b2 = phase4_upper_line_with_tolerance
    a3,b3 = phase2_left_line_with_tolerance
    a4,b4 = phase2_right_line_with_tolerance
    abline(a1,b1)
    abline(a2,b2)
    abline(a3,b3)
    abline(a4,b4)

def find_angle(M1, M2):
    """Returns the smallest angle between two intersecting lines"""
    # Store the tan value  of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))
    # Calculate tan inverse of the angle
    return atan(angle)

def intersection(L1, L2):
    """Returns the point of intersection of two lines. Return False if non-intersecting."""
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = -Dx / D
        y = -Dy / D
        return x,y
    else:
        return False
 
def distance(p1, p2):
    """Returns the distance between two points p1 and p2"""
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def best_fit(X, Y):
    """Returns the line of best-fit as slope, intercept between x-values X and y-values Y."""
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    #print(len(X))
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    if denum != 0:
        m = numer / denum
    else: 
        #print("hiccup")
        #print(numer)
        #print(denum)
        m = 10**8
    b = ybar - m * xbar

    return m, b

def slope_intercept(x1,y1,x2,y2, boost_ratio):
    """Calculates the slope and intercept for the line between x1, y1 and x2, y2.
       The boost_ratio is used to return the boost term, which can be added to the 
       y-intercept to increase the distance of the new line from the pre-boosted line 
       by a consistent amount (i.e. boost_ratio is consistent between 4 lines, therefore)
       boost distance is equivalent despite different boost return values."""
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1  
    boost = boost_ratio/np.sin(np.pi/2 - np.arctan(a)) #was 2.75
    return a,b, boost

def point_in_list_past_line(line, xlist, ylist, direction):
    """Returns true of any point in xlist, ylist is past the line (given as slope, intercept) in
       the indicated direction."""
    for x, y in zip(xlist, ylist):
        if direction == ">":
            if y > line[0]*x + line[1]:
                return True
        else:
            if y < line[0]*x + line[1]:
                return True
    return False