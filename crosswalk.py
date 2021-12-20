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
from utils import plot_bounds, find_angle, intersection, slope_intercept
from pedestrian import Pedestrian
import imageio
import os


class CrosswalkScene():
    """ Represents the static crosswalk scene (i.e. geometry of the intersection and crossings).
         Contains a drawing utility to display the scene with dynamic objects, and maintains positions
         of crossings, corners, boundaries, etc. 
         Estimates the locations of corners using LiDAR data."""
         
    def __init__(self, filepath):


        self.df = pd.read_csv(filepath)
        self.times = self.times_from_timestamps()

        self.df2 = pd.read_csv('Competition_Data/SPM/signal_two_part_one.csv')
        self.df4 = pd.read_csv('Competition_Data/SPM/signal_four_part_one.csv')

        self.road_user_to_color = ["black","red","purple","green","blue"]
        self.corners = [(5.74,-1.23),(-13.12,9.88),(12.35, 12.48),(-1.42,24)] #lower right, lower left, upper right, upper left
        self.lower_right = self.corners[0]
        self.lower_left = self.corners[1]
        self.upper_right = self.corners[2]
        self.upper_left = self.corners[3]
        self.bottom_line_midpoint = (.5*(self.corners[0][0]+self.corners[1][0]), .5*(self.corners[0][1] + self.corners[1][1]))
        self.top_line_midpoint = (.5*(self.corners[2][0]+self.corners[3][0]), .5*(self.corners[2][1] + self.corners[3][1]))
        self.left_line_midpoint = (.5*(self.corners[1][0]+self.corners[3][0]), .5*(self.corners[1][1] + self.corners[3][1]))
        self.right_line_midpoint = (.5*(self.corners[2][0]+self.corners[0][0]), .5*(self.corners[2][1] + self.corners[0][1]))
        self.pedestrian_list = self.df['ID'].unique()
        self.phase4_lower_line_with_tolerance, self.phase4_upper_line_with_tolerance, self.phase2_right_line_with_tolerance, self.phase2_left_line_with_tolerance = self.estimate_pedestrian_boundaries()
        self.lower_line_crossing, self.upper_line_crossing, self.right_line_crossing, self.left_line_crossing = self.estimate_crossing_boundaries()
        
        #self.corner_estimator() # Estimate locations of corners (takes a while to run, commented until needed.)

    def estimate_crossing_boundaries(self):
        """Returns the parameters for the inner boundary lines, where the crosswalk becomes the intersection center."""
        a1,b1,boost1 = slope_intercept(self.corners[0][0], self.corners[0][1], self.corners[1][0],self.corners[1][1], 2)
        a2,b2,boost2 = slope_intercept(self.corners[2][0], self.corners[2][1], self.corners[3][0],self.corners[3][1], 2)
        a3,b3,boost3 = slope_intercept(self.corners[0][0], self.corners[0][1], self.corners[2][0],self.corners[2][1], 2)
        a4,b4,boost4 = slope_intercept(self.corners[1][0], self.corners[1][1], self.corners[3][0],self.corners[3][1], 2)

        return (a1,b1+boost1),(a2,b2-boost2),(a3,b3+boost3),(a4,b4-boost4)

    def estimate_pedestrian_boundaries(self):
        """Returns the parameters for the outer boundary lines, where the crosswalk becomes the outer street."""
        a1,b1,boost1 = slope_intercept(self.corners[0][0], self.corners[0][1], self.corners[1][0],self.corners[1][1], 2.75)
        a2,b2,boost2 = slope_intercept(self.corners[2][0], self.corners[2][1], self.corners[3][0],self.corners[3][1], 2.75)
        a3,b3,boost3 = slope_intercept(self.corners[0][0], self.corners[0][1], self.corners[2][0],self.corners[2][1], 2.75)
        a4,b4,boost4 = slope_intercept(self.corners[1][0], self.corners[1][1], self.corners[3][0],self.corners[3][1], 2.75)

        return (a1,b1-boost1),(a2,b2+boost2),(a3,b3-boost3),(a4,b4+boost4)
        

    def get_corners_from_rectangle(self, center, dimensions, angle):
        # create the (normalized) perpendicular vectors, scale them appropriately by the dimensions
        v1 = [.5*dimensions[0]*np.cos(angle), .5*dimensions[0]*np.sin(angle)]
        v2 = [.5*dimensions[1]*-v1[1], .5*dimensions[1]*v1[0]]  # rotate by 90

        # return the corners by moving the center of the rectangle by the vectors
        corner1 = (center[0]+v1[0]+v2[0], center[1]+v1[1]+v2[1])
        corner2 = (center[0]-v1[0]+v2[0], center[1]-v1[1]+v2[1])
        corner3 = (center[0]-v1[0]-v2[0], center[1]-v1[1]-v2[1])
        corner4 = (center[0]+v1[0]-v2[0], center[1]+v1[1]-v2[1])

        return np.array([corner1, corner2, corner3, corner4])

    def add_to_heatmap(self, sum_map):
        for index,row in self.df.iterrows():
            bbx = row['BBox_Position_X']
            bby = row['BBox_Position_Y']

            road_user_color = self.road_user_to_color[int(row['Label'])]

            # Create a Rectangle patch. Cars red, bikes green, pedestrians yellow. Blue is unknown, black is none.  
            if road_user_color == "purple": #road_user_color != "black" and road_user_color != "blue":
                sum_map[86-int(bby)-40][int(bbx+40)] += 1

        return sum_map

    def corner_theta(self, v, w):
        return arccos(v.dot(w)/(norm(v)*norm(w)))

    def corner_estimator(self):
        # Calculate means, show them on scatter plot
        k = 4
        xs = self.df['BBox_Position_X']
        ys = self.df['BBox_Position_Y']
        X = np.array([list(a) for a in zip(xs,ys)])
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        print(kmeans.cluster_centers_)
        means = kmeans.cluster_centers_
    
        plt.scatter(xs,ys, color = 'black', s = 2)
        plt.scatter(means[:,0], means[:,1], color='cyan')
        # Create lines, show on scatter plot
        corner_start = means[0]
        corner1 = means[1]
        corner2 = means[2]
        corner3 = means[3]

        angle12 = self.corner_theta(means[1]-means[0], means[2]-means[0])
        angle23 = self.corner_theta(means[2]-means[0], means[3]-means[0])
        angle13 = self.corner_theta(means[1]-means[0], means[3]-means[0])
        angles = [angle12, angle23, angle13]

        if angle12 == max(angles):
            line1 = [corner_start, corner1]
            line2 = [corner_start, corner2]
            line3 = [corner1, corner3]
            line4 = [corner2, corner3]
        elif angle23 == max(angles):
            line1 = [corner_start, corner2]
            line2 = [corner_start, corner3]
            line3 = [corner1, corner3]
            line4 = [corner1, corner2]
        else:
            line1 = [corner_start, corner1]
            line2 = [corner_start, corner3]
            line3 = [corner1, corner2]
            line4 = [corner2, corner3]
        lines = [line1, line2, line3, line4]
        line_parameters = []
        # pick a corner to start
        # pick the two corners which maximize the angle
        # then, pair these two corners with the one that was left out. 
        for line in lines:
            x1 = line[0][0]
            x2 = line[1][0]
            y1 = line[0][1]
            y2 = line[1][1]
            xvals = [x1, x2]
            yvals = [y1, y2]
            a = y1-y2
            b = x2-x1
            c = (x1-x2)*y1+(y2-y1)*x1
            line_parameters += [[a,b,c]]
            #plt.plot(xvals, yvals)

        # Determine which points are close to which lines, color accordingly.
        point_clusters = [[],[],[],[]]
        for x, y in zip(xs, ys):
            dists = [abs(triple[0]*x+triple[1]*y+triple[2])/((triple[0]**2+triple[1]**2)**.5) for triple in line_parameters]
            closest = np.argmin(dists)
            if dists[closest] < 3.5:
                point_clusters[closest] += [[x,y]]

        diffs = []
        tolerance = .05

        filenames = []

        for j in range(20):
            filename = 'gif/'+f'{j}.png'
            filenames.append(filename)
            new_line_parameters = []
            for cluster in point_clusters:
                xs_c = [c[0] for c in cluster]
                slope, intercept = np.polyfit([c[0] for c in cluster], [c[1] for c in cluster], 1)
                b_coefficient = -1
                abline_values = [slope * i + intercept for i in xs_c]
                new_line_parameters += [[slope, b_coefficient, intercept]]

            # Find intersection to estimate corners
            if j > 0:
                old_intersections = intersections
            intersections = []
            for i in range(4):
                intersections += [intersection(new_line_parameters[i%4], new_line_parameters[(i+1)%4])]
            for intersection_pt in intersections:
                if j == 19:
                    plt.scatter(intersection_pt[0], intersection_pt[1], color='c', s = 100)
                else:
                    plt.scatter(intersection_pt[0], intersection_pt[1], color='pink')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(filename)

            if j > 0:
                difference_intersection = sum([((intersections[k][0] - old_intersections[k][0])**2 + (intersections[k][1] - old_intersections[k][1])**2)**.5 for k in range(4)])
                diffs += [difference_intersection]
                if difference_intersection < tolerance:
                    break

            # find nearest points now
            point_clusters = [[],[],[],[]]
            for x, y in zip(xs, ys):
                dists = [abs(triple[0]*x+triple[1]*y+triple[2])/((triple[0]**2+triple[1]**2)**.5) for triple in new_line_parameters]
                closest = np.argmin(dists)
                if j > 0:
                    boost = (3/j)/np.sin(np.pi/2 - np.arctan(new_line_parameters[closest][0]))
                else: 
                    boost = 3/np.sin(np.pi/2 - np.arctan(new_line_parameters[closest][0]))

                 
                if dists[closest] < 3.5 and y >= new_line_parameters[closest][0]*x + new_line_parameters[closest][2]-boost and y <= new_line_parameters[closest][0]*x + new_line_parameters[closest][2]+boost:
                    point_clusters[closest] += [[x,y]]

        print(diffs)

        for i in range(4):
            circle1 = plt.Circle(intersections[i], 2, color='r', fill=False)
            plt.gca().add_patch(circle1)
            plt.scatter(intersections[i][0], intersections[i][1], color='r')
        

        plt.gca().set_aspect('equal', adjustable='box')
        plt.gcf().savefig('scatter_ped.png')
        for j in range(21,30):
            filename = 'gif/'+f'{j}.png'
            filenames.append(filename)
            plt.gcf().savefig(filename)
        plt.show()
        # build gif
        with imageio.get_writer('gif/mygif.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                    
        # Remove files
        for filename in set(filenames):
            os.remove(filename)

        self.corners = intersections

    def display_scene_at_time(self, time):
        """ Includes ALL objects (cars, pedestrians, etc.) """
        # for the time, gather all rows
        timeset = self.df[self.df['Timestamp'] == self.times[time]]

        # for each object, draw a bounding box of that color
        plt.xlim([-40, 46])
        plt.ylim([-40, 46])
        plt.gca().set_aspect('equal', adjustable='box')
        for index,row in timeset.iterrows():
            bbx = row['BBox_Position_X']
            bby = row['BBox_Position_Y']
            bbsx = row['BBox_Size_X']
            bbsy = row['BBox_Size_Y']
            yaw = row['BBox_Yaw']

            road_user_color = self.road_user_to_color[int(row['Label'])]

            # Create a Rectangle patch. Cars red, bikes green, pedestrians yellow. Blue is unknown, black is none.  
            if road_user_color == "purple": #road_user_color != "black" and road_user_color != "blue":
                rect = patches.Rectangle((bbx-bbsx/2, bby-bbsy/2), bbsx, bbsy, linewidth=1, edgecolor=road_user_color, facecolor=road_user_color)
                t2 = mtransforms.Affine2D().rotate_around(bbx, bby, yaw)+ plt.gca().transData
                rect.set_transform(t2)


                #rect = patches.Rectangle((0, 0), .5, .5, linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                plt.gca().add_patch(rect)

            #rect = patches.Rectangle((bbx-bbsx/2, bby-bbsy/2), bbsx, bbsy, linewidth=1, edgecolor=road_user_color, facecolor=road_user_color)
            #plt.gca().add_patch(rect)
            # #corners = self.get_corners_from_rectangle((bbx,bby),(bbsx,bbsy),yaw)
            # #rr = cv2.minAreaRect(corners)
            # rot_rect = ((bbx,bby),(bbsx,bbsy),yaw)
            # box = cv2.boxPoints(rot_rect)
            # box = np.int0(box)
            # cv2.drawContours(im,[box],0,(0,0,255),2)

        plt.show(block=False)

        return 0

    def times_from_timestamps(self):
        timestamps = self.df['Timestamp'].unique()
        return timestamps