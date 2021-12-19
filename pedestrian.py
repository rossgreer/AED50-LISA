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
from utils import plot_bounds, find_angle, intersection, distance, best_fit, point_in_list_past_line
import heapq

class Pedestrian():
    def __init__(self, idval, snippet, cs):
        self.id = idval
        self.info = snippet

        self.xcoords = self.info['BBox_Position_X']
        self.ycoords = self.info['BBox_Position_Y']
        self.timesteps = self.info['Timestamp']

        self.cs = cs

        self.x_center = .25*sum([c[0] for c in self.cs.corners])
        self.y_center = .25*sum([c[1] for c in self.cs.corners])

        self.origin = None
        self.origin_ts = None
        self.destination = None
        self.info = snippet
        self.label = snippet['Label'].iloc[0]
        self.av_height = snippet['BBox_Size_Z'].mean()
        #self.find_origin()
        self.find_crossing_origin_and_destination()
        #print("origin: "+str(self.origin))
        #self.find_destination()
        self.stationary = self.stationary_at_corner()
        self.category = None # non-crossing
        self.classify()

        # path should be tuples of (time, xcoord, ycoord, velx, vely)

    def classify(self):
        # If all points of the trajectory are out of the crosswalk areas, then it is a 'non-crossing' pedestrian
        xcoords = self.info['BBox_Position_X']
        ycoords = self.info['BBox_Position_Y']

        # get average slope
        first_xcoord = xcoords.iloc[0]
        last_xcoord = xcoords.iloc[-1]
        first_ycoord = ycoords.iloc[0]
        last_ycoord = ycoords.iloc[-1]

        average_slope = (last_ycoord - first_ycoord)/(last_xcoord-first_xcoord)
        intercept = last_ycoord - average_slope*last_xcoord

        # upper, lower, left, right
        line_set = [self.cs.phase4_upper_line_with_tolerance, self.cs.phase4_lower_line_with_tolerance, self.cs.phase2_left_line_with_tolerance, self.cs.phase2_right_line_with_tolerance]
        intersections = [intersection((average_slope, -1, intercept), (line[0], -1, line[1])) for line in line_set]
        def biker_passing(intersections, corners, traj_point):
            #first, which set of corners are we collectively closest to?
            closer_to_midpoint_than_corners = False
            slope_near_perpendicular = False
            dist_lower_line_corners = distance(traj_point, corners[0]) + distance(traj_point, corners[1]) 
            dist_upper_line_corners = distance(traj_point, corners[2]) + distance(traj_point, corners[3])
            dist_left_line_corners = distance(traj_point, self.cs.lower_left) + distance(traj_point, self.cs.upper_left)
            dist_right_line_corners = distance(traj_point, corners[0]) + distance(traj_point, corners[2])
            dists = [dist_lower_line_corners, dist_upper_line_corners, dist_left_line_corners, dist_right_line_corners]
            angle_threshold = np.pi/8
            first_xcoord = xcoords.iloc[0]
            last_xcoord = xcoords.iloc[-1]
            first_ycoord = ycoords.iloc[0]
            last_ycoord = ycoords.iloc[-1]
            first = (first_xcoord, first_ycoord)
            last = (last_xcoord, last_ycoord)
            if min(dists) == dist_lower_line_corners:
                #intersection = intersections[1]
                #print("lower chosen")
                if (distance(first,corners[0]) >= distance(first,self.cs.bottom_line_midpoint) and distance(first, corners[1]) >= distance(first, self.cs.bottom_line_midpoint) and 
                    distance(last,corners[0]) >= distance(last,self.cs.bottom_line_midpoint) and distance(last, corners[1]) >= distance(last, self.cs.bottom_line_midpoint)):
                    closer_to_midpoint_than_corners = True
                both_angles = [(find_angle(line_set[1][0],average_slope)+2*np.pi)%(2*np.pi), np.pi/2]
                if (max(both_angles)-min(both_angles)) < angle_threshold: #pi/6
                    slope_near_perpendicular = True
                # print(slope_near_perpendicular)
                # print(closer_to_midpoint_than_corners)
                # print(((find_angle(line_set[1][0],average_slope)+2*np.pi)%(2*np.pi) - np.pi/2))

            elif min(dists) == dist_upper_line_corners:
                #intersection = intersections[0]

                if (distance(first,corners[2]) >= distance(first,self.cs.top_line_midpoint) and distance(first, corners[3]) >= distance(first, self.cs.top_line_midpoint) and 
                    distance(last,corners[2]) >= distance(last,self.cs.top_line_midpoint) and distance(last, corners[3]) >= distance(last, self.cs.top_line_midpoint)):
                    closer_to_midpoint_than_corners = True
                both_angles = [(find_angle(line_set[0][0],average_slope)+2*np.pi)%(2*np.pi), np.pi/2]
                if (max(both_angles)-min(both_angles)) < angle_threshold: #pi/6
                    slope_near_perpendicular = True
            elif min(dists) == dist_left_line_corners:
                #print("Left chosen")
                #intersection = intersections[2]
                if (distance(first,self.cs.lower_left) >= distance(first,self.cs.left_line_midpoint) and distance(first, self.cs.upper_left) >= distance(first, self.cs.left_line_midpoint) and 
                    distance(last,self.cs.lower_left) >= distance(last,self.cs.left_line_midpoint) and distance(last, self.cs.upper_left) >= distance(last, self.cs.left_line_midpoint)):
                    closer_to_midpoint_than_corners = True
                both_angles = [(find_angle(line_set[2][0],average_slope)+2*np.pi)%(2*np.pi), np.pi/2]
                if (max(both_angles)-min(both_angles)) < angle_threshold: #pi/6
                    slope_near_perpendicular = True
                # print("stats")
                # print(closer_to_midpoint_than_corners)
                # print(slope_near_perpendicular)
            elif min(dists) == dist_right_line_corners:
                #print("Right chosen")
                #intersection = intersections[3]
                if (distance(first,corners[0]) >= distance(first,self.cs.right_line_midpoint) and distance(first, corners[2]) >= distance(first, self.cs.right_line_midpoint) and 
                    distance(last,corners[0]) >= distance(last,self.cs.right_line_midpoint) and distance(last, corners[2]) >= distance(last, self.cs.right_line_midpoint)):
                    closer_to_midpoint_than_corners = True
                both_angles = [(find_angle(line_set[3][0],average_slope)+2*np.pi)%(2*np.pi), np.pi/2]
                if (max(both_angles)-min(both_angles)) < angle_threshold: #pi/6
                    slope_near_perpendicular = True

            return closer_to_midpoint_than_corners and slope_near_perpendicular

        def all_points_outside_boundaries():
            all_points_outside = True
            for ptx, pty in zip(xcoords, ycoords):
                # print("Statements")
                # print("Point: "+str(ptx)+" " + str(pty))
                # print("Values: "+str(self.cs.phase2_left_line_with_tolerance[0]))
                # print("Values: "+str(self.cs.phase2_left_line_with_tolerance[1]))

                # print(pty > self.cs.phase4_lower_line_with_tolerance[0]*ptx + self.cs.phase4_lower_line_with_tolerance[1])
                # print(pty < self.cs.phase4_upper_line_with_tolerance[0]*ptx + self.cs.phase4_upper_line_with_tolerance[1])
                # print(pty < self.cs.phase2_left_line_with_tolerance[0]*ptx + self.cs.phase2_left_line_with_tolerance[1])
                if (pty > self.cs.phase4_lower_line_with_tolerance[0]*ptx + self.cs.phase4_lower_line_with_tolerance[1] 
                    and pty < self.cs.phase4_upper_line_with_tolerance[0]*ptx + self.cs.phase4_upper_line_with_tolerance[1] 
                    and pty < self.cs.phase2_left_line_with_tolerance[0]*ptx + self.cs.phase2_left_line_with_tolerance[1] 
                    and pty > self.cs.phase2_right_line_with_tolerance[0]*ptx + self.cs.phase2_right_line_with_tolerance[1]):
                    #print("Here")
                    all_points_outside = False
                    break
            return all_points_outside

        def moves_to_pocket_away_from_origin():
            last_ptx = xcoords.iloc[-1]
            last_pty = ycoords.iloc[-1]
            #print(self.origin)
            #sprint(self.cs.lower_left)
            if self.origin == None:
                return False
            if (self.cs.corners[self.origin] == self.cs.lower_left and 
                last_pty < self.cs.phase4_lower_line_with_tolerance[0]*last_ptx + self.cs.phase4_lower_line_with_tolerance[1] and
                last_pty > self.cs.phase2_left_line_with_tolerance[0]*last_ptx + self.cs.phase2_left_line_with_tolerance[1]):
                return True
            elif (self.cs.corners[self.origin] == self.cs.lower_right and 
                last_pty < self.cs.phase4_lower_line_with_tolerance[0]*last_ptx + self.cs.phase4_lower_line_with_tolerance[1] and
                last_pty < self.cs.phase2_right_line_with_tolerance[0]*last_ptx + self.cs.phase2_right_line_with_tolerance[1]):
                return True
            elif (self.cs.corners[self.origin] == self.cs.upper_left and 
                last_pty > self.cs.phase4_upper_line_with_tolerance[0]*last_ptx + self.cs.phase4_upper_line_with_tolerance[1] and
                last_pty > self.cs.phase2_left_line_with_tolerance[0]*last_ptx + self.cs.phase2_left_line_with_tolerance[1]):
                return True
            elif (self.cs.corners[self.origin] == self.cs.upper_right and 
                last_pty > self.cs.phase4_upper_line_with_tolerance[0]*last_ptx + self.cs.phase4_upper_line_with_tolerance[1] and
                last_pty < self.cs.phase2_right_line_with_tolerance[0]*last_ptx + self.cs.phase2_right_line_with_tolerance[1]):
                return True
            else:
                return False

        def stays_closer_to_origin_than_midpoint():
            last_ptx = xcoords.iloc[-1]
            last_pty = ycoords.iloc[-1]
            midpoints = [self.cs.bottom_line_midpoint, self.cs.top_line_midpoint, self.cs.left_line_midpoint, self.cs.right_line_midpoint]
            for pt in midpoints:
                if distance((last_ptx,last_pty), self.cs.corners[self.origin]) > distance((last_ptx,last_pty),pt):
                    return False
            return True

        def starting_away():
            # if distance from first to center is further than all corners to center, with a little margin
            margin = 1
            first_xcoord = xcoords.iloc[0]
            first_ycoord = ycoords.iloc[0]
            x_center = .25*sum([c[0] for c in self.cs.corners])
            y_center = .25*sum([c[1] for c in self.cs.corners])

            dist_corners_to_center = min([distance(c,(x_center,y_center)) for c in self.cs.corners])
            if dist_corners_to_center + margin < distance((first_xcoord, first_ycoord), (x_center,y_center)):
                return True
            return False

        def segment_distance_similar():
            first_xcoord = xcoords.iloc[0]
            last_xcoord = xcoords.iloc[-1]
            first_ycoord = ycoords.iloc[0]
            last_ycoord = ycoords.iloc[-1]
            dist = distance((first_xcoord,first_ycoord),(last_xcoord,last_ycoord))
            corner_distance = distance((self.cs.corners[0][0], self.cs.corners[0][1]), (self.cs.corners[2][0], self.cs.corners[2][1]))
            #print(np.abs(dist-corner_distance))
            if np.abs(dist-corner_distance) < 5:
                return True
            return False

        def most_points_in_road():
            num_in_road = 0
            for ptx, pty in zip(xcoords, ycoords):
                if ((pty > self.cs.phase4_lower_line_with_tolerance[0]*ptx + self.cs.phase4_lower_line_with_tolerance[1] 
                    and pty < self.cs.phase4_upper_line_with_tolerance[0]*ptx + self.cs.phase4_upper_line_with_tolerance[1]) 
                    or (pty < self.cs.phase2_left_line_with_tolerance[0]*ptx + self.cs.phase2_left_line_with_tolerance[1] 
                    and pty > self.cs.phase2_right_line_with_tolerance[0]*ptx + self.cs.phase2_right_line_with_tolerance[1])):
                    #print("Here")
                    num_in_road += 1
            if num_in_road/len(xcoords) > .75:
                return True
            return False

        def every_point_further_than_corner_from_center():
            dist = max([distance(self.cs.corners[i], (self.x_center, self.y_center)) for i in range(4)])

            for ptx,pty in zip(self.xcoords,self.ycoords):
                if distance((ptx,pty), (self.x_center, self.y_center)) < dist:
                    return False
            return True


        if len(self.xcoords) < 3:
            self.category = "InsufficientData"

        elif biker_passing(intersections, self.cs.corners, ((xcoords.iloc[0]+xcoords.iloc[-1])/2, (ycoords.iloc[0]+ycoords.iloc[-1])/2)) or self.all_points_inside_inner(): #pi/6
            self.category = "BikerPassingThrough"

        elif ((segment_distance_similar() and most_points_in_road() and self.origin == None and self.destination == None) 
            or (self.starts_behind_origin_line() and self.distance_remaining_to_destination(int(self.timesteps.iloc[-1])) == 0 and self.cross_light_never_on())
            or (self.starts_behind_origin_line() and self.cross_light_never_on() and self.crosses_origin_line())):
            self.category = "Jaywalker"

        elif (all_points_outside_boundaries() or moves_to_pocket_away_from_origin() 
            or (self.origin != None and stays_closer_to_origin_than_midpoint() and starting_away()) 
            or self.stationary or every_point_further_than_corner_from_center()):
            #print(stays_closer_to_origin_than_midpoint() and starting_away())
            #print(moves_to_pocket_away_from_origin())
            #print(self.stationary)
            # print(all_points_outside_boundaries())
            # print(moves_to_pocket_away_from_origin())
            # print(self.origin != None and stays_closer_to_origin_than_midpoint() and starting_away())
            # print(self.stationary)
            self.category = "Non-Crossing"

        elif not self.starts_behind_origin_line() and self.cross_light_never_on():
            self.category = "Ambiguous"

        else:
            self.category = "Crossing"


        # IF THE average slope intersects the crosswalk line nearer the midpoint than an endpoint AND nearly perpendicular AND label 3, it is probably a cyclist passing through. Let's see if cyclist or not. 

    def plot_path(self, color):
        plt.xlim([-40, 46])
        plt.ylim([-40, 46])
        plt.gca().set_aspect('equal', adjustable='box')
        xcoords = self.info['BBox_Position_X']
        ycoords = self.info['BBox_Position_Y']
        plt.scatter(xcoords, ycoords, c=[i/len(xcoords) for i in range(len(xcoords))])
        plt.scatter([corner[0] for corner in self.cs.corners], [corner[1] for corner in self.cs.corners], color='b')
        plt.title(self.id)
        plt.show()

    def find_origin(self):
        xcoords = self.info['BBox_Position_X']
        ycoords = self.info['BBox_Position_Y']
        counter = 0
        # plt.xlim([-40, 46])
        # plt.ylim([-40, 46])
        # plt.gca().set_aspect('equal', adjustable='box')
        # for corner in self.cs.corners:
        #     circle1 = plt.Circle((corner[0], corner[1]), 2, color='m')
        #     plt.gca().add_patch(circle1)
        #plt.show()
        for ptx, pty in zip(xcoords,ycoords):
            if (pty > self.cs.phase4_lower_line_with_tolerance[0]*ptx + self.cs.phase4_lower_line_with_tolerance[1] 
                and pty < self.cs.phase4_upper_line_with_tolerance[0]*ptx + self.cs.phase4_upper_line_with_tolerance[1] 
                and pty < self.cs.phase2_left_line_with_tolerance[0]*ptx + self.cs.phase2_left_line_with_tolerance[1] 
                and pty > self.cs.phase2_right_line_with_tolerance[0]*ptx + self.cs.phase2_right_line_with_tolerance[1]):
                #print('within bounds')
                dists = [distance((ptx,pty),self.cs.corners[k]) for k in range(4)]
                self.origin = dists.index(min(dists))
                break
            else:
                for i, corner in enumerate(self.cs.corners):
                    if (ptx-corner[0])**2 + (pty-corner[1])**2 <= 56.25:
                        self.origin = i
                        self.origin_ts = counter
                        break
                counter += 1
                if self.origin != None:
                    break



        #return 0
    def all_points_inside_inner(self):
        for x,y in zip(self.xcoords, self.ycoords):
            if not (y < self.cs.upper_line_crossing[0]*x+self.cs.upper_line_crossing[1] and y > self.cs.lower_line_crossing[0]*x+self.cs.lower_line_crossing[1] 
                and y < self.cs.left_line_crossing[0]*x+self.cs.left_line_crossing[1] and y > self.cs.right_line_crossing[0]*x+self.cs.right_line_crossing[1]):
                return False
        return True

    def filter_remove_points_outside_restriction_box(self):
        xs = []
        ys = []
        for x, y in zip(self.xcoords,self.ycoords):
            if (y < self.cs.phase4_upper_line_with_tolerance[0]*x + self.cs.phase4_upper_line_with_tolerance[1] and
                y > self.cs.phase4_lower_line_with_tolerance[0]*x + self.cs.phase4_lower_line_with_tolerance[1] and 
                y < self.cs.left_line_crossing[0]*x + self.cs.left_line_crossing[1] and 
                y > self.cs.right_line_crossing[0]*x + self.cs.right_line_crossing[1]):
                xs += [x]
                ys += [y]
            elif (y < self.cs.upper_line_crossing[0]*x + self.cs.upper_line_crossing[1] and
                y > self.cs.lower_line_crossing[0]*x + self.cs.lower_line_crossing[1] and 
                y < self.cs.phase2_left_line_with_tolerance[0]*x + self.cs.phase2_left_line_with_tolerance[1] and 
                y > self.cs.phase2_right_line_with_tolerance[0]*x + self.cs.phase2_right_line_with_tolerance[1]): 
                xs += [x]
                ys += [y]
        return xs, ys


    def find_crossing_origin_and_destination(self):
        xcoords = self.info['BBox_Position_X']
        ycoords = self.info['BBox_Position_Y']
        # Make linear regression. Poll at x coordinates of each corner. See which pair it is closest to. 
        x_filtered,y_filtered = self.filter_remove_points_outside_restriction_box()
        if len(x_filtered)>4:
            #print("Enough filtered")
            best_fit_line_m, best_fit_line_b = best_fit(x_filtered,y_filtered)
            if x_filtered[-1] - x_filtered[0] != 0:
                av_slope = (y_filtered[-1]-y_filtered[0])/(x_filtered[-1] - x_filtered[0])
            else: 
                av_slope = (ycoords.iloc[-1]-ycoords.iloc[0])/(xcoords.iloc[-1] - xcoords.iloc[0])

        else:
            #print("Not enough filtered")
            best_fit_line_m, best_fit_line_b = best_fit(xcoords,ycoords)
            av_slope = (ycoords.iloc[-1]-ycoords.iloc[0])/(xcoords.iloc[-1] - xcoords.iloc[0])


            near_corner_y_vals = [abs(best_fit_line_m*self.cs.corners[i][0] + best_fit_line_b - self.cs.corners[i][1]) for i in range(4)]
            # find out if updown or leftright
            if (find_angle(av_slope,.5*(self.cs.phase2_left_line_with_tolerance[0]+self.cs.phase2_right_line_with_tolerance[0])) 
                <= find_angle(av_slope, .5*(self.cs.phase4_upper_line_with_tolerance[0]+self.cs.phase4_lower_line_with_tolerance[0]))):#abs(av_slope - self.cs.phase4_upper_line_with_tolerance[0]):
                updown = True
            else:
                updown = False
            # print("Updown")
            # print(updown)
            # print(find_angle(av_slope,.5*(self.cs.phase2_left_line_with_tolerance[0]+self.cs.phase2_right_line_with_tolerance[0])))
            # print(find_angle(av_slope, .5*(self.cs.phase4_upper_line_with_tolerance[0]+self.cs.phase4_lower_line_with_tolerance[0])))
            if updown:
                if ycoords.iloc[-1] > ycoords.iloc[0]:
                    moves_positive = True
                else:
                    moves_positive = False
            else:
                if xcoords.iloc[-1] > xcoords.iloc[0]:
                    moves_positive = True
                else:
                    moves_positive = False
            # print("MP")
            # print(moves_positive)

            threshold = 8.15
            # print(near_corner_y_vals)
            # print("Number of points")
            # print(len(self.xcoords))
            if near_corner_y_vals[0] < threshold and near_corner_y_vals[1] < threshold:
                if moves_positive:
                    self.origin = 1
                    self.destination = 0
                else:
                    self.origin = 0
                    self.destination = 1
            elif near_corner_y_vals[0] < threshold and near_corner_y_vals[2] < threshold:
                if moves_positive:
                    self.origin = 0
                    self.destination = 2
                else:
                    self.origin = 2
                    self.destination = 0
            elif near_corner_y_vals[1] < threshold and near_corner_y_vals[3] < threshold:
                if moves_positive:
                    self.origin = 1
                    self.destination = 3
                else:
                    self.origin = 3
                    self.destination = 1
            elif near_corner_y_vals[2] < threshold and near_corner_y_vals[3] < threshold:
                if moves_positive:
                    self.origin = 3
                    self.destination = 2
                else:
                    self.origin = 2
                    self.destination = 3

        if self.origin == None:
            self.find_origin()
            self.find_destination()
        # if self.destination == None:
        #     find_destination()



    def stationary_at_corner(self):
        xcoords = self.info['BBox_Position_X']
        ycoords = self.info['BBox_Position_Y']

        for ptx, pty in zip(xcoords,ycoords):
            if min([distance((ptx,pty),self.cs.corners[i]) for i in range(4)]) > 4:
                return False
            # if (ptx-self.cs.corners[self.origin][0])**2 + (pty-self.cs.corners[self.origin][1])**2 > 16:
            #     return False
        return True

    def find_destination(self):
        xcoords = self.info['BBox_Position_X']
        ycoords = self.info['BBox_Position_Y']
        counter = 0
        for ptx, pty in zip(reversed(list(xcoords)),reversed(list(ycoords))):
            if (pty > self.cs.phase4_lower_line_with_tolerance[0]*ptx + self.cs.phase4_lower_line_with_tolerance[1] 
                and pty < self.cs.phase4_upper_line_with_tolerance[0]*ptx + self.cs.phase4_upper_line_with_tolerance[1] 
                and pty < self.cs.phase2_left_line_with_tolerance[0]*ptx + self.cs.phase2_left_line_with_tolerance[1] 
                and pty > self.cs.phase2_right_line_with_tolerance[0]*ptx + self.cs.phase2_right_line_with_tolerance[1]):
                #print('within bounds')
                dists = [distance((ptx,pty),self.cs.corners[k]) for k in range(4)]
                self.destination = dists.index(min(dists))
                break
            else:
            #if counter > self.origin_ts:
                for i, corner in enumerate(self.cs.corners):
                    #if i != self.origin:
                    if (ptx-corner[0])**2 + (pty-corner[1])**2 <= 56.26:
                        self.destination = i
                        break
            if self.destination != None:
                break 
            #counter += 1

    def run_through_path_with_light_state(self, include_noncrossing_boundary=False):
        # gather the list of points and timesteps
        timesteps = self.info['Timestamp']
        #print(xcoords)
        #print(timesteps)
        # for each timestep, find the latest status of sign 2 or sign 4


        if include_noncrossing_boundary:
            plot_bounds(self.cs.corners, self.cs.phase4_lower_line_with_tolerance, self.cs.phase4_upper_line_with_tolerance, self.cs.phase2_left_line_with_tolerance, self.cs.phase2_right_line_with_tolerance)
            plot_bounds(self.cs.corners, self.cs.lower_line_crossing, self.cs.upper_line_crossing, self.cs.left_line_crossing, self.cs.right_line_crossing)
        plt.xlim([-40, 46])
        plt.ylim([-40, 46])
        plt.gca().set_aspect('equal', adjustable='box')
        for corner in self.cs.corners:
            circle1 = plt.Circle((corner[0], corner[1]), 7.5, color='m', fill=False)
            plt.gca().add_patch(circle1)
        corner_xs = [corner[0] for corner in self.cs.corners]
        corner_ys = [corner[1] for corner in self.cs.corners]
        plt.scatter(corner_xs, corner_ys, color='b')

        ##UNCOMMENT SOON
        # if self.origin != None:
        #     plt.scatter(self.cs.corners[self.origin][0], self.cs.corners[self.origin][1], color='yellow')
        # if self.destination != None:
        #     plt.scatter(self.cs.corners[self.destination][0], self.cs.corners[self.destination][1], color='lime')
        # plt.title(self.category + " - "+str(self.label))

        for i, timestep in enumerate(timesteps):
            dftemp = self.cs.df2[self.cs.df2['Timestamp'] <= timestep/1000]
            message2 = dftemp['Message'].iloc[-1]
            dftemp = self.cs.df4[self.cs.df4['Timestamp'] <= timestep/1000]
            message4 = dftemp['Message'].iloc[-1]
            # dftemp = self.cs.df2p[self.cs.df2p['Timestamp'] <= timestep/1000]
            # message2p = dftemp['Message'].iloc[-1]
            # dftemp = self.cs.df4p[self.cs.df4p['Timestamp'] <= timestep/1000]
            # message4p = dftemp['Message'].iloc[-1]
            # plot corners, then plot spot, then title with message
            
            # #UNCOMMENT
            # plt.scatter(self.xcoords.iloc[i], self.ycoords.iloc[i], color='r')
            # plt.scatter(self.xcoords.iloc[0], self.ycoords.iloc[0], color='cyan')
            # plt.scatter(self.xcoords.iloc[-1], self.ycoords.iloc[-1], color='cyan')

            # print("MESSAGING")
            # print(timestep)
            # print(dftemp['Timestamp'].iloc[-1])
            # print(message2)
            # print(message4)

            if message4 == 'Ped Begin Walk  (Ped 4)' or message4 == 'Ped Begin Clearance  (Ped 4)':#'Phase On  (Phase 4)':
                plt.plot(corner_xs[0:2],corner_ys[0:2],color='limegreen')
                plt.plot(corner_xs[2:4],corner_ys[2:4],color='limegreen')
            else:
                plt.plot(corner_xs[0:2],corner_ys[0:2],color='tomato')
                plt.plot(corner_xs[2:4],corner_ys[2:4],color='tomato')

                #plt.plot(self.cs.corners[2],self.cs.corners[3],color='g')
            if message2 == 'Ped Begin Walk  (Ped 2)' or message2 == 'Ped Begin Clearance  (Ped 2)': #'Phase On  (Phase 2)':
                plt.plot((corner_xs[1],corner_xs[3]),(corner_ys[1],corner_ys[3]),color='limegreen')
                plt.plot((corner_xs[2],corner_xs[0]),(corner_ys[2],corner_ys[0]),color='limegreen')
            else:
                plt.plot((corner_xs[1],corner_xs[3]),(corner_ys[1],corner_ys[3]),color='tomato')
                plt.plot((corner_xs[2],corner_xs[0]),(corner_ys[2],corner_ys[0]),color='tomato')    

            #plt.title(message2p[:-8]+  " - " + message4p[:-8])
            #plt.show()
            if i < len(timesteps)-1:
                plt.show(block=False)
                plt.pause(.025)
            if i == len(timesteps) - 1:
                print("DONE")
                if self.category == 'Crossing':
                    print("Did they cross in time?")
                    print("   "+str(not self.crossing_flag()))
                #print(self.timesteps)
                plt.gcf().savefig("crossings.png")
                plt.show()
            # HOW TO CLEAR PLOT?
            #plt.gca().close()

    def predicted_speed_to_destination(self,window_size, velocities_used):
        
        if (len(self.info)-window_size+1) < 0:
            print("Error ALERT BIG ISSUE")
            print(len(self.xcoords))
        # print("start end")
        # print(self.origin)
        # print(self.destination)
        velocities_towards_destination = []
        for i in range(len(self.info)-window_size):
            # take average slope over window_size points

            av_slope = (self.ycoords.iloc[i+window_size]-self.ycoords.iloc[i])/(self.xcoords.iloc[i+window_size]-self.xcoords.iloc[i])

            # make a vector which points from the endpoint to the destination
            enpoint_to_destination_slope = (self.cs.corners[self.destination][1]-self.ycoords.iloc[i])/(self.cs.corners[self.destination][0]-self.xcoords.iloc[i])

            # if the slopes are within pi/6, consider it in the right direction
            if find_angle(av_slope, enpoint_to_destination_slope) < np.pi/6:
                segment_distance = distance((self.xcoords.iloc[i],self.ycoords.iloc[i]), (self.xcoords.iloc[i+window_size],self.ycoords.iloc[i+window_size]))
                segment_time = (self.info['Timestamp'].iloc[i+window_size] - self.info['Timestamp'].iloc[i])/1000 #seconds 
                # calculate the velocity in that direction
                velocities_towards_destination += [segment_distance / segment_time] # meters per second
        
        # take the average of the top velocities_used of these velocities
        # using this velocity and the intersection distance, predict the time to cross the street
        av_velocity_used = np.mean(heapq.nlargest(velocities_used, velocities_towards_destination))
        # full_distance = distance(self.cs.corners[self.origin], self.cs.corners[self.destination])
        # print(full_distance)
        # print(av_velocity_used)
        # time_needed = full_distance/av_velocity_used

        return av_velocity_used

    def time_remaining_on_phase(self, current_time, current_phase):
        if current_phase == -1:
            return 0
        if current_phase == 2:
            remaining_events = self.cs.df2[self.cs.df2['Timestamp'] > current_time/1000]
            next_dontwalks = remaining_events[remaining_events['Message'] == 'Ped Begin Don’t Walk  (Ped 2)']
            next_dontwalk = next_dontwalks.iloc[0]
            return (next_dontwalk['Timestamp']-current_time/1000)
        elif current_phase == 4:
            remaining_events = self.cs.df4[self.cs.df4['Timestamp'] > current_time/1000]
            next_dontwalks = remaining_events[remaining_events['Message'] == 'Ped Begin Don’t Walk  (Ped 4)']
            next_dontwalk = next_dontwalks.iloc[0]
            return (next_dontwalk['Timestamp']-current_time/1000)
        return "TO BE COMPLETED"

    def starts_behind_origin_line(self):
        # need to find out if the user has made it to the destination already.
        current_history_x = self.xcoords
        current_history_y = self.ycoords
        starts_behind_origin_line_val = None
        if (self.origin == 0 and self.destination == 1) or (self.origin == 2 and self.destination == 3):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.right_line_crossing, current_history_x, current_history_y,"<")
        elif (self.origin == 0 and self.destination == 2) or (self.origin == 1 and self.destination == 3):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.lower_line_crossing, self.xcoords, self.ycoords,"<")
        elif (self.origin == 1 and self.destination == 0) or (self.origin == 3 and self.destination == 2):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.left_line_crossing, self.xcoords, self.ycoords,">")
        elif (self.origin == 2 and self.destination == 0) or (self.origin == 3 and self.destination == 1):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.upper_line_crossing, self.xcoords, self.ycoords,">")

        return starts_behind_origin_line_val

    def crosses_origin_line(self, x = None, y = None):
        if x == None and y == None:
            current_history_x = self.xcoords
            current_history_y = self.ycoords
        else:
            current_history_x = [x]
            current_history_y = [y]
        starts_behind_origin_line_val = None
        if (self.origin == 0 and self.destination == 1) or (self.origin == 2 and self.destination == 3):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.right_line_crossing, current_history_x, current_history_y,">")
        elif (self.origin == 0 and self.destination == 2) or (self.origin == 1 and self.destination == 3):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.lower_line_crossing, self.xcoords, self.ycoords,">")
        elif (self.origin == 1 and self.destination == 0) or (self.origin == 3 and self.destination == 2):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.left_line_crossing, self.xcoords, self.ycoords,"<")
        elif (self.origin == 2 and self.destination == 0) or (self.origin == 3 and self.destination == 1):
            starts_behind_origin_line_val = point_in_list_past_line(self.cs.upper_line_crossing, self.xcoords, self.ycoords,"<")  
        return starts_behind_origin_line_val      

    def distance_remaining_to_destination(self, time_index):
        # need to find out if the user has made it to the destination already.
        current_history_x = self.xcoords.iloc[:time_index]
        current_history_y = self.ycoords.iloc[:time_index]
        # print("Length")
        # print(len(current_history_x))
        sufficiently_close_to_destination = None
        if (self.origin == 0 and self.destination == 1) or (self.origin == 2 and self.destination == 3):
            sufficiently_close_to_destination = point_in_list_past_line(self.cs.left_line_crossing, current_history_x, current_history_y,">")
        elif (self.origin == 0 and self.destination == 2) or (self.origin == 1 and self.destination == 3):
            sufficiently_close_to_destination = point_in_list_past_line(self.cs.upper_line_crossing, self.xcoords, self.ycoords,">")
        elif (self.origin == 1 and self.destination == 0) or (self.origin == 3 and self.destination == 2):
            sufficiently_close_to_destination = point_in_list_past_line(self.cs.right_line_crossing, self.xcoords, self.ycoords,"<")
        elif (self.origin == 2 and self.destination == 0) or (self.origin == 3 and self.destination == 1):
            sufficiently_close_to_destination = point_in_list_past_line(self.cs.lower_line_crossing, self.xcoords, self.ycoords,"<")

        if sufficiently_close_to_destination:
            return 0
        else:
            return distance((current_history_x.iloc[-1], current_history_y.iloc[-1]),(self.cs.corners[self.destination][0], self.cs.corners[self.destination][1]))

    def phase_on(self, phase_number, timestep):

        dftemp = self.cs.df2[self.cs.df2['Timestamp'] <= timestep/1000]
        message2 = dftemp['Message'].iloc[-1]
        dftemp = self.cs.df4[self.cs.df4['Timestamp'] <= timestep/1000]
        message4 = dftemp['Message'].iloc[-1]

        if phase_number == 2:
            if message2 == 'Ped Begin Walk  (Ped 2)' or message2 == 'Ped Begin Clearance  (Ped 2)':
                return True
            return False
        else:
            if message4 == 'Ped Begin Walk  (Ped 4)' or message4 == 'Ped Begin Clearance  (Ped 4)':
                return True
            return False

    def cross_light_never_on(self):
        if (self.origin in [0,1] and self.destination in [0,1]) or (self.origin in [2,3] and self.destination in [2,3]):
            phase = 4
        else:
            phase = 2
        for timestep in self.timesteps:
            if self.phase_on(phase, timestep):
                return False
        return True

    def on_path_but_phase_off(self):
        crossing_time = None
        k = 0
        for x, y in zip(self.xcoords, self.ycoords):
            if self.crosses_origin_line(x,y):
                crossing_time = k
                break
            k += 1
        if crossing_time == None:
            return False
        finish_time = None
        k = crossing_time+1 
        while k < len(self.xcoords):
            if self.distance_remaining_to_destination(k) == 0:
                finish_time = k
                break
            k+=1
        if finish_time == None:
            finish_time = len(self.xcoords)
        for i in range(crossing_time,finish_time):
            if (self.origin in [0,1] and self.destination in [0,1]) or (self.origin in [2,3] and self.destination in [2,3]):
                if not self.phase_on(4, self.timesteps.iloc[i]):
                    return True
            else:
                if not self.phase_on(2,self.timesteps.iloc[i]):
                    return True
        return False

    def crossing_flag(self):

        current_phase = -1
        if (self.origin in [0,1] and self.destination in [0,1]) or (self.origin in [2,3] and self.destination in [2,3]):
            phase_is_on = self.phase_on(4, int(self.timesteps.iloc[-1]))
            if phase_is_on:
                current_phase = 4
        else:
            phase_is_on = self.phase_on(2, int(self.timesteps.iloc[-1])) 
            if phase_is_on:
                current_phase = 2 

        if self.on_path_but_phase_off() and not phase_is_on:
            #print("Case 0")
            #print("Time left: "+ str(self.time_remaining_on_phase(int(self.timesteps.iloc[-1]), current_phase)))

            return True

        elif not phase_is_on and self.distance_remaining_to_destination(int(self.timesteps.iloc[-1])) > 0:
            #print("Case 1")
            #print("Time left: "+ str(self.time_remaining_on_phase(int(self.timesteps.iloc[-1]), current_phase)))

            return True

        elif phase_is_on and self.distance_remaining_to_destination(int(self.timesteps.iloc[-1])) >= self.time_remaining_on_phase(int(self.timesteps.iloc[-1]), current_phase)*self.predicted_speed_to_destination(4,4):
            #print("Case 2")
            #print("Time left: "+ str(self.time_remaining_on_phase(int(self.timesteps.iloc[-1]), current_phase)))
            return True
        else:
            return False
        





# 4 is the bottom/top. Figure out when inactive vs active for timing. 

## IDEA: gather the pedestrians in an N second time context, and display all together. Do any seem to be continuations?
## IDEA: how confident are we about the street center locations? 
## Hough transform to find 4 lines, then get the intersections of those lines!!! those are the spots. Try PCA, RANSAC

## K means, use means as end points, estimate lines. 
## collect points closest to that line, use it to divide the groups to then make a linear regression estimate.

## From these estimates, adjust to determine the 'end points' / true corners

# speculation: 
## Then, use the mean of all points closest to these end points as the new end point
## Then, repeat the linear regression estimate again

## What is the nature of the outliers? Outside the ring by a certain amount. 
