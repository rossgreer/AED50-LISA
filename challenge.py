import csv
import pandas as pd
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import cv2
import numpy as np
import matplotlib.transforms as mtransforms
import time

class CrosswalkScene():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.times = self.times_from_timestamps()
        self.road_user_to_color = ["black","red","purple","green","blue"]
        self.corners = [(37,26),(22,37),(33,60),(46,45)] #pixel coordinates, row then column

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
            #print(row)
            bbx = row['BBox_Position_X']
            bby = row['BBox_Position_Y']

            road_user_color = self.road_user_to_color[int(row['Label'])]

            # Create a Rectangle patch. Cars red, bikes green, pedestrians yellow. Blue is unknown, black is none.  
            if road_user_color == "purple": #road_user_color != "black" and road_user_color != "blue":
                sum_map[86-int(bby)-40][int(bbx+40)] += 1

        return sum_map

    def display_scene_at_time(self, time):
        # for the time, gather all rows
        timeset = self.df[self.df['Timestamp'] == self.times[time]]
        #print(timeset)

        # for each object, draw a bounding box of that color
        # fig, ax = plt.subplots()
        plt.xlim([-40, 46])
        plt.ylim([-40, 46])
        plt.gca().set_aspect('equal', adjustable='box')
        for index,row in timeset.iterrows():
            #print(row)
            bbx = row['BBox_Position_X']
            bby = row['BBox_Position_Y']
            bbsx = row['BBox_Size_X']
            bbsy = row['BBox_Size_Y']
            yaw = row['BBox_Yaw']
            # print("Stats")
            # print(bbx)
            # print(bby)
            # print(bbsx)
            # print(bbsy)
            # print(self.get_corners_from_rectangle([bbx,bby],[bbsx,bbsy],yaw))
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

class Pedestrian():
    def __init__(self, filepath):
        self.id = 
        self.origin = -1
        self.destination = -1

    def find_origin(self):
        # scan through the file for any rows which contain 'id'
        # if it contains 'id'


    def find_destination(self):




cs1 = CrosswalkScene('Competition_Data/Lidar/collection_full.csv')
#cs2 = CrosswalkScene('Competition_Data/Lidar/collection_two.csv')
#cs3 = CrosswalkScene('Competition_Data/Lidar/collection_three.csv')
#cs = CrosswalkScene('Demo_Data/SAMPLE_LIDAR_OUTPUT.csv')

heatmap = np.zeros((86,86))
mm = cs1.add_to_heatmap(heatmap)
#mm = cs2.add_to_heatmap(mm)
#mm = cs3.add_to_heatmap(mm)

plt.imshow(mm, cmap='gray')
plt.show()


# for i in range(int(len(cs.times)/2)):
#   cs.display_scene_at_time(i)
#   #plt.pause(.1)
#   if i%100 == 0:
#       print(i)
#   if i == int(len(cs.times)/2)-1:
#       plt.savefig('sample2.png')
#   #print("here")
#   #plt.gca().close()




