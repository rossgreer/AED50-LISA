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
from utils import plot_bounds, find_angle, distance
from pedestrian import Pedestrian
from crosswalk import CrosswalkScene
from sklearn.decomposition import PCA



cs1 = CrosswalkScene('Competition_Data/Lidar/pedestrians_only_one.csv')  
cs2 = CrosswalkScene('Competition_Data/Lidar/pedestrians_only_two.csv')  
cs3 = CrosswalkScene('Competition_Data/Lidar/pedestrians_only_three.csv')  
#cs1 = CrosswalkScene('Competition_Data/Lidar/pedestrians_only.csv')  



def generate_heatmap():

    heatmap = np.zeros((86,86))
    mm = cs1.add_to_heatmap(heatmap)
    #mm = cs2.add_to_heatmap(mm)
    #mm = cs3.add_to_heatmap(mm)

    plt.imshow(mm, cmap='gray')
    plt.savefig('heatmap.png')
    plt.show()


#generate_heatmap()

xs = []
ys = []
zs_flagged = []
zs_noflagged = []
# for each pedestrian
df1 = pd.read_csv('Competition_Data/Lidar/pedestrians_only_one.csv')
df2 = pd.read_csv('Competition_Data/Lidar/pedestrians_only_two.csv')
df3 = pd.read_csv('Competition_Data/Lidar/pedestrians_only_three.csv')



def class_frequencies():
    classes = {}
    flagged = [0,0]
    flagged_velocities = []
    flagged_x = []
    flagged_y = []
    flagged_heights = []
    vectors = []
    for cs,df in zip([cs1, cs2, cs3],[df1, df2, df3]):
        pedestrian_list = df['ID'].unique()
        for i,ped in enumerate(pedestrian_list):
            current_pedestrian = Pedestrian(ped, df[df['ID'] == ped], cs)
            # if i == 0:
            #     current_pedestrian.run_through_path_with_light_state(True)
            # if i == -1:
            #     current_pedestrian.run_through_path_with_light_state(True)
            # predict the category
            if current_pedestrian.category in classes.keys():
                classes[current_pedestrian.category] += 1
            else:
                classes[current_pedestrian.category] = 1 
            if current_pedestrian.category == "Crossing":
                if current_pedestrian.crossing_flag():
                    flagged[0] += 1
                    vel = current_pedestrian.predicted_speed_to_destination(4,4)
                    height = df[df['ID']==ped]['BBox_Size_Z'].mean()
                    x = max(df[df['ID']==ped]['BBox_Size_X'].mean(), df[df['ID']==ped]['BBox_Size_Y'].mean())
                    y = min(df[df['ID']==ped]['BBox_Size_X'].mean(), df[df['ID']==ped]['BBox_Size_Y'].mean())
                    flagged_velocities += [vel]
                    flagged_heights += [height]
                    flagged_x += [x]
                    flagged_y += [y]
                    vectors += [[vel, height, x, y]]



                else:
                    flagged[1] += 1
            #     height = df[df['ID']==ped]['BBox_Size_X'].mean() * df[df['ID']==ped]['BBox_Size_Y'].mean()
            #     if current_pedestrian.crossing_flag():
            #         zs_flagged += [height]
            #     else:
            #         zs_noflagged += [height]
    print(classes)
    print(flagged)
    means = subclassify(vectors)
    plt.hist(flagged_velocities)
    plt.xlabel("Pedestrian Max Speed Toward Destination (m/s)")
    plt.ylabel("Count")
    plt.gcf().savefig('vel_hist.png')
    plt.show()
    plt.hist(flagged_x)
    plt.xlabel("Pedestrian Size X (m)")
    plt.ylabel("Count")
    plt.gcf().savefig('x_hist.png')
    plt.show()
    plt.hist(flagged_y)
    plt.xlabel("Pedestrian Size Y (m)")
    plt.ylabel("Count")
    plt.gcf().savefig('y_hist.png')
    plt.show()
    plt.hist(flagged_heights)
    plt.xlabel("Pedestrian Height (m)")
    plt.ylabel("Count")
    plt.gcf().savefig('height_hist.png')
    plt.show()
    return means

def build_results(means):
    with open('results.csv', 'w',newline="") as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(["Pedestrian Track ID", "Trajectory Category","Estimated Crossing Time","Extended Crossing Time Needed"])
        for cs,df in zip([cs1, cs2, cs3],[df1, df2, df3]):
            pedestrian_list = df['ID'].unique()
            for i,ped in enumerate(pedestrian_list):
                current_pedestrian = Pedestrian(ped, df[df['ID'] == ped], cs)
                # if i == 0:
                #     current_pedestrian.run_through_path_with_light_state(True)
                # if i == -1:
                #     current_pedestrian.run_through_path_with_light_state(True)
                # predict the category
                if current_pedestrian.category == "Crossing":
                    time_to_cross = current_pedestrian.predicted_speed_to_destination(4,4)*distance(current_pedestrian.cs.corners[current_pedestrian.origin], current_pedestrian.cs.corners[current_pedestrian.destination])
                    flag = current_pedestrian.crossing_flag()
                else:
                    time_to_cross = 0
                    flag = 0
                if flag != 0:
                    vel = current_pedestrian.predicted_speed_to_destination(4,4)
                    height = df[df['ID']==ped]['BBox_Size_Z'].mean()
                    x = max(df[df['ID']==ped]['BBox_Size_X'].mean(), df[df['ID']==ped]['BBox_Size_Y'].mean())
                    y = min(df[df['ID']==ped]['BBox_Size_X'].mean(), df[df['ID']==ped]['BBox_Size_Y'].mean())
                    subclass = means.transform([[vel,height,x,y]])[0]
                else:
                    subclass = 0
                print(subclass)
                row = [ped, current_pedestrian.category,time_to_cross,flag,int(subclass)+1]
                # write a row to the csv file
                writer.writerow(row)
                # if current_pedestrian.category == "Crossing":
                #     height = df[df['ID']==ped]['BBox_Size_X'].mean() * df[df['ID']==ped]['BBox_Size_Y'].mean()
                #     if current_pedestrian.crossing_flag():
                #         zs_flagged += [height]
                #     else:
                #         zs_noflagged += [height]
        #print(classes)

def subclassify(vectors):
    # PCA then display
    X = np.array(vectors)
    colors = ['red','green','blue','cyan','orange']
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.components_)
    y = pca.transform(X)
    for i,p in enumerate(y):
        plt.scatter(p[0],p[1],color = colors[int(kmeans.labels_[i])])
    plt.title("PCA Projection of K-Means Classes for Flagged Pedestrians")
    plt.gcf().savefig("pca.png")
    plt.show()
    return kmeans

means = class_frequencies()
build_results(means)

# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.hist(zs_flagged,bins=20)
# ax1.set_title('Sharing Y axis')
# ax2.hist(zs_noflagged,bins=20)
# plt.show()
# print(len(zs_noflagged))
# print(len(zs_flagged))



# if the category is crossing
# predict the flag
# if the flag is on, collect the size_xyz situation
# show these statistics

# store the pedestrian ID, category, flag










# steps for tuesday: figure out the future_destination of each trajectory (extend vector of velocity, find nearest node when stepping forward)
# estimate time to reach 
# figure out phases of signals
# MERGE PEDESTRIANS WHO HAVE THE SAME MATCHING END TO START POINTS

# Goal: pedestrian ID, then column whether crossing, noncrossing, bicycle, etc.
# If crossing, predict crossing time (regardless of whether crossing has completed)
# If crossing will not be completed, raise a flag.

## NOTE: some of the pedestrians that start on a red path midway through might be STUCK or might be JAYWALKING. should cluster by velocity to determine outliers. 


# REMOVE "JAYWALKERS" NOW, right now the fakeys are combined

