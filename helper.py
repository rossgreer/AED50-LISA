### Make pedestrian-only dataset

import pandas as pd

df = pd.read_csv('Competition_Data/Lidar/collection_full.csv')

pedestrian_set = df[df['Label'] == 2]

print(len(pedestrian_set))

