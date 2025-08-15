#read the u velocity
import pandas as pd
import numpy as np

prefix = "/workspace/ljl/Junle_PIV_data/SR3.25_AoA0/B"
suffix = ".csv"


def read_csv_and_transform(filename, colmn_num):
    df = pd.read_csv(filename, sep = ";",header=None, skiprows=[0])
    velocity = df.iloc[:, colmn_num].values
    velocity = velocity.reshape(383, 367)
    return velocity

# read all csv files and stack data
velocities_u = []
velocities_v = []
# vorcities = []

for i in range(1, 1201):
    filename = prefix + "{:04d}".format(i) + suffix
    
    velocity_u = read_csv_and_transform(filename, 2)
    velocity_v = read_csv_and_transform(filename, 3)
    #vortcity = read_csv_and_transform(filename, 9)
    velocities_u.append(velocity_u)
    velocities_v.append(velocity_v)
    #vorcities.append(vortcity)
# stack all data
velocities_u = np.stack(velocities_u, axis=0)
velocities_v = np.stack(velocities_v, axis=0)

np.save("velocities_v.npy", velocities_v)
np.save("velocities_u.npy", velocities_u)