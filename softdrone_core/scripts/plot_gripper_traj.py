#!/usr/bin/env python
"""Node for plotting gripper trajectory"""
import rospy
import rospkg
import numpy as np
import csv
import matplotlib.pyplot as plt

def main():
    #get path
    rospack = rospkg.RosPack()
    load_path = rospack.get_path('softdrone_core')
    load_path = load_path + "/data/data.csv"
    print(load_path)

    #load data
    file = open(load_path)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    data = np.asarray(rows)


    # for i in range(0, len(data[:,0])):
    #     for j in range(0, len(data[0,:])):
    #         data[i,j] = int(data[i,j])

    data = data.astype(float)
    data = data.astype(int)
    # print("type(data)")
    print(type(data))

    #plot
    fig = plt.figure()

    plt.title("Gripper Trajectory")

    # x = np.arange(0,len(data[:,0]))
    t = data[:,8]/1000.0
    plt.plot(t, data[:,0], color="black",label="arduino setpoint")
    plt.plot(t, data[:,4], color="red",label="position")
    plt.plot(t, data[:,9], color="green",label="python setpoint")

    plt.legend()

    # plt.plot(data[:,1], color="red")
    # plt.plot(data[:,2], color="blue")
    # plt.plot(data[:,3], color="green")

    # plt.plot(data[:,5], '--', color="red")
    # plt.plot(data[:,6], '--', color="blue")
    # plt.plot(data[:,7], '--', color="green")
   
    plt.show()


if __name__ == "__main__":
    main()
