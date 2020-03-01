#!/usr/bin/env python3

import helping_functions        as hf
import numpy                    as np
import  matplotlib.pyplot       as plt

def retrieveCorrespondingPoints(picturename1, picturename2):
    # INPUT: The filename of two pictures from different angles
    # OUTPUT: List with the pair of points in a tuple. 
    # Output format: [((x1,y1), (x2,y2)), ((x11, y11), (x21, y21)), ...]
    print("You must choose 16 corresponding points"
          " in the same order for the both pictures")
    npoints = 20
    im1 = plt.imread(picturename1)

    plt.imshow(im1)
    picture1 = np.array(plt.ginput(npoints, timeout = 0))


    im2 = plt.imread(picturename2)
    plt.imshow(im2)
    picture2 = np.array(plt.ginput(npoints, timeout = 0))

    list_of_points = []
    for i in range(npoints):
        pointp1 = (picture1[i][0], picture1[i][1],)
        pointp2 = (picture2[i][0], picture2[i][1],)
        pair = (pointp1, pointp2,)
        list_of_points.append(pair)

    plt.close()
    ans = input('Would you like to save the points to a file? (y/n): ')
    if ans.lower() == 'y':
        filename = input("Insert filename (default: list): ")
        if filename != '':
            hf.save_list(list_of_points, "{}.txt".format(filename))
        else:
            hf.save_list(list_of_points)

    return list_of_points

