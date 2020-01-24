#!/usr/bin/env python3

import helping_functions        as hf
import numpy                    as np
import  matplotlib.pyplot       as plt
from skimage import transform   as tf

def retrieveCorrespondingPoints(picturename1, picturename2):
    # INPUT: The filename of two pictures from different angles
    # OUTPUT: List with the pair of points in a tuple. 
    # Output format: [((x1,y1), (x2,y2)), ((x11, y11), (x21, y21)), ...]
    print("You must choose 8 corresponding points in the same order for the both pictures")
    npoints = 8
    im1 = plt.imread(picturename1)

    plt.imshow(im1)
    picture1 = np.array(plt.ginput(npoints))


    im2 = plt.imread(picturename2)
    plt.imshow(im2)
    picture2 = np.array(plt.ginput(npoints))

    list_of_points = []
    for i in range(npoints):
        pointp1 = (picture1[i][0], picture1[i][1],)
        pointp2 = (picture2[i][0], picture2[i][1],)
        pair = (pointp2, pointp1,)
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

def calculateHomographyTransformation(list_of_points): 
    # INPUT: All the corresponding pairs of point in the two pictures as a list with same index position
    # OUTPUT: Transformation matrix H
    # Here we want to find H that transforms the one picture to the other
    # transform from picture number 1 to picture number 2
    array_created = False
    for pair in list_of_points:
        x, y = pair[0]              # Picture 1
        x_prime, y_prime = pair[1]  # Picture 2

        if not array_created:
            A = np.array([[x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime],
                            [0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime]])
            B = np.array([x_prime, y_prime])
            array_created = True
        
        else:
            A = np.append(A, np.array([[x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime],
                                    [0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime]]), axis = 0)
            B = np.append(B, np.array([x_prime, y_prime]))
    
    # function return 8x1 with [h11 h12 h13 h21 h22 h23 h31 h32] here h33 = 1
    h = np.linalg.lstsq(A, B)[0] 
    h = np.append(h, 1)

    H = np.reshape(h, (3, 3))
    return H

def transformImage(imagename, desiredImg, matrix = np.empty((3,3), dtype = float)):
    # INPUT: The image that is to be transformed, the picture from the angle I should expect and the transformation matrix
    transformation = tf.ProjectiveTransform(matrix)
    print(transformation)
    img = plt.imread(imagename)
    img2 = plt.imread(desiredImg)


    transformedImage = tf.warp(img, transformation)
    
    fig = plt.figure(0)
    sub1 = fig.add_subplot(1,1,1)
    sub1.set_title('Original Picture')
    plt.imshow(img)
    fig2 = plt.figure(1)
    sub2 = fig2.add_subplot(1,1,1)
    sub2.set_title('Transformed picture')
    plt.imshow(transformedImage)
    fig3 = plt.figure(2)
    sub3 = fig3.add_subplot(1,1,1)
    sub3.set_title('Desired picture')
    plt.imshow(img2)
    plt.show()
    return 0
