#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from homography import retrieveCorrespondingPoints
from helping_functions import load_list
import cv2
"""
def findFMatrix(leftimage, rightimage):
    # You need atleast 7 points to find F_matrix. In this case we find 8 points for better calibration
    #list_of_corresponding_points = retrieveCorrespondingPoints(leftimage, rightimage)
    list_of_corresponding_points = load_list('test_checker.txt')
    is_array_created = False
    for points_pair in list_of_corresponding_points:
        x, y                = points_pair[0]
        x_prime, y_prime    = points_pair[1]
        if not is_array_created:
            A = np.array([[x_prime*x, x_prime*y, x_prime, y_prime*x, y_prime*y, y_prime, x, y, 1]])
            is_array_created = True
        else:
            A = np.append(A, np.array([[x_prime*x, x_prime*y, x_prime, y_prime*x, y_prime*y, y_prime, x, y, 1]]), axis = 0)
   
    # Calculate F using SVD
    u, d, v         = np.linalg.svd(A, full_matrices = True)

    f_vector        = v[:, -1] # F is last column in v_transposed 
    
    F               = np.reshape(f_vector, newshape = (3,3,))
    u, d, v_t       = np.linalg.svd(F, full_matrices = True)
    d_f_prime       = np.append(d[:-1], [0])
    F_prime         = np.matmul(np.matmul(u, np.diag(d_f_prime)), v_t)
    return F_prime
"""

def findFMatrix(leftimage, rightimage):
    # You need atleast 7 points to find F_matrix. In this case we find 8 points for better calibration
    #list_of_corresponding_points = retrieveCorrespondingPoints(leftimage, rightimage)
    list_of_corresponding_points = load_list('list.txt')
    pts1 = []
    pts2 = []
    for points_pair in list_of_corresponding_points:
        # Save function inverted the order of the points
        pts2.append(points_pair[0]) # left image
        pts1.append(points_pair[1]) # right image
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    print(fundamental_matrix)
    return fundamental_matrix

