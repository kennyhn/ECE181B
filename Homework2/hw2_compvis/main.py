#!/usr/bin/env python3

from homography import *

def main():
    # want to transform left picture to right
    #list_of_points = retrieveCorrespondingPoints('left.jpg', 'right.jpg')
    #homographymatrix = calculateHomographyTransformation(list_of_points)
    #transformImage('left.jpg','right.jpg', homographymatrix)
    
    list_of_points = retrieveCorrespondingPoints('bedleft.jpg', 'bedright.jpg')
    homographymatrix = calculateHomographyTransformation(list_of_points)
    transformImage('bedleft.jpg','bedright.jpg', homographymatrix)
    return 0

if __name__ == "__main__":
    main()

    