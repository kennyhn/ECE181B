#!/usr/bin/env python3

from homography import *
import helping_functions as hf
def main():
    # want to transform left picture to right
    #list_of_points = retrieveCorrespondingPoints('left.jpg', 'right.jpg')

    #list_of_points = hf.load_list('leftright1.txt')
    #homographymatrix = calculateHomographyTransformation(list_of_points)
    #transformImage('left.jpg','right.jpg', homographymatrix)
    
    #list_of_points = retrieveCorrespondingPoints('bedleft.jpg', 'bedright.jpg')
    #homographymatrix = calculateHomographyTransformation(list_of_points)
    #transformImage('bedleft.jpg','bedright.jpg', homographymatrix)
    list_of_points = hf.load_list('bedroom1.txt')
    calculateHomographyTransformation(list_of_points)
    
    return 0

if __name__ == "__main__":
    main()

    