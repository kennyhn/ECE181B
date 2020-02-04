#!/usr/bin/env python3
import matplotlib.pyplot as plt

def save_list(listname, filename = "list.txt"):
    with open(filename, 'w') as f:
        for item in listname:
            f.write("{}:{}\n".format(item[0], item[1]))

def load_list(filename = "list.txt"):
    list_of_points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip() # Remove whitespace \n etc
            p1, p2 = line.split(':')
            pointx1, pointy1 = (p1.strip('()')).split(',') # remove paranthesis
            pointx1 = float(pointx1)
            pointy1 = float(pointy1)

            pointx2, pointy2 = (p2.strip('()')).split(',') # Remove paranthesis
            pointx2 = float(pointx2)
            pointy2 = float(pointy2)

            list_of_points.append(((pointx1, pointy1,), (pointx2, pointy2,),))
    return list_of_points

def markPictures(image1, image2, list_of_points):
    list_of_points_image1 = []
    list_of_points_image2 = []

    for i in range(len(list_of_points)):
        list_of_points_image1.append(list_of_points[i][1])
        list_of_points_image2.append(list_of_points[i][0])
        
    list_of_x1 = []
    list_of_x2 = []
    list_of_y1 = []
    list_of_y2 = []

    for i in range(len(list_of_points_image1)):
        list_of_x1.append(list_of_points_image2[i][0])
        list_of_y1.append(list_of_points_image2[i][1])

    
    fig1, ax = plt.subplots()
    im1 = plt.imread(image2)
    plt.imshow(im1)
    for i in range(len(list_of_x1)): 
        ax.plot([list_of_x1[i]], [list_of_y1[i]], 'gx')
        if i == 0:
            plt.text(list_of_x1[i], list_of_y1[i], 
                "({ptx1:.2f},{pty1:.2f})".format(ptx1 = list_of_x1[i],
                                                pty1 = list_of_y1[i]),
                ha = 'center', va = 'bottom',
                color = 'red',
                transform = ax.transData)
        elif i==1:
            plt.text(list_of_x1[i], list_of_y1[i], 
                "({ptx1:.2f},{pty1:.2f})".format(ptx1 = list_of_x1[i],
                                                pty1 = list_of_y1[i]),
                ha = 'center', va = 'bottom',
                color = 'red',
                transform = ax.transData)
        elif i==4:
            plt.text(list_of_x1[i], list_of_y1[i], 
                "({ptx1:.2f},{pty1:.2f})".format(ptx1 = list_of_x1[i],
                                                pty1 = list_of_y1[i]),
                ha = 'center', va = 'bottom',
                color = 'red',
                transform = ax.transData)
        elif i == 5:
            plt.text(list_of_x1[i], list_of_y1[i], 
                "({ptx1:.2f},{pty1:.2f})".format(ptx1 = list_of_x1[i],
                                                pty1 = list_of_y1[i]),
                ha = 'center', va = 'bottom',
                color = 'red',
                transform = ax.transData)
        else:
            plt.text(list_of_x1[i], list_of_y1[i], 
                "({ptx1:.2f},{pty1:.2f})".format(ptx1 = list_of_x1[i],
                                                pty1 = list_of_y1[i]),
                ha = 'center', va = 'bottom',
                color = 'red',
                transform = ax.transData)
    plt.show()