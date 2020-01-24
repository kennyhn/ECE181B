#!/usr/bin/env python3


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