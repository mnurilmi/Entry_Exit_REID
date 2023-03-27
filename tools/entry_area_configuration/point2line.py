"""
deskripsi:
    - program untuk cari jarak terdekat dari 2 garis pada dan 1 titik (centroid) pada citra
        citra bila di bidang kartesian berada pada quadran IV sehingga
        diperlukan penyesuaian koordinat y dari masing masing parameter titik yang diketahui.
    - Operasi iteratif mengurangi big O dari program.
        program ini mengubah persamaan aljabar menjadi opeerasi matrix
input:
    centroids: list dari titik yang hendak dicari jaraknya dari garis
    p1: titik [x1, y1] dari garis
    p2: titik [x2, y2] dari garis
output:
    list jarak terdekat antara centroids dan garis (dari p1 dan p2)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json

def read_entry_line_config(cfg):
    # Opening JSON file
    with open(cfg, 'r') as openfile:
        # Reading from json file
        j = json.load(openfile)
    print(j)
    print(type(j["x1"]))
    return (int(j["img_h"]), int(j["img_w"])), [float(j["x1"]), float(j["y1"])], [float(j["x2"]), float(j["y2"])]
    # return [1000, 0], [0, 1000]
    # return [695, 232], [987, 1000]


    

def get_coef(p1, p2):
    #Definisi garis
    print(p1)
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    y = [-1 * i for i in y]
    # print(y)
    # Calculate the coefficients. This line answers the initial question. 
    coefficients = np.polyfit(x, y, 1) 
    return coefficients

def get_point_cartesian(points):
    points_img = np.array(points)
    M = np.array([[1, 0],[0, -1]])
    return np.dot(points_img,M)


def distance_point2line(point, coef):
    return ((coef[0]*point[0])-point[1]+coef[1])/math.sqrt((coef[0]*coef[0])+1)

def distance_points2line(points, coef):
    D = 0
    x = np.array([[coef[0]],[-1]])
    # print(x.shape)
    return ((np.dot(points, x)+coef[1]) * (1/math.sqrt((coef[0]*coef[0])+1))).flatten()

def visualize(img_shape, p1, p2, centroids, distance, coefficients):
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    y = [-1 * i for i in y]

    # print(x)
    # print(y)
    # Print the findings
    print('a =', coefficients[0])
    print('b =', coefficients[1])

    # Let's compute the values of the line...
    x_axis = []
    y_axis = []
    polynomial = np.poly1d(coefficients)
    # i = 0
    # for i in range(0, 2560):
    #     if polynomial(i) <= 0:
    #         print(polynomial(i))
    #         x_axis.append(i)
    #         y_axis.append(polynomial(i))
    #     i+=10
    x_axis = np.linspace(0,2560)
    y_axis = polynomial(x_axis)
    print("sasa", img_shape)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(1440*px, 2560*px))
    plt.text(0.5, 0.5, f'{img_shape[1]}px x {img_shape[0]}')
    plt.plot(x_axis, y_axis)
    plt.plot( x[0], y[0], 'go' )
    plt.plot( x[1], y[1], 'go' )
    plt.xlim([0, img_shape[1]])
    plt.ylim([-img_shape[0], 0])
    
    for i in range(len(centroids)):
        plt.plot(centroids[i][0], -1 * centroids[i][1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")
        plt.text(centroids[i][0]+10, -1 * centroids[i][1] - 50, 'p-'+str(i), fontsize = 10)
        plt.text(centroids[i][0]+10, -1 * centroids[i][1] - 100, 'jarak:'+str(distance[i]), fontsize = 10)
    
    plt.grid('on')
    plt.show()

# def test_iter_vs_matrix():
#     test_points = []
#     for _ in range(100):
#         test_points.append([1, 1])
#     test_points = np.array(test_points)
#     # print(test_points.shape)
#     t1 = time.time()
#     # # Cara Iterasi
#     # for p in test_points:
#     #     # print(np.array(p)[0])
#     #     print(distance_point2line(np.array(p)[0], coefficients))
#     # print(time.time()-t1)

#     # Cara matrix
#     print(distance_points2line(test_points, coefficients))
#     print(time.time()-t1)


if __name__ == "__main__":
    cfg = 'configs/entry_line_config2.json'
    # Definisi centroid dari bbox
    centroids = [
        [1500, 1200],
        [500, 1000],
        [0, 800],
        [0, 0],
        [2000, 0]
    ]
    #Definisi garis dari titik p1 dan p2
    img_shape, p1, p2 = read_entry_line_config(cfg)
    # print(type(img_shape[0]))
    coefficients = get_coef(p1, p2)
    points_cartesian = get_point_cartesian(centroids)
    # print(points_cartesian.shape) 

    t1 = time.time()
    # # Cara Iterasi
    # for p in points_cartesian:
    #     # print(np.array(p))
    #     print(distance_point2line(p, coefficients))
    # print(time.time()-t1)

    # Cara matrix
    d = -1 * distance_points2line(points_cartesian, coefficients)
    print(d)
    print(time.time()-t1)
    visualize(img_shape, p1, p2, centroids, d, coefficients)

