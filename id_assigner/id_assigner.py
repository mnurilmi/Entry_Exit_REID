"""
    ids:{
        "1":{
            "state":,
            "distance":,
            "count":
        },
        "2":{
            "state":,
            "distance":,
            "count":
        },
    }
    state       -> (string) "tracked","entry", "exit"
    distance    -> (float)
    count       ->
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json
import cv2

class track(object):
    # def __init__(self, tlbr):
    #     self.centroid = calculate_centroid(tlbr)
    pass

class IDAssigner(object):

    def __init__(self, entry_line_config, distance_treshold = 10):
        self.last_id = 0
        self.ids = []
        self.centroids = []
        self.distances = []
        self.distance_treshold = distance_treshold
        self.entry_line_coef = self.calculate_line_coef(entry_line_config)
        print("HOOHOHO")
        # Inisialisasi model
    
    def register_ids(self, im0, ot):
        """
        Steps:
        - kalkulasi centroids dan hitung jarak ke entry line
        - tentuin state
        -jika menyentuh garis pada treshold tertentu maka ekstrasi fitur
        """
        print(self.x)
        print(self.y)
        cv2.line(im0,(int(self.x[0]),int(self.y[0])),(int(self.x[1]),int(self.y[1])),(0,255,0),3)
        tlbrs = [t.tlbr for t in ot]
        self.centroids = self.calculate_centroids(tlbrs)
        self.distances = self.calculate_distance2line(self.centroids, self.entry_line_coef)
        
        print("D:", self.distances)
        self.tracks_passed = self.check_passed_the_line()
        print(self.tracks_passed)
        return self.distances, self.tracks_passed

    def check_passed_the_line(self):
        return [True if x <= self.distance_treshold else False for x in self.distances]
    

    # def calculate_centroid(tlbr):
    #     print(tlbr)
    #     w = int(tlbr[3]-tlbr[1])
    #     h = int(tlbr[2] - tlbr[0])
    #     return (int(tlbr[0] + h/2), int(tlbr[1] + w/2))

    def calculate_centroids(self, tlbrs):
        """
        input: list berukuran n yang berisi list 1x4 yang berisi data tlbr
        output: list berukuran n yang berisi centroid (cx, cy)
        """
        tlbrs_np = np.array(tlbrs)
        Mx = np.array([[1/2],[0],[1/2],[0]])
        My = np.array([[0],[1/2],[0],[1/2]])
        Cx = np.dot(tlbrs_np, Mx)
        Cy = np.dot(tlbrs_np, My)
        # print(Cx)
        # print(Cy)
        Cx= Cx.flatten()
        Cy = Cy.flatten()
        # print(Cx)
        return [(int(Cx[i]), int(Cy[i])) for i in range(len(Cx))]


    # =====Line to point functions
    def calculate_line_coef(self, config):
        # Opening JSON file
        with open(config, 'r') as openfile:
            # Reading from json file
            j = json.load(openfile)

        #Definisi garis
        x = [float(j["x1"]), float(j["x2"])]
        y = [float(j["y1"]), float(j["y2"])]
        self.x = x
        self.y = y
        y = [-1 * i for i in y] # penyesuaian pada bidang gambar
        # print(y)
        # Calculate the coefficients. This line answers the initial question. 
        coefficients = np.polyfit(x, y, 1) 
        return coefficients

    def calculate_points_cartesian(self, points):
        points_img = np.array(points)
        M = np.array([[1, 0],[0, -1]])
        return np.dot(points_img,M)


    # def distance_point2line(point, coef):
    #     return ((coef[0]*point[0])-point[1]+coef[1])/math.sqrt((coef[0]*coef[0])+1)

    def calculate_distance2line(self, p, coef):
        points = self.calculate_points_cartesian(p)
        x = np.array([[coef[0]],[-1]])
        # print(x.shape)
        return ((np.dot(points, x)+coef[1]) * (1/math.sqrt((coef[0]*coef[0])+1))).flatten()