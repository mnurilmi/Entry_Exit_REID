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
import sys
import uuid
import glob
import os
# sys.path.insert(0, "deep-person-reid")
# sys.path.append(".")

import torchreid
from torchreid.utils import FeatureExtractor
import torch
from scipy.spatial import distance



class IDAssigner(object):

    def __init__(self, entry_line_config, distance_treshold = 10, reid_model_name = "osnet_x1_0", reid_model_path = 'models/model.pth.tar-60'):
        print(torch.cuda.is_available())
        self.last_id = 0
        self.ids = []
        self.centroids = []
        self.distances = []
        self.distance_treshold = distance_treshold
        self.entry_line_coef = self.calculate_line_coef(entry_line_config)
        self.admin = {}
        self.max_intra = 0
        self.min_inter = 99

        # Inisialisasi model
        self.feat_extractor = FeatureExtractor(
            model_name = reid_model_name,
            model_path = reid_model_path,
            device='cuda'
        )

        self.reset()
        print("===ReID model and ID Assigner ready!===")
    
    def reset(self):
        files = glob.glob('temp/*')
        for f in files:
            os.remove(f)
        print("===Berhasil reset folder!===")

    def register_ids(self, im0, ot):
        """
        Steps:
        - kalkulasi centroids dan hitung jarak ke entry line
        - tentuin state
        """
        im1 = im0.copy()
        # print(self.x)
        # print(self.y)
        cv2.line(im0,(int(self.x[0]),int(self.y[0])),(int(self.x[1]),int(self.y[1])),(0,255,0),3)
        tlbrs = [t.tlbr for t in ot]

        # H, W, _ = np.shape(im0)
        # print(H, W)
        patches = [self.extract_patch(im1, tlbr) for tlbr in tlbrs]
        # print("p: ",patches)
        feats = self.extract_feature(patches)
        # print(feats[i].shape)

        self.centroids = self.calculate_centroids(tlbrs)
        self.distances = self.calculate_distance2line(self.centroids, self.entry_line_coef)
        # print("D:", self.distances)
        self.tracks_passed = self.check_passed_the_line()
        # print(self.tracks_passed)


        # Mulai register objek ke dictionary
        for i in range(len(ot)):
            tid = ot[i].get_id()

            f = []
            if tid in self.admin.keys():
                f = self.admin[tid]["feats"]
                if len(f)<5:
                    f.append(feats[i])
                else:
                    f.pop(0)
                    f.append(feats[i])
                self.admin[tid] = {
                  "objek": ot[i],
                  "feats":f,
                  "distance": self.distances[i],
                  "passed": self.tracks_passed[i],
                  "last_state": "tracked"
                }
                print("===cek state===")
                if self.admin[tid]["last_state"] == "tracked":
                    if self.admin[tid]["passed"]:
                        self.admin[tid]["last_state"] = "enter"
                if self.admin[tid]["last_state"] == "enter":
                    if not self.admin[tid]["passed"]:
                        self.admin[tid]["last_state"] = "exit"      
            else:
                f.append(feats[i])
                self.admin[tid] = {
                  "objek": ot[i],
                  "feats":f,
                  "distance": self.distances[i],
                  "passed": self.tracks_passed[i],
                  "last_state": "tracked"
                }


            print("admin: ", self.admin[tid]["distance"])
            print("admin: ", self.admin[tid]["passed"])
            print("admin: ", self.admin[tid]["last_state"])
            
            # print("objek: ", self.admin[tid]["objek"].get_id())
            print("feats length: ", len(self.admin[tid]["feats"]), "\n")

            
        # Cek asosiasi intra class feature
        for f in self.admin[1]["feats"]:
            for g in self.admin[1]["feats"]:
                d = distance.cdist(
                    f.cpu().data.numpy().reshape(1, -1),
                    g.cpu().data.numpy().reshape(1, -1), "euclidean"
                    )
                # Mencari maximum jarak euclidean intraclass
                if d > self.max_intra:
                    self.max_intra = d

        if len(self.admin[1]["feats"])==5:
            x = self.admin[1]["feats"].numpy()
            print(x.shape)
            # # print(x[0].shape)
            # for f in x:
            #     ds = distance.cdist(
            #         f.reshape(1, -1),
            #         x,
            #         "euclidean"
            #         ) 
            #     print("Ds: ", ds)
        print("max intra: ", self.max_intra) 

        # # Cek asosiasi inter class feature
        # if len(admin.keys()>1):
        #     for 


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

    # =====REID=====
    def extract_patch(self, img, tlbr):
        # print(tlbr)
        # print(img.shape)
        tlbr[0] = (max(0, tlbr[0]))
        tlbr[1] = (max(0, tlbr[1]))
        tlbr[2] = (min(img.shape[1] - 1, tlbr[2])) # relatif terhadap W
        tlbr[3] = (min(img.shape[0] - 1, tlbr[3])) # relatif terhadap H
        # print(tlbr)
        patch = img[
            int(tlbr[1]):int(tlbr[3]),
            int(tlbr[0]):int(tlbr[2])
        ]
        cv2.resize(patch, (128, 256))
        
        ## Function to save the features locally
        # cv2.imwrite("temp/{0}.jpg".format(uuid.uuid4()), patch)
        # print(len(glob.glob("temp/*")))
        return patch

    def extract_feature(self, patches, verbose = False):
        t1 = time.time()
        features = self.feat_extractor(patches)
        if verbose:
            print("feats:", features)
            print("waktu ekstrak feats: ", time.time() - t1)
            # Debug matching fitur
            for f in features:
                for g in features:
                    # print(distance.cdist(f.cpu().data.numpy(), g.cpu().data.numpy()))
                    print(
                          distance.cdist(
                              f.cpu().data.numpy().reshape(1, -1),
                              g.cpu().data.numpy().reshape(1, -1), "euclidean"
                              ))
        return features
