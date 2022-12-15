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
    state       -> (string) "tracked","in", "exit"
    distance    -> (float)
    count       ->
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json
import cv2
# import sys
# import uuid
import glob
import os
from numpy.lib import DataSource
# sys.path.insert(0, "deep-person-reid")
# sys.path.append(".")

from tracker.basetrack import TrackState

# import torchreid
from torchreid.utils import FeatureExtractor
import torch
from scipy.spatial import distance

class IDAssigner(object):

    def __init__(self, entry_line_config, distance_treshold = 0, reid_model_name = "osnet_x1_0", reid_model_path = 'models/model.pth.tar-60'):
        print("CUDA active?: ", torch.cuda.is_available())
        self.last_id = 0
        self.ids = []
        self.centroids = []
        self.distances = []
        self.distance_treshold = distance_treshold
        self.entry_line_coef = self.calculate_line_coef(entry_line_config)
        self.db = {}
        self.max_intra = 0
        self.min_inter = 99
        self.tracked_objects = []
        self.states = []
        self.event_logger = {
            "saving_db":0,
            "matching_db":0,
            "match":0
        }

        # Reid model inisialization
        self.feat_extractor = FeatureExtractor(
            model_name = reid_model_name,
            model_path = reid_model_path,
            device='cuda'
        )

        # Reset folder for saving patches
        self.reset()

        print("===ReID model and ID Assigner ready!===")
    

    def reset(self):
        files = glob.glob('temp/*')
        for f in files:
            os.remove(f)
        print("===Berhasil reset folder!===")

    def next_id(self):
        self.last_id += 1
        return self.last_id 

    def update(self, im0, ot):
        """
        Steps:
        - kalkulasi centroids dan hitung jarak ke entry line
        - tentuin state
        """
        # if len(self.tracked_objects) == 0:
        #     self.tracked_objects = ot
        #     self.states = ["new" for _ in range(len(ot))]
        # else:
        #     for t in ot:
        #         if t not in self.tracked_objects:
        #             self.tracked_objects.append(t)
        # print("TO: ", self.tracked_objects)

        if len(ot) == 0:
            return [], [], []
            
        im1 = im0.copy()
        tlbrs = [t.tlbr for t in ot]
        centroids = self.calculate_centroids(tlbrs)
        distances = self.calculate_distance2line(centroids, self.entry_line_coef)
        patches = [self.extract_patch(im1, tlbr) for tlbr in tlbrs]
        feats = self.extract_feature(patches)
        self.update_feats(ot, feats)
        self.set_ids(ot, distances)

        # visualize the entry line
        cv2.line(im0,(int(self.x[0]),int(self.y[0])),(int(self.x[1]),int(self.y[1])),(0,255,0),3)
        
        # print(self.x)
        # print(self.y)
        # H, W, _ = np.shape(im0)
        # print(H, W)
        # print("p: ",patches)
        # print(feats[i].shape)
        # print("D:", self.distances)
        tracks_passed = self.check_passed_the_line(distances)
        # print(self.tracks_passed)

        return distances, tracks_passed, centroids


    def check_passed_the_line(self, distances):
        track_passed = []
        for d in distances:
            track_passed.append(self.is_passed(d))
        return track_passed
    
    def is_passed(self, distance, entry_area = "left"):
        # First ssumption the location enter on the left side of the picture
        # The user can setting it on the config
        if entry_area == "left":
            if distance <= self.distance_treshold:
                if self.gradient == "positive":
                    return False
                else:
                    return True
            else:
                if self.gradient == "positive":
                    return True
                else:
                    return False
        else:
            if distance <= self.distance_treshold:
                if self.gradient == "negative":
                     return False
                else:
                    return True
            else:
                if self.gradient == "negative":
                    return True
                else:
                    return False

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

        # gradient calculation for determining entry area in another function
        if x[1]-x[0] == 0:
            m = np.inf
        else:
            m = -1*(y[1]-y[0])/(x[1]-x[0])
        if  m > 0:
            self.gradient = "positive"
        elif m < 0:
            self.gradient = "negative"
        elif m == 0:
            self.gradient = "horizontal"
        elif m == np.inf: 
            self.gradient = "vertical"

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
    
    def update_feats(self, ot, feats):
        # print("feats length: ", feats.shape)  
        feats_np = feats.cpu().data.numpy()
        for i in range(len(ot)):
            # ot[i].set_id(99)
            # ot[i].set_state(TrackState.In)
            # print("id: ", ot[i].get_id())
            # print("state: ", ot[i].get_state())
            # print(TrackState.In)
            # print("hoho",len(feats_np[i]))
            ot[i].update_feat(feats_np[i])


    def set_ids(self, ot, distances):
        print("===set ids===   ", len(ot), "tracks online")
        for i in range(len(ot)):
            # print("{}={}={}".format(ot[i].get_id(), ot[i].get_state(), distances[i]))
            # print(self.calculate_intra_class_distance(ot[i]))
    
            self.algorithm_scene1(ot, distances, i)
    
            # print("<=======================>")  
            # print("{}={}={}".format(ot[i].get_id(), ot[i].get_state(), distances[i]))
            # print("\n")
        print("data in database: ", self.db.keys())

    def algorithm_scene1(self, ot, distances, i):
        """
        algorithm notes:
            * default state: Tracked
            * features are saved just once when the track pass the line for the first time
            * Impossible condition are:
                * id -1; any state except tracked

        """
        if ot[i].get_id() == -1:
            if not self.is_passed(distances[i]):
                # set new id, state unchanged
                ot[i].set_id(self.next_id())
            else:
                # matching db
                is_match, registered_object = self.matching_db(ot[i].get_feat())
                if is_match:
                    ot[i].set_id(registered_object.get_id())
                    ot[i].set_state(TrackState.Matching)
                else:
                    # set new id and start tracking as usual
                    ot[i].set_id(self.next_id())

                    self.save_2_db(ot[i])
                    ot[i].set_state(TrackState.In) # transition Tracked to In
        else:
            if not self.is_passed(distances[i]):
                # tracking as usual until cross/pass the entry line
                ot[i].set_state(TrackState.Tracked)
            else:
                if ot[i].get_state() == TrackState.Tracked:
                    # saving to db
                    self.save_2_db(ot[i])
                    ot[i].set_state(TrackState.In) # transition Tracked to In
                elif ot[i].get_state() == TrackState.Matching:
                    # matching db until pass the entry line again
                    is_match, registered_object = self.matching_db(ot[i].get_feat())
                    if is_match:
                        ot[i].set_id(registered_object.get_id())
                    else:
                        # tracking as usual with current id
                        pass
                else:
                    # tracking as usual
                    pass

        #===end function===
    
    def matching_db(self, feat_):
        self.log_event("matching_db")
        # for o in self.db.keys():


        # return True, registered_object

        return False, None
            


    def save_2_db(self, t):
        # saving feat to db
        print(t.get_id(), " SAVE FITUR 2 DB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log_event("saving_db")
        self.db[t.get_id()] = {
            "registered_object": t
        }
        

    
    def calculate_intra_class_distance(self, t):
        # Cek asosiasi intra class feature
        ds = []
        print("len feat: ", len(t.get_feat()))
        print("feat", np.array(t.get_feat()).shape)
        for f in t.get_feat():
            for g in t.get_feat():
                ds.append(int(
                    distance.cdist(
                    f.reshape(1, -1),
                    g.reshape(1, -1), "euclidean"
                ).tolist()[-1][-1]
                ))
        print("intra class distance: ")
        # print(ds)
        return ds

    def log_event(self, event):
        if event not in self.event_logger.keys():
            self.event_logger[event] = 1
        else:    
            self.event_logger[event] += 1

    def log_report(self):
        print("\n=====REPORT EVENT=====")
        print(self.event_logger)
    
    def sample_db(self):
        print("\n===== SAMPLE DATABASE=====")
        print(self.db)
        for i in self.db.keys():
            print("feat shape: ", self.db[i]["registered_object"].get_feat().shape) 