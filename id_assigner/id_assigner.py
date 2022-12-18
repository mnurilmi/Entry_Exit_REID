"""
Description:
    Class: ID_Assigner
    This class is used for entry exit reid implementation.
    The implementation combains yolov7 (detector), bytetrack
    (tracker), and id assigner it self to solve the problem.
    This class will manage each tracker object ids according 
    to each track states.
author:
    muhammad nur ilmi-gadjah mada university
    mnurilmi18@gmail.com
"""
import numpy as np
import math
import time
import json
import cv2
import glob
import os
# import sys
# sys.path.insert(0, "deep-person-reid")
# sys.path.append(".")

from tracker.basetrack import TrackState

from torchreid.utils import FeatureExtractor
import torch
from scipy.spatial import distance

class TrackState_(TrackState):
    In = 4
    Matching = 5

class ID_Assigner(object):
    def __init__(
        self,
        entry_line_config, 
        entry_area_position,
        distance_treshold = 0, 
        reid_model_name = "osnet_x1_0", 
        reid_model_path = 'models/model.pth.tar-60',
        save_feats = False
        ):
        print("CUDA is active?: ", torch.cuda.is_available())
        
        self.db = {}
        self.frame_id = 0
        self.last_id = 0
        self.distance_treshold = distance_treshold
        self.entry_area_position = entry_area_position
        self.entry_line_coef = self.calculate_line_coef(entry_line_config)
        self.output = []
        self.max_intra = 0
        self.event_logger = {
            "saving_db":0,
            "matching_db":0,
            "match":0,
            "frame_indices":[]
        }
        # Reid model inisialization
        self.feat_extractor = FeatureExtractor(
            model_name = reid_model_name,
            model_path = reid_model_path,
            device='cuda'
        )

        if save_feats:
            # Reset folder to save patches
            self.reset()

        print("===ReID model and ID Assigner ready!===")

    def reset(self):
        files = glob.glob('temp/*')
        for f in files:
            os.remove(f)
        print("===folder reset success!===")

    def next_id(self):
        self.last_id += 1
        return self.last_id

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

    def update(self, ot, im0):
        """
        Steps:
        - kalkulasi centroids dan hitung jarak ke entry line
        - tentuin state
        """
        self.frame_id +=1

        if len(ot) == 0:
            return []
            
        im1 = im0.copy()
        centroids, patches = self.get_centroids_and_patches(ot, im1)     # matrix operation and iteration process
        distances = self.calculate_distance2line(centroids)         # matrix operation calculation
        feats = self.extract_feature(patches)                       # One inference process
        print(len(patches))
        print(feats.shape)
        self.update_tracks(ot, feats, distances, centroids)                                # iteration process
        self.set_ids(ot, distances)                                 # iteration process

        # visualize entry line
        cv2.line(im0,(int(self.x[0]),int(self.y[0])),(int(self.x[1]),int(self.y[1])),(0,255,0),1)

        # tracks_passed = self.check_passed_the_line(distances)
        return ot

    def get_centroids_and_patches(self, ot, im1):
        """
        input: list berukuran n yang berisi list 1x4 yang berisi data tlbr
        output: list berukuran n yang berisi centroid (cx, cy)
        """
        patches = []
        tlbrs = []
        for i in range(len(ot)):
            tlbrs.append(ot[i].tlbr)
            patches.append(self.extract_patch(im1, ot[i].tlbr))

        tlbrs_np = np.array(tlbrs)
        Mx = np.array([[1/2],[0],[1/2],[0]])
        My = np.array([[0],[1/2],[0],[1/2]])
        Cx = np.dot(tlbrs_np, Mx).astype(int)
        Cy = np.dot(tlbrs_np, My).astype(int)
        # print(Cx)
        # print(Cy)
        Cx= Cx.flatten()
        Cy = Cy.flatten()
        # print(Cx)
        centroids = np.transpose(np.array([Cx, Cy]))

        return centroids, patches

    def calculate_distance2line(self, p):
        coef = self.entry_line_coef
        points = self.calculate_points_cartesian(p)
        x = np.array([[coef[0]],[-1]])
        # print(x.shape)
        return ((np.dot(points, x)+coef[1]) * (1/math.sqrt((coef[0]*coef[0])+1))).flatten()

    def calculate_points_cartesian(self, points):
        points_img = np.array(points)
        M = np.array([[1, 0],[0, -1]])
        return np.dot(points_img,M)

    def extract_patch(self, img, tlbr, target_size = (128, 256)):
        # print(tlbr)
        # print(img.shape)
        tlbr = np.array(tlbr).astype(int)
        tlbr[0] = (max(0, tlbr[0]))
        tlbr[1] = (max(0, tlbr[1]))
        tlbr[2] = (min(img.shape[1] - 1, tlbr[2])) # relatif terhadap W
        tlbr[3] = (min(img.shape[0] - 1, tlbr[3])) # relatif terhadap H
        # print(tlbr)
        patch = img[
            tlbr[1]:tlbr[3],
            tlbr[0]:tlbr[2]
        ]
        cv2.resize(patch, target_size)
        
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

    def update_tracks(self, ot, feats, distances, centroids):
        # print("feats length: ", feats.shape)  
        feats_np = feats.cpu().data.numpy()

        for i in range(len(ot)):
            ot[i].update_feat(feats_np[i])
            ot[i].set_distance(distances[i])
            ot[i].set_centroid((centroids[i][0],centroids[i][1]))

    def set_ids(self, ot, distances):
        print("===set ids===   ", len(ot), "tracks online")
        for i in range(len(ot)):
            # print("{}={}={}".format(ot[i].get_id(), ot[i].get_last_state(), distances[i]))
            # print(self.calculate_intra_class_distance(ot[i]))
            # self.algorithm1(ot, i)
            self.algorithm2(ot, i)
            
            # print("<=======================>")
            # print("{}={}={}".format(ot[i].get_id(), ot[i].get_last_state(), distances[i]))
            # print("\n")
        print("data in database: ", self.db.keys())

    def algorithm1(self, ot, i):
        """
        algorithm notes:
            * default state: Tracked
            * features are saved just once when the track pass the line for the first time
            * Impossible condition are:
                * id -1; any state except tracked

        """
        tid = ot[i].get_id()
        tls = ot[i].get_last_state()    # track last state
        is_passed = self.is_passed(ot[i].get_distance()) 
        feat = ot[i].get_feat()

        if tid == -1:
            if not is_passed:
                # set new id, last state unchanged
                ot[i].set_id(self.next_id())
            else:
                # matching db
                is_match, registered_object = self.matching_db(feat)
                if is_match:
                    ot[i].set_id(registered_object.get_id())
                    # ot[i].set_id(99)
                    # print(feat)
                    ot[i].set_last_state(TrackState_.Matching)
                else:
                    # set new id and start tracking as usual
                    ot[i].set_id(self.next_id())

                    self.save_2_db(ot[i])
                    ot[i].set_last_state(TrackState_.In) # transition Tracked to In
        else:
            if not is_passed:
                if tls == TrackState_.Matching:
                    print(feat)
                    # Exit event condition; append output
                    self.output.append(ot[i].get_id())

                # tracking as usual until cross/pass the entry line
                ot[i].set_last_state(TrackState_.Tracked)
            else:
                if  tls == TrackState_.Tracked:
                    # saving to db
                    self.save_2_db(ot[i])
                    ot[i].set_last_state(TrackState_.In) # transition Tracked to In
                elif tls == TrackState_.Matching:
                    # matching db until pass the entry line again
                    is_match, registered_object = self.matching_db(feat)
                    if is_match:
                        ot[i].set_id(registered_object.get_id())
                        # ot[i].set_id(99)
                    else:
                        # tracking as usual with current id
                        pass
                else:
                    # tracking as usual
                    pass

    def algorithm2(self, ot, i):
        """
        algorithm notes:
            * default state: Tracked
            * features are saved just once when the track pass the line for the first time
            * Impossible condition are:
                * id -1; any state except tracked

        """
        tid = ot[i].get_id()
        tls = ot[i].get_last_state()    # track last state
        is_passed = self.is_passed(ot[i].get_distance()) 
        feat = ot[i].get_feat()

        if tid == -1:
            # matching db
            is_match, registered_object = self.matching_db(feat)
            if is_match:
                ot[i].set_id(registered_object.get_id())
                # ot[i].set_id(99)
                # print(feat)
                ot[i].set_last_state(TrackState_.Matching)
            else:
                # set new id and start tracking as usual
                ot[i].set_id(self.next_id())

                self.save_2_db(ot[i])
                ot[i].set_last_state(TrackState_.In) # transition Tracked to In
        else:
            if not is_passed:
                if tls == TrackState_.Matching:
                    print(feat)
                    # Exit event condition; append output
                    self.output.append(ot[i].get_id())

                # tracking as usual until cross/pass the entry line
                ot[i].set_last_state(TrackState_.Tracked)
            else:
                if  tls == TrackState_.Tracked:
                    # saving to db
                    self.save_2_db(ot[i])
                    ot[i].set_last_state(TrackState_.In) # transition Tracked to In
                elif tls == TrackState_.Matching:
                    # matching db until pass the entry line again
                    is_match, registered_object = self.matching_db(feat)
                    if is_match:
                        ot[i].set_id(registered_object.get_id())
                        # ot[i].set_id(99)
                    else:
                        # tracking as usual with current id
                        pass
                else:
                    # tracking as usual
                    pass


    def is_passed(self, distance):
        # First ssumption the location enter on the left side of the picture
        # The user can setting it on the config
        if self.entry_area_position == "left":
            if distance >= self.distance_treshold:
                if self.gradient == "positive":
                    return True
                else:
                    return False
            else:
                if self.gradient == "positive":
                    return False
                else:
                    return True
        elif self.entry_area_position == "right":
            if distance < self.distance_treshold:
                if self.gradient == "positive":
                    return True
                else:
                    return False
            else:
                if self.gradient == "positive":
                    return False
                else:
                    return True
        elif self.entry_area_position == "above":
            if distance >= self.distance_treshold:
                if self.gradient == "positive":
                    return True
                else:
                    return False
            else:
                if self.gradient == "positive":
                    return False
                else:
                    return True
        elif self.entry_area_position == "below":
            if distance < self.distance_treshold:
                if self.gradient == "positive":
                    return True
                else:
                    return False
            else:
                if self.gradient == "positive":
                    return False
                else:
                    return True

    def matching_db(self, feat_):
        print("MATCHING DB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log_event("matching_db")
        ids = list(self.db.keys())
        print("data in database: ", ids)

        if len(ids) == 0:
            return False, None

        query = feat_
        gallery = []
        for id_ in ids:
            gallery.extend(self.db[id_]["registered_object"].get_feat())
        # calculate similarity
        query = np.array(query)
        gallery = np.array(gallery)
        # print(query)
        # print(gallery)
        ds = distance.cdist(
                    query,
                    gallery, 
                    "euclidean"
                )[-1]
        ds = ds.tolist()
        print(ds)
        if min(ds) <20:
            print(ds)
            match = ids[self.argmin(ds)]
            self.log_event("match")
            self.event_logger["frame_indices"].append(self.frame_id)
            print("MATCHED ID: ", match)
            o = self.db[match]["registered_object"]
            # del self.db["match"]
            return True, o
        else:
            return False, None  

    def argmin(self, lst):
      return lst.index(min(lst))
    
    def argmax(self, lst):
      return lst.index(max(lst))

    def save_2_db(self, t):
        # saving feat to db
        print(t.get_id(), " SAVE FITUR 2 DB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log_event("saving_db")
        self.db[t.get_id()] = {
            "registered_object": t
        }


    def check_passed_the_line(self, distances):
        track_passed = []
        for d in distances:
            track_passed.append(self.is_passed(d))
        return track_passed

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
    
    def log_output(self):
        print("Output:", self.output)
