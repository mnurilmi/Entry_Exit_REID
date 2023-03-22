"""
Description:
    Class: Entry_Exit_REID
    This class is used for entry exit reid implementation.
    The implementation combains yolov7 (detector), bytetrack
    (tracker), and OsNet reid model to solve the problem.
    This class will manage each track ids when a person enter 
    or exit a entry line area
author:
    muhammad nur ilmi-Gadjah Mada University, Yogyakarta, Indonesia
    mnurilmi18@gmail.com
"""
import numpy as np
import math
import time
import cv2
import glob
import os
import csv

from tracker.basetrack import TrackState

import torch
from scipy.spatial import distance

class TrackState_(TrackState):
    In = 4
    Matching = 5

class Entry_Exit_REID(object):
    def __init__(
        self,
        entry_line_config,        
        feat_extractor,
        feat_match_thresh,
        save_patch_dir = None,
        distance_treshold = 0, 
        ):
        print("CUDA is active?: ", torch.cuda.is_available())
        
        self.db = {}
        self.output = []
        self.frame_id = 0
        self.last_id = 0
        self.distance_treshold = distance_treshold
        self.entry_area_position = entry_line_config["entry_area_position"]
        self.entry_line_coef = self.calculate_line_coef(entry_line_config)
        self.system_recorder = {
            "total_person_in_frame": [],
            "saving_db":0,
            "matching_db":0,
            "match":0,
            "frame_indices":[]
        }
        self.EER_recorder = []
        
        # Reid model inisialization
        self.feat_extractor = feat_extractor
        self.feat_match_thresh = feat_match_thresh 
        if save_patch_dir != None:
            self.save_patch_dir = save_patch_dir
            os.mkdir(str(save_patch_dir)+"/sample_patch")
        print("feature matching treshold:", self.feat_match_thresh)
        print("===Entry Exit REID ready!===")
        

    def next_id(self):
        self.last_id += 1
        return self.last_id

    def calculate_line_coef(self, j):
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
        
        # Determine upper and lower bounds of entry line
        if y[0] > y[1]:
            self.top_line_points = [(0,-int(y[0])), (int(x[0]),-int(y[0]))]
            self.bottom_line_points = [(0,-int(y[1])), (int(x[1]),-int(y[1]))]
        else:
            self.top_line_points = [(0,-int(y[1])), (int(x[1]),-int(y[1]))]
            self.bottom_line_points = [(0,-int(y[0])), (int(x[0]),-int(y[0]))]
            
        return coefficients
    
    def get_top_bottom_line(self):
        return self.top_line_points, self.bottom_line_points
    
    def update(self, ot, im0):
        """
        Steps:
        - kalkulasi centroids dan hitung jarak ke entry line
        - tentuin state
        """
        self.frame_id +=1
        
        if len(ot) == 0:
            return []
        self.system_recorder["total_person_in_frame"].append(len(ot))
        
        im1 = im0.copy()
        centroids, patches = self.get_centroids_and_patches(ot, im1)    # matrix operation and iteration process
        distances = self.calculate_distance2line(centroids)             # matrix operation calculation
        feats = self.extract_feature(patches)                           # One inference process (many objects)
        feats_np = feats.cpu().data.numpy()
        self.set_ids(ot, feats_np, patches, distances, centroids)                # iteration process
        
        # print("feats length: ", feats.shape)  
        # print(len(patches))
        # print(feats.shape)
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
        # bottom centroid
        Mx = np.array([[1/2],[0],[1/2],[0]]) # left,right
        My = np.array([[0],[0],[0],[1]])     # top bottom

        # # Middle centroid
        # Mx = np.array([[1/2],[0],[1/2],[0]]) # left,right
        # My = np.array([[0],[0],[0],[1]])     # top bottom

        Cx = np.dot(tlbrs_np, Mx).astype(int)
        Cy = np.dot(tlbrs_np, My).astype(int)
        # print(Cx)
        # print(Cy)
        Cx= Cx.flatten()
        Cy = Cy.flatten()
        # print(Cx)
        centroids = np.transpose(np.array([Cx, Cy]))
        return centroids, patches
    
    def save_patch(self, fname, patch):
        cv2.imwrite(f"{fname}.jpg", patch)

    def calculate_distance2line(self, p):
        """Distance centroid point to entry_exit line Calculation

        Args:
            p (list): list of online track centroids

        Returns:
            distances: list distance centroid point to entry_exit line 
        """
        coef = self.entry_line_coef
        points = self.calculate_points_cartesian(p)
        x = np.array([[coef[0]],[-1]])
        # print(x.shape)
        return ((np.dot(points, x)+coef[1]) * (1/math.sqrt((coef[0]*coef[0])+1))).flatten()

    def calculate_points_cartesian(self, points):
        """Convert entry line points in the image to cartesian field (quadran I)

        Args:
            points (list): entry line points

        Returns:
            numpy array: points in cartesian field
        """
        points_img = np.array(points)
        M = np.array([[1, 0],[0, -1]])
        return np.dot(points_img,M)

    def extract_patch(self, img, tlbr, target_size = (128, 256)):
        # print(tlbr)
        # print(img.shape)
        tlbr = np.array(tlbr).astype(int)
        tlbr[0] = (max(0, tlbr[0]))
        tlbr[1] = (max(0, tlbr[1]))
        tlbr[2] = (min(img.shape[1] - 1, tlbr[2])) # relative to W
        tlbr[3] = (min(img.shape[0] - 1, tlbr[3])) # relative to H
        # print(tlbr)
        patch = img[
            tlbr[1]:tlbr[3],
            tlbr[0]:tlbr[2]
        ]
        cv2.resize(patch, target_size)
        return patch

    def extract_feature(self, patches):
        """Feature extraction of all online track bounding boxes
            The REID model will do inferencing to all patches and return features of every patch
        Args:
            patches (list): list of person patches
        """
        # t1 = time.time()
        features = self.feat_extractor(patches)
        if torch.cuda.is_available():
                torch.cuda.synchronize()
        return features

    def update_track(self, index, ot, feats_np, distances, centroids):   
        ot[index].update_feat(feats_np[index])
        ot[index].set_distance(distances[index])
        ot[index].set_centroid((centroids[index][0],centroids[index][1]))

    def set_ids(self, ot, feats_np, patches, distances, centroids):
        """Set IDs (ID Assigner)

        Args:
            - ot (list of object): list of online tracks
            - feats_np (numpy array): features from previous reid model inference process
            - distances (list): centroid distances of online tracks
            - centroids (list): centroid position of each online tracks
        
        Tasks:
            - update online track attributes
            - set each online track id's
        """
        # print("===set ids===   ", len(ot), "tracks online")
        for i in range(len(ot)):
            # update current track attribute tobe evaluated
            self.update_track(i, ot, feats_np, distances, centroids)
            
            # procedure to set each track id
            self.id_assigner_procedure(ot, i, patches)
        # print("data in database: ", self.db.keys())

    def id_assigner_procedure(self, ot, i, patches):
        """
        Args:
            - ot (list of object): list of online tracks
            - i (int): index of ot  
        algorithm notes:
            * default state: Tracked
            * features are saved just once when the track pass the line for the first time
            * Impossible condition are:
                * id -1; any state except tracked

        """
        print(i, "-", end = " ")
        t_last_id = ot[i].get_id()
        t_last_state = ot[i].get_last_state()    # track last state
        is_passed = self.is_passed(ot[i].get_distance(), ot[i].get_centroid()) 
        feat = ot[i].get_feat()

        if t_last_id == -1:
            if not is_passed:
                # set new id, last state unchanged (tracked)
                ot[i].set_id(self.next_id())
            else:
                ot[i].set_last_state(TrackState_.Matching)
                # matching db
                is_match, registered_object = self.matching_db(feat)
                if is_match and not self.db[registered_object.get_id()]["being_used"]:
                    ot[i].set_id(registered_object.get_id())
                    self.db[registered_object.get_id()]["being_used"] = True
                else:
                    # set new id and start tracking as usual
                    ot[i].set_id(self.next_id())
        else:
            if not is_passed:
                if t_last_state == TrackState_.Matching:
                    # Exit event condition, transition Matching to Tracked; save result
                    self.output.append(t_last_id)
                    self.set_EER_recorder(t_last_id, "Exit", self.get_local_time())
                    # if t_last_id in self.db.keys():
                    #     self.db[t_last_id]["matched"] = True
                    if self.save_patch_dir != None:
                        self.save_patch(f"{self.save_patch_dir}/sample_patch/{t_last_id} exit",patches[i])

                # tracking as usual until cross/pass the entry line
                ot[i].set_last_state(TrackState_.Tracked)
            else:
                if  t_last_state == TrackState_.Tracked:
                    # Enter event condition, transition Tracked to In
                    # saving to db
                    self.save_2_db(ot[i])
                    ot[i].set_last_state(TrackState_.In)
                    if self.save_patch_dir != None:
                        self.save_patch(f"{self.save_patch_dir}/sample_patch/{t_last_id} in",patches[i])
                elif t_last_state == TrackState_.Matching:
                    # matching db until pass the entry line again
                    is_match, registered_object = self.matching_db(feat)
                    if is_match and not self.db[registered_object.get_id()]["being_used"]:
                        ot[i].set_id(registered_object.get_id())
                        self.db[registered_object.get_id()]["being_used"] = True
                        
                        if t_last_id in self.db.keys():
                            self.db[t_last_id]["being_used"] = False
                    else:
                        # tracking as usual with current id
                        pass
                else:
                    # tracking as usual
                    pass
            

    def in_bounds(self, centroid):
        # check centroid to both line x coordinates 
        # print(centroid)
        # print(self.top_line_points)
        # print(self.bottom_line_points)
        # print()
        # print("z", (-centroid[1] <= self.top_line_points[1][1] and -centroid[1] > self.bottom_line_points[1][1]))
        if (centroid[1] >= self.top_line_points[1][1] and centroid[1] <= self.bottom_line_points[1][1]):
            return True
        else:
            return False  
                 
    def is_passed(self, distance, centroid):
        # First ssumption the location enter on the left side of the picture
        # The user can setting it on the config
        # print("c:", centroid)
        # print(self.in_bounds(centroid))
        if self.entry_area_position == "left":
            if distance >= self.distance_treshold:
                if self.gradient == "positive":
                    if self.in_bounds(centroid):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                if self.gradient == "positive":
                    return False
                else:
                    if self.in_bounds(centroid):
                        return True
                    else:
                        return False
        elif self.entry_area_position == "right":
            if distance < self.distance_treshold:
                if self.gradient == "positive":
                    if self.in_bounds(centroid):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                if self.gradient == "positive":
                    return False
                else:
                    if self.in_bounds(centroid):
                        return True
                    else:
                        return False
        # elif self.entry_area_position == "above":
        #     if distance >= self.distance_treshold:
        #         if self.gradient == "positive":
        #             return True
        #         else:
        #             return False
        #     else:
        #         if self.gradient == "positive":
        #             return False
        #         else:
        #             return True
        # elif self.entry_area_position == "below":
        #     if distance < self.distance_treshold:
        #         if self.gradient == "positive":
        #             return True
        #         else:
        #             return False
        #     else:
        #         if self.gradient == "positive":
        #             return False
        #         else:
        #             return True

    def matching_db(self, feat_):
        # print("\nMATCHING DB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log_event("matching_db")
        ids = list(self.db.keys())
        # print("data in database: ", ids)

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
        # ds = distance.cdist(
        #             query,
        #             gallery, 
        #             "sqeuclidean"
        #         )[-1]
        ds = distance.cdist(
                    query,
                    gallery, 
                    "cosine"
                )[-1]
        ds = ds.tolist()
        print("\tSCORE: ", ds)
        match = ids[self.argmin(ds)]
        print(min(ds), "-", ids[self.argmin(ds)])
        
        if  min(ds) <= self.feat_match_thresh:
            # print(ds) 
            self.log_event("match")
            self.system_recorder["frame_indices"].append(self.frame_id)
            # print("MATCHED ID: ", match)
            o = self.db[match]["registered_object"]
            # del self.db["match"]
            return True, o
        else:
            return False, None  

    def save_2_db(self, t):
        # saving feat to db
        # print(t.get_id(), " SAVE FITUR 2 DB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log_event("saving_db")
        self.db[t.get_id()] = {
            "registered_object": t,
            "being_used": False
        }
        self.set_EER_recorder(t.get_id(), "In", self.get_local_time())
    
    def argmin(self, lst):
      return lst.index(min(lst))
    
    def argmax(self, lst):
      return lst.index(max(lst))
  
    def get_local_time(self):
      # get local time in UTC format
      return time.asctime(time.gmtime())

    def get_total_person(self):
        if len(self.system_recorder["total_person_in_frame"]) == 0:
            return 0
        else:
            return self.system_recorder["total_person_in_frame"][-1]
    
    def set_EER_recorder(self, id, event, UTC_time):
        self.EER_recorder.append(
            {   
                'id': id,
                'event': event,
                'UTC_time': UTC_time
            }
        )
        
    def get_EER_recorder(self):
        return self.EER_recorder
    
    def generate_EER_recorder_csv(self, fname):
        fname = fname+".csv"
        with open(fname, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = ['id', 'event', 'UTC_time'])
            writer.writeheader()
            writer.writerows(self.get_EER_recorder())
        print("=====EER recorder generated=====")
    
    def log_event(self, event):
        if event not in self.system_recorder.keys():
            self.system_recorder[event] = 1
        else:    
            self.system_recorder[event] += 1

    def log_report(self):
        print("\n=====REPORT EVENT=====")
        print(self.system_recorder)
    
    def sample_db(self):
        print("\n===== SAMPLE DATABASE=====")
        print(self.db)
        for i in self.db.keys():
            print("feat shape: ", self.db[i]["registered_object"].get_feat().shape)
    
    def log_output(self):
        print(f"ID to evaluate (Exit condition):{len(self.output)}\n", self.output)
        print(self.get_EER_recorder())
        # min_total = min(self.system_recorder["total_person_in_frame"])
        max_total = max(self.system_recorder["total_person_in_frame"])
        print(f"\nmax person in frame: {max_total} " )