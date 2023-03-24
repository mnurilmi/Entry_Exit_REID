"""
Description:
    Class: Entry_Exit_REID
    This class is used for entry exit reid implementation.
    The implementation combains yolov7 (detector), bytetrack
    (tracker), and OsNet reid model to solve the problem.
    This class will manage each track ids when a person enter 
    or exit a entry-exit area
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
        entry_area_config,        
        feat_extractor,
        feat_match_thresh,
        save_patch_dir = None,
        feat_history = 5
        ):
        self.frame_id = 0
        self.last_id = 0

        self.db = {}
        self.output = []
        self.system_recorder = {
            "total_person_in_frame": [],
            "saving_db":0,
            "matching_db":0,
            "match":0,
            "frame_indices":[]
        }
        self.EER_recorder = []
        self.contours = self.get_contours(entry_area_config)
        
        # Reid model inisialization        
        print("CUDA is active?: ", torch.cuda.is_available())
        self.feat_extractor = feat_extractor
        self.feat_match_thresh = feat_match_thresh 
        self.feat_history = feat_history
        
        if save_patch_dir != None:
            self.save_patch_dir = save_patch_dir
            os.mkdir(str(save_patch_dir)+"/sample_patch")
            
        print("Feature matching treshold:", self.feat_match_thresh)
        print("===Entry Exit REID ready!===")
        
    def next_id(self):
        self.last_id += 1
        return self.last_id

    def get_contours(self, config):
        self.img_h = config["img_h"]
        self.img_w = config["img_w"]
        src = np.zeros((config["img_h"], config["img_w"]), dtype=np.uint8) 
        cv2.line(src, config["points"][0], config["points"][1],(255), 3)
        cv2.line(src, config["points"][1], config["points"][2],(255), 3)
        cv2.line(src, config["points"][2], config["points"][3],(255), 3)
        cv2.line(src, config["points"][3], config["points"][0],(255), 3)
        contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # cv2.imwrite(f"a.jpg", src)
        return contours
        
    def update(self, ot, im0):
        """
        Steps:
        - bbox centroid calculation and feature extraction
        - Entry Area configuration to determine a centroid in entry area or not
        - Assigning ID based on state each tracks
        """
        self.frame_id +=1
        self.system_recorder["total_person_in_frame"].append(len(ot))
        if len(ot) == 0:
            return []
        
        im1 = im0.copy()
        centroids, patches = self.get_centroids_and_patches(ot, im1)    # matrix operation and iteration process
        feats = self.extract_feature(patches)                           # One inference process (many objects)
        feats_np = feats.cpu().data.numpy()
        self.set_ids(ot, feats_np, patches, centroids)                # iteration process
        return ot

    def get_centroids_and_patches(self, ot, im1):
        """Generating Centroids and Patches

        Args:
            ot (object): online tracks from tracker
            im1 (numpy array): current frame image

        Returns:
            centroids, patches: list of centroids, list of patches
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

        # # Middle centroid (center of bbox)
        # Mx = np.array([[1/2],[0],[1/2],[0]]) # left,right
        # My = np.array([[0],[0],[0],[1]])     # top bottom

        Cx = np.dot(tlbrs_np, Mx).astype(int)
        Cy = np.dot(tlbrs_np, My).astype(int)

        Cx= Cx.flatten()
        Cy = Cy.flatten()
        centroids = np.transpose(np.array([Cx, Cy]))
        return centroids, patches
    
    def save_patch(self, fname, patch):
        cv2.imwrite(f"{fname}.jpg", patch)

    def extract_patch(self, img, tlbr, target_size = (128, 256)):
        
        tlbr = np.array(tlbr).astype(int)
        tlbr[0] = (max(0, tlbr[0]))
        tlbr[1] = (max(0, tlbr[1]))
        tlbr[2] = (min(img.shape[1] - 1, tlbr[2])) # relative to W
        tlbr[3] = (min(img.shape[0] - 1, tlbr[3])) # relative to H

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
        features = self.feat_extractor(patches)
        if torch.cuda.is_available():
                torch.cuda.synchronize()
        return features

    def update_track(self, index, ot, feats_np, centroids, bias = 30):   
        """Track update

        Args:
            index (int): index of each online tracks
            ot (object): online tracks from tracker
            feats_np (numpy array): feature of each tracks
            centroids (list): list of centroid tuple (x, y) in the image
            bias (int, optional): Bias to centroid if it beyond the image. Defaults to 30.
        """
        if centroids[index][0] > self.img_w - bias:
            centroids[index][0] = self.img_w -bias
        if centroids[index][1] > self.img_h - bias:
            centroids[index][1] = self.img_h - bias
        ot[index].set_centroid((centroids[index][0],centroids[index][1]))
        ot[index].update_feat(feats_np[index])
        # print(type(ot[index].get_feat()))

    def set_ids(self, ot, feats_np, patches, centroids):
        """Set IDs (ID Assigner)

        Args:
            - ot (list of object): list of online tracks
            - feats_np (numpy array): features from previous reid model inference process
            - centroids (list): centroid position of each online tracks
        
        Tasks:
            - update online track attributes
            - set each online track id's
        """
        # print("===set ids===   ", len(ot), "tracks online")
        for i in range(len(ot)):
            # update current track attribute
            self.update_track(i, ot, feats_np, centroids)
            
            # procedure for assigning ID of each tracks
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
        t_last_id = ot[i].get_id()                          # track last ID
        t_last_state = ot[i].get_last_state()               # track last state
        is_passed = self.is_passed(ot[i].get_centroid())    # whether track passed the entry area or not
        feat = ot[i].get_feat()[-1]                             # track feature
        # print(len(ot[i].get_feat()))
        if t_last_id == -1:
            if not is_passed:
                # set new id, last state unchanged (tracked)
                ot[i].set_id(self.next_id())
            else:
                ot[i].set_last_state(TrackState_.Matching)
                # matching db
                ot[i] = self.matching_db(feat, ot[i])
                # if is_match and not self.db[registered_object.get_id()]["being_used"]:
                # if is_match:
                #     ot[i].set_id(match_id)
                #     ot[i].set_val_counts(match_id)
                    # self.db[registered_object.get_id()]["being_used"] = True
                # else:
                    # set new id and start tracking as usual
                    # ot[i].set_id(self.next_id())
                # ot[i].set_id_validation(ot[i].get_id())
        else:
            if not is_passed:
                if t_last_state == TrackState_.Matching:
                    # Exit event condition, transition Matching to Tracked; save result
                    # ot[i].set_id(ot[i].get_valid_id())
                    # ot[i].set_id_validation = {}
                    valid_id, count = ot[i].get_valid_val_counts()
                    print("masuk:", valid_id)
                    if valid_id != None and count > 5:
                        ot[i].set_id(valid_id)
                        
                    self.output.append(ot[i].get_id())
                    self.set_EER_recorder(t_last_id, "Exit", self.get_local_time())
                    if self.save_patch_dir != None:
                        self.save_patch(f"{self.save_patch_dir}/sample_patch/{t_last_id} exit",patches[i])

                # tracking as usual until cross/pass the entry area
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
                    # matching db until pass the entry area again
                    ot[i] = self.matching_db(feat, ot[i])
                    # print(self.db)
                    # if is_match and not self.db[registered_object.get_id()]["being_used"]:
                    # if is_match:
                    #     ot[i].set_id(match_id)
                    #     ot[i].set_val_counts(match_id)
                        # self.db[registered_object.get_id()]["being_used"] = True
                        
                        # if t_last_id in self.db.keys():
                        #     self.db[t_last_id]["being_used"] = False
                        # if t_last_id != registered_object.get_id():
                        #     ot[i].reset_id_validation(t_last_id)
                    # else:
                        # tracking as usual with current id
                        # pass
                    # ot[i].set_id_validation(ot[i].get_id())
                else:
                    # tracking as usual
                    pass
        # print(ot[i].get_id_validation())
        # print("v:",ot[i].get_valid_id())
        print(ot[i].get_val_counts())
                
            
                    
    def is_passed(self, centroid):
        in_entry_area = cv2.pointPolygonTest(self.contours[0], (int(centroid[0]), int(centroid[1])), True)
        if in_entry_area < 0:
            # if negative it means the centroid is outside the entry area
            return False
        else:
            return True

    def matching_db(self, feat_, ot):
        self.log_event("matching_db")
        # ids = list(self.db.keys())
        # print("data in database: ", ids)  

        # if still empty
        val_counts = ot.get_val_counts()
        if not bool(val_counts):
            # print("masuk")
            val_idx = 0
            ot, match = self.is_query_match_gallery(feat_, val_idx, ot)
            if match != None:
                ot.set_val_counts(match)
            else:
                ot.set_id(self.next_id())
        else:
            # print("masuk2")
            match_ids = []
            for key, val_idx in val_counts.items(): 
                print(key)
                ot, match = self.is_query_match_gallery(feat_, val_idx, ot)
                match_ids.append(match)
                
            for match in match_ids:
                if match != None:
                    ot.set_val_counts(match)
                else:
                    ot.set_id(self.next_id())

        return ot
            
            
    def is_query_match_gallery(self, feat_, val_idx, ot, metric = "cosine"):
        query = np.array([feat_])
        gallery = []
        gallery_ids = []
        # print(self.db.keys())
        print("v:", val_idx)
        for id_ in self.db.keys():
            feats = np.array(self.db[id_]["registered_object"].get_feat())
            # print(feats.shape)
            if  val_idx < feats.shape[0]:
                gallery_ids.append(id_)
                gallery.append(feats[val_idx])

        if len(gallery_ids) == 0:
            return ot, None
            
        query = np.array(query)
        gallery = np.array(gallery)
        # print(query.shape)
        # print(gallery.shape)
        # Similarity calculation (default cosine)
        ds = distance.cdist(
                    query,
                    gallery, 
                    metric
                )[-1]
        ds = ds.tolist()
        match = gallery_ids[self.argmin(ds)] # get the gallery ID that most similar to the query

        print("\tSCORE: ", ds)
        print(min(ds), "-", gallery_ids[self.argmin(ds)])

        if  min(ds) <= self.feat_match_thresh:
            self.log_event("match")
            self.system_recorder["frame_indices"].append(self.frame_id)
            ot.set_id(match)
            return ot, match
        else:
            return ot, None


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
        max_total = max(self.system_recorder["total_person_in_frame"])
        print(f"\nmax person in frame: {max_total} " )