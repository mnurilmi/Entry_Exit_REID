<!-- This content will not appear in the rendered Markdown -->
# Entry-Exit Monitoring System
Entry-Exit Monitoring System is a part of intelligent surveillance systems. This issue aims to monitor a person’s activity using wide-ranging surveillance cameras in both public and private spaces, such as restrooms, changing rooms, and baby care areas. This study presents the development and evaluation of a single-camera entry-exit monitoring system equipped with person re-identification to maintain a consistent identity while the persons going in and out of an area. The system employs a comprehensive pipeline, including YOLOv7 (original and tiny versions), ByteTrack, and OSNet-x1 as the person detector, tracker, and re-identifier.
The entry-exit event detection component using the YOLOv7+ByteTrack achieved the highest accuracy, scoring 1 for both entry and exit event detection. Conversely, when using YOLOv7-tiny+ByteTrack, the system showed slightly lower performance with the entry event detection score of 0.975. The best person re-identification performance was observed with YOLOv7+ByteTrack+OSNet-x1, achieving an f1-score of 0.864 and accuracy of 0.826, whereas YOLOv7-tiny+ByteTrack+OSNet-x1 exhibited reduced values of 0.826 and 0.780, respectively. Evaluation of FPS on Google Colab and Nvidia Jetson Xavier AGX revealed peak averages of 26.7 FPS and 13 FPS when using YOLOv7-tiny and 21.8 FPS and 10.6 FPS when using YOLOv7, respectively.

# System Design

<picture>
  <source media="(prefers-color-scheme: dark)"> 
  <img alt="system diagram" src="assets/images/System_Diagram.png">
</picture>
The system design can be seen in the picture above. The system consists of three subsystems, person detection, person tracking, and person re-identification. Each subsystem uses various models such as YOLOv7 (original and tiny), ByteTrack, and OSNet-x1. Person detection and person tracking are used to detect, track, and provide an initial ID to individuals in the frame. Meanwhile, the person re-identification subsystem is used to store descriptive information from individuals and recognize the same person when they enter or leave the area monitored by surveillance cameras. The model performance in the person detection and person re-identification subsystems is evaluated through the best hyperparameter tuning, while in the person tracking task, no training process is carried out because it uses a model-free tracking to reduce computational cost. For the Entry-Exit Re-ID system algorithm, the track results from the tracker will be the input for the system. The system will extract the centroid points (state determinator) and features of each track. Then, the tracklet attributes will be iteratively updated and each attribute will be assigned a corresponding ID based on feature matching (cosine similarity). The updated track will be returned from the system that consists valid ID.

# Running
An example running the system on loc1_1 video with YOLOv7+OSnet model
```
python3 "tools/demo.py" \
--name "testing" \
--yolo_weight "models/yolo_15k_300_rep.pt" \
--reid_model_weight "models/rf_120.pth.tar-120" \
--img-size 640 \
--conf-thres 0.65 \
--agnostic-nms \
--feature_match_thresh 0.2 \
--match_count_thresh 3 \
--source "testing_datas/loc1_1.mp4" \
--entry_area_config "testing_datas/loc1_1.json" \
# --without_EER
```

# Deployment
```
// put docker file outside of the "entry_exit_reid" folder
// go to the directory that contains "DockerFile"

sudo docker build .

// after image has been build, check the image ID with command below
sudo docker images

//run the docker image, enjoy the environment
sudo docker run -it --runtime nvidia --network host <image ID>
```


# Samples
These two pictures below are the result samples of the system that running with YOLOv7+ByteTrack+OSNet-x1
## True Positive and True Negative Samples
<picture>
  <source media="(prefers-color-scheme: dark)"> 
  <img alt="TP TN samples" src="assets/images/Results_TP_TN.png">
</picture>

## False Positive and False Negative Samples

<picture>
  <source media="(prefers-color-scheme: dark)"> 
  <img alt="FP FN samples" src="assets/images/Results_FP_FN.png">
</picture>

# Acknowledgement
This work has been supported by DIKE lab, FMIPA (Faculty of Natural Science and Mathematics), Gadjah Mada University.

# Citations

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```
```
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
```
@article{torchreid,
  title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
  author={Zhou, Kaiyang and Xiang, Tao},
  journal={arXiv preprint arXiv:1910.10093},
  year={2019}
}

@inproceedings{zhou2019osnet,
  title={Omni-Scale Feature Learning for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  booktitle={ICCV},
  year={2019}
}
```