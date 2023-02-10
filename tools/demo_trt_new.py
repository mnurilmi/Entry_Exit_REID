"""
author: muhammad nur ilmi
Adapted from:
    https://github.com/ifzhang/ByteTrack/blob/main/tools/demo_track.py
    https://github.com/NirAharon/BoT-SORT/blob/main/tools/mc_demo_yolov7.py
"""
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, './yolov7')
sys.path.append(".")

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, TracedModel

from tracker.byte_tracker import BYTETracker
from timer.timer import Timer

from id_assigner.id_assigner import ID_Assigner
import json

import tensorrt as trt
import numpy as np
from collections import OrderedDict,namedtuple
from yolov7.utils.datasets import letterbox

def read_config(config):
    # Opening JSON file
    with open(config, 'r') as openfile:
        # Reading from json file
        j = json.load(openfile)
    return j

def get_entry_line_points(j):
    return [float(j["x1"]), float(j["x2"])], [float(j["y1"]), float(j["y2"])]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

# def postprocess(boxes,r,dwdh):
#     dwdh = torch.tensor(dwdh*2).to(boxes.device)
#     boxes -= dwdh
#     boxes /= r
#     return boxes

def preprocess(image, device):
    print(image.shape)
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im = torch.from_numpy(im).to(device)
    im/=255
    return im, ratio, dwdh

# Start Code
def main():
    timer_init = Timer()
    timer_init.tic()

    save_img = not opt.nosave and not opt.source.endswith(".txt")
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or opt.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok = opt.exist_ok))
    (save_dir / "labels" if opt.save_txt else save_dir).mkdir(parents = True, exist_ok = True)

    #Inisialization
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"

    # Model Loading (TensorRT)
    w = opt.yolo_weight[-1]
    # print(w)
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

    stride = 32
    img_size = check_img_size(
        opt.img_size,
        s = stride
        )

    if opt.trace:
        model = TracedModel(
            model,
            device,
            opt.img_size
        )

    names = ["person"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Tracker Inisialization
    tracker = BYTETracker(
        opt,
        frame_rate = 30.0
    )

    jfile = read_config(opt.entry_line_config)
    x, y = get_entry_line_points(jfile)
    # ID Assigner Inisialization
    if not opt.without_id_assigner:
        """
            https://github.com/KaiyangZhou/deep-person-reid
            code reference: reid model type: 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0'
        """ 
        id_assigner = ID_Assigner(
            entry_line_config = jfile,
            reid_model_path = opt.osnet_weight,
            reid_model_name = "osnet_ibn_x1_0"
        )
    
    # Start Detection and Tracking
    # if device.type!= "cpu":
    #     model(torch.zeros(1, 3, img_size).to(device).type_as(next(model.parameters())))

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=img_size, stride=stride)
    else:
        dataset = LoadImages(opt.source, img_size=img_size, stride=stride)
        
    timer_init.toc()
    print("Inizialization Time: ", timer_init.average_time, " s\n\n")

    frame_id, min_FPS, max_FPS = 1, 999, -1
    timer = Timer()
    t0 = time.time()

    for path, img, im0s, vid_cap, in dataset:
        timer.tic()

        img_ = img.copy().transpose(1,2,0)
        im, ratio, dwdh = preprocess(img_, device)
        binding_addrs['images'] = int(im.data_ptr())
        context.execute_v2(list(binding_addrs.values()))

        # Model inference
        nums = bindings['num_dets'].data
        boxes = bindings['det_boxes'].data
        scores = bindings['det_scores'].data
        classes = bindings['det_classes'].data
        boxes = boxes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]

        scores = torch.transpose(scores[np.newaxis, :], 0, 1)
        classes = torch.transpose(classes[np.newaxis, :], 0, 1)
        pred =  torch.cat((boxes, scores, classes), 1)
        # pred = [torch.tensor(pred)]

        print("\n=========DETEKSI========== ")
        # print(pred)
        # print(boxes)
        # print(scores)

        # Tracking
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        
        # visualize entry line
        cv2.line(im0,(int(x[0]),int(y[0])),(int(x[1]),int(y[1])),(0,255,0),1)

        det = pred
        print(pred)
        print(pred.size())
        print(len(pred))
        print(len(det))
        # print(len(det))
        det[:, 1] = det[:, 1]-dwdh[1]
        det[:, 3] = det[:, 3]-dwdh[1]
        det[:, 0] = det[:, 0]-dwdh[0]
        det[:, 2] = det[:, 2]-dwdh[0]
        results = []
        
        detections = []
        confs = []
        online_tlwhs = []
        online_tid = []
        online_tlbr = []

        if  det.size()[0] != 0:
            print(img.shape)
            boxes = scale_coords(img.shape[1:], det[:, :4], im0.shape)
            boxes = boxes.cpu().numpy()
            detections = det.cpu().numpy()
            detections[:, :4] = boxes
            confs = det[:, 4]

        online_targets = tracker.update(
            detections
        )

        if not opt.without_id_assigner:
            online_targets= id_assigner.update(
                online_targets,
                im0
            )

        for i in range(0, len(online_targets)):
            tlwh = online_targets[i].tlwh
            tlbr = online_targets[i].tlbr

            if tlwh[2] * tlwh[3] > opt.min_box_area:
                conf = confs[i]
                if not opt.without_id_assigner:
                    tid = online_targets[i].get_id()
                    td = online_targets[i].get_distance()  
                    tls = online_targets[i].decode_state(online_targets[i].get_last_state())
                    tc = online_targets[i].get_centroid()
                    label = '{0:}-{1:.2f}-{2:.2f}-{3:}'.format(tid, conf, td, tls)
                    cv2.circle(im0, tc, radius=2, color=(0, 0, 255), thickness=2)
                else:
                    tid = online_targets[i].track_id
                    label = '{0:}-{1:.2f}'.format(tid, conf)
                
                if save_img or view_img:  # Add bbox to image
                    plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=1)

                online_tlwhs.append(tlwh)
                online_tid.append(tid)
                online_tlbr.append(tlbr)
                #save results
                results.append(
                    f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{online_targets[i].score:.2f},-1,-1,-1\n"
                )      
        timer.toc()
        FPS = int(1. / max(1e-5, timer.average_time))

        if FPS < min_FPS and not FPS == 0:
            min_FPS = FPS 
        if FPS > max_FPS:
            max_FPS = FPS
        print(f"FPS:{FPS}; MIN: {min_FPS}; MAX: {max_FPS}")
        cv2.putText(im0, f"Frame: {frame_id}; FPS:{FPS}; MIN: {min_FPS}; MAX: {max_FPS}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
                
        p = Path(p)
        save_path = str(save_dir / p.name)
        # Stream results
        if opt.view_img:
            cv2.imshow('ByteTrack', im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        frame_id += 1
        
    if opt.save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if not opt.without_id_assigner:
        id_assigner.log_report()
        id_assigner.sample_db()
        id_assigner.log_output()

    print(f'Done. ({time.time() - t0:.3f}s)\t\tfolder name: {save_dir}')
    print(f"total_frame/total_time: {frame_id/(time.time() - t0)} \t min FPS: {min_FPS} \t max FPS: {max_FPS}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # YOLOv7 parser
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--yolo_weight', nargs='+', type=str, help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # ByteTrack Parser
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=20, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # REID Model Parser (OSNET model)
    parser.add_argument('--osnet_weight', type=str, help='model.pt path')
    

    # ID_Assigner Parser
    parser.add_argument('--without_id_assigner', action='store_true', help='without id assigner')
    parser.add_argument('--entry_line_config', type=str, default='test/testing_vid1/testing_vid1.json', help='entry line config path')
    # parser.add_argument('--entry_area_position', type=str, default='right', help='entry area position POV (right/left)')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        main()