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

from entry_exit_reid.entry_exit_reid import Entry_Exit_REID
import json

def read_config(config):
    # Opening JSON file
    with open(config, 'r') as openfile:
        # Reading from json file
        j = json.load(openfile)
    return j

def get_entry_line_points(j):
    return [float(j["x1"]), float(j["x2"])], [float(j["y1"]), float(j["y2"])]

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

    #Model Loading
    model = attempt_load(
        opt.yolo_weight,
        map_location = device
        )
    stride = int(model.stride.max())
    # print(stride)
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
    
    if half:
        model.half()

    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Tracker Inisialization
    tracker = BYTETracker(
        opt,
        frame_rate = 30.0
    )

    jfile = read_config(opt.entry_line_config)
    x, y = get_entry_line_points(jfile)
    # ID Assigner Inisialization

    if not opt.without_EER:
        """
            https://github.com/KaiyangZhou/deep-person-reid
            reid model type: 
                        'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 
                        'osnet_ibn_x1_0', 
                        'osnet_ain_x1_0'
        """ 
        EER = Entry_Exit_REID(
            entry_line_config = jfile,
            reid_model_path = opt.osnet_weight,
            reid_model_name = "osnet_x1_0"
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
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print(img.shape)
        print("\n=========DETEKSI========== ")
        # Model inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(
                img,
                augment = opt.augment)[0]

        # NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes = opt.classes,
            agnostic = opt.agnostic_nms
        )
        print(pred)

        # Tracking
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        
        results = []
        det = pred[0]
        detections = []
        confs = []
        online_tlwhs = []
        online_tid = []
        online_tlbr = []

        if len(det):
            boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
            boxes = boxes.cpu().numpy()
            detections = det.cpu().numpy()
            detections[:, :4] = boxes
            confs = det[:, 4]

        online_targets = tracker.update(
            detections
        )

        if not opt.without_EER:
            online_targets= EER.update(
                online_targets,
                im0
            )

        for i in range(0, len(online_targets)):
            tlwh = online_targets[i].tlwh
            tlbr = online_targets[i].tlbr

            if tlwh[2] * tlwh[3] > opt.min_box_area:
                conf = confs[i]
                if not opt.without_EER:
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
                    # visualize entry line
                    cv2.line(im0,(int(x[0]),int(y[0])),(int(x[1]),int(y[1])),(0,255,0),1)

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

    if not opt.without_EER:
        EER.log_report()
        EER.sample_db()
        EER.log_output()

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
    

    # Entry Exit REID Parser
    parser.add_argument('--without_EER', action='store_true', help='without id assigner')
    parser.add_argument('--entry_line_config', type=str, default='test/testing_vid1/testing_vid1.json', help='entry line config path')
    # parser.add_argument('--entry_area_position', type=str, default='right', help='entry area position POV (right/left)')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ["yolov7.pt"]:
                main()
                strip_optimizer(opt.weights)
        else:
            main()