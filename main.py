# --------------------------------------------------------
# ZBS: Zero-shot Background Subtraction
# Written by Yongqi An
# --------------------------------------------------------

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import warnings
import cv2
import tqdm
import sys
import mss
import torch
import json
from scipy.signal import medfilt

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures.instances import Instances

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detectron2.utils.visualizer import VisImage
from detic.config import add_detic_config
from sort_zbs import *
from collections import defaultdict
from vibe_gpu import ViBeGPU
from video_roi import select_roi


from detic.predictor import VisualizationDemo

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "ZBS"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--parallel", action='store_true')
    parser.add_argument("--visual", action='store_true')
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--noiof", action='store_true', help="Don't use the IoF.")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="The categories to the custom vocabulary file.",
    )
    parser.add_argument(
        "--white_list",
        default="",
        help="The categories to be excluded from the vocabulary.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--delta_conf",
        type=float,
        default=0.2,
        help="the difference between stage3 and stage2(stage3>stage2)",
    )
    parser.add_argument(
        "--fore_threshold",
        type=float,
        default=0.8,
        help="Maximum score for foreground selection to not be shown",
    )
    parser.add_argument(
        "--move_threshold",
        type=float,
        default=0.5,
        help="Minimum score for background modeling to be added into instance-level background model",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    
    # Sort Parameters
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=20)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=5)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.4)
    
    # ViBe Parameters
    parser.add_argument("--pixel", action='store_true', help="Add pixel-level background as a supplementary.")    
    parser.add_argument("--num_sam", type=int, default=30)
    parser.add_argument("--min_match", type=int, default=2)
    parser.add_argument("--radiu", type=int, default=10)
    parser.add_argument("--rand_sum", type=int, default=16)
    parser.add_argument("--fake_thres", type=int, default=50)
    
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def IoU(box1: torch.tensor, box2: torch.tensor) -> float:
    """Calculate the IoU (Intersection over Union) of box1 and box2.
    """
    box_1 = box1.tolist()
    box_2 = box2.tolist()
    ax = max(box_1[0], box_2[0])
    ay = max(box_1[1], box_2[1])
    bx = min(box_1[2], box_2[2])
    by = min(box_1[3], box_2[3])
    
    area_N = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_M = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    
    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h
    
    return area_X / (area_N + area_M - area_X)


def IoF(box1: torch.tensor, box2: torch.tensor) -> float:
    """Calculate the IoF (Intersection over Foreground) of box1 and box2. Among them, box1 is the foreground.
    """
    box_1 = box1.tolist()
    box_2 = box2.tolist()
    ax = max(box_1[0], box_2[0])
    ay = max(box_1[1], box_2[1])
    bx = min(box_1[2], box_2[2])
    by = min(box_1[3], box_2[3])
    
    area_N = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    
    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h
    
    return area_X / area_N


def cal_maxIoU(det_box: torch.tensor, target_boxes: torch.tensor) -> float:
    """Given an input of candidate bounding boxes and target bounding boxes, 
    compute the IoU (Intersection over Union) for each pair, and return the maximum IoU.

    Args:
        det_box (torch.tensor): the all-instance detection box of the current frame.
        target_boxes (torch.tensor): the same class of rec_box in background model.
    """
    max_IoU = 0
    for box in target_boxes:
        max_IoU = max(max_IoU, IoU(det_box, box))
    return max_IoU

def foreground_selection(rec_box: torch.tensor, bg_boxes: torch.tensor, fore_threshold: float) -> bool:
    """Select the moving foreground by comparing with the instance-level background model.

    Args:
        rec_box (torch.tensor): the all-instance detection box of the current frame.
        bg_boxes (torch.tensor): the same class of rec_box in background model.
        fore_threshold (float): the threshold in foreground instance selection.

    Returns:
        bool: True means this instance should be a foreground. Otherwise, it is a background.
    """
    IoU_list = []
    for bg_box in bg_boxes:
        IoU_list.append(IoU(rec_box, bg_box))
    # IoF uses a larger threshold than IoU, because the foreground less than the union of foreground and background. 
    # Rule 2 introduced in the paper.
    if np.max(IoU_list) < fore_threshold and IoF(rec_box, bg_boxes[np.argmax(IoU_list)]) < 0.5 + fore_threshold / 2:
        return True
    return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    
    # Build Model
    cfg = setup_cfg(args)
    if not args.pixel:
        demo = VisualizationDemo(cfg, args, parallel=args.parallel)
    
    # Build White List
    with open('datasets/metadata/lvis_v1_train_cat_info.json', encoding="utf-8") as f:
        data = json.load(f)
        classes = [data[i].get('name') for i in range(len(data))]
        white_list = [classes.index(cls) for cls in args.white_list.split(',')] if args.white_list else []

    # Create the SORT tracker
    mot_tracker = Sort(max_age=args.max_age, 
                min_hits=args.min_hits,
                iou_threshold=args.iou_threshold)
    total_trks = np.array([]).reshape(0, 7) # an array to record all tracks for background modeling
    moved_ids = set()    # Store the existing motion trajectories' trk_id in [:, 4].
    cls_bg = defaultdict(list)  # an instance-level background model
    gt_flag = True
    init_flag = True
    
    # Video mode
    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_frames = num_frames // 10    # We choose 1/10 frames to initialize the instance-level background model
        basename = os.path.basename(args.video_input)
        
        # Mouse interaction to draw polygons ROI
        _, img = video.read() 
        roi_binary = select_roi(img)  # create a mask
        
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output_dir:
            if os.path.isdir(args.output_dir):
                output_fname = os.path.join(args.output_dir, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output_dir
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for i, frame_det in enumerate(tqdm.tqdm(demo.detic_detect_video(video), total=num_frames)):
            frame = frame_det[0]    # current frame in video
            # ========= Start: all-instance detection =========
            predictions = frame_det[1]  # the all-instance detection of current frame
            # ========= End: all-instance detection =========
            
            # ========= Start: instance-level background modeling =========
            if i <= bg_frames:  # collect data for the instance-level background model
                # all-instance detection results for Sort
                np_box = predictions.pred_boxes.tensor.numpy()
                np_score = predictions.scores.numpy().reshape(-1, 1)
                np_cls = predictions.pred_classes.numpy().reshape(-1, 1)
                dets = np.concatenate((np_box, np_score, np_cls), axis=1)   # (x1, y1, x2, y2, score, cls)
                trackers = mot_tracker.update(dets)     # (x1, y1, x2, y2, trk_id, id_map, cls)
                total_trks = np.concatenate((total_trks,trackers), axis=0) # an array to record all tracks for background modeling
                
            else:
                if gt_flag:   # initialize the instance-level background model
                    total_trk_ids = set(total_trks[:,4].astype('uint16'))
                    for id in total_trk_ids:
                        trks = total_trks[total_trks[:, 4]==id]    # find trks of each id
                        arg_bbox = np.average(trks[:, :4], axis=0) # calculate the arg_box of all boxes of this id
                        
                        IoU_trk = np.round([IoU(bbox, arg_bbox) for bbox in trks[:, :4]], 2)    # calculate the IoU between each box of this id and arg_box
                        kernel = len(IoU_trk) // 9 if (len(IoU_trk) // 9) % 2 else len(IoU_trk) // 9 + 1    # use the 1/9 length as a kernel
                        IoU_trk = medfilt(IoU_trk, kernel)  # median filter, remove outliers
                        if np.min(IoU_trk) >  1 - args.move_threshold:  # min(IoU) means the max movement of this track
                            for cls in set(trks[:, -1]):
                                cls_bg[int(cls)].append(arg_bbox)   # add it into the instance-level background model
                        
                    gt_flag = False   # only initialize once
                    
                    # visualize the instance-level background model
                    if args.visual:
                        boxes, classes = [], []
                        if cls_bg != defaultdict(list):
                            for cls in cls_bg.keys():
                                for box in cls_bg[cls]:
                                    boxes.append(torch.Tensor(box))
                                    classes.append(torch.tensor(cls))
                        img = video.read()[1]
                        instance_bg = Instances(image_size=(width, height), pred_boxes=torch.stack(boxes), pred_classes=torch.stack(classes))
                        background_vis_output = demo.background_vis(image=img, predictions=instance_bg) # use the visualization function of Detic
                        if args.output_dir:
                            background_vis_output.save('/'.join(args.output_dir.split('/')[:-1]) + '/cls_bg.jpg')
                    
                    # Using a larger confidence during foreground selection
                    args.confidence_threshold += args.delta_conf
                    cfg = setup_cfg(args)
                    demo = VisualizationDemo(cfg, args, parallel=args.parallel)
                # ========= End: instance-level background modeling =========
                    
                # ========= Start: foreground instance selection =========
                masks = np.zeros_like(frame[:,:,0])
                for i, cls_id in enumerate(predictions.pred_classes):
                    if cls_id in white_list:    # skip the classes in white_list
                        continue
                    box_ids = torch.nonzero(predictions.pred_classes == cls_id) # find all box_ids for the cls_id of current frame's detection results
                    for box_id in box_ids:
                        rec_mask = predictions.pred_masks[box_id.item()]
                        
                        if cls_bg[int(cls_id)] == []:
                            masks += np.asarray(predictions.pred_masks[i])
                        else:
                            rec_box = predictions.pred_boxes[box_id.item()].tensor[0]
                            bg_boxes = torch.Tensor(cls_bg[int(cls_id)])
                            if foreground_selection(rec_box, bg_boxes, args.fore_threshold):
                                masks += np.asarray(predictions.pred_masks[box_id.item()])
                masks = (masks != 0)
                # ========= End: foreground instance selection =========
                
                # save as a video
                results = masks & roi_binary
                vis_frame = VisImage(results)    
                vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)  
                if args.output_dir:
                    output_file.write(vis_frame)
                else:
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.imshow(basename, vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
        video.release()
        if args.output_dir:
            output_file.release()
        else:
            cv2.destroyAllWindows()

    # Image mode for CDnet 2014 or other image sequences
    elif args.input_dir:
        # Using a pixel-level background model.
        if args.pixel:
            vibe = ViBeGPU(num_sam=args.num_sam, min_match=args.min_match, radiu=args.radiu, rand_sam=args.rand_sum, fake_thres=args.fake_thres)
            flag = 0
            
        # read the ROI
        roi = read_image(args.input_dir + 'ROI.bmp')
        if roi.shape[-1] == 3:
            roi = roi[:, :, 0]
        if roi.sum() == 0:
            roi_binary = (roi == 0)
        else:
            roi_binary = (roi != 0)

        for input_path in sorted(glob.glob(args.input_dir + 'input/*.jpg')):
            print(input_path)
            if args.pixel:
                frame = read_image(input_path, format="BGR")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if flag == 0:
                    vibe.ProcessFirstFrame(gray)
                    flag = 1
                vibe.Update(gray)
                segMat = vibe.getFGMask()
                segMat = segMat.cpu().numpy().astype(np.uint8)
                results = segMat & roi_binary
                results[results != 0] = 255
                
                if args.output_dir:
                    output_dir = '/'.join(input_path.split('/')[:-2]) + args.output_dir
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = output_dir + input_path.split('/')[-1]
                    cv2.imwrite(output_path, results, [int(cv2.IMWRITE_PNG_STRATEGY)])
            
            else:
                split_dir = input_path.split('/')
                split_dir[-2] = 'groundtruth'
                split_dir[-1] = 'gt' + split_dir[-1][2:-3] + 'png'
                
                # ========= Start: all-instance detection =========
                img = read_image(input_path, format="BGR")
                predictions = demo.detic_detect(img)    # all-instance detection
                # ========= End: all-instance detection =========
                     
                gt_img = read_image('/'.join(split_dir))
                # ========= Start: instance-level background modeling =========
                if (gt_img==85).all() or (gt_img==170).all():   # the ground truth frames to initize
                    # all-instance detection results for Sort
                    np_box = predictions.pred_boxes.tensor.numpy()
                    np_score = predictions.scores.numpy().reshape(-1, 1)
                    np_cls = predictions.pred_classes.numpy().reshape(-1, 1)
                    dets = np.concatenate((np_box, np_score, np_cls), axis=1)   # (x1, y1, x2, y2, score, cls)
                    trackers = mot_tracker.update(dets)     # (x1, y1, x2, y2, trk_id, id_map, cls)
                    total_trks = np.concatenate((total_trks,trackers), axis=0) # an array to record all tracks for background modeling

                    if args.noiof:
                        all_trk_ids = set(trackers[:,4].astype('uint16'))   # Set of all track IDs contained in the single image
                        sure_ids = all_trk_ids & moved_ids  # Track IDs that were previously confirmed as foreground, no additional calculations needed
                        unsure_ids = all_trk_ids - sure_ids # Track IDs in the current image that are uncertain
                        for trk_id in unsure_ids:
                            trks = total_trks[total_trks[:, 4]==trk_id, :4]    # Filter data for the specific trk_id
                            delta_trk = np.max(trks, axis=0) - np.min(trks, axis=0) 
                            delta_trk[[0, 2]] /= img.shape[0]
                            delta_trk[[1, 3]] /= img.shape[1]
                            if sum(delta_trk > args.move_threshold):
                                moved_ids.add(trk_id)   # Add to the set of confirmed moving IDs
                else:
                    
                    if gt_flag:   # initialize the instance-level background model
                        total_trk_ids = set(total_trks[:,4].astype('uint16'))
                        if args.noiof:
                            unmoved_ids = total_trk_ids - moved_ids
                            for id in unmoved_ids:
                                trks = total_trks[total_trks[:, 4]==id]    # Filter data for trk_id with no motion
                                bbox = np.average(trks[:, :4], axis=0) # Obtain average bounding box based on the track
                                for cls in set(trks[:, -1]):
                                    cls_bg[int(cls)].append(bbox)   # Build background model based on track category
                        else:
                            for id in total_trk_ids:
                                trks = total_trks[total_trks[:, 4]==id]    # find trks of each id
                                arg_bbox = np.average(trks[:, :4], axis=0) # calculate the arg_box of all boxes of this id
                                
                                IoU_trk = np.round([IoU(bbox, arg_bbox) for bbox in trks[:, :4]], 2)    # calculate the IoU between each box of this id and arg_box
                                kernel = len(IoU_trk) // 9 if (len(IoU_trk) // 9) % 2 else len(IoU_trk) // 9 + 1    # use the 1/9 length as a kernel
                                IoU_trk = medfilt(IoU_trk, kernel)  # median filter, remove outliers
                                if np.min(IoU_trk) >  1 - args.move_threshold:  # min(IoU) means the max movement of this track
                                    for cls in set(trks[:, -1]):
                                        cls_bg[int(cls)].append(arg_bbox)   # add it into the instance-level background model
                            
                        gt_flag = False   # only initialize once
                            
                        # visualize the instance-level background model
                        if args.visual:
                            boxes, classes = [], []
                            if cls_bg != defaultdict(list):
                                for cls in cls_bg.keys():
                                    for box in cls_bg[cls]:
                                        boxes.append(torch.Tensor(box))
                                        classes.append(torch.tensor(cls))
                            instance_bg = Instances(image_size=(roi.shape[0],roi.shape[1]), pred_boxes=torch.stack(boxes), pred_classes=torch.stack(classes))
                            background_vis_output = demo.background_vis(image=img, predictions=instance_bg) # use the visualization function of Detic
                            if args.output_dir:
                                background_vis_output.save('/'.join(input_path.split('/')[:-2]) + f'{args.output_dir[:-1]}.jpg')
                        
                        # Using a larger confidence during foreground selection
                        args.confidence_threshold += args.delta_conf
                        cfg = setup_cfg(args)
                        demo = VisualizationDemo(cfg, args, parallel=args.parallel)
                        # ========= End: instance-level background modeling =========

                    # ========= Start: foreground instance selection =========
                    masks = np.zeros_like(img[:,:,0])
                    for i, cls_id in enumerate(predictions.pred_classes):
                        box_ids = torch.nonzero(predictions.pred_classes == cls_id) # find all box_ids for the cls_id of current frame's detection results
                        for box_id in box_ids:
                            rec_mask = predictions.pred_masks[box_id.item()]
                            if cls_bg[int(cls_id)] == []:
                                masks += np.asarray(predictions.pred_masks[i])
                            else:
                                rec_box = predictions.pred_boxes[box_id.item()].tensor[0]
                                bg_boxes = torch.Tensor(cls_bg[int(cls_id)])
                                if args.noiof:
                                    if cal_maxIoU(rec_box, bg_boxes) < args.fore_threshold:
                                        masks += np.asarray(predictions.pred_masks[box_id.item()])
                                else:
                                    if foreground_selection(rec_box, bg_boxes, args.fore_threshold):
                                        masks += np.asarray(predictions.pred_masks[box_id.item()])
                    masks = (masks != 0)
                    # ========= End: foreground instance selection =========
                    
                    # save as a image
                    results = masks & roi_binary
                    visualized_output = VisImage(results)  
                            
                    if args.output_dir:
                        output_dir = '/'.join(input_path.split('/')[:-2]) + args.output_dir
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = output_dir + input_path.split('/')[-1]
                        visualized_output.save(output_path)            