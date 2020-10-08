# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json

from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from ffmpeg_main import pre_process
from np_to_json import convert_json

# constants
WINDOW_NAME = "COCO detections"

basepath = f'/app'

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.WEIGHTS = f'{basepath}/model_final_cafdb1.pkl'

    # cfg.MODEL.WEIGHTS = '/data1/code_base/mnt_data/ODbatch/model_final_cafdb1.pkl'

    # Set score_threshold for builtin models
    #confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5 
    # confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        # default = '/app/docker_files/detectron2/configs/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml',

        default = '/app/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml',
        # default = '/data1/code_base/mnt_data/visd2/d2sourcecode/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml',
        # default = '/app/docker_files/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml',

        # default = f'{basepath}/panoptic_fpn_R_101_3x.yaml',

        metavar="FILE",

        help="path to config file",
    )

    return parser

def load_model():
    args, unknown = get_parser().parse_known_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))    
    cfg = setup_cfg(args)
    model = VisualizationDemo(cfg)
    return model


def visual_od(video_id=None, model = None):

    video = cv2.VideoCapture(f'{basepath}/{video_id}.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fname = f'{basepath}/{video_id}out.webm'
    # assert not os.path.isfile(output_fname), output_fname
    output_file = cv2.VideoWriter(
        filename=output_fname,
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc('V','P','8','0'),
        # fourcc = -1,
        # fourcc = 0x00000021,
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )
    frames = []
    for vis_frame, frame in tqdm.tqdm(model.run_on_video(video), total=num_frames):
        output_file.write(vis_frame)
        frames.append(frame)

    video.release()
    output_file.release()
    cv2.destroyAllWindows()

    data = convert_json(video_id=video_id, basepath=basepath, width=width, height=height, \
    frames_per_second= frames_per_second, num_frames=num_frames, all_preds=frames)
    with open(f'{basepath}/{video_id}.json', 'w') as f:
        json.dump(data,f)

    # print(out)


# visual_od(video_id='15341_', model=load_model())
# pre_process(video_id='15341')