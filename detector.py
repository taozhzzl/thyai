import matplotlib.pyplot as plt
import hashlib
from PIL import Image
import os
import shutil
import json
from io import BytesIO
import cv2
import numpy as np
import argparse
# this makes our figures bigger
from time import time
import datetime
import torch
from maskrcnn_benchmark.config import cfg
from predictor_xray import XrayDetector


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm' , ".tif"]
def is_image_file(filename):
    """Checks if a file is an image.
      Args:
          filename (string): path to a file
      Returns:
          bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

folder_path = os.getcwd()
output_dir = os.path.join(folder_path,"vis_outputs")
is_save_json = True # 本地是否保存json
marginExpandRatio = 0.5 # 扩充像素大小
isSegmentMask = True #是否是显示分割
output_json_dir = os.path.join(folder_path,"vis_outputs/json_results")
isDrawBbox = False # 是否显示bbox
isShowClassname = False # 是否显示名字和概率



isExists = os.path.exists(output_dir)
if not isExists:
    os.mkdir(output_dir)
# else:
#     shutil.rmtree(output_dir)

## output json formate 输出mask_anotation 能识别并显示的文件
isExists = os.path.exists(output_json_dir)
if is_save_json == True:
    if not isExists:
        os.makedirs(output_json_dir)
    # else:
    #     shutil.rmtree(output_json_dir)
    #     os.rmdir(output_json_dir)

config_file = "./configs/medical_seg/e2e_mask_rcnn_R_50_FPN_1x_1_class.yaml"



# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options, 将下面的cpu换成cuda即可以升级为显卡
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

detector = XrayDetector(
    cfg,
    confidence_threshold=0.5,
    show_mask_heatmaps=False,
    masks_per_dim=2,
    size=(512,512),#(h,w)(288,384)(480,640)
    save_json=is_save_json,
    isDrawBbox=isDrawBbox,
    isSegmentMask=isSegmentMask,
    marginExpandRatio=marginExpandRatio,
    isShowClassname = isShowClassname
)

if not os.path.exists(output_dir + "/none/"):
    os.makedirs(output_dir + "/none/")
if not os.path.exists(output_dir + "/have/"):
    os.makedirs(output_dir + "/have/")

export_truth = []

def detect(image):
    dt_ms = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.%f')

    dt_ms_md5_val = hashlib.md5(dt_ms.encode('utf8')).hexdigest()
    start = time()
    output_name = ""
    output_name_segs = []
    output_name_expand_segs = []
    output_name_boxs = []
    output_name_expandboxs =[]
    output_name_cams = []
    if isSegmentMask:
        predictions, dict_list, boxnum,seg_images,seg_images_expand,seg_images_box,seg_images_expand_box,cam_list = detector.run_on_opencv_image(image)
        # 保存局部结果图，不扩充边缘
        for k,seg in enumerate(seg_images):
            output_name_seg = "have/"+ dt_ms_md5_val+ '_{}.'.format(k) + "jpg"
            output_name_seg = os.path.join(output_dir, '{}'.format(output_name_seg))
            cv2.imwrite(output_name_seg, seg)
            output_name_seg = dt_ms_md5_val+ '_{}'.format(k)
            output_name_segs.append(output_name_seg)
        # 保存局部结果图，扩充边缘
        for k,seg in enumerate(seg_images_expand):
            output_name_expand_seg = "have/"+ dt_ms_md5_val + '_expand_seg_{}.'.format(k) + "jpg"
            output_name_expand_seg = os.path.join(output_dir, '{}'.format(output_name_expand_seg))
            cv2.imwrite(output_name_expand_seg, seg)
            output_name_expand_seg = dt_ms_md5_val + '_expand_seg_{}'.format(k)
            output_name_expand_segs.append(output_name_expand_seg)
        # box保存局部结果图，不扩充边缘
        for k,seg in enumerate(seg_images_box):
            output_name_box = "have/" + dt_ms_md5_val+  '_box_{}.'.format(k) + "jpg"
            output_name_box = os.path.join(output_dir, '{}'.format(output_name_box))
            cv2.imwrite(output_name_box, seg)
            output_name_box =dt_ms_md5_val + '_box_{}'.format(k)
            output_name_boxs.append(output_name_box)
        # box保存局部结果图，扩充边缘
        for k,seg in enumerate(seg_images_expand_box):
            output_name_expandbox = "have/" + dt_ms_md5_val+  '_expandbox_{}.'.format(k) + "jpg"
            output_name_expandbox = os.path.join(output_dir, '{}'.format(output_name_expandbox))
            cv2.imwrite(output_name_expandbox, seg)
            output_name_expandbox = dt_ms_md5_val + '_expandbox_{}'.format(k)
            output_name_expandboxs.append(output_name_expandbox)
        # cam保存局部结果图，扩充边缘
        for k,seg in enumerate(cam_list):
            output_name_cam = "have/" + dt_ms_md5_val+  '_cam_{}.'.format(k) + "jpg"
            output_name_cam = os.path.join(output_dir, '{}'.format(output_name_cam))
            cv2.imwrite(output_name_cam, seg)
            output_name_cam = dt_ms_md5_val + '_cam_{}'.format(k)
            output_name_cams.append(output_name_cam)
    else:
        predictions,dict_list ,boxnum = detector.run_on_opencv_image(image)
    # save result images
    stop = time()
    print(str(stop-start) + "s")


    output_name = "have/"+ dt_ms_md5_val + '.' + "jpg"
    output_name = os.path.join(output_dir, '{}'.format(output_name))
    cv2.imwrite(output_name, predictions)
    if is_save_json:
        output_namej = dt_ms_md5_val + '.json'

        total_dict = []
        for k,item in enumerate(dict_list):
            dic = dict(label=item['label'], id=item['id'], date=item['date'], deleted=item['deleted'], draw=item['draw'], user=item['user'],
                       verified=item['verified'], bbox=item['bbox'], scores=item['scores'], shape_type=item['shape_type'], probarray=item['probarray'],
                       seg_filename=output_name_segs[k],
                       expand_seg_filename=output_name_expand_segs[k],
                       box_filename=output_name_boxs[k],
                       expandbox_filename=output_name_expandboxs[k],
                       cam_filename=output_name_cams[k],
                       )
            total_dict.append(dic)
        jsPath = os.path.join(output_json_dir, '{}'.format(output_namej))
        djson = {"result_filename": dt_ms_md5_val,
                 "imgHeight": image.shape[0],
                 "imgWidth": image.shape[1],
                 "objects": total_dict
                 }
        with open(jsPath, "w") as f:
            json.dump(djson, f, sort_keys=True, indent=4)
        print("%s saved." % jsPath)
    return  predictions,djson

