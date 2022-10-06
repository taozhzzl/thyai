#!/usr/bin/env python

import argparse
import pathlib
import time
import cv2

import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import torchvision
from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from PIL import Image
from fvcore.common.checkpoint import Checkpointer as CheckPointer
from pytorch_image_classification.utils import (
    compute_accuracy,
    AverageMeter,
    create_logger,
    get_rank,
)
from pytorch_image_classification.transforms.transforms import Normalize,ToTensor


def load_config():
    config = get_default_config()
    config.merge_from_file('./configs/se_resnext_50.yaml')

    update_config(config)
    config.freeze()
    return config




config = load_config()

if config.test.output_dir is None:
    output_dir = pathlib.Path(config.test.checkpoint).parent
else:
    output_dir = pathlib.Path(config.test.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

logger = create_logger(name=__name__, distributed_rank=get_rank())

model = create_model(config)
print(model)
model = apply_data_parallel_wrapper(config, model)
checkpointer = CheckPointer(model,
                            checkpoint_dir=output_dir,
                            logger=logger,
                            distributed_rank=get_rank())
checkpointer.load(config.test.checkpoint)
model.eval()

# class_to_idx= ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '4', '5', '6', '7', '8', '9']
# image = cv2.imread('/home/shining/work/Projects/work/medical_maskrcnn/Genemerge-all/classfication/bbox//test/1/122.jpg')
# image = cv2.resize(image, (224, 224))
# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# img = img / 255.0
# img = (img - mean) / std
# img = img.astype(np.float32)

# img = torch.from_numpy(img).cuda()
# img = img.unsqueeze(0)
# img = img.permute(0, 3, 1, 2)

transforms = []

transforms.append(torchvision.transforms.Resize(224))

# transforms.append(torchvision.transforms.CenterCrop(224))
transforms += [
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensor(),
]
transform = torchvision.transforms.Compose(transforms)
# def doClassfier(image):
#     transform = torchvision.transforms.Compose(transforms)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = transform(Image.fromarray(np.uint8(image)))  # 归一化到 3x1080x1920(通道*高*宽)，数值[0.0,1.0]
#     img = img.unsqueeze(0)
#     with torch.no_grad():
#         start=time.time()
#         out=model.forward(img)
#         # out = model.forward(img)
#         out = F.softmax(out, dim=1)
#         print("total time:{}".format(time.time()-start))
#
#         result=out.cpu().numpy()[0]
#         ind=np.argmax(out.cpu().numpy())
#         # ind=np.argsort(result,axis=1)
#         return  ind,result

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def doClassfier(image,origin_image,box):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(Image.fromarray(np.uint8(image)))  # 归一化到 3x1080x1920(通道*高*宽)，数值[0.0,1.0]
    img = img.unsqueeze(0)
    with torch.no_grad():
        start=time.time()
        out,features=model.forward(img)
        # out = model.forward(img)
        out = F.softmax(out, dim=1)
        print("total time:{}".format(time.time()-start))

        result=out.cpu().numpy()[0]
        ind=np.argmax(out.cpu().numpy())
        # ind=np.argsort(result,axis=1)
        probs, idx = out.cpu().data.squeeze().sort(0, True)
        idx = idx.numpy()
        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features.cpu().numpy(), weight_softmax, [idx[0]])

        # render the CAM and output
        print('output CAM.jpg for the top1 prediction: %s'%idx[0])
        img = image
        height, width, _ = img.shape
        x, y, w, h = box
        heatmap_full = np.zeros([origin_image.shape[0], origin_image.shape[1]], dtype=np.uint8)
        heatmap_full[int(y):int(y + h), int(x):int(x + w)] = cv2.resize(CAMs[0],(w, h))
        # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(heatmap_full, cv2.COLORMAP_JET)

        cam_result = heatmap * 0.3 + origin_image * 0.5

        # origin_image_cpoy = origin_image.copy()
        # origin_image_cpoy[int(y):int(y+h),int(x):int(x+w)] = cam_result
        # cam_result = heatmap_full * 0.3 + origin_image * 0.5
        return  ind,result,cam_result
