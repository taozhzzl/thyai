import cv2
import torch
from torchvision import transforms as T
import json

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
import classfier

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return np.array(keep)

class XrayDetector(object):
    # COCO categories for pretty print
    # CATEGORIES = ['__background__', 'handgun', 'knife', 'small knife', 'scissors', 'pliers', 'wrench', 'tools', 'bullet group',
    #  'grenade', 'carbine', 'firecrackers', 'battery', 'cigarette lighter']
    # CATEGORIES = {0: '__background__', 1: 'malignant_i_lnmeta_yes', 2: 'malignant_i_lnmeta_no' ,3: 'benign_i_lnmeta_no'}
    CATEGORIES = {0: '__background__',1: 'node'}
    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        size=(224,224),#(h, w)
        save_json =True,
        isDrawBbox=False,
        isSegmentMask = True,
        marginExpandRatio=0.5,
        isShowClassname=False
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        print(self.model)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.size = size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim
        #for json

        self.save_json = save_json
        self.isDrawBbox = isDrawBbox
        self.isSegmentMask = isSegmentMask
        if marginExpandRatio < 1 and marginExpandRatio > 0:
            self.marginExpandRatio = marginExpandRatio
        else:
            self.marginExpandRatio = 0.001
        self.isShowClassname = isShowClassname
    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self,image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        origin_image = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)

        # if self.cfg.MODEL.MASK_ON:
        #     result = self.overlay_boxes(result, top_predictions)
        #     result = self.overlay_mask(result, top_predictions)
        # else:
        #     result = self.overlay_boxes(result, top_predictions)
        if self.isSegmentMask:
            result,boxnum,seg_images,seg_images_expand,seg_images_box,seg_images_expand_box,seg_expand_box = self.overlay_annotations(result, top_predictions)
        else:
            result,boxnum,seg_images,seg_images_expand,seg_images_box,seg_images_expand_box,seg_expand_box = self.overlay_annotations(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        result, dict_list,cam_list = self.overlay_class_names(result,origin_image, top_predictions,seg_images_expand , seg_images_expand_box , seg_expand_box)
        if self.isSegmentMask:
            return result,dict_list,boxnum,seg_images,seg_images_expand,seg_images_box,seg_images_expand_box,cam_list
        else:
            return result, dict_list, boxnum


    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        if(image.size() == 3):
            image=image.unsqueeze(0)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        threshold = self.confidence_threshold
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        # 当检测结果没有目标的时候，阈值自动减0.1
        while(len(keep.detach().cpu().numpy()) == 0 ):
            threshold = threshold - 0.05
            keep = torch.nonzero(scores > threshold).squeeze(1)
            if(len(scores.detach().cpu().numpy()) == 0):
                break
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        predictions = predictions[idx]
        # # 增加额外的后处理，如果bbox有重叠就合并bbox
        boxes = predictions.bbox
        bbox = torch.cat((boxes,scores.unsqueeze(1)),dim = 1)
        bbox = bbox.detach().cpu().numpy()
        keep = py_cpu_nms(bbox, 0.05)
        if len(keep) == 0:
            return predictions
        keep = torch.from_numpy(keep)
        predictions = predictions[keep]

        return predictions

    # def compute_colors_for_labels(self, labels):
    #     """
    #     Simple function that adds fixed colors depending on the class
    #     """
    #     colors = [[34, 139, 34], [0, 165, 255], [0, 0, 255]]
    #     return colors

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def inverse_color(self,color):
        new_color = []
        for item in color:
            inverse = 255 - item
            new_color.append(inverse)
        return new_color

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 8
            )

        return image



    def overlay_annotations(self, image, predictions):
        """
        Adds the predicted boxes and mask on top of the image
        and save json formate annotations on
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        scores = predictions.get_field("scores").detach().cpu().numpy()
        boxes = predictions.bbox
        origin_image = image.copy()

        # for save json
        # dict_list = []
        boxnum = 0
        colors = self.compute_colors_for_labels(labels).tolist()


        composite = image
        if self.cfg.MODEL.MASK_ON:
            masks = predictions.get_field("mask").numpy()
            mask_img = np.copy(image)
            # 根据轮廓信息提取掩模
            seg_mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
            seg_mask_expand = np.zeros((image.shape[0], image.shape[1]), np.uint8)

            seg_images = []
            seg_images_box = []
            seg_images_expand = []
            seg_images_expand_box = []
            seg_expand_box = []
            for bbox, mask, color, idx in zip(boxes, masks, colors, labels):
                thresh = mask[0, :, :, None]
                thresh = thresh.astype(np.uint8)
                _ ,contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # 创建矩形结构单元
                bbox_width = bbox[2] -bbox[0]
                bbox_height = bbox[3] - bbox[1]
                if(float(bbox_height) < 10 or float(bbox_width) < 10):
                    continue
                marginExpandRatio_h = int(.5 * bbox_height)
                marginExpandRatio_w = int(.5 * bbox_width)
                #（宽，高）
                g = cv2.getStructuringElement(cv2.MORPH_RECT, (marginExpandRatio_w, marginExpandRatio_h))
                # 腐蚀图像，迭代次数采用默认1
                img_erode = cv2.dilate(thresh, g)
                _, expand_contours, hierarchy = cv2.findContours(
                    img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # _, expand_contours, hierarchy = cv2.findContours(
                #     img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                # )
                mask_img = cv2.drawContours(mask_img, expand_contours, -1, color, -1)


                # 下面是扩大的图像
                color = self.inverse_color(color)
                # _, contours, hierarchy = cv2.findContours(
                #     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                # )
                _, contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                mask_img = cv2.drawContours(mask_img, contours, -1, color, -1)

                # if self.save_json:
                #     for c in contours:
                #         # 过滤掉小面积的目标，因为制作样本不方便
                #         if (cv2.contourArea(c) < 10):
                #             continue
                #         b = (c.reshape((-1, 2))).tolist()
                #         # print(b)
                #         ln = len(b)
                #         lnum = (int(ln / 10))
                #         bpoint = []
                #         for l in range(0, lnum):
                #             bpoint.append(b[l * 10])
                #         # print (num)
                #         import datetime
                #         nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
                #         dic = dict(label=self.CATEGORIES[int(idx)], id=int(idx), date=nowTime, deleted=0, draw=True, user="None",shape_type = "POLYGON",
                #                    verified=0,
                #                    polygon=bpoint)
                #     dict_list.append(dic)
                alpha = 0.45
                composite = cv2.addWeighted(image, 1.0 - alpha, mask_img, alpha, 0)
                if self.isSegmentMask and len(contours) > 0:
                    # 原图像名字为gray，由掩模抠出来的图像为next_gray
                    # 这里一定要使用参数 -1, 绘制填充的的轮廓
                    cv2.drawContours(seg_mask, contours, 0, 255, -1)
                    white = cv2.bitwise_not(seg_mask)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    next_gray = cv2.bitwise_and(origin_image, origin_image, mask=seg_mask)
                    white = cv2.cvtColor(white,cv2.COLOR_GRAY2RGB)
                    next_gray = next_gray + white
                    seg_gray = next_gray[y:y+h,x:x+w]
                    # cv2.imshow('next_gray', seg_gray)
                    seg_images.append(seg_gray)

                if self.isSegmentMask and len(contours) > 0:
                    # 原图像名字为gray，由掩模抠出来的图像为next_gray
                    # 这里一定要使用参数 -1, 绘制填充的的轮廓
                    x, y, w, h = cv2.boundingRect(contours[0])
                    seg_gray = origin_image[y:y+h,x:x+w]
                    # cv2.imshow('next_gray', seg_gray)
                    seg_images_box.append(seg_gray)

                if self.isSegmentMask and len(expand_contours) > 0:
                    # 原图像名字为gray，由掩模抠出来的图像为next_gray
                    # 这里一定要使用参数 -1, 绘制填充的的轮廓
                    cv2.drawContours(seg_mask_expand, expand_contours, 0, 255, -1)
                    white = cv2.bitwise_not(seg_mask_expand)
                    x, y, w, h = cv2.boundingRect(expand_contours[0])
                    next_gray = cv2.bitwise_and(origin_image, origin_image, mask=seg_mask_expand)
                    white = cv2.cvtColor(white,cv2.COLOR_GRAY2RGB)
                    next_gray = next_gray + white
                    seg_gray = next_gray[y:y+h,x:x+w]
                    # cv2.imshow('next_gray', seg_gray)
                    seg_images_expand.append(seg_gray)

                if self.isSegmentMask and len(expand_contours) > 0:
                    # 原图像名字为gray，由掩模抠出来的图像为next_gray
                    # 这里一定要使用参数 -1, 绘制填充的的轮廓

                    x, y, w, h = cv2.boundingRect(expand_contours[0])
                    seg_expand_box.append([x, y, w, h])
                    seg_gray = origin_image[y:y+h,x:x+w]
                    # cv2.imshow('next_gray', seg_gray)
                    seg_images_expand_box.append(seg_gray)

            return composite, boxnum,seg_images,seg_images_expand,seg_images_box,seg_images_expand_box ,seg_expand_box

        return composite,boxnum

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            _, contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        #color = color or [random.randint(0, 255) for _ in range(3)]
        if self.isDrawBbox:
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label and self.isShowClassname:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def overlay_class_names(self, image,origin_image, predictions , seg_images_expand ,seg_images_expand_box , seg_expand_box):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        classes = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in classes]
        # names = ['malignant_i_lnmeta_yes', 'malignant_i_lnmeta_no', 'benign_i_lnmeta_no']
        names = ['BN ', 'MN-LN(-)', 'MN-LN(+)']
        boxes = predictions.bbox
        # 橙色 R值: 22, G: 07, B: 201
        colors = [[34,139,34], [0,165,255], [0,0,255]]
        template = "{}: {:.2f}"
        dict_list = []
        cam_list = []
        for index,(box, score, label,class_,seg_image_expand_box,expand_box) in enumerate(zip(boxes, scores, labels, classes,seg_images_expand_box , seg_expand_box)):
            # x, y = box[:2]

            # 先做结点检测，然后再做分割图像的分类
            index,result,cam_result = classfier.doClassfier(seg_image_expand_box,origin_image,expand_box)
            cam_list.append(cam_result)
            label = names[index]
            s = template.format(label, result[index])
            self.plot_one_box(box, image, colors[index], s)
            # cv2.putText(
            #     image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            # )
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            if self.save_json:
                bbox = [[int(top_left[0]), int(top_left[1])], [int(bottom_right[0]), int(bottom_right[1])]]
                import datetime
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在

                # dic = dict(label=self.CATEGORIES[int(idx)], id=int(idx), date=nowTime, deleted=0, draw=True, user="None", verified=0, bbox=bbox, scores=str(scores[index]), shape_type="RECT")
                # dict_list.append(dic)
                probarray = []
                for index_name,name in enumerate(names):
                    probarray.append({"{}".format(name) : str(result[index_name])})
                dic = dict(label=self.CATEGORIES[1], id=int(index), date=nowTime, deleted=0, draw=True, user="None",
                           verified=0, bbox=bbox, scores=score, shape_type="RECT",probarray=probarray,

                )

                dict_list.append(dic)

        return image , dict_list,cam_list

import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

def vis_keypoints(img, kps, kp_thresh=5, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_buttocks')] +
        kps[:2, dataset_keypoints.index('left_buttocks')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_buttocks')],
        kps[2, dataset_keypoints.index('left_buttocks')])
    nose_idx = dataset_keypoints.index('head')
    # if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
    #     cv2.line(
    #         kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
    #         color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    # if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
    #     cv2.line(
    #         kp_mask, tuple(mid_shoulder), tuple(mid_hip),
    #         color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
    # p2 = kps[0, 4], kps[1, 4]
    # if kps[2, 4] > kp_thresh:
    #     cv2.circle(
    #         kp_mask, p2,
    #         radius=3, color=colors[0], thickness=-1, lineType=cv2.LINE_AA)
    # p2 = kps[0, 10], kps[1, 10]
    # if kps[2, 10] > kp_thresh:
    #     cv2.circle(
    #         kp_mask, p2,
    #         radius=3, color=colors[0], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
