import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d, check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import affine_transform_extend
from lib.datasets.kitti_utils import compute_box_3d
import pdb

import cv2 as cv
import torchvision.ops.roi_align as roi_align
import math
from lib.datasets.kitti_utils import Object3d


class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        '''
        ##h.w.l
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191462, 1.62856739989, 3.88311640418],
                                       [1.73698127, 0.59706367, 1.76282397]])

        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(root_dir, cfg['data_dir'], 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, cfg['data_dir'], 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.dense_depth_dir = cfg['dense_depth_dir']

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)  # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        img_size = np.array(img.size)

        d = cv.imread('{}/{:0>6}.png'.format(self.dense_depth_dir, index), -1) / 256.
        dst_W, dst_H = img_size
        pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
        pad_wr = dst_W - pad_w - d.shape[1]
        d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
        d = Image.fromarray(d)

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False

        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                d = d.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        d_trans = d.transform(tuple(self.resolution.tolist()),
                              method=Image.AFFINE,
                              data=tuple(trans_inv.reshape(-1).tolist()),
                              resample=Image.BILINEAR)
        d_trans = np.array(d_trans)
        down_d_trans = cv.resize(d_trans,
                                 (self.resolution[0] // self.downsample, self.resolution[1] // self.downsample),
                                 interpolation=cv.INTER_AREA)

        coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        calib = self.get_calib(index)


        features_size = self.resolution // self.downsample  # W * H
        #  ============================   get labels   ==============================
        if self.split != 'test':
            objects = self.get_label(index)

            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi

            calib_P2 = calib.P2.copy()
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32)  # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            # if torch.__version__ == '1.10.0+cu113':
            if torch.__version__ in ['1.10.0+cu113', '1.10.0', '1.6.0', '1.4.0', '1.7.0']:
                mask_2d = np.zeros((self.max_objs), dtype=np.bool)
            else:
                mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            roi_h2ds = np.zeros((self.max_objs, 5, 5), dtype=np.float32)
            roi_h3ds = np.zeros((self.max_objs, 5, 5), dtype=np.float32)
            att_depths = np.zeros((self.max_objs, 5, 5), dtype=np.float32)
            grd_depths = np.zeros((self.max_objs, 60, 60, 1), dtype=np.float32)
            surf_depths = np.zeros((self.max_objs, 5, 5), dtype=np.float32)
            grd_cord_2ds = np.zeros((self.max_objs, 60, 60, 1, 2), dtype=np.float32)
            roi_surface_masks = np.zeros((self.max_objs, 5, 5), dtype=np.bool)
            roi_surface_net_masks = np.zeros((self.max_objs, 5, 5), dtype=np.bool)
            grd_masks = np.zeros((self.max_objs, 60, 60, 1), dtype=np.bool)

            roi_cord_2ds = np.zeros((self.max_objs, 5, 5, 2), dtype=np.float32)

            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue

                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                bbox_2d_ori = objects[i].box2d.copy()

                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                box_2d_scaled_ratio = (bbox_2d[3] - bbox_2d[1]) / (bbox_2d_ori[3] - bbox_2d_ori[1])

                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                     dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue

                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h

                # encoding depth
                depth[i] = objects[i].pos[-1]

                # encoding heading angle
                # heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                    mask_2d[i] = 1

                roi_depth_grd = roi_align(torch.from_numpy(down_d_trans).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                      [torch.tensor(bbox_2d).unsqueeze(0)], [60, 60]).numpy()[0, 0]
                # 原始图像之中的坐标
                # x为列坐标
                roi_depth_grd[roi_depth_grd == 0] = 0.1
                interval_x = (bbox_2d_ori[2] - bbox_2d_ori[0]) / 60
                interval_y = (bbox_2d_ori[3] - bbox_2d_ori[1]) / 60
                roi_cord_x = np.linspace(bbox_2d_ori[0] + interval_x / 2, bbox_2d_ori[2] - interval_x / 2, 60)
                roi_cord_y = np.linspace(bbox_2d_ori[1] + interval_y / 2, bbox_2d_ori[3] - interval_y / 2, 60)
                x, y = np.meshgrid(roi_cord_x, roi_cord_y)
                roi_cord_2d = np.stack([x, y], -1)
                roi_cord = torch.from_numpy(
                    np.concatenate([roi_cord_2d, torch.from_numpy(roi_depth_grd).unsqueeze(-1).numpy()], -1)).view(-1, 3)
                roi_cord_cam = project2rect(torch.from_numpy(calib_P2).unsqueeze(0).repeat(roi_cord.shape[0], 1, 1),
                                            roi_cord).view(60, 60, 3)

                roi_cord_cam_y = roi_cord_cam[:, :, 1]

                roi_h3d = (objects[i].pos[1] - roi_cord_cam_y).numpy()

                # ground points(x,y,z)
                roi_cord_cam_x = roi_cord_cam[:, :, 0]
                roi_cord_cam_z = roi_cord_cam[:, :, 2]
                roi_cord_cam_y = torch.tensor(objects[i].pos[1]).unsqueeze(0).repeat(60, 60)
                roi_cord_cam = torch.stack([roi_cord_cam_x, roi_cord_cam_y, roi_cord_cam_z], -1)
                grd_cord_2d, _ = calib.rect_to_img(roi_cord_cam.view(-1, 3))
                grd_cord_2d = torch.from_numpy(grd_cord_2d).view(60, 60, 2)
                grd_depth = torch.from_numpy(roi_depth_grd).unsqueeze(-1).repeat(1, 1, 1).numpy()
                grd_cord_2d_x = grd_cord_2d[:, :, 0].unsqueeze(-1)
                grd_cord_2d_y = grd_cord_2d[:, :, 1].unsqueeze(-1)

                grd_cord_2d = torch.stack([grd_cord_2d_x, grd_cord_2d_y], -1)
                grd_cord_2d = grd_cord_2d.view(-1, 2).numpy()
                for j in range(grd_cord_2d.shape[0]):
                    grd_cord_2d[j] = affine_transform(grd_cord_2d[j], trans)
                grd_cord_2d = (torch.from_numpy(grd_cord_2d).view(60, 60, 1, 2) / self.downsample).numpy()
                # maintain interested points
                roi_surface_mask = (roi_depth_grd > (depth[i] - 3)) & \
                                   (roi_depth_grd < (depth[i] + 3)) & \
                                   (roi_depth_grd > 0.1) & \
                                   (roi_h3d > 0)
                grd_mask = torch.from_numpy(roi_surface_mask).unsqueeze(-1).repeat(1, 1, 1).numpy()
                grd_depth[~grd_mask] = 0


                roi_depth = roi_align(torch.from_numpy(down_d_trans).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                      [torch.tensor(bbox_2d).unsqueeze(0)], [5, 5]).numpy()[0, 0]

                # 原始图像之中的坐标
                roi_depth[roi_depth == 0] = 0.1
                interval_x = (bbox_2d_ori[2] - bbox_2d_ori[0]) / 5
                interval_y = (bbox_2d_ori[3] - bbox_2d_ori[1]) / 5
                roi_cord_x = np.linspace(bbox_2d_ori[0]+interval_x/2, bbox_2d_ori[2]-interval_x/2, 5)
                roi_cord_y = np.linspace(bbox_2d_ori[1]+interval_y/2, bbox_2d_ori[3]-interval_y/2, 5)
                x, y = np.meshgrid(roi_cord_x, roi_cord_y)
                roi_cord_2d = np.stack([x, y], -1)

                roi_cord = torch.from_numpy(
                    np.concatenate([roi_cord_2d, torch.from_numpy(roi_depth).unsqueeze(-1).numpy()], -1)).view(-1, 3)
                roi_cord_cam = project2rect(torch.from_numpy(calib_P2).unsqueeze(0).repeat(roi_cord.shape[0], 1, 1),
                                            roi_cord).view(5, 5, 3)

                roi_cord_cam_y = roi_cord_cam[:, :, 1]

                roi_h3d = objects[i].pos[1] - roi_cord_cam_y
                roi_h3d = roi_h3d.numpy()



                # ground points(x,y,z)
                roi_cord_cam_x = roi_cord_cam[:, :, 0]
                roi_cord_cam_z = roi_cord_cam[:, :, 2]
                roi_cord_cam_y = torch.tensor(objects[i].pos[1]).unsqueeze(0).repeat(5, 5)
                roi_cord_cam = torch.stack([roi_cord_cam_x, roi_cord_cam_y, roi_cord_cam_z], -1)
                grd_cord_2d_tmp, _ = calib.rect_to_img(roi_cord_cam.view(-1, 3))
                grd_cord_2d_tmp = torch.from_numpy(grd_cord_2d_tmp).view(5, 5, 2)
                roi_h2d = grd_cord_2d_tmp[:, :, 1] - roi_cord_2d[:, :, 1]

                roi_cord_2d = torch.from_numpy(roi_cord_2d).view(-1, 2).numpy()
                for k in range(roi_cord_2d.shape[0]):
                    roi_cord_2d[k] = affine_transform(roi_cord_2d[k], trans)
                roi_cord_2d = (torch.from_numpy(roi_cord_2d).view(5, 5, 2) / self.downsample)
                roi_h2d = roi_h2d * box_2d_scaled_ratio
                roi_h2d = (roi_h2d / self.downsample).numpy()

                # maintain interested points
                roi_surface_mask = (roi_depth > (depth[i] - 3)) & \
                                   (roi_depth < (depth[i] + 3)) & \
                                   (roi_depth > 0.1) & \
                                   (roi_h2d > 0)
                roi_depth_ind_net = (roi_depth > depth[i] - 3) & \
                                    (roi_depth < depth[i] + 3)

                roi_h2d[~roi_surface_mask] = 0
                roi_h3d[~roi_surface_mask] = 0
                att_depth = depth[i] - roi_depth

                roi_h2ds[i] = roi_h2d
                roi_h3ds[i] = roi_h3d

                att_depths[i] = att_depth
                surf_depths[i] = roi_depth

                roi_surface_masks[i] = roi_surface_mask
                roi_surface_net_masks[i] = roi_depth_ind_net
                grd_depths[i] = grd_depth
                grd_cord_2ds[i] = grd_cord_2d
                grd_masks[i] = grd_mask
                roi_cord_2ds[i] = roi_cord_2d

            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d,

                       'roi_h2ds': roi_h2ds,
                       'roi_h3ds': roi_h3ds,
                       'att_depths': att_depths,
                       'grd_depths': grd_depths,
                       'surf_depths': surf_depths,
                       'grd_cord_2ds': grd_cord_2ds,
                       'roi_surface_masks': roi_surface_masks,
                       'roi_surface_net_masks': roi_surface_net_masks,
                       'grd_masks': grd_masks,
                       'roi_cord_2ds': roi_cord_2ds
                       }
        else:
            targets = {}

        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size
                }

        return inputs, calib.P2, coord_range, targets, info  # calib.P2


def project2rect(calib, point_img):
    c_u = calib[:, 0, 2]
    c_v = calib[:, 1, 2]
    f_u = calib[:, 0, 0]
    f_v = calib[:, 1, 1]
    b_x = calib[:, 0, 3] / (-f_u)  # relative
    b_y = calib[:, 1, 3] / (-f_v)
    x = (point_img[:, 0] - c_u) * point_img[:, 2] / f_u + b_x
    y = (point_img[:, 1] - c_v) * point_img[:, 2] / f_v + b_y
    z = point_img[:, 2]
    centre_by_obj = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1)
    return centre_by_obj

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    cfg = {'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.4, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist': ['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center': False}
    dataset = KITTI('../../data', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
