# File heavily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple, Literal

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from src.utils.posemap import get_coco_body25_mapping
from src.utils.posemap import kpoint_to_heatmap


class EngProjDataset(data.Dataset):
    def __init__(self,
                 dataroot_path: str,
                 radius=5,
                 size: Tuple[int, int] = (512, 384),
                 ):

        super(EngProjDataset, self).__init__()
        self.dataroot = dataroot_path
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        im_names = []
        c_names = []
        dataroot_names = []

        dataroot = self.dataroot

        # Enumerate all files in {dataroot}/person_image
        for im_name in os.listdir(os.path.join(dataroot, 'person_image')):
            if im_name.endswith('.jpg'):
                # Append to im_names, c_names, dataroot_names
                im_names.append(im_name)
                c_names.append('input_cloth.jpg')
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        category = 'upper_body'

        # Cloth image
        cloth = Image.open(os.path.join(dataroot, c_name))
        cloth = cloth.resize((self.width, self.height))
        cloth = self.transform(cloth)  # [-1,1]

        # Person image
        image = Image.open(os.path.join(dataroot, 'person_image', im_name))
        image = image.resize((self.width, self.height))
        image = self.transform(image)  # [-1,1]

        # labels = {
        #     0: ['background', [0, 10]],  # 0 is background, 10 is neck
        #     1: ['hair', [1, 2]],  # 1 and 2 are hair
        #     2: ['face', [4, 13]],
        #     3: ['upper', [5, 6, 7]],
        #     4: ['bottom', [9, 12]],
        #     5: ['left_arm', [14]],
        #     6: ['right_arm', [15]],
        #     7: ['left_leg', [16]],
        #     8: ['right_leg', [17]],
        #     9: ['left_shoe', [18]],
        #     10: ['right_shoe', [19]],
        #     11: ['socks', [8]],
        #     12: ['noise', [3, 11]]
        # }

        # Label Map
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(os.path.join(dataroot, 'parsing', parse_name))
        im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
        im_parse_final = transforms.ToTensor()(im_parse) * 255
        parse_array = np.array(im_parse)

        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 4).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32)

        parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                            (parse_array == 2).astype(np.float32) + \
                            (parse_array == 18).astype(np.float32) + \
                            (parse_array == 19).astype(np.float32)

        parser_mask_changeable = (parse_array == 0).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 6).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32)
        parse_mask = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 6).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32)

        parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32)  # the lower body is fixed

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()

        # Shape
        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
        shape = self.transform2D(parse_shape)  # [-1,1]

        # Load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(dataroot, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

            # rescale keypoints on the base of height and width
            pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
            pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

        pose_mapping = get_coco_body25_mapping()

        # point_num = pose_data.shape[0]
        point_num = len(pose_mapping)

        pose_map = torch.zeros(point_num, self.height, self.width)
        r = self.radius * (self.height / 512.0)
        im_pose = Image.new('L', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)
        neck = Image.new('L', (self.width, self.height))
        neck_draw = ImageDraw.Draw(neck)
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)

            point_x = np.multiply(pose_data[pose_mapping[i], 0], 1)
            point_y = np.multiply(pose_data[pose_mapping[i], 1], 1)

            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                if i == 2 or i == 5:
                    neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                        'white')
            one_map = self.transform2D(one_map)
            pose_map[i] = one_map[0]

        d = []

        for idx in range(point_num):
            ux = pose_data[pose_mapping[idx], 0]  # / (192)
            uy = (pose_data[pose_mapping[idx], 1])  # / (256)

            # scale posemap points
            px = ux  # * self.width
            py = uy  # * self.height

            d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

        pose_map = torch.stack(d)

        # just for visualization
        im_pose = self.transform2D(im_pose)

        im_arms = Image.new('L', (self.width, self.height))
        arms_draw = ImageDraw.Draw(im_arms)

        # do in any case because i have only upperbody
        with open(os.path.join(dataroot, 'pose', pose_name), 'r') as f:
            data = json.load(f)
            data = data['people'][0]['pose_keypoints_2d']
            data = np.array(data)
            data = data.reshape((-1, 3))[:, :2]

            # rescale keypoints on the base of height and width
            data[:, 0] = data[:, 0] * (self.width / 768)
            data[:, 1] = data[:, 1] * (self.height / 1024)

            shoulder_right = tuple(data[pose_mapping[2]])
            shoulder_left = tuple(data[pose_mapping[5]])
            elbow_right = tuple(data[pose_mapping[3]])
            elbow_left = tuple(data[pose_mapping[6]])
            wrist_right = tuple(data[pose_mapping[4]])
            wrist_left = tuple(data[pose_mapping[7]])

            ARM_LINE_WIDTH = int(90 / 512 * self.height)
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

            hands = np.logical_and(np.logical_not(im_arms), arms)
            parse_mask += im_arms
            parser_mask_fixed += hands

        # delete neck
        parse_head_2 = torch.clone(parse_head)

        parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                np.logical_not(
                                                                    np.array(parse_head_2, dtype=np.uint16))))

        # tune the amount of dilation here
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        im_mask = image * parse_mask_total
        inpaint_mask = 1 - parse_mask_total
        inpaint_mask = inpaint_mask.unsqueeze(0)
        parse_mask_total = parse_mask_total.numpy()
        parse_mask_total = parse_array * parse_mask_total
        parse_mask_total = torch.from_numpy(parse_mask_total)

        outputlist = ["image", "inpaint_mask", "pose_map", "category", "cloth", "im_mask", "im_name"]
        result = {}
        for k in outputlist:
            result[k] = vars()[k]

        return result

    def __len__(self):
        return len(self.c_names)
