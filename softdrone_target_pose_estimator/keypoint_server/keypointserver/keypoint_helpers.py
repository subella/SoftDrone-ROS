import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import numpy as np
from trt_pose.coco import *
from trt_pose.parse_objects import ParseObjects
from PIL import Image
from torch2trt import TRTModule

# TODO(sam): these values should not be hardcoded
WIDTH = 912
HEIGHT = 256
IMAGE_AREA = 234000


def load_model_from_json(self, filename, model_weights):
    with open(filename, 'r') as f:
        target_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(target_pose)

    num_parts = len(target_pose['keypoints'])
    num_links = len(target_pose['skeleton'])

    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(model_weights))
    return model

def optimize_model(model):
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    return model_trt

class KeypointDetector():

    def __init__(self, config_file, optimized_model_file):
        self.transform = torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize([0.42174017, 0.39773076, 0.40150955], 
                                                          [0.25029679, 0.23943031, 0.25025475])
                         ])
        self.load_config(config_file)
        self.model = self.load_optimized_model(optimized_model_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            target_pose = json.load(f)

        self.num_keypoints = len(target_pose["keypoints"])
        self.topology = trt_pose.coco.coco_category_to_topology(target_pose)
        self.parse_objects = ParseObjects(self.topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, \
                                          line_integral_samples=7, max_num_parts=100, max_num_objects=100)

    
    def load_optimized_model(self, filename):
        model = TRTModule()
        model.load_state_dict(torch.load(filename))
        return model
    
    def detect_keypoints(self, input_image, from_cv=True, vertical_crop=0.5):
        if from_cv:
            orig_image = Image.fromarray(input_image).convert("RGB")
        else:
            orig_image = input_image

    
        orig_image = orig_image.crop((0, 
                                      vertical_crop * orig_image.height,
                                      orig_image.width, 
                                      orig_image.height))
    
        resize_factor = np.sqrt((orig_image.width * orig_image.height) / IMAGE_AREA)
        image_shape = (int(orig_image.width / resize_factor), int(orig_image.height / resize_factor))

        quad = get_quad(0.0, (0, 0), 1.0, aspect_ratio=1.0)
        image = transform_image(orig_image, image_shape, quad)
        data = self.transform(image).cuda()[None, ...]
        cmap = self.model(data)
        cmap = cmap.cpu()
        object_counts, objects, peaks = self.parse_objects(cmap, cmap)
        object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]
        kps = [0] * (self.num_keypoints * 3)
        for i in range(object_counts):
            object = objects[i]
            cnt = 0
            for j in range(self.num_keypoints):
                k = object[j]
                if k >= 0:
                    peak = peaks[j][k]
                    x = (peak[1] - 0.5) + 0.5
                    y = peak[0]
                    x = round(float(orig_image.width * x))
                    y = round(float(orig_image.height + orig_image.height * y))
                    kps[j * 3 + 0] = x
                    kps[j * 3 + 1] = y
                    kps[j * 3 + 2] = 2
        
        kps = np.array(kps).reshape((self.num_keypoints, 3))
        return kps
