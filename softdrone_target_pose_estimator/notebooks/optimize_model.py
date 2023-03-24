# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: trt_venv
#     language: python
#     name: envname
# ---

import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt

config_location = ""
model_weights = ""
output_location = ""

# +
with open(config_location, 'r') as f:
    target_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(target_pose)
# -

num_parts = len(target_pose['keypoints'])
num_links = len(target_pose['skeleton'])
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

model.load_state_dict(torch.load(model_weights))

WIDTH = 912
HEIGHT = 256
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

torch.save(model_trt.state_dict(), output_location)
