# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: keypoint_venv
#     language: python
#     name: keypoint_venv
# ---

# # Keypoint Evaluation

# +
# %load_ext autoreload
# %autoreload 2

import glob
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import teaserpp_python
import seaborn as sns
import pandas as pd
import random
import json
import matplotlib.gridspec as gridspec

from scipy import stats
from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation as R
from keypointserver.keypoint_helpers import *
from notebook_helpers import *

sns.set(font_scale=1)
sns.set_style("ticks",{'axes.grid' : True})
# -

model = load_model_from_json(10, "/home/subella/src/KeypointTraining/tasks/obj_000012/obj_000012_pose.json", "/home/subella/src/KeypointTraining/tasks/obj_000012/obj_000012.json.checkpoints/epoch_75.pth")
opt_model = optimize_model(model)

OPTIMIZED_MODEL = 'cleaner_model.pth'
torch.save(opt_model.state_dict(), OPTIMIZED_MODEL)

target_name = "obj_000012"

# +
# config_file = "../models/{}_pose.json".format(target_name)
# model_file = "../models/{}_model.pth".format(target_name)
# dataset_path = "/home/subella/src/AutomatedAnnotater/Data/"

config_file = "/home/subella/src/KeypointTraining/tasks/obj_000012/{}_pose.json".format(target_name)
model_file = OPTIMIZED_MODEL
dataset_path = "/home/subella/src/AutomatedAnnotater/Data/"
# -

model = KeypointDetector(config_file, model_file)
# model = KeypointDetector(config_file, "medkit_softdrone.pth")

# TODO: standardize this.
training_folder = dataset_path + "CleanerPBRAugmented" + "/Datasets/Training/"
validation_folder = dataset_path + "Medkit" + "/Datasets/Validation/"
test_folder = dataset_path + "CleanerPBRAugmented" + "/Datasets/Test/"

working_folder = test_folder
# working_folder = validation_folder
# working_folder = training_folder

# +
def load_rgb_image(rgb_image_filename):
    return Image.open(rgb_image_filename).convert('RGB')

def load_depth_image(depth_image_filename):
    return cv2.imread(depth_image_filename, cv2.IMREAD_UNCHANGED)


# +
def write_metrics(metrics, name):
    # !mkdir metrics
    data = json.dumps(metrics)
    filename = "metrics/{}.json".format(name)
    if not os.path.isfile(filename):
        f = open(filename,"w")
        f.write(data)
        f.close()
    else:
        print("File exists, please manually delete it first to update it.")

def load_metrics(name):
    with open("metrics/{}.json".format(name)) as json_file:
        metrics = json.load(json_file)
        return convert_metrics_to_np(metrics)
    return None

def convert_metrics_to_np(metrics):
    for metric in metrics:
        metric["gt_pixel_keypoints"] = np.array(metric["gt_pixel_keypoints"])
        metric["est_pixel_keypoints"] = np.array(metric["est_pixel_keypoints"])
        if "est_interp_pixel_keypoints" in metric:
            metric["est_interp_pixel_keypoints"] = np.array(metric["est_interp_pixel_keypoints"])
        metric["gt_world_keypoints"] = np.array(metric["gt_world_keypoints"])
        metric["gt_teaser_pose"] = np.array(metric["gt_teaser_pose"])
        metric["est_teaser_pose"] = np.array(metric["est_teaser_pose"])
        if "est_interp_teaser_pose" in metric:
            metric["est_interp_teaser_pose"] = np.array(metric["est_interp_teaser_pose"])
        if "gt_pixel_est_depth_teaser_pose" in metric:
            metric["gt_pixel_est_depth_teaser_pose"] = np.array(metric["gt_pixel_est_depth_teaser_pose"])
        if "interp_cad_keypoints" in metric:
            metric["interp_cad_keypoints"] = np.array(metric["interp_cad_keypoints"])
        if "cast_pose" in metric:
            metric["cast_pose"] = np.array(metric["cast_pose"])
        if "inliers" in metric:
            metric["inliers"] = np.array(metric["inliers"])
        if "interp_inliers" in metric:    
            metric["interp_inliers"] = np.array(metric["interp_inliers"])
        if "gt_pixel_est_depth_inliers" in metric:
            metric["gt_pixel_est_depth_inliers"] = np.array(metric["gt_pixel_est_depth_inliers"])
        if "est_all_inliers_teaser_pose" in metric:
            metric["est_all_inliers_teaser_pose"] = np.array(metric["est_all_inliers_teaser_pose"])
        if "K_ros" in metric:
            metric["K_ros"] = np.array(metric["K_ros"])
        if "K_mat" in metric:
            metric["K_mat"] = np.array(metric["K_mat"])
    return metrics


# +
class SingleEvaluator():
    def __init__(self, model, cad_keypoints, 
                 rgb_image_filename, depth_image_filename, 
                 annotation, K_ros, K_mat, verbose=False):
        self.rgb_image_filename = rgb_image_filename
        self.rgb_image = load_rgb_image(rgb_image_filename)
        self.depth_image_filename = depth_image_filename
        self.depth_image = load_depth_image(depth_image_filename)
        self.width = self.rgb_image.width
        self.height = self.rgb_image.height
        self.annotation = annotation
        self.model = model
        self.K_ros = K_ros
        self.K_mat = K_mat
        self.cad_keypoints = cad_keypoints
        self.verbose = verbose
        self.solver_params = self.get_teaser_params()
        self.solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)
        
        self.all_inliers_solver_params = self.get_teaser_params()
        self.all_inliers_solver_params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.NONE
        self.all_inliers_solver = teaserpp_python.RobustRegistrationSolver(self.all_inliers_solver_params)
        
        self.gt_pixel_keypoints = self.get_gt_pixel_keypoints()
        self.est_pixel_keypoints = self.get_est_pixel_keypoints()
        self.gt_world_keypoints = self.get_gt_world_keypoints()
        print("GT", self.gt_world_keypoints)
        self.est_world_keypoints = self.get_est_world_keypoints()
        print("EST", self.est_world_keypoints)
#         self.est_interp_pixel_keypoints, self.interp_cad_keypoints = self.get_interp_keypoints()
        self.est_interp_pixel_keypoints, self.interp_cad_keypoints = self.get_spherical_keypoints()
        self.est_interp_world_keypoints = self.get_est_interp_world_keypoints()
        self.gt_pixel_est_depth_world_keypoints = self.get_gt_pixel_est_depth_world_keypoints()
        print("GT DEPTH", self.gt_pixel_est_depth_world_keypoints)
        self.gt_teaser_pose = self.get_gt_teaser_pose()
        self.est_teaser_pose = self.get_est_teaser_pose()
#         print("GT", self.gt_teaser_pose)
#         print("EST", self.est_teaser_pose)
        self.est_interp_teaser_pose = self.get_est_interp_teaser_pose()
        self.gt_pixel_est_depth_teaser_pose = self.get_gt_pixel_est_depth_teaser_pose()
#         print("GT DEPTH", self.gt_pixel_est_depth_teaser_pose)
        self.est_all_inliers_teaser_pose = self.get_est_all_inliers_teaser_pose()
        self.distance = np.linalg.norm(self.gt_teaser_pose[:3,3])
        
        self.pixel_keypoints_error = self.compute_pixel_keypoints_error()
        self.world_keypoints_error = self.compute_world_keypoints_error()
        self.rotation_error, self.translation_error = self.compute_pose_error(self.est_teaser_pose)
        self.all_inliers_rotation_error, self.all_inliers_translation_error = self.compute_pose_error(self.est_all_inliers_teaser_pose)

    def get_teaser_params(self):
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 10
        solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-6
        
        return solver_params
        
    def get_gt_pixel_keypoints(self):
        return np.array(self.annotation["ground_truth_pixel_keypoints"])
    
    def get_est_pixel_keypoints(self):
        return np.array(self.model.detect_keypoints(self.rgb_image.copy(), from_cv=False))
    
    def get_interp_keypoints(self):
        # TODO: read from json
        pixels = self.est_pixel_keypoints.copy()
        cad_keypoints = self.cad_keypoints.copy()
        skeleton = np.array([[11,12],[12,13],[8,11],[7,11],[9,13],[10,13],\
                             [1,8],[3,10],[7,9],[7,8],[7,0],[1,14],[8,14],\
                             [14,15],[15,3],[15,10],[9,11],[10,11],[7,13],\
                             [8,13],[0,4],[1,4],[0,1],[4,5],[5,6],[6,2],[2,3],[3,6],\
                             [0,6],[1,6],[2,4],[3,4]])
#         skeleton = np.array([[1,8],[3,10],[7,9],[7,8],[7,0],[1,14],[8,14],[14,15],[15,3],[15,10]])
        num_points = 4
        for pair in skeleton:
            if (self.est_pixel_keypoints[pair[0]][0] != 0 \
                and self.est_pixel_keypoints[pair[0]][1] != 0 \
                and self.est_pixel_keypoints[pair[1]][0] !=0 \
                and self.est_pixel_keypoints[pair[1]][1] !=0):
                
                pixel_points = np.linspace(self.est_pixel_keypoints[pair[0]], self.est_pixel_keypoints[pair[1]], num_points, endpoint=False)[1:]
                cad_keypoints_points = np.linspace(self.cad_keypoints[pair[0]], self.cad_keypoints[pair[1]], num_points, endpoint=False)[1:]
                pixels = np.append(pixels, pixel_points, axis=0)
                cad_keypoints = np.append(cad_keypoints, cad_keypoints_points, axis=0)
                
#             else:
#                 pixel_points = np.linspace(np.array([0,0,0]), np.array([0,0,0]), num_points, endpoint=False)[1:]
#                 cad_keypoints_points = np.linspace(np.array([0,0,0]), np.array([0,0,0]), num_points, endpoint=False)[1:]
#                 pixels = np.append(pixels, pixel_points, axis=0)
#                 cad_keypoints = np.append(cad_keypoints, cad_keypoints_points, axis=0)
        return pixels, cad_keypoints

    def get_spherical_keypoints(self):
        pixels = self.est_pixel_keypoints.copy()
        cad_keypoints = self.cad_keypoints.copy()
        radius = 2
        directions = 2 * np.array([(1,1), (-1,-1), (1,-1), (-1,1)])
        for pixel, cad in zip(self.est_pixel_keypoints, self.cad_keypoints):
            for direction in directions:
                if (pixel[0] != 0 and pixel[1] != 0):
                    new_pixel = np.array([[pixel[0] + direction[0], pixel[1] + direction[1], pixel[2]]])
                else:
                    new_pixel = np.array([[0,0,0]])
                pixels = np.append(pixels, new_pixel, axis=0)
                cad_keypoints = np.append(cad_keypoints, [cad], axis=0)
        
        
#         for pixel, cad in zip(self.est_pixel_keypoints, self.cad_keypoints):
#             if (pixel[0] != 0 and pixel[1] != 0):
#                 for i in range(-radius, radius):
#                     for j in range(-radius, radius):
#                         new_pixel = np.array([[pixel[0] + i, pixel[1] + j, pixel[2]]])
#                         pixels = np.append(pixels, new_pixel, axis=0)
#                         cad_keypoints = np.append(cad_keypoints, [cad], axis=0)
#             else:
#                 for i in range(-radius, radius):
#                     for j in range(-radius, radius):
#                         new_pixel = np.array([[0,0,0]])
#                         pixels = np.append(pixels, new_pixel, axis=0)
#                         cad_keypoints = np.append(cad_keypoints, [cad], axis=0)
        return pixels, cad_keypoints
                    
    
    def get_est_interp_world_keypoints(self):
        pxs = np.clip(self.est_interp_pixel_keypoints[:,0].astype(int), 0, self.width - 1)
        pys = np.clip(self.est_interp_pixel_keypoints[:,1].astype(int), 0 , self.height - 1)
        z = self.depth_image[pys,pxs]
        est_world_keypoints = reproject(pxs, pys, z, self.K_ros)
        return est_world_keypoints
            

    def get_gt_world_keypoints(self):
        return np.array(self.annotation["ground_truth_keypoints"])
    
    def get_est_world_keypoints(self):
        pxs = np.clip(self.est_pixel_keypoints[:,0].astype(int), 0, self.width - 1)
        pys = np.clip(self.est_pixel_keypoints[:,1].astype(int), 0 , self.height - 1)
        z = self.depth_image[pys,pxs]
        est_world_keypoints = reproject(pxs, pys, z, self.K_ros)
        return est_world_keypoints
    
    def get_gt_pixel_est_depth_world_keypoints(self):
#         keypoints = np.array(self.annotation["keypoints"])
#         keypoints = keypoints.reshape((self.annotation["num_keypoints"], 3))
#         nonvisible_ids = np.where(keypoints[:,2] == 0)[0]
        
        pxs = np.clip(self.gt_pixel_keypoints[:,0].astype(int), 0, self.width - 1)
        pys = np.clip(self.gt_pixel_keypoints[:,1].astype(int), 0 , self.height - 1)
        
        
        z = self.depth_image[pys,pxs]
        est_world_keypoints = reproject(pxs, pys, z, self.K_ros)
#         est_world_keypoints[nonvisible_ids] = 0
        return est_world_keypoints
    
    def get_gt_teaser_pose(self):
        self.solver.reset(self.solver_params)
        self.solver.solve(self.cad_keypoints.T, self.gt_world_keypoints.T)
        solution = self.solver.getSolution()
        tf = make_tf(solution.rotation, solution.translation)
        return tf
      
    def get_base_est_teaser_pose(self, solver_params, solver):
        solver.reset(solver_params)
        solver.solve(self.cad_keypoints.T, self.est_world_keypoints.T)
        solution = solver.getSolution()
        tf = make_tf(solution.rotation, solution.translation)
        inliers = self.solver.getInlierMaxClique()
#         self.num_inliers = len(inliers)
#         print("Num Inliers", self.num_inliers)
#         # If 0 points included as inliers, not a valid detection
#         if np.any(~self.est_pixel_keypoints[:,:2].any(axis=1)):
#             self.is_valid = False
#         else:
#             self.is_valid = True
            
#         self.is_valid = self.is_valid and self.is_target_visible()
            
        return tf, inliers
    
    def get_est_teaser_pose(self):
        tf, inliers = self.get_base_est_teaser_pose(self.solver_params, self.solver)
        self.inliers = inliers
        self.num_inliers = len(inliers)
        if self.num_inliers == 0:
            self.is_valid = False
        else:
            self.is_valid = True
        return tf
    
    def get_est_interp_teaser_pose(self):
        self.solver.reset(self.solver_params)
        self.solver.solve(self.interp_cad_keypoints.T, self.est_interp_world_keypoints.T)
        solution = self.solver.getSolution()
        self.interp_inliers = self.solver.getInlierMaxClique()
        tf = make_tf(solution.rotation, solution.translation)
        return tf
    
        
    def get_est_all_inliers_teaser_pose(self):
        tf, inliers = self.get_base_est_teaser_pose(self.all_inliers_solver_params, self.all_inliers_solver)
        return tf
    
    def get_gt_pixel_est_depth_teaser_pose(self):
        self.solver.reset(self.solver_params)
        self.solver.solve(self.cad_keypoints.T, self.gt_pixel_est_depth_world_keypoints.T)
        solution = self.solver.getSolution()
        self.gt_pixel_est_depth_inliers = self.solver.getInlierMaxClique()
        tf = make_tf(solution.rotation, solution.translation)
        return tf
    
    def compute_pixel_keypoints_error(self):
        self.pixel_keypoints_error = 0
        num = 0
        for gt_keypoint, est_keypoint in zip(self.gt_pixel_keypoints, self.est_pixel_keypoints):
            if np.any(est_keypoint):
                self.pixel_keypoints_error += np.linalg.norm(gt_keypoint - est_keypoint[:2])
                num += 1
        if num > 0:
            return self.pixel_keypoints_error / num
        return None
        
    def compute_world_keypoints_error(self):
        self.world_keypoints_error = 0
        num = 0
        for gt_keypoint, est_keypoint in zip(self.gt_world_keypoints, self.est_world_keypoints):
            if np.any(est_keypoint):
                self.world_keypoints_error += np.linalg.norm(gt_keypoint - est_keypoint)
                num += 1
        if num > 0:
            return self.world_keypoints_error / num
        return None
    
    def compute_pose_error(self, est_pose):
        gt_rotation = self.gt_teaser_pose[:3,:3]
        est_rotation = est_pose[:3,:3]
        rotation_error = np.arccos((np.trace(est_rotation.T.dot(gt_rotation)) - 1) / 2)
        rotation_error = np.rad2deg(rotation_error)
        translation_error = np.linalg.norm(self.gt_teaser_pose[:3,3] - est_pose[:3,3])
        return rotation_error, translation_error
        
    def is_target_visible(self):
        keypoints = np.array(self.annotation["keypoints"])
        keypoints = keypoints.reshape((self.annotation["num_keypoints"], 3))
        num_visible = np.where(keypoints[:,2]==2)[0].size
        if num_visible < 3:
            return False
        return True
        
    def get_metrics(self):
        metrics = {}
        metrics["gt_pixel_keypoints"] = self.gt_pixel_keypoints.tolist()
        metrics["est_pixel_keypoints"] = self.est_pixel_keypoints.tolist()
        metrics["est_interp_pixel_keypoints"] = self.est_interp_pixel_keypoints.tolist()
        metrics["gt_world_keypoints"] = self.gt_world_keypoints.tolist()
        metrics["est_world_keypoints"] = self.est_world_keypoints.tolist()
        metrics["est_interp_world_keypoints"] = self.est_interp_world_keypoints.tolist()
        metrics["gt_teaser_pose"] = self.gt_teaser_pose.tolist()
        metrics["est_teaser_pose"] = self.est_teaser_pose.tolist()
        metrics["est_interp_teaser_pose"] = self.est_interp_teaser_pose.tolist()
        metrics["interp_cad_keypoints"] = self.interp_cad_keypoints.tolist()
        metrics["gt_pixel_est_depth_teaser_pose"] = self.gt_pixel_est_depth_teaser_pose.tolist()
        metrics["est_all_inliers_teaser_pose"] = self.est_all_inliers_teaser_pose.tolist()
        metrics["rgb_image_filename"] = self.rgb_image_filename
        metrics["depth_image_filename"] = self.depth_image_filename
        metrics["average_keypoint_pixel_error"] = self.pixel_keypoints_error
        metrics["average_keypoint_world_error"] = self.world_keypoints_error
        metrics["translation_error"] = self.translation_error
        metrics["all_inliers_translation_error"] = self.all_inliers_translation_error
        metrics["rotation_error"] = self.rotation_error
        metrics["all_inliers_rotation_error"] = self.all_inliers_rotation_error
        metrics["inliers"] = self.inliers
        metrics["interp_inliers"] = self.interp_inliers
        metrics["gt_pixel_est_depth_inliers"] = self.gt_pixel_est_depth_inliers
        metrics["distance"] = self.distance
        metrics["K_ros"] = self.K_ros.tolist()
        metrics["K_mat"] = self.K_mat.tolist()
        metrics["is_valid"] = self.is_valid
        metrics["num_inliers"] = self.num_inliers
        return metrics       


# +
class SinglePlotter():
    def __init__(self, single_evaluator_metrics):
        self.metrics = single_evaluator_metrics
        self.rgb_image = load_rgb_image(self.metrics["rgb_image_filename"])
        self.depth_image = load_depth_image(self.metrics["depth_image_filename"])
        self.upscale = 2
        width, height = self.rgb_image.size
        self.rgb_image = self.rgb_image.resize((width * self.upscale, height*self.upscale))
        self.depth_image = cv2.resize(self.depth_image, (width * self.upscale, height*self.upscale), interpolation = cv2.INTER_LINEAR)
#         self.depth_image = self.depth_image.resize((width * self.upscale, height*self.upscale))
        self.drawing = ImageDraw.Draw(self.rgb_image)

        self.K_ros = self.metrics["K_ros"]
        self.K_mat = self.metrics["K_mat"]
        self.fig_3d = None
        self.ax_3d = None
    
    def init_fig_3d(self):
        # %matplotlib inline
        self.fig_3d = plt.figure(figsize=(10, 10))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        
    def finalize_fig_3d(self):
        self.ax_3d.legend()
        plt.show()
        
    def show_image(self):
        display(self.rgb_image)
        
    def write_image(self, folder, id):
        file = folder + "/frame_" + "{:04d}".format(id) + ".jpg"
        self.rgb_image.save(file)
        
    def crop_image(self):
        center = np.mean(self.upscale * self.metrics["gt_pixel_keypoints"], axis=0)
        half_width = 150 * self.upscale
        half_height = 150 * self.upscale
        
        if center[0] - half_width < 0:
            center[0] = half_width
        elif center[0] + half_width > self.rgb_image.width:
            center[0] = self.rgb_image.width - half_width
            
        if center[1] - half_height < 0:
            center[1] = half_height
        elif center[1] + half_height > self.rgb_image.height:
            center[1] = self.rgb_image.height - half_height
            
        cropped_image = self.rgb_image.crop((center[0] - half_width, center[1] - half_height, 
                                                  center[0] + half_width, center[1] + half_height))
        
        self.rgb_image = ImageOps.contain(cropped_image, (self.rgb_image.width, self.rgb_image.height))
        
    def plot_usable_space(self):
        mask = np.where(self.depth_image != 0, 1, 0)
        mask = mask.astype('uint8')
        rgb_cv = np.array(self.rgb_image)
        rgb_cv = cv2.bitwise_and(rgb_cv, rgb_cv, mask=mask)
        self.rgb_image = Image.fromarray(rgb_cv)
        self.drawing = ImageDraw.Draw(self.rgb_image)
        
    def plot_keypoints(self, keypoints, r=4, color=(0,255,0)):
        r = r * self.upscale
        for keypoint in keypoints:
            x, y = keypoint[0] * self.upscale, keypoint[1] * self.upscale
            self.drawing.ellipse((x-r, y-r, x+r, y+r), fill=color)
        
    def plot_gt_pixel_keypoints(self, r=2, color=(0,255,0)):
        self.plot_keypoints(self.metrics["gt_pixel_keypoints"], color=color)

    def plot_est_pixel_keypoints(self, r=2, color=(255,0,0)):
        self.plot_keypoints(self.metrics["est_pixel_keypoints"], color=color)      
        
    def plot_est_interp_pixel_keypoints(self, r=2, color=(255,0,0)):
        self.plot_keypoints(self.metrics["est_interp_pixel_keypoints"], color=color) 
        
    def plot_est_inlier_pixel_keypoints(self, r=2, color=(0,0,255)):
        print(self.metrics["inliers"])
        self.plot_keypoints(self.metrics["est_pixel_keypoints"][self.metrics["inliers"]], color=color)
  
    def plot_est_interp_inlier_pixel_keypoints(self, r=2, color=(0,0,255)):
        self.plot_keypoints(self.metrics["est_interp_pixel_keypoints"][self.metrics["interp_inliers"]], color=color)
        
    def plot_gt_pixel_est_depth_inlier_pixel_keypoints(self, r=2, color=(0,0,255)):
        self.plot_keypoints(self.metrics["gt_pixel_keypoints"][self.metrics["gt_pixel_est_depth_inliers"]], color=color)
  

    def plot_keypoints_depth(self):
        for i, keypoint in enumerate(self.metrics["est_pixel_keypoints"]):
            depth = self.metrics["est_world_keypoints"][i][2]
            x, y = keypoint[0] * self.upscale, keypoint[1] * self.upscale
#             self.drawing.ellipse((x-r, y-r, x+r, y+r), fill=color)
            self.drawing.text((x+5, y+5), str(depth), (0,0,0))

        
    def plot_gt_world_keypoints(self):
        if not self.fig_3d:
            self.init_fig_3d()
        points = self.metrics["gt_world_keypoints"].T
        self.ax_3d.scatter(points[0], points[1], points[2],
                           c='green', label='Ground Truth Point Cloud')        
    
    def plot_est_world_keypoints(self):
        if not self.fig_3d:
            self.init_fig_3d()
        points = self.metrics["est_world_keypoints"].T
        self.ax_3d.scatter(points[0], points[1], points[2],
                           c='red', label='Estimated Point Cloud')
        
    def plot_correspondences(self):
        for est_keypoint, gt_keypoint in zip(self.metrics["est_pixel_keypoints"], self.metrics["gt_pixel_keypoints"]):
            self.drawing.line((est_keypoint[0], est_keypoint[1], \
                               gt_keypoint[0], gt_keypoint[1]), \
                               fill=(165, 42, 42), width=1)

    def plot_gt_teaser_pose(self, scale=100):
        print(self.metrics["gt_teaser_pose"])
        plot_pose(self.drawing, self.metrics["gt_teaser_pose"], self.K_mat, resize_factor=self.upscale)
    
    def plot_est_teaser_pose(self, scale=100):
        plot_pose(self.drawing, self.metrics["est_teaser_pose"], self.K_ros, 
                  x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139), resize_factor=self.upscale)
  
    def plot_est_interp_teaser_pose(self):
        try:
            plot_pose(self.drawing, self.metrics["est_interp_teaser_pose"], self.K_ros, 
                      x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139), resize_factor=self.upscale)
        except:
            pass
        
    def plot_gt_pixel_est_depth_teaser_pose(self):
        print(self.metrics["gt_pixel_est_depth_teaser_pose"])
        plot_pose(self.drawing, self.metrics["gt_pixel_est_depth_teaser_pose"], self.K_ros, 
                  x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139), resize_factor=self.upscale)
        
    def plot_cast_pose(self):
#         print(self.metrics["cast_pose"])
        if np.any(self.metrics["cast_pose"][:3,:3]):
            x = self.metrics["cast_pose"].copy()
            
            theta_degrees = 0

            # Convert angle to radians
            theta_radians = np.radians(theta_degrees)

            # Rotation matrix about z-axis
            R_z = np.array([
                [np.cos(theta_radians), -np.sin(theta_radians), 0],
                [np.sin(theta_radians), np.cos(theta_radians), 0],
                [0, 0, 1]
            ])

            # Original rotation matrix
            R_original = x[:3,:3]  # Your original rotation matrix

            # Rotate the original rotation matrix about z-axis
            x[:3,:3] = np.dot(R_original, R_z)

#             x[:,[2,1]] = x[:,[1,2]]
#             x[:3,3] = x[:3,3] * 1000
            plot_pose(self.drawing, x, self.K_mat, 
                      x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139), resize_factor=self.upscale)
    
# -

class MultiEvaluator():
    def __init__(self, folder, verbose=False):
        self.folder = folder
        self.metrics = []
        self.verbose = verbose
        
    def parse_metadata(self, folder):
        metadata_path = folder + "/metadata.json"
        f = open(metadata_path)
        metadata = json.load(f)
        # Distinction here:
        # K_mat is calibrated with matlab specifically on this dataset
        # (it is more accurate for ground truth / overfit)
        # K_ros is what ros gives for the camera 
        # (what we actually use on the robot)
        K_ros = np.array(metadata["ros_intrinsics"]["intrinsic_matrix"])
        K_mat = np.array(metadata["intrinsics"]["intrinsic_matrix"])
        cad_keypoints = np.array(metadata["cad_frame"])
        return metadata, K_mat, K_ros, cad_keypoints
        
    def eval_annotation(self, folder, annotation, K_ros, K_mat, cad_keypoints):
        rgb_image_filename = folder + "/" + annotation["rgb_file_name"]
        depth_image_filename = folder + "/" + annotation["depth_file_name"]
#         rgb_image_filename = annotation["rgb_file_name"]
#         depth_image_filename = annotation["depth_file_name"]
        evaluator = SingleEvaluator(model, cad_keypoints, 
                                    rgb_image_filename, depth_image_filename, 
                                    annotation, K_ros, K_mat, verbose=self.verbose)
        self.metrics.append(evaluator.get_metrics())

        
    def run(self):
        for folder in glob.glob(self.folder + "/*"):
            metadata, K_mat, K_ros, cad_keypoints = self.parse_metadata(folder)
            print("Starting folder :", folder)
            for annotation in tqdm.tqdm(metadata["annotations"]):
                self.eval_annotation(folder, annotation, K_mat, K_ros, cad_keypoints)


# +
class MultiPlotter():
    def __init__(self, multi_evaluator_metrics, verbose=False):
        self.metrics = multi_evaluator_metrics
        self.df = pd.DataFrame.from_records(self.metrics,
                                            columns=["average_keypoint_pixel_error", 
                                                     "average_keypoint_world_error", 
                                                     "translation_error",
                                                     "all_inliers_translation_error",
                                                     "rotation_error",
                                                     "all_inliers_rotation_error",
                                                     "distance",
                                                     "is_valid",
                                                     "num_inliers"])
        self.verbose = verbose

    def plot_keypoint_pixel_error_vs_distance(self, ax=None):
#         pruned_df = self.df[self.df["is_valid"] == True]
        pruned_df = self.df
        data_used = int(len(pruned_df) / len(self.df) * 100)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
#             ax.set_title("Average Keypoint Error (using {}% of data)".format(data_used))
            ax.set(xlabel="Distance [m]", ylabel="Keypoint Error [px]")

        
        values = np.vstack([pruned_df["distance"], pruned_df["average_keypoint_pixel_error"]])
        z = stats.gaussian_kde(values)(values)
        idx = z.argsort()
        x = np.array(pruned_df["distance"]) / 1000.
        y = np.array(pruned_df["average_keypoint_pixel_error"])
        x, y, z = x[idx], y[idx], z[idx]

        ax = sns.scatterplot(
            x=x,
            y=y,
            c=z,
            cmap="viridis",
            ax=ax,
            linewidth=0
            )
        plt.savefig("keypoint.svg")
        
    def plot_translation_error_vs_distance(self, ax=None):
#         pruned_df = self.df[self.df["is_valid"] == True]
        pruned_df = self.df
        data_used = int(len(pruned_df) / len(self.df) * 100)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
#             ax.set_title("Translation Error (using {}% of data)".format(data_used))
            ax.set(xlabel="Distance [m]", ylabel="Translation Error [mm]")

        values = np.vstack([pruned_df["distance"], pruned_df["translation_error"]])
        z = stats.gaussian_kde(values)(values)
        idx = z.argsort()
        x = np.array(pruned_df["distance"]) / 1000.
        y = np.array(pruned_df["translation_error"])
        x, y, z = x[idx], y[idx], z[idx]
   
        ax = sns.scatterplot(
            x=x,
            y=y,
            c=z,
            cmap="viridis",
            ax=ax,
            linewidth=0
            )
    
#         ax = sns.scatterplot(
#             x=x,
#             y=y,
# #             c=z,
# #             cmap="viridis",
#             ax=ax,
#             linewidth=0
#             )
        plt.savefig("translation.svg")

        
    def plot_rotation_error_vs_distance(self, ax=None):
#         pruned_df = self.df[self.df["is_valid"] == True]
        pruned_df = self.df
        data_used = int(len(pruned_df) / len(self.df) * 100)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
#             ax.set_title("Rotation Error (using {}% of data)".format(data_used))
            ax.set(xlabel="Distance [m]", ylabel="Rotation Error [Deg]")

        values = np.vstack([pruned_df["distance"], pruned_df["rotation_error"]])
        z = stats.gaussian_kde(values)(values)
        idx = z.argsort()
        x = np.array(pruned_df["distance"]) / 1000.
        y = np.array(pruned_df["rotation_error"])
        x, y, z = x[idx], y[idx], z[idx]
        
        sns.scatterplot(
            x=x,
            y=y,
            c=z,
            cmap="viridis",
            ax=ax,
            linewidth=0
            )
        plt.savefig("rotation.svg")
          
    def plot_multiple(self, rows=3, cols=2):
        # %matplotlib inline

        fig = plt.figure(1,(7 * rows, 7 * cols))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(rows,cols),
                         axes_pad=0.1,
                         share_all=True
                         )
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])

        for i in range(rows * cols):
            metric = random.choice(self.metrics)
            plotter = SinglePlotter(metric)
            plotter.plot_gt_pixel_keypoints()
            plotter.plot_est_pixel_keypoints()
            plotter.plot_gt_teaser_pose()
            plotter.plot_est_teaser_pose()
            plotter.plot_correspondences()
            plotter.crop_image()
            grid[i].imshow(plotter.rgb_image)
        
    def plot_sequence(self):
        i = 0
        step_size = 1
        while i < len(self.metrics):
            print("plotting", i)
            
            metric = self.metrics[i]
            fig = plt.figure()
            plotter = SinglePlotter(metric)
#             plotter.plot_usable_space()
            plotter.plot_gt_pixel_keypoints()
            plotter.plot_est_pixel_keypoints()
#             plotter.plot_keypoints_depth()
#             plotter.plot_est_inlier_pixel_keypoints()
#             plotter.plot_est_interp_pixel_keypoints()
#             plotter.plot_est_interp_inlier_pixel_keypoints()
#             plotter.plot_gt_pixel_est_depth_inlier_pixel_keypoints()
#             plotter.plot_correspondences()
            plotter.plot_gt_teaser_pose()
            plotter.plot_est_teaser_pose()
#             plotter.plot_est_interp_teaser_pose()
#             plotter.plot_gt_pixel_est_depth_teaser_pose()
#             plotter.plot_cast_pose()
    #             plotter.crop_image()
#             plotter.show_image()
            plotter.write_image("temp_vid", i)

            fig = plt.figure()
    #             plotter.crop_image()
#             plotter.show_image()

            if self.verbose:
                print("Average Keypoint Pixel Error: ", metric["average_keypoint_pixel_error"])
                print("Average Keypoint World Error: ", metric["average_keypoint_world_error"])
                print("Euler Rotation Error: ", metric["rotation_error"])
                print("Translation Error", metric["translation_error"])
                print("gt", metric["gt_teaser_pose"])
                print("est", metric["est_teaser_pose"])
            i += step_size

    def plot_composite(self):
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(12, 28, figure=fig)
        kpt = fig.add_subplot(gs[:4,9:])
        trans = fig.add_subplot(gs[4:8,9:])    
        rot = fig.add_subplot(gs[8:12,9:])
    
        self.plot_keypoint_pixel_error_vs_distance(kpt)
        self.plot_translation_error_vs_distance(trans)
        self.plot_rotation_error_vs_distance(rot)
        
        kpt.set(xlabel=None, ylabel="Keypoint Error [px]")
        trans.set(xlabel=None, ylabel="Translation Error [mm]")
        rot.set(xlabel="Distance [mm]", ylabel="Rotation Error [mm]")
        kpt.tick_params(labelbottom=False)
        trans.tick_params(labelbottom=False)
                       
              
        def fill_ax(ax):
            metric = random.choice(self.metrics)
            plotter = SinglePlotter(metric)
            plotter.plot_gt_pixel_keypoints()
            plotter.plot_est_pixel_keypoints()
            plotter.plot_gt_teaser_pose()
            plotter.plot_est_teaser_pose()
            plotter.plot_correspondences()
            plotter.crop_image()
            ax.imshow(plotter.rgb_image, aspect="auto")
            ax.grid(False)
            ax.tick_params(labelbottom=False, labelleft=False)
            
        ax1 = fig.add_subplot(gs[:3,:7])
        fill_ax(ax1)
        ax2 = fig.add_subplot(gs[3:6,:7])
        fill_ax(ax2)
        ax3 = fig.add_subplot(gs[6:9,:7])
        fill_ax(ax3)
        ax4 = fig.add_subplot(gs[9:12,:7])
        fill_ax(ax4)
#         ax5 = fig.add_subplot(gs[24:30, :6])
#         fill_ax(ax5)
#         ax6 = fig.add_subplot(gs[20:30,10:20])
#         fill_ax(ax6)
    
    def plot_inliers(self):
        
        xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        filtered_inliers = []
        all_inliers = []
        dfs = []
        for i, inlier in enumerate(xs):
            df = (self.df[self.df['num_inliers'] == inlier])
            dfs.append(df)
            
            
#             filtered_inliers.append(np.mean(df["translation_error"]))
#             all_inliers.append(np.mean(df["all_inliers_translation_error"]))
            fig, ax = plt.subplots(figsize=(9,4))
            ax.set_title("Translation Errors for {} inlier count ({} data points)".format(i, len(df)))
            sns.boxplot(data=df[["translation_error", "all_inliers_translation_error"]], orient="h")
        
        
        
        for i, inlier in enumerate(xs):
            df = (self.df[self.df['num_inliers'] == inlier])
            dfs.append(df)
            
            
#             filtered_inliers.append(np.mean(df["translation_error"]))
#             all_inliers.append(np.mean(df["all_inliers_translation_error"]))
            fig, ax = plt.subplots(figsize=(9,4))
            ax.set_title("Rotation Errors for {} inlier count ({} data points)".format(i, len(df)))
            sns.boxplot(data=df[["rotation_error", "all_inliers_rotation_error"]], orient="h")
        
#         self.df['num_inliers'].value_counts()[1]
#         xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         ys = []
#         for inlier in xs:
#             ys.append((self.df['num_inliers'] == inlier)["translation_error"])
 
#         fig, ax = plt.subplots(1,1, figsize=(5,4))
#         N = 10
#         ind = np.arange(N)
#         width = 0.6    

#         # Plotting
#         # ax.bar(ind, vision , width, label='Vision')
#         ax.bar(xs, ys, width)
#         ax.set_ylabel('Count')
#         # plt.title('Here goes title of the plot')
# #         ax.set_xticks(ind, ('Slow', 'Fast', '0.5m/s', '1.0m/s', '1.5m/s'))
        # plt.xticks(rotation = 45)

        ax.legend(loc='best')
        
        
        fig.tight_layout()

# -

multi_evaluator = MultiEvaluator(working_folder, verbose=False)
multi_evaluator.run()

write_metrics(multi_evaluator.metrics, "bleach_hard_00_03_chaitanya")

name = "pepsi_bottle_metrics"
metrics = load_metrics(name)

metrics = convert_metrics_to_np(multi_evaluator.metrics)

multi_plotter.df

# +
multi_plotter = MultiPlotter(metrics, verbose=False)
multi_plotter.plot_sequence()
# multi_plotter.plot_multiple()

# multi_plotter.plot_keypoint_pixel_error_vs_distance()
# multi_plotter.plot_translation_error_vs_distance()
# multi_plotter.plot_rotation_error_vs_distance()
# multi_plotter.plot_inliers()
# multi_plotter.plot_composite()
# -

# # Plot components average

def get_free_error(pose_error, total_error, max_error):
    free_error = (max_error - (total_error - pose_error))
    return free_error


# +
from scipy.optimize import curve_fit
    
def plot_threshold(metrics, free_errors_dict, name):
    fig, ax = plt.subplots()
    errors = []
    distances = []

    for metric in metrics:
#         if metric["is_valid"] == True:
#         if "2023-02-20-09-14-59" in metric["rgb_image_filename"] \
#         or "2023-01-09-12-53-11" in metric["rgb_image_filename"] \
#         or "2023-02-09-16-09-23" in metric["rgb_image_filename"]:
        est_wrt_gt = np.linalg.inv(metric["gt_teaser_pose"]).dot(metric["est_teaser_pose"])
        error_t = abs(est_wrt_gt[:3,3])
        if np.all(error_t < 300):
            errors.append(error_t / 10.)

            distance = metric["distance"] / 1000.
            distances.append(distance)
    
    errors = np.array(errors)
    distances = np.array(distances)

    x = errors[:,0]
    y = errors[:,1]
    z = errors[:,2]

    # Sort the data based on distance
    sorted_indices = np.argsort(distances)
    distance_sorted = distances[sorted_indices]
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    z_sorted = z[sorted_indices]
    


    def func(x, a, b, c):
        return -b + b* np.exp(c * x)



    distance_sampled = np.linspace(np.min(distance_sorted), np.max(distance_sorted), 100)
    degree = 2
    
#     coefficients = np.polyfit(distance_sorted, x_sorted, degree)
#     polynomial = np.poly1d(coefficients)
#     longitudinal = polynomial(x_new)
#     plt.plot(x_new, longitudinal, label='Longitudinal')
    
    popt, pcov = curve_fit(func, distance_sorted, x_sorted, [-1, 3, 1], maxfev=5000)
    longitudinal = func(distance_sampled, *popt)
    ax.plot(distance_sampled, longitudinal, label='Longitudinal') 

#     coefficients = np.polyfit(distance_sorted, y_sorted, degree)
#     polynomial = np.poly1d(coefficients)
#     lateral = polynomial(x_new)
#     plt.plot(x_new, lateral, label='Lateral')
    popt, pcov = curve_fit(func, distance_sorted, y_sorted, [-1, 3, 1], maxfev=5000)
    lateral = func(distance_sampled, *popt)
    ax.plot(distance_sampled, lateral, label='Lateral') 

#     coefficients = np.polyfit(distance_sorted, z_sorted, degree)
#     polynomial = np.poly1d(coefficients)
#     vertical = polynomial(x_new)
#     plt.plot(x_new, vertical, label='Vertical')
    popt, pcov = curve_fit(func, distance_sorted, z_sorted, [-1, 3, 1], maxfev=5000)
    vertical = func(distance_sampled, *popt)
    ax.plot(distance_sampled, vertical, label='Vertical') 
    
    for speed, free_errors in free_errors_dict.items():
        try:
            threshold_long_dist = np.where(longitudinal > free_errors[0])[0][0]
        except:
            threshold_long_dist = np.inf
        try:
            threshold_lateral_dist = np.where(lateral > free_errors[1])[0][0]
        except:
            threshold_lateral_dist = np.inf
        try:
            threshold_vertical_dist = np.where(vertical > free_errors[2])[0][0]
        except:
            threshold_vertical_dist = np.inf

        threshold = min(threshold_long_dist, threshold_lateral_dist, threshold_vertical_dist)
        threshold_idx = np.argmin([threshold_long_dist, threshold_lateral_dist, threshold_vertical_dist])
#         print(speed)
#         print(threshold)
        if threshold_idx == 0:
            ax.plot(distance_sampled[threshold], longitudinal[threshold], 'ro', markersize=10)

        if threshold_idx == 1:
            ax.plot(distance_sampled[threshold], lateral[threshold], 'ro', markersize=10)

        if threshold_idx == 2:
            ax.plot(distance_sampled[threshold], vertical[threshold], 'ro',  markersize=10)


        plt.axvline(x=distance_sampled[threshold], color="black")

    # Add labels and legend
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Errors [cm]')
    ax.set_title(name)
    ax.legend()

#     plt.show()
    plt.savefig("figures/{}_max_distance.svg".format(name))

# +
medkit_metrics = load_metrics("medkit_metrics")
medkit_long_free = get_free_error(pose_error=.42, total_error=1.48, max_error=2.95)
medkit_lat_free = get_free_error(pose_error=3.98, total_error=4.78, max_error=6.85)
medkit_vert_free = get_free_error(pose_error=1.44, total_error=6.1, max_error=7.6)
medkit_free_errors = {"0.5 m/s" : [medkit_long_free, medkit_lat_free, medkit_vert_free]}
plot_threshold(medkit_metrics, medkit_free_errors, "Medkit")

cardboard_metrics = load_metrics("cardboard_box_metrics")
cardboard_long_free = get_free_error(pose_error=2.39, total_error=3.4, max_error=5.45)
cardboard_lat_free = get_free_error(pose_error=4.07, total_error=5.51, max_error=7.85)
cardboard_vert_free = get_free_error(pose_error=1.65, total_error=7.58, max_error=7.6)
cardboard_free_errors = {"0.5 m/s" : [medkit_long_free, medkit_lat_free, medkit_vert_free]}
plot_threshold(cardboard_metrics, cardboard_free_errors, "Cardboard")

pepsi_metrics = load_metrics("pepsi_bottle_metrics")
pepsi_long_free_05 = get_free_error(pose_error=0.93, total_error=2.01, max_error=6.7)
pepsi_lat_free_05 = get_free_error(pose_error=3.6, total_error=4.72, max_error=10.6)
pepsi_vert_free_05 = get_free_error(pose_error=1.32, total_error=6.14, max_error=7.6)

pepsi_long_free_125 = get_free_error(pose_error=1.13, total_error=3.34, max_error=6.7)
pepsi_lat_free_125 = get_free_error(pose_error=3.73, total_error=5.3, max_error=10.6)
pepsi_vert_free_125 = get_free_error(pose_error=3.71, total_error=6.06, max_error=7.6)

pepsi_long_free_2 = get_free_error(pose_error=0.47, total_error=4.07, max_error=6.7)
pepsi_lat_free_2 = get_free_error(pose_error=3.07, total_error=4.92, max_error=10.6)
pepsi_vert_free_2 = get_free_error(pose_error=2.57, total_error=6.07, max_error=7.6)

pepsi_long_free_3 = get_free_error(pose_error=0.47, total_error=7.21, max_error=6.7)
pepsi_lat_free_3 = get_free_error(pose_error=3.07, total_error=8.39, max_error=10.6)
pepsi_vert_free_3 = get_free_error(pose_error=2.57, total_error=10.95, max_error=7.6)

pepsi_free_errors = {"0.5 m/s" : [pepsi_long_free_05, pepsi_lat_free_05, pepsi_vert_free_05],
#                      "1.25 m/s": [pepsi_long_free_125, pepsi_lat_free_125, pepsi_vert_free_125],
#                      "2 m/s" : [pepsi_long_free_2, pepsi_lat_free_2, pepsi_vert_free_2],
#                      "3 m/s" : [pepsi_long_free_3, pepsi_lat_free_3, pepsi_vert_free_3]
                    }
plot_threshold(pepsi_metrics, pepsi_free_errors, "Pepsi")

# +
errors = []
distances = []

for metric in metrics:
#     if metric["is_valid"] == True:
    if "2023-01-09-12-53-11s" not in metric["rgb_image_filename"]\
    and "2023-01-09-12-48-04" not in metric["rgb_image_filename"] \
    and "2023-02-20-09-14-59s" not in metric["rgb_image_filename"] \
    and "2023-02-10-13-32-22" not in metric["rgb_image_filename"] \
    and "2023-02-09-16-50-26" not in metric["rgb_image_filename"] \
    and "2023-02-09-16-09-23s" not in metric["rgb_image_filename"] \
    and "2023-02-10-13-30-47" not in metric["rgb_image_filename"] \
    and "2023-02-09-16-36-54" not in metric["rgb_image_filename"]:

        est_R = metric["est_teaser_pose"][:3,:3]
        est_t = metric["est_teaser_pose"][:3,3]
        gt_t = metric["gt_teaser_pose"][:3,3]

        est_wrt_gt = np.linalg.inv(metric["gt_teaser_pose"]).dot(metric["est_teaser_pose"])
        error_t = abs(est_wrt_gt[:3,3])
        if np.all(error_t < 300):
            errors.append(error_t)
            distance = metric["distance"] / 1000.
            distances.append(distance)
    
errors = np.array(errors)
distances = np.array(distances)

fig, ax = plt.subplots(figsize=(6,6))
ax.set(xlabel="Distance [m]", ylabel="Translation Error [mm]")
    
values = np.vstack([distances, errors[:,0]])
z = stats.gaussian_kde(values)(values)
idx = z.argsort()
x = np.array(distances)
long = np.array(errors[:,0])
lat = np.array(errors[:,1])
vert = np.array(errors[:,2])

y = lat
x, y, z = x[idx], y[idx], z[idx]

ax = sns.scatterplot(
    x=x,
    y=y,
    c=z,
    cmap="viridis",
    ax=ax,
    linewidth=0
    )

sorted_indices = np.argsort(distances)
distance_sorted = distances[sorted_indices]
long_sorted = long[sorted_indices]
lat_sorted = lat[sorted_indices]
vert_sorted = vert[sorted_indices]

x_new = np.linspace(np.min(distance_sorted), np.max(distance_sorted), 100)
degree = 2

coefficients = np.polyfit(distance_sorted, vert_sorted, degree)
polynomial = np.poly1d(coefficients)
vals = polynomial(x_new)
plt.plot(x_new, vals, label='Longitudinal')

from scipy.optimize import curve_fit

def func(x, a, b, c):
    return -b + b* np.exp(c * x)

popt, pcov = curve_fit(func, distance_sorted, lat_sorted, [-1, 3, 1], maxfev=5000)
plt.plot(distance_sorted, func(distance_sorted, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

print(func(distance_sorted, *popt))
# -

print(errors)


def plot_translation_error_vs_distance(self, ax=None):
        pruned_df = self.df[self.df["is_valid"] == True]
        data_used = int(len(pruned_df) / len(self.df) * 100)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
#             ax.set_title("Translation Error (using {}% of data)".format(data_used))
            ax.set(xlabel="Distance [m]", ylabel="Translation Error [mm]")

        values = np.vstack([pruned_df["distance"], pruned_df["translation_error"]])
        z = stats.gaussian_kde(values)(values)
        idx = z.argsort()
        x = np.array(pruned_df["distance"]) / 1000.
        y = np.array(pruned_df["translation_error"])
        x, y, z = x[idx], y[idx], z[idx]
   
        ax = sns.scatterplot(
            x=x,
            y=y,
            c=z,
            cmap="viridis",
            ax=ax,
            linewidth=0
            )
    
        plt.savefig("translation.svg")


# +
sorted_indices = np.argsort(distance)
distance_sorted = distance[sorted_indices]
errors_sorted = errors[sorted_indices]

df = pd.DataFrame({'distances': distance_sorted, 
                  'x': errors_sorted[:,0],
                  'y': errors_sorted[:,1],
                  'z': errors_sorted[:,2]})
df["mean_x"] = df['x'].rolling(window=1000).mean()
df["mean_y"] = df['y'].rolling(window=1000).mean()
df["mean_z"] = df['z'].rolling(window=1000).mean()

df.plot(x="distances",y=["mean_x", "mean_y", "mean_z"])
# -

threshold_long_dist[0]


