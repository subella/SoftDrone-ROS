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
import pickle

from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation as R
from keypointserver.keypoint_helpers import *
from notebook_helpers import *
# -

target_name = "pepsi"

config_file = "../models/{}_pose.json".format(target_name)
model_file = "../models/{}_model.pth".format(target_name)
dataset_path = "/home/subella/src/AutomatedAnnotater/Data/"

model = KeypointDetector(config_file, model_file)

# TODO: standardize this.
training_folder = dataset_path + "FormattingTest" + "/Datasets/Training/"
validation_folder = dataset_path + "FormattingTest" + "/Datasets/Validation/"

working_folder = training_folder


# +
class SingleEvaluator():
    def __init__(self, model, cad_keypoints, rgb_image, depth_image, 
                 annotation, K, verbose=False):
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.width = rgb_image.width
        self.height = rgb_image.height
        self.annotation = annotation
        self.model = model
        self.K = K
        self.cad_keypoints = cad_keypoints
        self.verbose = verbose

        self.solver = self.get_teaser()
        
        self.gt_pixel_keypoints = self.get_gt_pixel_keypoints()
        self.est_pixel_keypoints = self.get_est_pixel_keypoints()
        self.gt_world_keypoints = self.get_gt_world_keypoints()
        self.est_world_keypoints = self.get_est_world_keypoints()
        self.gt_teaser_pose = self.get_gt_teaser_pose()
        self.est_teaser_pose = self.get_est_teaser_pose()
        self.distance = np.linalg.norm(self.gt_teaser_pose[:3,3])
        
        self.pixel_keypoints_error = self.compute_pixel_keypoints_error()
        self.world_keypoints_error = self.compute_world_keypoints_error()
        self.rotation_error, self.translation_error = self.compute_pose_error()
        
#         self.return_stats(verbose=True)

    def get_teaser(self):
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 15
        solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-6
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        return solver
        
    def get_gt_pixel_keypoints(self):
        return np.array(self.annotation["ground_truth_pixel_keypoints"])
    
    def get_est_pixel_keypoints(self):
        return np.array(self.model.detect_keypoints(self.rgb_image.copy(), from_cv=False))
    
    def get_gt_world_keypoints(self):
        return np.array(self.annotation["ground_truth_keypoints"])
    
    def get_est_world_keypoints(self):
        pxs = np.clip(self.est_pixel_keypoints[:,0].astype(int), 0, self.width - 1)
        pys = np.clip(self.est_pixel_keypoints[:,1].astype(int), 0 , self.height - 1)
        z = self.depth_image[pys,pxs]
        est_world_keypoints = reproject(pxs, pys, z, self.K)
        return est_world_keypoints
    
    def get_gt_teaser_pose(self):
        self.solver.solve(self.cad_keypoints.T, self.gt_world_keypoints.T)
        solution = self.solver.getSolution()
        tf = make_tf(solution.rotation, solution.translation)
        return tf
      
    def get_est_teaser_pose(self):
        self.solver.solve(self.cad_keypoints.T, self.est_world_keypoints.T)
        solution = self.solver.getSolution()
        tf = make_tf(solution.rotation, solution.translation)
        return tf
        
    def compute_pixel_keypoints_error(self):
        self.pixel_keypoints_error = 0
        for gt_keypoint, est_keypoint in zip(self.gt_pixel_keypoints, self.est_pixel_keypoints):    
            self.pixel_keypoints_error += np.linalg.norm(gt_keypoint - est_keypoint[:2])
        return self.pixel_keypoints_error / len(self.gt_pixel_keypoints)
        
    def compute_world_keypoints_error(self):
        self.world_keypoints_error = 0
        for gt_keypoint, est_keypoint in zip(self.gt_world_keypoints, self.est_world_keypoints):    
            self.world_keypoints_error += np.linalg.norm(gt_keypoint - est_keypoint)
        return self.world_keypoints_error / len(self.gt_pixel_keypoints)
    
    def compute_pose_error(self):
        gt_rotation = self.gt_teaser_pose[:3,:3]
        est_rotation = self.est_teaser_pose[:3,:3]
        rotation_error = np.arccos((np.trace(est_rotation.T.dot(gt_rotation)) - 1) / 2)
        rotation_error = np.rad2deg(rotation_error)
        translation_error = np.linalg.norm(self.gt_teaser_pose[:3,3] - self.est_teaser_pose[:3,3])
        return rotation_error, translation_error
        
    def get_df(self, verbose=False):
        stats = {}
        stats["average_keypoint_pixel_error"] = [self.pixel_keypoints_error]
        stats["average_keypoint_world_error"] = [self.world_keypoints_error]
        stats["translation_error"] = [self.translation_error]
        stats["rotation_error"] = [self.rotation_error]
        stats["distance"] = [self.distance]
        if self.verbose:
            print("Average Keypoint Pixel Error: ", self.pixel_keypoints_error)
            print("Average Keypoint World Error: ", self.world_keypoints_error)
            print("Euler Rotation Error: ", self.rotation_error)
            print("Translation Error", self.translation_error)
            print("gt", self.gt_teaser_pose)
            print("est", self.est_teaser_pose)
        return pd.DataFrame(stats)
        

# -

class SinglePlotter():
    def __init__(self, evaluator):
        self.drawing = ImageDraw.Draw(evaluator.rgb_image)
        self.image = evaluator.rgb_image.copy()
        self.eval = evaluator
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
        display(self.image)
        
    def crop_image(self):
        center = np.mean(self.eval.gt_pixel_keypoints, axis=0)
        half_width = 150
        half_height = 150
        
        if center[0] - half_width < 0:
            center[0] = half_width
        elif center[0] + half_width > self.image.width:
            center[0] = self.image.width - half_width
            
        if center[1] - half_height < 0:
            center[1] = half_height
        elif center[1] + half_height > self.image.height:
            center[1] = self.image.height - half_height
            
        cropped_image = self.eval.rgb_image.crop((center[0] - half_width, center[1] - half_height, 
                                                  center[0] + half_width, center[1] + half_height))
        self.image = ImageOps.contain(cropped_image, (self.eval.width, self.eval.height))
        
    def plot_keypoints(self, keypoints, r=2, color=(0,255,0)):
        for keypoint in keypoints:
            x, y = keypoint[0], keypoint[1]
            self.drawing.ellipse((x-r, y-r, x+r, y+r), fill=color)
        
    def plot_gt_pixel_keypoints(self, r=2, color=(0,255,0)):
        self.plot_keypoints(self.eval.gt_pixel_keypoints, r, color)

    def plot_est_pixel_keypoints(self, r=2, color=(255,0,0)):
        self.plot_keypoints(self.eval.est_pixel_keypoints, r, color)        
        
    def plot_gt_world_keypoints(self):
        if not self.fig_3d:
            self.init_fig_3d()
        points = self.eval.gt_world_keypoints.T
        self.ax_3d.scatter(points[0], points[1], points[2],
                           c='green', label='Ground Truth Point Cloud')        
    
    def plot_est_world_keypoints(self):
        if not self.fig_3d:
            self.init_fig_3d()
        points = self.eval.est_world_keypoints.T
        self.ax_3d.scatter(points[0], points[1], points[2],
                           c='red', label='Estimated Point Cloud')
    
    def plot_gt_teaser_pose(self, scale=100):
        plot_pose(self.drawing, self.eval.gt_teaser_pose, self.eval.K)
    
    def plot_est_teaser_pose(self):
        plot_pose(self.drawing, self.eval.est_teaser_pose, self.eval.K, 
                  x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139))



# +
class MultiEvaluator():
    def __init__(self, folder, verbose=False):
        self.folder = folder
        self.evaluators = []
        self.df = pd.DataFrame({"average_keypoint_pixel_error":[],
                                "average_keypoint_world_error":[],
                                "translation_error":[],
                                "rotation_error":[],
                                "distance":[]})
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
        K_mat = np.array(metadata["intrinsics"]["intrinsic_matrix"])
        K_ros = np.array(metadata["ros_intrinsics"]["intrinsic_matrix"])
        cad_keypoints = np.array(metadata["cad_frame"])
        return metadata, K_mat, K_ros, cad_keypoints
        
    def eval_annotation(self, folder, annotation, K_mat, K_ros, cad_keypoints):
        rgb_image_path = folder + "/" + annotation["rgb_file_name"]
        rgb_image = Image.open(rgb_image_path).convert('RGB')
        depth_image_path = folder + "/" + annotation["depth_file_name"]
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        evaluator = SingleEvaluator(model, cad_keypoints, rgb_image, depth_image, 
                                    annotation, K_mat, verbose=self.verbose)
        self.evaluators.append(evaluator)
        self.df = self.df.append(evaluator.get_df())

        
    def run(self):
        for folder in glob.glob(self.folder + "/*"):                                       
            metadata, K_mat, K_ros, cad_keypoints = self.parse_metadata(folder)
            print("Starting folder :", folder)
            for annotation in tqdm.tqdm(metadata["annotations"]):
                self.eval_annotation(folder, annotation, K_mat, K_ros, cad_keypoints)
    
    def evaluate(self):
        pass
        
    

# -

class MultiPlotter():
    def __init__(self, multi_evaluator):
        self.multi_evaluator = multi_evaluator
        
    def eval_annotation(self, folder, annotation):
        evaluator = super().eval_annotation(folder, annotation)
        Single
        if self.verbose:
            self.cur_evaluator.plot_gt_pixel_keypoints()
            self.cur_evaluator.plot_est_pixel_keypoints()
            self.cur_evaluator.plot_gt_teaser_pose()
            self.cur_evaluator.plot_est_teaser_pose()
            self.cur_evaluator.plot_gt_world_keypoints()
            self.cur_evaluator.plot_est_world_keypoints()
            self.cur_evaluator.finalize_fig_3d()
            image = self.cur_evaluator.crop_object()
            display(image)
        
    
    def plot_keypoint_pixel_error_vs_distance(self):
        fig = plt.figure()
        self.multi_evaluator.df.plot(x="distance", y="average_keypoint_pixel_error")
        
    def plot_translation_error_vs_distance(self):
        fig = plt.figure()
        self.multi_evaluator.df.plot(x="distance", y="translation_error")
    
    def plot_rotation_error_vs_distance(self):
        fig = plt.figure()
        self.multi_evaluator.df.plot(x="distance", y="rotation_error")
          
    def plot_multiple(self, rows=3, cols=3):
        # %matplotlib inline

        fig = plt.figure(1,(20,20))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(rows,cols),
                         axes_pad=0.1,
                         share_all=True
                         )
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])

        for i in range(rows * cols):
            evaluator = random.choice(self.multi_evaluator.evaluators)
            plotter = SinglePlotter(evaluator)
            plotter.plot_gt_pixel_keypoints()
            plotter.plot_est_pixel_keypoints()
            plotter.plot_gt_teaser_pose()
            plotter.plot_est_teaser_pose()
            plotter.crop_image()
            grid[i].imshow(plotter.image)

data_folder_array = []
for data_folder in glob.glob(working_folder + "/*"):                                       
    data_folder_array.append(data_folder)

multi_evaluator = MultiEvaluator(working_folder, verbose=False)
multi_evaluator.run()

multi_evaluator.df

multi_plotter = MultiPlotter(multi_evaluator)
multi_plotter.plot_multiple()
multi_plotter.plot_keypoint_pixel_error_vs_distance()
multi_plotter.plot_translation_error_vs_distance()
multi_plotter.plot_rotation_error_vs_distance()

multi_plotter.df

for folder in data_folder_array:
    metadata_path = folder + "/metadata.json"
    f = open(metadata_path)
    metadata = json.load(f)
    K = np.array(metadata["intrinsics"]).T
    cad_keypoints = np.array(metadata["cad_frame"])
    for annotation in metadata["annotations"]:
        rgb_image_path = folder + "/" + annotation["rgb_file_name"]
        rgb_image = Image.open(rgb_image_path).convert('RGB')
        depth_image_path = folder + "/" + annotation["depth_file_name"]
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        
        drawing = ImageDraw.Draw(rgb_image)
        plotter = SinglePlotter(model, cad_keypoints, rgb_image, depth_image, 
                                annotation, K, drawing)
        plotter.plot_gt_pixel_keypoints()
        plotter.plot_est_pixel_keypoints()
        plotter.plot_gt_teaser_pose()
        plotter.plot_est_teaser_pose()
        plotter.plot_gt_world_keypoints()
        plotter.plot_est_world_keypoints()
        plotter.finalize_fig_3d()
        image = plotter.crop_object()
        display(image)




