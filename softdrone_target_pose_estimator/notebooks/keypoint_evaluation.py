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
import json

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

sns.set(font_scale=1.5)
# -

target_name = "cardboard_box"

config_file = "../models/{}_pose.json".format(target_name)
model_file = "../models/{}_model.pth".format(target_name)
dataset_path = "/home/subella/src/AutomatedAnnotater/Data/"

model = KeypointDetector(config_file, model_file)

# TODO: standardize this.
training_folder = dataset_path + "Medkit" + "/Datasets/Training/"
validation_folder = dataset_path + "Medkit" + "/Datasets/Validation/"

working_folder = validation_folder


# +
def load_rgb_image(rgb_image_filename):
    return Image.open(rgb_image_filename).convert('RGB')

def load_depth_image(depth_image_filename):
    return cv2.imread(depth_image_filename, cv2.IMREAD_UNCHANGED)


# +
def write_metrics(metrics):
    data = json.dumps(metrics)
    filename = "{}_metrics.json".format(target_name)
    if not os.path.isfile(filename):
        f = open(filename,"w")
        f.write(data)
        f.close()
    else:
        print("File exists, please manually delete it first to update it.")

def load_metrics():
    with open("{}_metrics.json".format(target_name)) as json_file:
        metrics = json.load(json_file)
        return convert_metrics_to_np(metrics)
    return None

def convert_metrics_to_np(metrics):
    for metric in metrics:
        metric["gt_pixel_keypoints"] = np.array(metric["gt_pixel_keypoints"])
        metric["est_pixel_keypoints"] = np.array(metric["est_pixel_keypoints"])
        metric["gt_world_keypoints"] = np.array(metric["gt_world_keypoints"])
        metric["gt_teaser_pose"] = np.array(metric["gt_teaser_pose"])
        metric["est_teaser_pose"] = np.array(metric["est_teaser_pose"])
        metric["K"] = np.array(metric["K"])
    return metrics


# -

class SingleEvaluator():
    def __init__(self, model, cad_keypoints, 
                 rgb_image_filename, depth_image_filename, 
                 annotation, K, verbose=False):
        self.rgb_image_filename = rgb_image_filename
        self.rgb_image = load_rgb_image(rgb_image_filename)
        self.depth_image = load_depth_image(depth_image_filename)
        self.width = self.rgb_image.width
        self.height = self.rgb_image.height
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
        
    def is_valid(self):
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
        metrics["gt_world_keypoints"] = self.gt_world_keypoints.tolist()
        metrics["gt_teaser_pose"] = self.gt_teaser_pose.tolist()
        metrics["est_teaser_pose"] = self.est_teaser_pose.tolist()
        metrics["rgb_image_filename"] = self.rgb_image_filename
        metrics["average_keypoint_pixel_error"] = self.pixel_keypoints_error
        metrics["average_keypoint_world_error"] = self.world_keypoints_error
        metrics["translation_error"] = self.translation_error
        metrics["rotation_error"] = self.rotation_error
        metrics["distance"] = self.distance
        metrics["K"] = self.K.tolist()
        metrics["is_valid"] = self.is_valid()
        return metrics       



class SinglePlotter():
    def __init__(self, single_evaluator_metrics):
        self.metrics = single_evaluator_metrics
        self.rgb_image = load_rgb_image(self.metrics["rgb_image_filename"])
        self.drawing = ImageDraw.Draw(self.rgb_image)
        self.K = self.metrics["K"]
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
        
    def crop_image(self):
        center = np.mean(self.metrics["gt_pixel_keypoints"], axis=0)
        half_width = 150
        half_height = 150
        
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
        
    def plot_keypoints(self, keypoints, r=2, color=(0,255,0)):
        for keypoint in keypoints:
            x, y = keypoint[0], keypoint[1]
            self.drawing.ellipse((x-r, y-r, x+r, y+r), fill=color)
        
    def plot_gt_pixel_keypoints(self, r=2, color=(0,255,0)):
        self.plot_keypoints(self.metrics["gt_pixel_keypoints"], r, color)

    def plot_est_pixel_keypoints(self, r=2, color=(255,0,0)):
        self.plot_keypoints(self.metrics["est_pixel_keypoints"], r, color)        
        
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
        plot_pose(self.drawing, self.metrics["gt_teaser_pose"], self.K)
    
    def plot_est_teaser_pose(self):
        plot_pose(self.drawing, self.metrics["est_teaser_pose"], self.K, 
                  x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139))


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
        K_mat = np.array(metadata["intrinsics"]["intrinsic_matrix"])
        K_ros = np.array(metadata["ros_intrinsics"]["intrinsic_matrix"])
        cad_keypoints = np.array(metadata["cad_frame"])
        return metadata, K_mat, K_ros, cad_keypoints
        
    def eval_annotation(self, folder, annotation, K_mat, K_ros, cad_keypoints):
        rgb_image_filename = folder + "/" + annotation["rgb_file_name"]
        depth_image_filename = folder + "/" + annotation["depth_file_name"]
        evaluator = SingleEvaluator(model, cad_keypoints, 
                                    rgb_image_filename, depth_image_filename, 
                                    annotation, K_mat, verbose=self.verbose)
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
                                                     "rotation_error",
                                                     "distance",
                                                     "is_valid"])
        self.verbose = verbose
    
    def plot_keypoint_pixel_error_vs_distance(self):
        fig = plt.figure(figsize=(15,8))
        pruned_df = self.df[self.df["average_keypoint_pixel_error"] < 10]
#         pruned_df = self.df[self.df["is_valid"] == True]
        pruned_df = self.df
        data_used = int(len(pruned_df) / len(self.df) * 100)
        
        values = np.vstack([pruned_df["distance"], pruned_df["average_keypoint_pixel_error"]])
        kernel = stats.gaussian_kde(values)(values)
        ax = sns.scatterplot(
            data=pruned_df,
            x="distance",
            y="average_keypoint_pixel_error",
            c=kernel,
            cmap="viridis"
            )
#         ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
#         density = ax.scatter_density(pruned_df["distance"], pruned_df["average_keypoint_pixel_error"], cmap=white_viridis)
#         fig.colorbar(density, label='Number of points per pixel')
#         ax = sns.kdeplot(data=pruned_df, x="distance", y="average_keypoint_pixel_error")
        ax.set_title("Average Keypoint Error (using {}% of data)".format(data_used))
        ax.set(xlabel="Distance [mm]", ylabel="Keypoint Error [px]")
        
    def plot_translation_error_vs_distance(self):
        fig = plt.figure(figsize=(15,8))
        pruned_df = self.df[self.df["translation_error"] < 250]
#         pruned_df = self.df[self.df["is_valid"] == True]
        pruned_df = self.df
        data_used = int(len(pruned_df) / len(self.df) * 100)
        values = np.vstack([pruned_df["distance"], pruned_df["translation_error"]])
        kernel = stats.gaussian_kde(values)(values)
        ax = sns.scatterplot(
            data=pruned_df,
            x="distance",
            y="translation_error",
            c=kernel,
            cmap="viridis"
            )
        ax.set_title("Translation Error (using {}% of data)".format(data_used))
        ax.set(xlabel="Distance [mm]", ylabel="Translation Error [mm]")
        
    def plot_rotation_error_vs_distance(self):
        fig = plt.figure(figsize=(15,8))
        pruned_df = self.df[self.df["rotation_error"] < 50]
#         pruned_df = self.df[self.df["is_valid"] == True]
        pruned_df = self.df
        data_used = int(len(pruned_df) / len(self.df) * 100)
        values = np.vstack([pruned_df["distance"], pruned_df["rotation_error"]])
        kernel = stats.gaussian_kde(values)(values)
        ax = sns.scatterplot(
            data=pruned_df,
            x="distance",
            y="rotation_error",
            c=kernel,
            cmap="viridis"
            )
        
#         ax = sns.scatterplot(data=pruned_df, x="distance", y="rotation_error")
        ax.set_title("Rotation Error (using {}% of data)".format(data_used))
        ax.set(xlabel="Distance [mm]", ylabel="Rotation Error [Deg]")
          
    def plot_multiple(self, rows=3, cols=3):
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
        for metric in self.metrics:
            fig = plt.figure()
            plotter = SinglePlotter(metric)
            plotter.plot_gt_pixel_keypoints()
            plotter.plot_est_pixel_keypoints()
            plotter.plot_correspondences()
            plotter.plot_gt_teaser_pose()
            plotter.plot_est_teaser_pose()
            plotter.crop_image()
            plotter.show_image()
            
            if self.verbose:
                print("Average Keypoint Pixel Error: ", metric["average_keypoint_pixel_error"])
                print("Average Keypoint World Error: ", metric["average_keypoint_world_error"])
                print("Euler Rotation Error: ", metric["rotation_error"])
                print("Translation Error", metric["translation_error"])
                print("gt", metric["gt_teaser_pose"])
                print("est", metric["est_teaser_pose"])


# -

multi_evaluator = MultiEvaluator(working_folder, verbose=False)
multi_evaluator.run()

write_metrics(multi_evaluator.metrics)

metrics = load_metrics()

metrics = convert_metrics_to_np(multi_evaluator.metrics)

multi_plotter = MultiPlotter(metrics, verbose=True)
# multi_plotter.plot_sequence()
multi_plotter.plot_multiple()
multi_plotter.plot_keypoint_pixel_error_vs_distance()
multi_plotter.plot_translation_error_vs_distance()
multi_plotter.plot_rotation_error_vs_distance()


