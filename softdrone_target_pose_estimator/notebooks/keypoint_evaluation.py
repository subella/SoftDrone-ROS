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

model = load_model_from_json(10, "../models/medkit_pose.json", "/home/subella/src/trt_pose/tasks/target_pose/configs/medkit/target.json.checkpoints/epoch_75.pth")
opt_model = optimize_model(model)

OPTIMIZED_MODEL = 'medkit_softdrone.pth'
torch.save(opt_model.state_dict(), OPTIMIZED_MODEL)

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

target_name = "medkit"

config_file = "../models/{}_pose.json".format(target_name)
model_file = "../models/{}_model.pth".format(target_name)
dataset_path = "/home/subella/src/AutomatedAnnotater/Data/"

# model = KeypointDetector(config_file, model_file)
model = KeypointDetector(config_file, "medkit_softdrone.pth")

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
        metric["est_all_inliers_teaser_pose"] = np.array(metric["est_all_inliers_teaser_pose"])
        metric["K_ros"] = np.array(metric["K_ros"])
        metric["K_mat"] = np.array(metric["K_mat"])
    return metrics


# +
class SingleEvaluator():
    def __init__(self, model, cad_keypoints, 
                 rgb_image_filename, depth_image_filename, 
                 annotation, K_ros, K_mat, verbose=False):
        self.rgb_image_filename = rgb_image_filename
        self.rgb_image = load_rgb_image(rgb_image_filename)
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
        self.est_world_keypoints = self.get_est_world_keypoints()
        self.gt_teaser_pose = self.get_gt_teaser_pose()
        self.est_teaser_pose = self.get_est_teaser_pose()
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
    
    def get_gt_world_keypoints(self):
        return np.array(self.annotation["ground_truth_keypoints"])
    
    def get_est_world_keypoints(self):
        pxs = np.clip(self.est_pixel_keypoints[:,0].astype(int), 0, self.width - 1)
        pys = np.clip(self.est_pixel_keypoints[:,1].astype(int), 0 , self.height - 1)
        z = self.depth_image[pys,pxs]
        est_world_keypoints = reproject(pxs, pys, z, self.K_ros)
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
#         print("Inliers", inliers)
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
        self.num_inliers = len(inliers)
        if self.num_inliers == 0:
            self.is_valid = False
        else:
            self.is_valid = True
        return tf
        
    def get_est_all_inliers_teaser_pose(self):
        tf, inliers = self.get_base_est_teaser_pose(self.all_inliers_solver_params, self.all_inliers_solver)
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
        metrics["gt_world_keypoints"] = self.gt_world_keypoints.tolist()
        metrics["est_world_keypoints"] = self.est_world_keypoints.tolist()
        metrics["gt_teaser_pose"] = self.gt_teaser_pose.tolist()
        metrics["est_teaser_pose"] = self.est_teaser_pose.tolist()
        metrics["est_all_inliers_teaser_pose"] = self.est_all_inliers_teaser_pose.tolist()
        metrics["rgb_image_filename"] = self.rgb_image_filename
        metrics["average_keypoint_pixel_error"] = self.pixel_keypoints_error
        metrics["average_keypoint_world_error"] = self.world_keypoints_error
        metrics["translation_error"] = self.translation_error
        metrics["all_inliers_translation_error"] = self.all_inliers_translation_error
        metrics["rotation_error"] = self.rotation_error
        metrics["all_inliers_rotation_error"] = self.all_inliers_rotation_error
        metrics["distance"] = self.distance
        metrics["K_ros"] = self.K_ros.tolist()
        metrics["K_mat"] = self.K_mat.tolist()
        metrics["is_valid"] = self.is_valid
        metrics["num_inliers"] = self.num_inliers
        return metrics       



# -

class SinglePlotter():
    def __init__(self, single_evaluator_metrics):
        self.metrics = single_evaluator_metrics
        self.rgb_image = load_rgb_image(self.metrics["rgb_image_filename"])
        
        self.upscale = 2
        width, height = self.rgb_image.size
        self.rgb_image = self.rgb_image.resize((width * self.upscale, height*self.upscale))
        

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
        
    def plot_keypoints(self, keypoints, r=4, color=(0,255,0)):
        r = r * self.upscale
        for keypoint in keypoints:
            x, y = keypoint[0] * self.upscale, keypoint[1] * self.upscale
            self.drawing.ellipse((x-r, y-r, x+r, y+r), fill=color)
        
    def plot_gt_pixel_keypoints(self, r=2, color=(0,255,0)):
        self.plot_keypoints(self.metrics["gt_pixel_keypoints"], color=color)

    def plot_est_pixel_keypoints(self, r=2, color=(255,0,0)):
        self.plot_keypoints(self.metrics["est_pixel_keypoints"], color=color)        
        
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
        plot_pose(self.drawing, self.metrics["gt_teaser_pose"], self.K_mat, resize_factor=self.upscale)
    
    def plot_est_teaser_pose(self):
        plot_pose(self.drawing, self.metrics["est_teaser_pose"], self.K_ros, 
                  x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139), resize_factor=self.upscale)


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
        pruned_df = self.df[self.df["is_valid"] == True]
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
#             c=z,
#             cmap="viridis",
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
        pruned_df = self.df[self.df["is_valid"] == True]
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
        step_size = 100
        while i < len(self.metrics):
            print(self.metrics[i]["rgb_image_filename"])
            metric = self.metrics[i]
            fig = plt.figure()
            plotter = SinglePlotter(metric)
            plotter.plot_gt_pixel_keypoints()
            plotter.plot_est_pixel_keypoints()
#             plotter.plot_correspondences()
            plotter.plot_gt_teaser_pose()
            plotter.plot_est_teaser_pose()
    #             plotter.crop_image()
            plotter.show_image()

            fig = plt.figure()
    #             plotter.crop_image()
            plotter.show_image()

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

write_metrics(multi_evaluator.metrics)

metrics = load_metrics()

metrics2 = convert_metrics_to_np(multi_evaluator.metrics)

multi_plotter.df

multi_plotter = MultiPlotter(metrics, verbose=True)
# multi_plotter.plot_sequence()
# multi_plotter.plot_multiple()
multi_plotter.plot_keypoint_pixel_error_vs_distance()
multi_plotter.plot_translation_error_vs_distance()
multi_plotter.plot_rotation_error_vs_distance()
multi_plotter.plot_inliers()
# multi_plotter.plot_composite()

# +
import numpy as np

# load image as grayscale
img = cv2.imread("/home/subella/src/AutomatedAnnotater/Data/Medkit/Datasets/Validation/2023-02-15-11-29-06/depth_image_615.png", cv2.IMREAD_ANYDEPTH)
max_ = np.max(img)
print(max_)
print(img)
img = (img * 255.0/(max_/4)).astype(np.uint8)
print(np.max(img))
im = Image.fromarray(img)
display(im)

# +
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis, dpi=10)
    fig.colorbar(density, label='Number of points per pixel')

fig = plt.figure()
using_mpl_scatter_density(fig, x, y)
plt.show()

# +
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd


def using_datashader(ax, x, y):

    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=0,
        vmax=35,
        norm="linear",
        aspect="auto",
        ax=ax,
    )

    plt.colorbar(dsartist)


fig, ax = plt.subplots()
using_datashader(ax, x, y)
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

# Generate fake data
x = np.random.normal(size=1000)
y = x * 3 + np.random.normal(size=1000)

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax_ = ax.scatter(x, y, c=z, s=50)
plt.colorbar(ax_)

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax

def convolve(xs, ys, box_height=5, box_width=50):
    y_range = max(ys) - min(ys)
    x_range = max(xs) - min(xs)
    
    data = np.array([xs, ys]).T
    colors = []
    box_size = 0.05
    box_width = x_range * box_size
    box_height = y_range * box_size
    for (x,y) in data:
        c = -1
        for (nx, ny) in data:
            if nx < x + box_width and nx > x - box_width and ny < y + box_height and ny > y - box_height:
                c += 1
        colors.append(c)
    colors = np.array(colors).astype(np.float)
    colors /= np.max(colors)
    return colors
    
    

if "__main__" == __name__ :

    x = np.random.normal(size=1000)
    y = x * 3 + np.random.normal(size=1000)
#     print(x)
#     print(y)
    convolve(x,y)
#     density_scatter( x, y, bins = [30,30] )
# -

a = np.array([[0,1]])
plt.figure(figsize=(.3, 3))
img = plt.imshow(a, cmap="viridis")
plt.gca().set_visible(False)
cax = plt.axes()
plt.colorbar( cax=cax, ticks=[0,1])
plt.savefig("colorbar.svg")


