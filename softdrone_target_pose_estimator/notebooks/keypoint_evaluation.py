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
from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch
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
        
        self.return_stats(verbose=True)

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
        gt_rotation = R.from_matrix(self.gt_teaser_pose[:3,:3])
        est_rotation = R.from_matrix(self.est_teaser_pose[:3,:3])
        gt_euler = gt_rotation.as_euler('zyx', degrees=True)
        est_euler = est_rotation.as_euler('zyx', degrees=True)
        error_euler = gt_euler - est_euler
        error_translation = self.gt_teaser_pose[:3,3] - self.est_teaser_pose[:3,3]
        return error_euler, error_translation
        
    def return_stats(self, verbose=False):
        stats = {}
        stats["average_keypoint_pixel_error"] = self.pixel_keypoints_error
        stats["average_keypoint_world_error"] = self.world_keypoints_error
        stats["translation_error"] = self.translation_error
        stats["rotation_error"] = self.rotation_error
        stats["distance"] = self.distance
        if self.verbose:
            print("Average Keypoint Pixel Error: ", self.pixel_keypoints_error)
            print("Average Keypoint World Error: ", self.world_keypoints_error)
            print("Euler Rotation Error: ", self.rotation_error)
            print("Translation Error", self.translation_error)
            print("gt", self.gt_teaser_pose)
            print("est", self.est_teaser_pose)
        

# -

class SinglePlotter(SingleEvaluator):
    def __init__(self, drawing, *args, **kwargs):
        self.drawing = drawing
        self.fig_3d = None
        self.ax_3d = None
        super().__init__(*args, **kwargs)
    
    def init_fig_3d(self):
        # %matplotlib inline
        self.fig_3d = plt.figure(figsize=(10, 10))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        
    def finalize_fig_3d(self):
        self.ax_3d.legend()
        plt.show()
        
    def crop_object(self):
        min_x, min_y = np.min(self.gt_pixel_keypoints, axis=0)
        max_x, max_y = np.max(self.gt_pixel_keypoints, axis=0)
        x_padding = 2 * (max_x - min_x)
        y_padding = 2 * (max_y - min_y)
        cropped_image = self.rgb_image.crop((min_x - x_padding, min_y - y_padding, 
                                         max_x + x_padding, max_y + y_padding))
        image = ImageOps.contain(cropped_image, (self.width, self.height))
        return image
        
    def plot_keypoints(self, keypoints, r=2, color=(0,255,0)):
        for keypoint in keypoints:
            x, y = keypoint[0], keypoint[1]
            self.drawing.ellipse((x-r, y-r, x+r, y+r), fill=color)
        
    def plot_gt_pixel_keypoints(self, r=2, color=(0,255,0)):
        self.plot_keypoints(self.gt_pixel_keypoints, r, color)

    def plot_est_pixel_keypoints(self, r=2, color=(255,0,0)):
        self.plot_keypoints(self.est_pixel_keypoints, r, color)        
        
    def plot_gt_world_keypoints(self):
        if not self.fig_3d:
            self.init_fig_3d()
        points = self.gt_world_keypoints.T
        self.ax_3d.scatter(points[0], points[1], points[2],
                           c='green', label='Ground Truth Point Cloud')        
    
    def plot_est_world_keypoints(self):
        if not self.fig_3d:
            self.init_fig_3d()
        points = self.est_world_keypoints.T
        self.ax_3d.scatter(points[0], points[1], points[2],
                           c='red', label='Estimated Point Cloud')
    
    def plot_gt_teaser_pose(self, scale=100):
        plot_pose(self.drawing, self.gt_teaser_pose, self.K)
    
    def plot_est_teaser_pose(self):
        plot_pose(self.drawing, self.est_teaser_pose, self.K, 
                  x_color=(139,0,0), y_color=(0,139,0), z_color=(0,0,139))



# +
class MultiEvaluator():
    def __init__(self, verbose=False):
        self.data = []
#         self.run(folder, verbose)
        
    def run(self, folder, verbose=False):
        for folder in glob.glob(working_folder + "/*"):                                       
            metadata_path = folder + "/metadata.json"
            f = open(metadata_path)
            metadata = json.load(f)
            K = np.array(metadata["intrinsics"]).T
            cad_keypoints = np.array(metadata["cad_frame"])
            print("Starting folder :", folder)
            for annotation in tqdm.tqdm(metadata["annotations"]):
                rgb_image_path = folder + "/" + annotation["rgb_file_name"]
                rgb_image = Image.open(rgb_image_path).convert('RGB')
                depth_image_path = folder + "/" + annotation["depth_file_name"]
                depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

                drawing = ImageDraw.Draw(rgb_image)
                plotter = SinglePlotter(drawing, model, cad_keypoints, rgb_image, depth_image, 
                                        annotation, K, verbose=verbose)
                self.data.append(plotter.return_stats())
                if verbose:
                 plotter.plot_gt_pixel_keypoints()
                 plotter.plot_est_pixel_keypoints()
                 plotter.plot_gt_teaser_pose()
                 plotter.plot_est_teaser_pose()
                 plotter.plot_gt_world_keypoints()
                 plotter.plot_est_world_keypoints()
                 plotter.finalize_fig_3d()
                 image = plotter.crop_object()
                 display(image)
    
    def evaluate(self):
        pass
        
    


# +
class MultiPlotter(MultiEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def plot_keypoint_error_vs_distance(self):
        fig = plt.figure()
        for data_point in self.data:
            
#         for data_point in self.


# -

data_folder_array = []
for data_folder in glob.glob(working_folder + "/*"):                                       
    data_folder_array.append(data_folder)

multi_plotter = MultiPlotter()
multi_plotter.run(working_folder)

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




