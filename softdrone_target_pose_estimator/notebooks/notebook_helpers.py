import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.transform import Rotation as Rotation
from PIL import Image, ImageDraw


# Camera params for the d455 with bad connector
d455_model_1 = np.array([[637.08242287, 0.0, 646.6643606],
                         [0.0, 636.93770311, 374.53418],
                         [0.0, 0.0, 1.0]])

def project(point, K):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    px = int((fx * point[0] / point[2] + cx))
    py = int((fy * point[1] / point[2] + cy))
    return [px, py]

def reproject(px, py, z, K):
    """
        Convert pixel coordinates + depth to world coordinates.
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy
    pos = np.array([x, y, z]).T
    return pos

def quat_to_rot_matrix(quat):
    """
    Converts a quaternion to rotation matrix.

    Parameters
    ----------
    quat : 4, np array
        Quaternion in numpy format x, y, z, w.

    Returns
    -------
    3,3 np array
        Rotation matrix.

    """
    r = Rotation.from_quat(quat)
    rot_np = r.as_matrix()
    return rot_np

def make_tf(rot, trans):
    """
    Creates a transformation matrix.

    Parameters
    ----------
    rot : 3,3 np array
        Rotation matrix.
    tras : 3, np array
        Translation vector.

    Returns
    -------
    4,4 np array
        Transformation matrix.

    """
    transformation = np.empty((4,4))
    transformation[:3, :3] = rot
    transformation[:3, 3] = trans
    transformation[3, :] = [0, 0, 0, 1]
    return transformation

def make_tf_from_quat(quat, pos):
    rot = quat_to_rot_matrix(quat)
    tf = make_tf(rot, pos)
    return tf


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_pose(drawing, tf, K, x_color=(255,0,0), y_color=(0,255,0), z_color=(0,0,255), scale=150, resize_factor=1):
     center = tf[:3,3].T
     center_px = resize_factor * np.array(project(center, K))
     drawing.ellipse((center_px[0] - 2, center_px[1] - 2,
                           center_px[0] + 2, center_px[1] + 2),
                           fill=(0,0,0))
     x_tip = center +  tf[:3,0] * scale
     x_tip_px = resize_factor * np.array(project(x_tip, K))
     drawing.line((center_px[0], center_px[1], x_tip_px[0], x_tip_px[1]), \
                   fill=x_color, width=5*resize_factor)
     y_tip = center + tf[:3,1] * scale
     y_tip_px = resize_factor * np.array(project(y_tip, K))
     drawing.line((center_px[0], center_px[1], y_tip_px[0], y_tip_px[1]), \
                   fill=y_color, width=5*resize_factor)
     z_tip = center + tf[:3,2] * scale
     z_tip_px = resize_factor * np.array(project(z_tip, K))
     drawing.line((center_px[0], center_px[1], z_tip_px[0], z_tip_px[1]), \
                   fill=z_color, width=5*resize_factor)

