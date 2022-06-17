import numpy as np
from scipy.spatial.transform import Rotation as R

class Pose(object):
    def __init__(self, 
                 translation=None, 
                 rotation=None,
                 tf_matrix=None):

        if translation is not None and rotation is not None:
            self.init_from_euler(translation, rotation)
        elif tf_matrix is not None:
            self.init_from_tf_matrix(tf_matrix)
        else:
            self.tf_matrix = np.eye((4))

    def init_from_euler(self, translation, rotation):
        r = np.array(R.from_euler('xyz', rotation, degrees=True).as_dcm())
        t = np.array(translation).reshape(3,1)
        T = np.hstack((r, t))
        self.tf_matrix = np.vstack((T, [0,0,0,1]))

    def init_from_tf_matrix(self, tf_matrix):
        self.tf_matrix = tf_matrix

    def get_translation(self):
        translation = self.tf_matrix[:3,3].T
        return translation.tolist()

    def get_rotation(self):
        r = self.tf_matrix[:3,:3]
        rotation = R.from_dcm(r).as_euler('xyz', degrees=True)
        return rotation.tolist()

def pose_from_dict(params):
    return Pose(translation=params["translation"],
                rotation=params["rotation"])

def make_tf_list(tf_list, parser):
    if parser.parent is None:
        return tf_list
    tf_list.append(pose_from_dict(parser.top_params))
    return make_tf_list(tf_list, parser.parent)

def point_to_world(tf_list, point_wrt_child=Pose()):
    child_wrt_world = Pose()
    for pose in tf_list[::-1]:
        child_wrt_world.tf_matrix = child_wrt_world.tf_matrix.dot(pose.tf_matrix)
    point_wrt_world = Pose()    
    point_wrt_world.tf_matrix = child_wrt_world.tf_matrix.dot(point_wrt_child.tf_matrix)
    return point_wrt_world.get_translation(), point_wrt_world.get_rotation()




