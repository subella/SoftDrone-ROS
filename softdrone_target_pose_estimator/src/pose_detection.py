import numpy as np
# Converts Unity's left-handed coordinate system to ENU
# by swapping the y and z axes.
UNITY_TO_ENU = np.array([[1,0,0,0],
                         [0,0,1,0],
                         [0,1,0,0],
                         [0,0,0,1]])

from scipy.spatial.transform import Rotation as Rotation

def transform_kpts(src, transformation):
    """
    Transforms array of positions by the transformation.

    Parameters
    ----------
    src : 3,x np array
        Positions in the original frame.
    transformation : 4,4 np array
        Transformation matrix.

    Returns
    -------
    3,x np array
        Positions in the transformed frame.

    """
    ones = np.ones(src.shape[1])
    src_homo_T = np.vstack((src, ones)).T
    dst = (src_homo_T.dot(transformation.T)).T[:3]
    return dst

def kpts_unity_to_enu_np(kpts_unity):
    """
    Converts a Unity keypoints dict to numpy array in ENU.

    Parameters
    ----------
    unity_keypoints : dict
        Contains keys "x", "y", "z" for positions
        in Unity's left-handed frame.

    Returns
    -------
    3x20 np array
        Array of keypoint positions.
        Rows are x,y,z.

    """
    kpts_np = np.zeros((3,20))
    for kpt_id in range(len(kpts_unity)):
        kpt = kpts_unity[kpt_id]
        kpt_np = pos_unity_to_np(kpt)
        kpts_np[:,kpt_id] = kpt_np.reshape((3,))
    kpts_enu = transform_kpts(kpts_np, UNITY_TO_ENU)
    return kpts_enu

def pos_unity_to_np(pos_unity):
    """
    Converts a Unity position dict to numpy array.

    Parameters
    ----------
    pos_unity : dict
        Contains keys "x", "y", "z" for position
        in Unity's left-handed frame.

    Returns
    -------
    3, np array
        Position vector in np format.

    """
    pos_np = np.array([pos_unity["x"], pos_unity["y"], pos_unity["z"]])
    return pos_np

def quat_unity_to_np(quat_unity):
    """
    Converts a Unity quaternion dict to numpy array.

    Parameters
    ----------
    quat_unity : dict
        Contains keys "x", "y", "z", "w" for quaternion
        in Unity's left-handed frame.

    Returns
    -------
    4, np array
        Quaternion in np format.

    """
    quat_np = np.array([quat_unity["x"], quat_unity["y"], quat_unity["z"], quat_unity["w"]])
    return quat_np

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

def make_transformation_matrix(rot, trans):
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


def unity_to_enu_transformation(pos_unity, quat_unity):
    """
    Creates a transformation matrix from Unity to ENU.

    Parameters
    ----------
    pos_unity : dict
        Contains keys "x", "y", "z" for position
        in Unity's left-handed frame.
    quat_unity : dict
        Contains keys "x", "y", "z", "w" for quaternion
        in Unity's left-handed frame.

    Returns
    -------
    4,4 np array
        Transformation matrix.

    """

    quat_np = quat_unity_to_np(quat_unity)
    rot_matrix = quat_to_rot_matrix(quat_np)
    if pos_unity is not None:
        pos = pos_unity_to_np(pos_unity)
    else:
        pos = np.zeros((3,))

    transformation = make_transformation_matrix(rot_matrix, pos)
    unity_to_enu = UNITY_TO_ENU.dot(transformation).dot(UNITY_TO_ENU)
    return unity_to_enu

def plot_teaser(src, dst, est_dst=None, dst_gt_pose=None):
    """
    Plots keypoints for teaser visualization.

    Parameters
    ----------
    src : 3,x np array
        Keypoints in the target frame.
    dst : 3,x np array
        Ground truth keypoints wrt the camera frame.
    est_dst: 3,x np array
        dst keypoints computed using Teaser's pose
        estimate and transforming each src keypoint.
    dst_gt_pose : 3,x np array
        dst keypoints computed using the ground truth
        pose and transforming each src keypoint.

    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1, 1.5)
    ax.set_ylim3d(-0.5, 2)
    ax.set_zlim3d(-0.5, 2)
    ax.scatter(src[0], src[1], src[2], c='green')
    ax.scatter(dst[0], dst[1], dst[2], c='red')
    if est_dst is not None:
        ax.scatter(est_dst[0], est_dst[1], est_dst[2], c='purple')
    if dst_gt_pose is not None:
        ax.scatter(dst_gt_pose[0], dst_gt_pose[1], dst_gt_pose[2], c='orange')

def evaluate_teaser(est_transformation_arr, gt_transformation_arr):
    """
    Computes error metrics on Teaser's performance.

    Parameters
    ----------
    est_transformation_arr : x,4,4 np array
        Estimated positions for each image.
    gt_transformation_arr : x,4,4 np array
        Ground truth positions for each image.

    """
    est_rot_arr = est_transformation_arr[:,:3,:3]
    gt_rot_arr = gt_transformation_arr[:,:3,:3]
    est_pos_arr = est_transformation_arr[:,:3,3]
    gt_pos_arr = gt_transformation_arr[:,:3,3]

    # Rotation error represents the angle between estimated and ground truth matrices:
    # theta = arccos( (tr(R) - 1) / 2), R = Rd'R
    R = np.matmul(gt_rot_arr.transpose((0,2,1)), est_rot_arr)

    cos_theta = np.clip((np.trace(R, axis1=1, axis2=2) - 1)/2, -1, 1)
    # rot_err_norms ia a (n,) array with each entry the rotation error formula above.
    rot_error_norms = np.arccos(cos_theta)
    rot_error_avg = np.mean(rot_error_norms)

    # pos_error_norms is a (n,) array with each entry the error norm for the
    # corresponding image.
    pos_error_norms = np.linalg.norm(est_pos_arr - gt_pos_arr, axis=1)
    # pos_error_avg averages all of the error norms for each image.
    pos_error_avg = np.mean(pos_error_norms)
    print("Average Position Error", pos_error_avg)
    print("Average Rotation Error", rot_error_avg)

import teaserpp_python

# Initialize Teaser Solver
solver_params = teaserpp_python.RobustRegistrationSolver.Params()
solver_params.cbar2 = 1
solver_params.noise_bound = 0.00001
solver_params.estimate_scaling = True
solver_params.rotation_estimation_algorithm = (
    teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
)
solver_params.rotation_gnc_factor = 1.4
solver_params.rotation_max_iterations = 100
solver_params.rotation_cost_threshold = 1e-12
print("TEASER++ Parameters are:", solver_params)
teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

solver = teaserpp_python.RobustRegistrationSolver(solver_params)

src = np.array([[ 0.28649801,  0.28060094, -0.01200118, -0.29429922, -0.3186965 , -0.29710305,
                 -0.01500078,  0.27899802,  0.0668972 , -0.00940074, -0.08550046, -0.00789654,
                 -0.29969981, -0.28470284, -0.02579845,  0.26600331,  0.28299904,  0.26999822,
                 -0.01100318, -0.28660125],
                [ 0.09929965,  0.09290027,  0.09500089,  0.10449973,  0.0995018 ,  0.09759869,
                  0.10329929,  0.08889883,  0.13289766,  0.12540029,  0.12469915,  0.13820116,
                 -0.10209939, -0.10790227, -0.11089812, -0.10019808, -0.11090039, -0.11589961,
                 -0.11370148, -0.11309941],
                [ 0.184898  ,  0.37230077,  0.37549677,  0.36800182,  0.22080141,  0.0544985,
                  0.02500032,  0.04659964,  0.20920083,  0.2828986 ,  0.2040012 ,  0.12280256,
                  0.2233991 ,  0.37870058,  0.37869859,  0.37870073,  0.21009934,  0.04249672,
                  0.02689845,  0.05179719]])


def estimate_pose(dst):
    # Populate the parameters
    solver.solve(src, dst)
    solution = solver.getSolution()
    return solution.translation.copy(), solution.rotation.copy()
