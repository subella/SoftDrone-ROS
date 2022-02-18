// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoints_3D.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/reproject_keypoints.hpp>

#define DEPTH_CONVERSION ((double) 1)

namespace sdrone
{

ReprojectKeypoints::
ReprojectKeypoints()
{
  is_initialized_ = false;
};

ReprojectKeypoints::
ReprojectKeypoints(Eigen::Matrix3d& camera_intrinsics)
{
  init(camera_intrinsics);
};

void ReprojectKeypoints::
init(Eigen::Matrix3d& camera_intrinsics)
{
    camera_intrinsics_ = camera_intrinsics;
    is_initialized_ = true;
};

int ReprojectKeypoints::
reprojectSingleKeypoint(Eigen::Vector2i& px_py, double z, Eigen::Vector3d& keypoint_3D_vec)
{
    if (!is_initialized_)
      return 0;
    double fx = camera_intrinsics_(0,0);
    double fy = camera_intrinsics_(1,1);
    double cx = camera_intrinsics_(0,2);
    double cy = camera_intrinsics_(1,2);
    int px = px_py(0);
    int py = px_py(1);
    double x = (px - cx) * z / fx;
    double y = (py - cy) * z / fy;
    keypoint_3D_vec << x, y, z;
    return 1;
}

int ReprojectKeypoints::
reprojectKeypoints(Eigen::MatrixX2i& px_py_mat, cv::Mat& depth_img, Eigen::MatrixX3d& keypoints_3D_mat)
{
  if (!is_initialized_)
    return 0;

  for (int i=0; i < px_py_mat.rows(); i++)
  {
    Eigen::Vector2i px_py = px_py_mat.row(i);
    unsigned short depth_img_z = depth_img.at<unsigned short>(px_py[1], px_py[0]);
    double z = DEPTH_CONVERSION * depth_img_z;
    Eigen::Vector3d keypoint_3D_vec;
    reprojectSingleKeypoint(px_py, z, keypoint_3D_vec);
    keypoints_3D_mat.row(i) = keypoint_3D_vec;
  }

  return 1;
}

}