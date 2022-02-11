// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoints_3D.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/reproject_keypoints.hpp>

namespace sdrone
{

ReprojectKeypoints::
ReprojectKeypoints()
{
  is_initialized_ = false;
};

ReprojectKeypoints::
ReprojectKeypoints(Eigen::Matrix4d& camera_intrinsics)
{
  init(camera_intrinsics);
};

void ReprojectKeypoints::
init(Eigen::Matrix4d& camera_intrinsics)
{
    camera_intrinsics_ = camera_intrinsics;
    is_initialized_ = true;
};

void ReprojectKeypoints::
reprojectTo3D(double px, double py, double z, Eigen::Vector3d& point_3D)
{
    double fx = camera_intrinsics_(0,0);
    double fy = camera_intrinsics_(1,1);
    double cx = camera_intrinsics_(0,2);
    double cy = camera_intrinsics_(1,2);
    double x = (px - cx) * z / fx;
    double y = (py - cy) * z / fy;
    point_3D << x, y, z;
}

void ReprojectKeypoints::
reprojectKeypoints(Eigen::MatrixX2d& keypoints_2D, Eigen::MatrixX3d& keypoints_3D)
{

}

}