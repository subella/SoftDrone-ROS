// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoints_3D.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/keypoints_3D.hpp>

namespace softdrone
{

Keypoints3D::
Keypoints3D()
{
  is_initialized_ = false;
};

Keypoints3D::
Keypoints3D(Matrix4d& camera_intrinsics)
{
  is_initialized_ = false;
  init(camera_intrinsics);

};

void Keypoints3D::
init(Matrix4d& camera_intrinsics)
{
    camera_intrinsics_ = camera_intrinsics;
};

void Keypoints3D::
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

void reprojectKeypoints(Eigen::ArrayXXd& keypoints_2D, Eigen::ArrayXXd& keypoints_3D)
{
    for(auto row : keypoints_2D.rowwise())
        std::cout << row;

}
}