// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    reproject_keypoints.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Class for converting 2D keypoints in pixels to 3D
 *        coordinates in meters given depth.
 */
//-----------------------------------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>

#ifndef REPROJECT_KEYPOINTS_HPP
#define REPROJECT_KEYPOINTS_HPP


namespace sdrone
{

class ReprojectKeypoints {

  public:

    ReprojectKeypoints();

    ReprojectKeypoints(Eigen::Matrix4d& camera_intrinsics);

    ~ReprojectKeypoints() = default;

  protected:

    bool is_initialized_;

    Eigen::Matrix4d camera_intrinsics_;

    void init(Eigen::Matrix4d& camera_intrinsics);

    void reprojectTo3D(double px, double py, double z, Eigen::Vector3d& point_3D);

    void reprojectKeypoints(Eigen::MatrixX2d& keypoints_2D, Eigen::MatrixX3d& keypoints_3D);

};

}; //namespace sdrone

#endif