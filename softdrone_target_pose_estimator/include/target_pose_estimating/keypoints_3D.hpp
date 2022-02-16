// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoints_3D.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Class for converting 2D keypoints in pixels to 3D
 *        coordinates in meters given depth.
 */
//-----------------------------------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>

#ifndef Keypoints_3D_HPP
#define Keypoints_3D_HPP


namespace softdrone
{

class Keypoints3D {
  public:
    typedef Eigen::Matrix<double, 4, 4> Matrix4d;

    Keypoints3D();

    Keypoints3D(Matrix4d& camera_intrinsics);

    ~Keypoints3D() = default;

  protected:

    bool is_initialized_;

    Matrix4d camera_intrinsics_;

    void init(Matrix4d& camera_intrinsics);

    void reprojectTo3D(double px, double py, double z, Eigen::Vector3d& point_3D);

    void reprojectKeypoints(Eigen::ArrayXXd& keypoints_2D, Eigen::ArrayXXd& keypoints_3D);

};

}; //namespace softdrone

#endif