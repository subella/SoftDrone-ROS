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
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#ifndef REPROJECT_KEYPOINTS_HPP
#define REPROJECT_KEYPOINTS_HPP


namespace sdrone
{

class ReprojectKeypoints {

  public:

    ReprojectKeypoints();

    ReprojectKeypoints(Eigen::Matrix3d& camera_intrinsics);

    ~ReprojectKeypoints() = default;

    int reprojectKeypoints(Eigen::MatrixX2i& px_py_mat, cv::Mat& depth_img, Eigen::MatrixX3d& keypoints_3D_mat);

  protected:

    bool is_initialized_;

    Eigen::Matrix3d camera_intrinsics_;

    void init(Eigen::Matrix3d& camera_intrinsics);

    int reprojectSingleKeypoint(Eigen::Vector2i& px_py, double z, Eigen::Vector3d& keypoint_3D_vec);

    void depthImgToZ(Eigen::MatrixX2i px_py_mat, cv::Mat& depth_img, Eigen::Vector3d& keypoint_3D_vec);

};

}; //namespace sdrone

#endif