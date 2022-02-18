// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Class for solving point cloud rigid transformation.
 */
//-----------------------------------------------------------------------------
#include <fstream>
#include <teaser/registration.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef Pose_Estimator_HPP
#define Pose_Estimator_HPP


namespace sdrone
{

typedef teaser::RobustRegistrationSolver::Params TeaserParams;
typedef teaser::RobustRegistrationSolver TeaserSolver;

class PoseEstimator {

  public:

    PoseEstimator();

    PoseEstimator(const std::string&  cad_frame_file_name,
                  const TeaserParams& params);

    int solveTransformation(Eigen::Matrix3Xd& keypoints_3D, Eigen::Matrix3d& R, Eigen::Vector3d& t);

  protected:

    bool is_initialized_;

    Eigen::Matrix3Xd cad_frame_keypoints_;

    teaser::RobustRegistrationSolver solver_;

    TeaserParams params_;

    void init(const std::string& cad_frame_file_name);

    void initCadFrame(const std::string& cad_frame_file_name);   

};

}; //namespace sdron

#endif // Pose_Estimator_HPP