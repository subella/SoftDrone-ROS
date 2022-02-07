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


namespace softdrone
{

class PoseEstimator {
  public:

    PoseEstimator();

    PoseEstimator(const std::string& cad_frame_file_name);

  protected:

    bool is_initialized_;

    teaser::RobustRegistrationSolver::Params params;

    teaser::RobustRegistrationSolver solver;

    void init(const std::string& cad_frame_file_name);

};

}; //namespace soft

#endif // Tracker_HPP