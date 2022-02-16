// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/pose_estimator.hpp>

namespace softdrone
{

PoseEstimator::
PoseEstimator()
{
  is_initialized_ = false;
};

PoseEstimator::
PoseEstimator(const std::string& cad_frame_file_name)
{
  is_initialized_ = false;

};

void PoseEstimator::
init(const std::string& cad_frame_file_name)
{
    std::ifstream file(cad_frame_file_name);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            printf("%s", line.c_str());
        }
        file.close();
    }
};



}