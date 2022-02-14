// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/pose_estimator.hpp>

#define MAXBUFFERSIZE  ((int) 300)

namespace sdrone
{

PoseEstimator::
PoseEstimator()
    : solver_()
{
  is_initialized_ = false;
};

PoseEstimator::
PoseEstimator(const std::string&  cad_frame_file_name,
              const TeaserParams& params)
    :solver_(params)
{
    init(cad_frame_file_name);
    is_initialized_ = true;
};

void PoseEstimator::
init(const std::string&  cad_frame_file_name)
{
    initCadFrame(cad_frame_file_name);
};

void PoseEstimator::
initCadFrame(const std::string& cad_frame_file_name)
{
    int rows = 0;
    int cols = 3;
    double buffer[MAXBUFFERSIZE];
    std::ifstream file(cad_frame_file_name);
    double x, y, z;
    while (file >> x >> y >> z)
    {
        buffer[rows * cols] = x;
        buffer[rows * cols + 1] = y;
        buffer[rows * cols + 2] = z;
        rows++;
    }

    Eigen::MatrixX3d matrix(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix(i,j) = buffer[cols * i + j];

    cad_frame_keypoints_ = matrix.transpose();
}

int PoseEstimator::
solveTransformation(Eigen::Matrix3Xd& keypoints_3D, Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
    if (!is_initialized_)
        return 0;

    try
    {
        solver_.solve(cad_frame_keypoints_, keypoints_3D);
        auto solution = solver_.getSolution();
        R = solution.rotation;
        t = solution.translation;
        return 1;
    }
    catch (...)
    {
        return 0;
    }
}

}