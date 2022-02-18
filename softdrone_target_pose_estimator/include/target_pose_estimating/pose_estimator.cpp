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
    :solver_()
{
  is_initialized_ = false;
};

PoseEstimator::
PoseEstimator(const std::string&  cad_frame_file_name,
              const TeaserParams& params)
    : solver_(params)
{
    init(cad_frame_file_name);
    params_ = params;

    // Supress all cout output because we can't turn off 
    // pmc output.
    //std::cout.rdbuf(NULL);
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
        // Eigen::Matrix3Xd keypoints_3D2 (3, 13);
        // keypoints_3D2.row(0) << 33.2724982, 4.9671052, 1.25381468, -31.26401346, 21.82148281,
        //                        20.08924026, -14.09728151, -66.51182183, -69.04602762, -105.7634699,
        //                        -126.35688104, 21.54286407, 9.59531259;

        // keypoints_3D2.row(1) << 164.79990612, 168.29259231, 154.91091854, 159.63966852, 239.96312074,
        //                        219.90923897, 224.50828476, 232.98404034, 253.64538866, 254.1953732,
        //                        216.87030347, 268.9356993,  270.24513745;

        // keypoints_3D2.row(2) << 780.78155908, 747.98341003, 813.90712514, 785.87050527, 774.19316156,
        //                        771.4787762,  724.36339773, 680.92319297, 679.28279643, 676.00235736,
        //                        676.99352248, 822.48078072, 809.36966347;
        std::cout << keypoints_3D << std::endl << std::flush;
        //TeaserSolver solver(params_);
        solver_.solve(cad_frame_keypoints_, keypoints_3D);
        auto solution = solver_.getSolution();
        R = solution.rotation;
        t = solution.translation;
        solver_.reset(params_);
        return 1;
    }
    catch (...)
    {
        return 0;
    }
}

}