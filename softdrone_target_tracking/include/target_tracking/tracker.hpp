// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    tracker.hpp
 * @author  Jared Strader
 * 
 * @brief Class for tracking the 7D pose of a target using an EKF.
 */
//-----------------------------------------------------------------------------

#ifndef Tracker_HPP
#define Tracker_HPP

#include <mrpt/math/CQuaternion.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPose3DPDFGaussian.h>
#include <mrpt/poses/CPose3DQuat.h>
#include <mrpt/poses/CPose3DQuatPDFGaussian.h>
#include <mrpt/poses/CPoseRandomSampler.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>

namespace sdrone
{

class Tracker {
  public:
    typedef mrpt::math::CQuaternionDouble Quat;
    typedef mrpt::poses::CPose3D Pose6D;
    typedef mrpt::poses::CPose3DQuat Pose7D;
    typedef mrpt::poses::CPose3DPDFGaussian Belief6D;
    typedef mrpt::poses::CPose3DQuatPDFGaussian Belief7D;
    typedef Eigen::Matrix<double,7,1> Mat71;
    typedef Eigen::Matrix<double,7,7> Mat77;
    typedef Eigen::Matrix<double,6,1> Mat61;
    typedef Eigen::Matrix<double,6,6> Mat66;
    typedef mrpt::math::CMatrixDouble66 CMat66;
    typedef mrpt::math::CMatrixDouble77 CMat77;
    typedef mrpt::poses::CPoseRandomSampler Sampler;

    /** \brief  */
    Tracker();

    /** \brief  */
    Tracker(const Belief7D &b_target_meas);
    Tracker(const Belief6D &b_target_meas);

  protected:
    /** \brief */
    bool is_initialized_;

    /** \brief */
    Belief7D b_target_;
    Belief6D b_target_6D_;

    /** \brief */
    void init(const Belief7D &b_target_meas);
    void init(const Belief6D &b_target_meas);

    /** \brief */
    void update(const Belief7D &b_target_meas);
    void update(const Belief6D &b_target_meas);

    /** \brief */
    void copyMeanToMat71(const Belief7D &b, Mat71 &mu);
    void copyMeanToMat61(const Belief6D &b, Mat61 &mu);

    /** \brief */
    void copyCovarianceToMat77(const Belief7D &b, Mat77 &cov);
    void copyCovarianceToMat66(const Belief6D &b, Mat66 &cov);

    /** \brief */
    void setBelief(const Mat71 &mu, const Mat77 &cov);
    void setBelief(const Mat61 &mu, const Mat66 &cov);
};

}; //namespace sdrone

#endif // Tracker_HPP