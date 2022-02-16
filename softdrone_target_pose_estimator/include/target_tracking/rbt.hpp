// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    rbt.hpp
 * @author  Jared Strader
 * 
 * @brief Class of utilities for conversions between representations of rigid 
 *        body transformations. Requires Eigen for matrix operations. Does not 
 *        include operations on covariance matrices (e.g., error propagation).
 */
//-----------------------------------------------------------------------------

#ifndef RBT_HPP
#define RBT_HPP

#include <iostream>
#include <vector>
#include <cmath>

#include <Eigen/Core>

namespace rbt
{
    /** \brief Constant for euler angle convention R = RxRyRz. Note, the
     * order of rotations, this is not fixed-axis (i.e., not extrinisic, 
     * but instead intrinsic). Note: Intrinsic XYZ is equivalent extrinsic
     * ZYX (sometimes referred to as YPR) */
    static const int XYZ = 0;

    /** \brief Constant for euler angle convention R = RxRyRz. Note, the
     * order of rotations, this is not fixed-axis (i.e., not extrinisic,
     * but instead intrinsic). Note: Intrinsic ZYX is equivalent extrinsic 
     * XYZ (sometimes referred to as RPY) */
    static const int ZYX = 1;

    /** \brief Constant pi with 16 decimel places */
    static const double PI = 3.1415926535897932;

    /** \brief Convert degrees to radians */
    double deg2rad(const double &x);

    /** \brief Convert radians to degrees */
    double rad2deg(const double &x);

    /** \brief Convert euler angles to rotation matrix, the sequence is intrinsic
     * (i.e., not fixed-axis) */
    Eigen::Matrix3d eul2Rot(const Eigen::Vector3d &t, const int &seq);

    /** \brief Convert euler angles to quaternion, the sequence is not fixed-axis */
    Eigen::Vector4d eul2Quat(const Eigen::Vector3d &t, const int &seq);

    /** \brief Convert rotation matrix to euler angles, the sequence is intrinsic
     * (i.e., not fixed-axis) */
    Eigen::Vector3d rot2Eul(const Eigen::Matrix3d &R, const int &seq);

    /** \brief Convert rotation matrix to quaternion */
    Eigen::Vector4d rot2Quat(const Eigen::Matrix3d &R);

    /** \brief Convert quaternion to rotation matrix */
    Eigen::Matrix3d quat2Rot(const Eigen::Vector4d &q);

    /** \brief Convert quaternion to euler angles, the sequence is intrinsic
     * (i.e., not fixed-axis) */
    Eigen::Vector3d quat2Eul(const Eigen::Vector4d &q, const int &seq);

    /** \brief Convert pose defined by 3D point and quaternion to 4x4 
     * transformation */
    Eigen::Matrix4d compose4x4(const Eigen::Vector3d &a, const Eigen::Vector4d &q);

    /** \brief Convert pose defined by 3D point and euler angles to 4x4 
     * transformation */
    Eigen::Matrix4d compose4x4(const Eigen::Vector3d &a, const Eigen::Vector3d &t, const int &seq);

    /** \brief Convert pose defined by 3D point and rotation matrix to 4x4 
     * transformation */
    Eigen::Matrix4d compose4x4(const Eigen::Vector3d &a, const Eigen::Matrix3d &R);

    /** \brief Convert 7D pose p = [x,y,z,qw,qx,qy,qz] to 6D pose 
     * p = [x,y,z, thetax, thetay, thetaz], where thetax, thetay, and thetaz
     * are the euler angles for sequence ZYX (not fixed-axis). Note: ZYX is
     * equivalent to fixed-axes XYZ. */
    Eigen::VectorXd pose7DFrom6D(const Eigen::VectorXd &p);

    /** \brief Convert 6D pose p = [x,y,z, thetax, thetay, thetaz] to 7D pose 
     * p = [x,y,z,qw,qx,qy,qz], where thetax, thetay, and thetaz are the euler 
     * angles for sequence ZYX (not fixed-axis). Note: ZYX is equivalent to
     * fixed-axes XYZ. */
    Eigen::VectorXd pose6DFrom7D(const Eigen::VectorXd &p);

    /** \brief Convert pose represeted as 4x4 transformation to 7D pose
     * p = [x,y,z,qw,qx,qy,qz] */
    Eigen::Matrix4d pose4x4From7D(const Eigen::VectorXd &p);

    /** \brief Convert pose represeted as 4x4 transformation to 6D pose
     * p = [x,y,z, thetax, thetay, thetaz] where thetax, thetay, thetaz,
     * are the euler angles for sequence ZYX (not fixed-axis). Note: ZYX
     * is equivaelent to fixed-axis XYZ */
    Eigen::Matrix4d pose4x4From6D(const Eigen::VectorXd &p);

    /** \brief Converts 7D pose p = [x,y,z,qw,qx,qy,qz] to 4x4 transformation */
    Eigen::VectorXd pose7DFrom4x4(const Eigen::Matrix4d &T);

    /** \brief Converts 6D pose p = [x,y,z, thetax, thetay, thetaz] to 4x4 
     * transformation where thetax, thetay, thetaz are the euler angles for 
     * sequence ZYX (not fixed-axis). Note: ZYX is equivaelent to fixed-axis 
     * XYZ */
    Eigen::VectorXd pose6DFrom4x4(const Eigen::Matrix4d &T);

} //namespace rbt

#endif // Utils_HPP