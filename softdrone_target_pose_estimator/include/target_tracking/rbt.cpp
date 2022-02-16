// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    rbt.cpp
 * @author  Jared Strader
 */
//-----------------------------------------------------------------------------

#include <target_tracking/rbt.hpp>

namespace rbt
{
    double deg2rad(const double &x)
    {
        return x*PI/180.0;
    }

    double rad2deg(const double &x)
    {
        return x*180.0/PI;
    }

    Eigen::Matrix3d eul2Rot(const Eigen::Vector3d &t, const int &seq)
    {
        std::vector<double> ct = {std::cos(t[0]), std::cos(t[1]), std::cos(t[2])};
        std::vector<double> st = {std::sin(t[0]), std::sin(t[1]), std::sin(t[2])};

        Eigen::Matrix3d R;
        switch(seq)
        {
            case ZYX: //R=RzRyRx, p'=R*p
                R(0,0) = ct[1]*ct[0];
                R(0,1) = st[2]*st[1]*ct[0] - ct[2]*st[0];
                R(0,2) = ct[2]*st[1]*ct[0] + st[2]*st[0];
                R(1,0) = ct[1]*st[0];
                R(1,1) = st[2]*st[1]*st[0] + ct[2]*ct[0];
                R(1,2) = ct[2]*st[1]*st[0] - st[2]*ct[0];
                R(2,0) = -st[1];
                R(2,1) = st[2]*ct[1];
                R(2,2) = ct[2]*ct[1];
                return R;
                break;
            case XYZ: //R=RxRyRz, p'=R*p
                R(0,0) = ct[1]*ct[2];
                R(0,1) = -ct[1]*st[2];
                R(0,2) = st[1];
                R(1,0) = ct[0]*st[2] + ct[2]*st[0]*st[1];
                R(1,1) = ct[0]*ct[2] - st[0]*st[1]*st[2];
                R(1,2) = -ct[1]*st[0];
                R(2,0) = st[0]*st[2] - ct[0]*ct[2]*st[1];
                R(2,1) = ct[2]*st[0] + ct[0]*st[1]*st[2];
                R(2,2) = ct[0]*ct[1];
                return R;
                break;
            default:
                std::cout << "Error! Invalid sequence for eul2Rot." << std::endl;
                break;
        }
    }

    //https://www.geometrictools.com/Documentation/EulerAngles.pdf
    Eigen::Vector3d rot2Eul(const Eigen::Matrix3d &R, const int &seq)
    {
        double tx, ty, tz;
        switch(seq)
        {
            case ZYX: //R=RzRyRx
                if(R(2,0) < 1)
                {
                    if(R(2,0) > -1)
                    {
                        tx = std::atan2(R(2,1),R(2,2));
                        ty = std::asin(-R(2,0));
                        tz = std::atan2(R(1,0),R(0,0));
                    }
                    else
                    {
                        tx = 0;
                        ty = PI*0.5;
                        tz = -std::atan2(-R(1,2),R(1,1));
                    }
                }
                else
                {
                    tx = 0;
                    ty = -PI*0.5;
                    tz = std::atan2(-R(1,2),R(1,1));
                }
                return Eigen::Vector3d(tz, ty, tx);
                break;
            case XYZ: //R=RxRyRz
                if(R(0,2) < 1)
                {
                    if(R(0,2) > -1)
                    {
                        tx = std::atan2(-R(1,2),R(2,2));
                        ty = std::asin(R(0,2));
                        tz = std::atan2(-R(0,1),R(0,0));
                    }
                    else
                    {
                        //not unique, tz - tx = atan2(R10,R11)
                        tx = -atan2(R(1,0),R(1,1));
                        ty = -PI*0.5;
                        tz = 0;
                    }
                }
                else
                {
                    //not unique, tz + tx = atan2(R10,R11)
                    tx = atan2(R(1,0),R(1,1));
                    ty = PI*0.5;
                    tz = 0;
                }
                return Eigen::Vector3d(tx, ty, tz);
                break;
            default:
                std::cout << "Error! Invalid sequence for rot2Eul." << std::endl;
                break;
        }
    }

    Eigen::Matrix3d quat2Rot(const Eigen::Vector4d &q)
    {
        Eigen::Matrix3d R;
        R(0,0) = q(0)*q(0) + q(1)*q(1) - q(2)*q(2) - q(3)*q(3);
        R(0,1) = 2*(q(1)*q(2) - q(0)*q(3));
        R(0,2) = 2*(q(0)*q(2) + q(1)*q(3));
        R(1,0) = 2*(q(1)*q(2) + q(0)*q(3));
        R(1,1) = q(0)*q(0) - q(1)*q(1) + q(2)*q(2) - q(3)*q(3);
        R(1,2) = 2*(q(2)*q(3) - q(0)*q(1));
        R(2,0) = 2*(q(1)*q(3) - q(0)*q(2));
        R(2,1) = 2*(q(0)*q(1) + q(2)*q(3));
        R(2,2) = q(0)*q(0) - q(1)*q(1) - q(2)*q(2) + q(3)*q(3);
        return R;
    }

    Eigen::Vector4d rot2Quat(const Eigen::Matrix3d &R)
    {
        double qw, qx, qy, qz;

        double tr = R(0,0) + R(1,1) + R(2,2);
        double S;
        if (tr > 0) 
        { 
          S = std::sqrt(tr+1.0)*2; // S=4*qw 
          qw = 0.25*S;
          qx = (R(2,1) - R(1,2))/S;
          qy = (R(0,2) - R(2,0))/S; 
          qz = (R(1,0) - R(0,1))/S; 
        } else if ((R(0,0) > R(1,1))&(R(0,0) > R(2,2))) 
        { 
          S = std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2))*2; // S=4*qx 
          qw = (R(2,1) - R(1,2))/S;
          qx = 0.25 * S;
          qy = (R(0,1) + R(1,0))/S; 
          qz = (R(0,2) + R(2,0))/S; 
        } else if (R(1,1) > R(2,2)) 
        { 
          S = std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2))*2; // S=4*qy
          qw = (R(0,2) - R(2,0))/S;
          qx = (R(0,1) + R(1,0))/S; 
          qy = 0.25*S;
          qz = (R(1,2) + R(2,1))/S; 
        } 
        else 
        { 
          S = std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1))*2; // S=4*qz
          qw = (R(1,0) - R(0,1))/S;
          qx = (R(0,2) + R(2,0))/S;
          qy = (R(1,2) + R(2,1))/S;
          qz = 0.25*S;
        }
        return Eigen::Vector4d(qw,qx,qy,qz);
    }

    //https://ntrs.nasa.gov/citations/19770024290
    Eigen::Vector3d quat2Eul(const Eigen::Vector4d &q, const int &seq)
    {
        double tx, ty, tz;
        double sinp, delta;
        switch(seq)
        {
            case ZYX: //R=RzRyRx
                tx = std::atan2(2*(q[0]*q[1] + q[2]*q[3]), 
                                q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]);
                tz = std::atan2(2*(q[0]*q[3] + q[1]*q[2]), 
                                q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]);
                sinp = 2*(q[0]*q[2] - q[3]*q[1]);
                if(std::abs(sinp) >= 1)
                {
                    // ty = std::copysign(PI/2.0,sinp);
                    ty = std::copysign(2*std::atan2(q[1],q[0]),sinp);
                }
                else
                {
                    ty = std::asin(sinp);
                }
                return Eigen::Vector3d(tz, ty, tx);
                break;
            case XYZ: //R=RxRyRz
                tx = std::atan2(-2*(q[2]*q[3] - q[1]*q[0]), 
                                q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]);
                tz = std::atan2(-2*(q[1]*q[2] - q[3]*q[0]), 
                                q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]);
                sinp = 2*(q[1]*q[3] + q[2]*q[0]);
                if(std::abs(sinp) >= 1)
                {
                    ty = std::copysign(2*std::atan2(q[1],q[0]),sinp);
                }
                else
                {
                    ty = std::asin(sinp);
                }
                return Eigen::Vector3d(tx, ty, tz);
                break;
            default:
                std::cout << "Error! Invalid sequence for quat2Eul." << std::endl;
                break;
        }
    }

    Eigen::Vector4d eul2Quat(const Eigen::Vector3d &t, const int &seq)
    {
        std::vector<double> ct = {std::cos(t[0]*0.5), std::cos(t[1]*0.5), std::cos(t[2]*0.5)};
        std::vector<double> st = {std::sin(t[0]*0.5), std::sin(t[1]*0.5), std::sin(t[2]*0.5)};

        double qw, qx, qy, qz;
        switch(seq)
        {
            case ZYX: //R=RzRyRx
                qw = st[0]*st[1]*st[2] + ct[0]*ct[1]*ct[2];
                qx = -st[0]*st[1]*ct[2] + st[2]*ct[0]*ct[1];
                qy = st[0]*st[2]*ct[1] + st[1]*ct[0]*ct[2];
                qz = st[0]*ct[1]*ct[2] - st[1]*st[2]*ct[0];
                return Eigen::Vector4d(qw,qx,qy,qz);
                break;
            case XYZ: //R=RxRyRz
                qw = -st[0]*st[1]*st[2] + ct[0]*ct[1]*ct[2];
                qx = st[0]*ct[1]*ct[2] + st[1]*st[2]*ct[0];
                qy = -st[0]*st[2]*ct[1] + st[1]*ct[0]*ct[2];
                qz = st[0]*st[1]*ct[2] + st[2]*ct[0]*ct[1];
                return Eigen::Vector4d(qw,qx,qy,qz);
                break;
            default:
                std::cout << "Error! Invalid sequence for eul2Quat." << std::endl;
                break;
        }
    }

    Eigen::Matrix4d compose4x4(const Eigen::Vector3d &a, const Eigen::Vector4d &q)
    {   
        Eigen::Matrix3d R = quat2Rot(q);
        Eigen::Matrix4d T;
        T <<  R(0,0), R(0,1), R(0,2), a(0),
              R(1,0), R(1,1), R(1,2), a(1),
              R(2,0), R(2,1), R(2,2), a(2),
              0.0,    0.0,    0.0,    1.0;
        return T;
    }

    Eigen::Matrix4d compose4x4(const Eigen::Vector3d &a, const Eigen::Vector3d &t, const int &seq)
    {   
        Eigen::Matrix3d R = eul2Rot(t, seq);
        Eigen::Matrix4d T;
        T <<  R(0,0), R(0,1), R(0,2), a(0),
              R(1,0), R(1,1), R(1,2), a(1),
              R(2,0), R(2,1), R(2,2), a(2),
              0.0,    0.0,    0.0,    1.0;
        return T;
    }

    Eigen::Matrix4d compose4x4(const Eigen::Vector3d &a, const Eigen::Matrix3d &R)
    {   
        Eigen::Matrix4d T;
        T <<  R(0,0), R(0,1), R(0,2), a(0),
              R(1,0), R(1,1), R(1,2), a(1),
              R(2,0), R(2,1), R(2,2), a(2),
              0.0,    0.0,    0.0,    1.0;
        return T;
    }

    Eigen::VectorXd pose7DFrom6D(const Eigen::VectorXd &p)
    {
        Eigen::Vector4d q;
        q = eul2Quat(Eigen::Vector3d(p(5), p(4), p(3)), ZYX);

        Eigen::VectorXd p7D(7,1);
        p7D << p(0), p(1), p(2), q(0), q(1), q(2), q(3);

        return p7D;
    }

    Eigen::VectorXd pose6DFrom7D(const Eigen::VectorXd &p)
    {
        Eigen::Vector3d zyx;
        zyx = quat2Eul(Eigen::Vector4d(p(3),p(4),p(5),p(6)), ZYX);

        Eigen::VectorXd p6D(6,1);
        p6D << p(0), p(1), p(2), zyx(2), zyx(1), zyx(0);

        return p6D;
    }

    Eigen::Matrix4d pose4x4From7D(const Eigen::VectorXd &p)
    {
        Eigen::Matrix4d T;
        T.block<3,3>(0,0) = quat2Rot(Eigen::Vector4d(p(3),p(4),p(5),p(6)));
        T.block<3,1>(0,3) = Eigen::Vector3d(p(0),p(1),p(2));
        T(3,0) = 0.0;
        T(3,1) = 0.0;
        T(3,2) = 0.0;
        T(3,3) = 1.0;
        return T;
    }

    Eigen::Matrix4d pose4x4From6D(const Eigen::VectorXd &p)
    {
        Eigen::Matrix4d T;
        T.block<3,3>(0,0) = eul2Rot(Eigen::Vector3d(p(5),p(4),p(3)), ZYX);
        T.block<3,1>(0,3) = Eigen::Vector3d(p(0),p(1),p(2));
        T(3,0) = 0.0;
        T(3,1) = 0.0;
        T(3,2) = 0.0;
        T(3,3) = 1.0;
        return T;
    }

    Eigen::VectorXd pose7DFrom4x4(const Eigen::Matrix4d &T)
    {
        Eigen::VectorXd p(7,1);
        p(0) = T(0,3);
        p(1) = T(1,3);
        p(2) = T(2,3);

        Eigen::VectorXd q = rot2Quat(T.block<3,3>(0,0));
        p(3) = q(0);
        p(4) = q(1);
        p(5) = q(2);
        p(6) = q(3);

        return p;
    }

    Eigen::VectorXd pose6DFrom4x4(const Eigen::Matrix4d &T)
    {
        Eigen::VectorXd p(6,1);
        p(0) = T(0,3);
        p(1) = T(1,3);
        p(2) = T(2,3);

        Eigen::VectorXd t = rot2Eul(T.block<3,3>(0,0), ZYX);
        p(3) = t(2);
        p(4) = t(1);
        p(5) = t(0);

        return p;
    }

} //namespace rbt