#include <target_tracking/rbt.hpp>
#include <gtest/gtest.h>
 
TEST(Conversions, Euler) { 
    double tx = rbt::deg2rad(14.7);
    double ty = rbt::deg2rad(2.2);
    double tz = rbt::deg2rad(-60.7);

    Eigen::Matrix3d R;
    Eigen::Vector4d q;

    //ZYX 
    Eigen::Vector3d zyx(tz,ty,tx);

    R = rbt::eul2Rot(zyx, rbt::ZYX);
    Eigen::Vector3d t1 = rbt::rot2Eul(R, rbt::ZYX);
    EXPECT_NEAR(tz, t1[0], 1e-15);
    EXPECT_NEAR(ty, t1[1], 1e-15);
    EXPECT_NEAR(tx, t1[2], 1e-15);

    q = rbt::eul2Quat(zyx, rbt::ZYX);
    Eigen::Vector3d t2 = rbt::quat2Eul(q, rbt::ZYX);
    EXPECT_NEAR(tz, t2[0], 1e-15);
    EXPECT_NEAR(ty, t2[1], 1e-15);
    EXPECT_NEAR(tx, t2[2], 1e-15);

    //XYZ
    Eigen::Vector3d xyz(tx,ty,tz);

    R = rbt::eul2Rot(xyz, rbt::XYZ);
    Eigen::Vector3d t3 = rbt::rot2Eul(R, rbt::XYZ);
    EXPECT_NEAR(tx, t3[0], 1e-15);
    EXPECT_NEAR(ty, t3[1], 1e-15);
    EXPECT_NEAR(tz, t3[2], 1e-15);

    q = rbt::eul2Quat(xyz, rbt::XYZ);
    Eigen::Vector3d t4 = rbt::quat2Eul(q, rbt::XYZ);
    EXPECT_NEAR(tx, t4[0], 1e-15);
    EXPECT_NEAR(ty, t4[1], 1e-15);
    EXPECT_NEAR(tz, t4[2], 1e-15);
}

TEST(Conversions, Quaternion) { 
    double qw = 1.2;
    double qx = -0.3;
    double qy = 0.1;
    double qz = 0.2;

    double d = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw=qw/d;
    qx=qx/d;
    qy=qy/d;
    qz=qz/d;

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Vector4d q(qw,qx,qy,qz);

    R = rbt::quat2Rot(q);
    q = rbt::rot2Quat(R);;
    EXPECT_NEAR(qw, q[0], 1e-15);
    EXPECT_NEAR(qx, q[1], 1e-15);
    EXPECT_NEAR(qy, q[2], 1e-15);
    EXPECT_NEAR(qz, q[3], 1e-15);

    d = std::sqrt(q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3));
    EXPECT_NEAR(1.0, d, 1e-15);

    t = rbt::quat2Eul(q, rbt::ZYX);
    q = rbt::eul2Quat(t, rbt::ZYX);
    EXPECT_NEAR(qw, q[0], 1e-15);
    EXPECT_NEAR(qx, q[1], 1e-15);
    EXPECT_NEAR(qy, q[2], 1e-15);
    EXPECT_NEAR(qz, q[3], 1e-15);

    d = std::sqrt(q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3));
    EXPECT_NEAR(1.0, d, 1e-15);

    t = rbt::quat2Eul(q, rbt::XYZ);
    q = rbt::eul2Quat(t, rbt::XYZ);
    EXPECT_NEAR(qw, q[0], 1e-15);
    EXPECT_NEAR(qx, q[1], 1e-15);
    EXPECT_NEAR(qy, q[2], 1e-15);
    EXPECT_NEAR(qz, q[3], 1e-15);

    d = std::sqrt(q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3));
    EXPECT_NEAR(1.0, d, 1e-15);
}

TEST(Conversions, Pose7D) { 
    Eigen::VectorXd p(7,1);
    p(0) = 1.2;
    p(1) = -0.7;
    p(2) = 11.1;
    p(3) = 0.4;
    p(4) = 0.11;
    p(5) = 0.34;
    p(6) = 0.7;

    double d = std::sqrt(p(3)*p(3) + p(4)*p(4) + p(5)*p(5) + p(6)*p(6));
    p(3)=p(3)/d;
    p(4)=p(4)/d;
    p(5)=p(5)/d;
    p(6)=p(6)/d;

    Eigen::VectorXd p6D(6,1);
    Eigen::VectorXd p7D(7,1);
    p6D = rbt::pose6DFrom7D(p);
    p7D = rbt::pose7DFrom6D(p6D);

    EXPECT_NEAR(p(0), p7D(0), 1e-15);
    EXPECT_NEAR(p(1), p7D(1), 1e-15);
    EXPECT_NEAR(p(2), p7D(2), 1e-15);
    EXPECT_NEAR(p(3), p7D(3), 1e-15);
    EXPECT_NEAR(p(4), p7D(4), 1e-15);
    EXPECT_NEAR(p(5), p7D(5), 1e-15);
    EXPECT_NEAR(p(6), p7D(6), 1e-15);

    Eigen::MatrixXd T = rbt::pose4x4From7D(p);
    p7D = rbt::pose7DFrom4x4(T);
    EXPECT_NEAR(p(0), p7D(0), 1e-15);
    EXPECT_NEAR(p(1), p7D(1), 1e-15);
    EXPECT_NEAR(p(2), p7D(2), 1e-15);
    EXPECT_NEAR(p(3), p7D(3), 1e-15);
    EXPECT_NEAR(p(4), p7D(4), 1e-15);
    EXPECT_NEAR(p(5), p7D(5), 1e-15);
}

TEST(Conversions, Pose6D) { 
    Eigen::VectorXd p(7,1);
    p(0) = 1.2;
    p(1) = -0.2;
    p(2) = 10.2;
    p(3) = 0.3;
    p(4) = 0.75;
    p(5) = 0.1;

    Eigen::VectorXd p7D(7,1);
    Eigen::VectorXd p6D(6,1);
    p7D = rbt::pose7DFrom6D(p);
    p6D = rbt::pose6DFrom7D(p7D);

    EXPECT_NEAR(p(0), p6D(0), 1e-15);
    EXPECT_NEAR(p(1), p6D(1), 1e-15);
    EXPECT_NEAR(p(2), p6D(2), 1e-15);
    EXPECT_NEAR(p(3), p6D(3), 1e-15);
    EXPECT_NEAR(p(4), p6D(4), 1e-15);
    EXPECT_NEAR(p(5), p6D(5), 1e-15);

    Eigen::MatrixXd T = rbt::pose4x4From6D(p);
    p6D = rbt::pose6DFrom4x4(T);
    EXPECT_NEAR(p(0), p6D(0), 1e-15);
    EXPECT_NEAR(p(1), p6D(1), 1e-15);
    EXPECT_NEAR(p(2), p6D(2), 1e-15);
    EXPECT_NEAR(p(3), p6D(3), 1e-15);
    EXPECT_NEAR(p(4), p6D(4), 1e-15);
    EXPECT_NEAR(p(5), p6D(5), 1e-15);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}