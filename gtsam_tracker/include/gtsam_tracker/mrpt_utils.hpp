#include <mrpt/poses/CPose3DQuatPDFGaussian.h>
#include <mrpt/poses/CPose3DQuat.h>
#include <mrpt/poses/CPose3DPDFGaussian.h>
#include <mrpt/poses/CPose3D.h>

#include <geometry_msgs/PoseWithCovariance.h>


typedef mrpt::poses::CPose3DQuatPDFGaussian Belief7D;
typedef mrpt::poses::CPose3DQuat Pose7D;
typedef mrpt::poses::CPose3DPDFGaussian Belief6D;
typedef mrpt::poses::CPose3D Pose6D;
typedef mrpt::math::CMatrixDouble66 CMat66;
typedef geometry_msgs::PoseWithCovariance PoseWCov;


Pose6D inline getPose6DFromPoseWCov(const PoseWCov &pwc) {
  Pose7D p7D;
  p7D.m_coords[0] = pwc.pose.position.x;
  p7D.m_coords[1] = pwc.pose.position.y;
  p7D.m_coords[2] = pwc.pose.position.z;
  p7D.m_quat.r() = pwc.pose.orientation.w;
  p7D.m_quat.x() = pwc.pose.orientation.x;
  p7D.m_quat.y() = pwc.pose.orientation.y;
  p7D.m_quat.z() = pwc.pose.orientation.z;
  return Pose6D(p7D);
};

CMat66 inline getCov6DFromPoseWCov(const PoseWCov &pwc)
{
  //The order for the euler angles are different between
  //Pose6D and PoseWCov
  CMat66 cov6D;
  const int map[6] = {0,1,2,5,4,3};
  for(int i=0; i<6; i++)
  {
    for(int j=0; j<6; j++)
    {
      cov6D(i,j) = pwc.covariance[map[i]*6 + map[j]];
    }
  }
  return cov6D;
};



Belief6D inline belief6DFromPoseWCov(const PoseWCov &pwc) {
  Pose6D p = getPose6DFromPoseWCov(pwc);
  CMat66 P = getCov6DFromPoseWCov(pwc);
  return Belief6D(p,P);
};
 

