#include "estimator_utils.h"

Pose6D getPose6DFromPoseWCov(const geometry_msgs::PoseWithCovarianceStamped &pwc)
{
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

