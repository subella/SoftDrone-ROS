// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    tracker_ros.cpp
 * @author  Jared Strader
 */
//-----------------------------------------------------------------------------

#include <target_tracking/tracker_ros.hpp>

namespace sdrone
{

TrackerROS::
TrackerROS(const ros::NodeHandle &nh)
  : nh_(nh), 
    agent_sub_(nh_, "agent_odom", 1),
    target_rel_sub_(nh_, "estimated_relative_pose", 1),
    sync_(SyncPolicy(10), agent_sub_, target_rel_sub_)
{
  is_initialized_ = false;
  target_pub_ = nh_.advertise<PoseWCovStamp>("target_global_pose_estimate",  1);
  sync_.registerCallback(boost::bind(&TrackerROS::syncCallback, this, _1, _2));
};

TrackerROS::Pose TrackerROS::
samplePose(const Eigen::VectorXd &mu,
           const Eigen::MatrixXd &cov)
{
  //belief
  Pose7D mu_mrpt7D(mu(0),mu(1),mu(2),Quat(mu(3),mu(4),mu(5),mu(6)));
  Pose6D mu_mrpt6D(mu_mrpt7D);

  CMat66 cov_mrpt;
  cov_mrpt = cov;

  Belief6D b(mu_mrpt6D, cov_mrpt);

  //sample
  Sampler sampler;
  sampler.setPosePDF(b);

  Pose6D p6D;
  sampler.drawSample(p6D);

  //convert sample to PoseWCov
  Pose7D p7D(p6D);
  Pose pose = getPoseFromPose7D(p7D);

  return pose;
};

void TrackerROS::
syncCallback(const Odom::ConstPtr &odom, const PoseWCovStamp::ConstPtr &pwcs)
{
  time_stamp_ = odom->header.stamp;
  frame_id_ = odom->header.frame_id;

  if(odom->child_frame_id != pwcs->header.frame_id)
  {
    ROS_ERROR("Error: odom->child_frame_id != pwcs->header.frame_id");
  }

  //don't use 7D, left/right multiplication by jacobian to convert
  //covariance from 6D to 7D may cause singular covariance matrix
  // Belief7D b_agent = belief7DFromPoseWCov(odom->pose);
  // Belief7D b_target_rel = belief7DFromPoseWCov(pwcs->pose);
  // Belief7D b_target_meas = b_agent + b_target_rel;
  // update(b_target_meas);
  // publishResults7D();

  //use 6D
  Belief6D b_agent = belief6DFromPoseWCov(odom->pose);
  Belief6D b_target_rel = belief6DFromPoseWCov(pwcs->pose);
  Belief6D b_target_meas = b_agent + b_target_rel;
  update(b_target_meas);
  publishResults6D();
};

void TrackerROS::
publishResults7D()
{
  PoseWCovStamp pwcs;
  pwcs.header.stamp = time_stamp_;
  pwcs.header.frame_id = frame_id_;
  pwcs.pose = poseWCovFromBelief(b_target_);
  target_pub_.publish(pwcs);
};

void TrackerROS::
publishResults6D()
{
  PoseWCovStamp pwcs;
  pwcs.header.stamp = time_stamp_;
  pwcs.header.frame_id = frame_id_;
  pwcs.pose = poseWCovFromBelief(b_target_6D_);
  pwcs.pose.pose.position.x *= 1000.;
  pwcs.pose.pose.position.y *= 1000.;
  pwcs.pose.pose.position.z *= 1000.;
  target_pub_.publish(pwcs);
};

TrackerROS::Belief7D TrackerROS::
belief7DFromPoseWCov(const PoseWCov &pwc)
{
  Pose6D p = getPose6DFromPoseWCov(pwc);
  CMat66 P = getCov6DFromPoseWCov(pwc);
  Belief6D b6D(p,P);
  return Belief7D(b6D);
};

TrackerROS::Belief6D TrackerROS::
belief6DFromPoseWCov(const PoseWCov &pwc)
{
  Pose6D p = getPose6DFromPoseWCov(pwc);
  CMat66 P = getCov6DFromPoseWCov(pwc);
  return Belief6D(p,P);
};

TrackerROS::PoseWCov TrackerROS::
poseWCovFromBelief(const Belief7D &b)
{
  PoseWCov pwc;

  //mean
  Belief6D b6D(b);

  Pose7D p7D;
  b.getMean(p7D);

  pwc.pose.position.x = p7D.m_coords[0];
  pwc.pose.position.y = p7D.m_coords[1];
  pwc.pose.position.z = p7D.m_coords[2];
  pwc.pose.orientation.w = p7D.m_quat.r();
  pwc.pose.orientation.x = p7D.m_quat.x();
  pwc.pose.orientation.y = p7D.m_quat.y();
  pwc.pose.orientation.z = p7D.m_quat.z();

  //covariance
  CMat66 cov6D;
  b6D.getCovariance(cov6D);

  const int map[6] = {0,1,2,5,4,3};
  for(int i=0; i<6; i++)
  {
    for(int j=0; j<6; j++)
    {
      pwc.covariance[map[i]*6 + map[j]] = cov6D(i,j);
    }
  }

  return pwc;
};

TrackerROS::PoseWCov TrackerROS::
poseWCovFromBelief(const Belief6D &b)
{
  PoseWCov pwc;

  //mean
  Belief7D b7D(b);

  Pose7D p7D;
  b7D.getMean(p7D);

  pwc.pose.position.x = p7D.m_coords[0];
  pwc.pose.position.y = p7D.m_coords[1];
  pwc.pose.position.z = p7D.m_coords[2];
  pwc.pose.orientation.w = p7D.m_quat.r();
  pwc.pose.orientation.x = p7D.m_quat.x();
  pwc.pose.orientation.y = p7D.m_quat.y();
  pwc.pose.orientation.z = p7D.m_quat.z();

  //covariance
  CMat66 cov6D;
  b.getCovariance(cov6D);

  const int map[6] = {0,1,2,5,4,3};
  for(int i=0; i<6; i++)
  {
    for(int j=0; j<6; j++)
    {
      pwc.covariance[map[i]*6 + map[j]] = cov6D(i,j);
    }
  }

  return pwc;
};

TrackerROS::Pose6D TrackerROS::
getPose6DFromPoseWCov(const PoseWCov &pwc)
{
  Pose7D p7D;
  p7D.m_coords[0] = pwc.pose.position.x/1000.;
  p7D.m_coords[1] = pwc.pose.position.y/1000.;
  p7D.m_coords[2] = pwc.pose.position.z/1000.;
  p7D.m_quat.r() = pwc.pose.orientation.w;
  p7D.m_quat.x() = pwc.pose.orientation.x;
  p7D.m_quat.y() = pwc.pose.orientation.y;
  p7D.m_quat.z() = pwc.pose.orientation.z;
  return Pose6D(p7D);
};

TrackerROS::CMat66 TrackerROS::
getCov6DFromPoseWCov(const PoseWCov &pwc)
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

TrackerROS::Pose TrackerROS::
getPoseFromPose7D(const Pose7D &pose)
{
  Pose p;
  p.position.x = pose.m_coords[0];
  p.position.y = pose.m_coords[1];
  p.position.z = pose.m_coords[2];
  p.orientation.w = pose.m_quat.r();
  p.orientation.x = pose.m_quat.x();
  p.orientation.y = pose.m_quat.y();
  p.orientation.z = pose.m_quat.z();
  return p;
};

}; //namespace sdrone
