// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator_node.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Given detected 3D keypoints, estimates the pose of the 
 *        target using point cloud registration with Teaser.
 */
//-----------------------------------------------------------------------------

#include <ros/ros.h>
#include <target_pose_estimating/pose_estimator_ros.hpp>

std::string keypoints_3D_topic_;
std::string pose_topic_;
std::string cad_frame_file_name_;
sdrone::TeaserParams params_;

void load_params(const ros::NodeHandle& nh)
{
  // nh.param("pose_estimator_params/keypoints_3D_topic", keypoints_3D_topic_);
  // nh.param("pose_estimator_params/pose_topic", pose_topic_);
  // nh.param("pose_estimator_params/cad_frame_file_name", cad_frame_file_name_);
  // nh.param("pose_estimator_params/noise_bound", params_.noise_bound, (double) 5);
  // nh.param("pose_estimator_params/cbar2", params_.cbar2, (double) 1);
  // nh.param("pose_estimator_params/estimate_scaling", params_.estimate_scaling, false);
  // // nh.param can't handle unsigned long.
  // int rotation_max_iterations;
  // nh.param("pose_estimator_params/rotation_max_iterations", rotation_max_iterations, 100);
  // params_.rotation_max_iterations = (unsigned long) rotation_max_iterations;
  // nh.param("pose_estimator_params/rotation_gnc_factor", params_.rotation_gnc_factor, (double) 1.4);
  // nh.param("pose_estimator_params/rotation_cost_threshold", params_.rotation_cost_threshold, (double) 0.005);

  nh.getParam("pose_estimator_params/keypoints_3D_topic", keypoints_3D_topic_);
  nh.getParam("pose_estimator_params/pose_topic", pose_topic_);
  nh.getParam("pose_estimator_params/cad_frame_file_name", cad_frame_file_name_);
  nh.getParam("pose_estimator_params/noise_bound", params_.noise_bound);
  nh.getParam("pose_estimator_params/cbar2", params_.cbar2);
  nh.getParam("pose_estimator_params/estimate_scaling", params_.estimate_scaling);
  int rotation_max_iterations;
  nh.param("pose_estimator_params/rotation_max_iterations", rotation_max_iterations);
  params_.rotation_max_iterations = (unsigned long) rotation_max_iterations;
  nh.getParam("pose_estimator_params/rotation_gnc_factor", params_.rotation_gnc_factor);
  nh.getParam("pose_estimator_params/rotation_cost_threshold", params_.rotation_cost_threshold);

  // TODO: We can expose this to the yaml if needed.
  params_.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "pose_estimator_node");
	ros::NodeHandle nh;

	ROS_INFO("pose_estimator_node running...");

	load_params(nh);
	sdrone::PoseEstimatorROS pose_estimator(nh, 
                                          keypoints_3D_topic_,
                                          pose_topic_,
                                          cad_frame_file_name_,
                                          params_);
	ros::spin();

	return 0;
}