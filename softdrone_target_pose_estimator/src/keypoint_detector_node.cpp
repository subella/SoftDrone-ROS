// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoint_detector_node.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Runs an neural net to detect pre-labelled keypoints from an rgb image.
 *        Subscribes to the Image topic and publishes a custom Keypoints message.
 *        Specify the file location of the jitted model (model.ts).
 */
//-----------------------------------------------------------------------------

#include <ros/ros.h>
#include <target_pose_estimating/keypoint_detector_ros.hpp>

std::string rgb_img_topic_;
std::string model_file_name_;
bool should_publish_annotated_img_;
std::string annotated_img_topic_;
std::string keypoints_topic_;

void load_params(const ros::NodeHandle& nh)
{
  nh.getParam("keypoint_detector_params/rgb_img_topic", rgb_img_topic_);
  nh.getParam("keypoint_detector_params/model_file_name", model_file_name_);
  nh.getParam("keypoint_detector_params/should_publish_annotated_img", should_publish_annotated_img_);
  nh.getParam("keypoint_detector_params/keypoints_topic", keypoints_topic_);
  nh.getParam("keypoint_detector_params/annotated_img_topic", annotated_img_topic_);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "keypoint_detector_node");
	ros::NodeHandle nh;

	ROS_INFO("keypoint_detector_node running...");


	load_params(nh);
	softdrone::KeypointDetectorROS keypoint_detector(nh, 
                                                   rgb_img_topic_, 
                                                   model_file_name_, 
                                                   should_publish_annotated_img_,
                                                   keypoints_topic_,
                                                   annotated_img_topic_);
	ros::spin();

	return 0;
}
