// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoint_detector_ros.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/keypoint_detector_ros.hpp>

namespace softdrone
{

KeypointDetectorROS::
KeypointDetectorROS(const ros::NodeHandle& nh)
  : nh_(nh)
{
  is_initialized_ = false;
};

KeypointDetectorROS::
KeypointDetectorROS(const ros::NodeHandle& nh,
                    const std::string&     rgb_image_topic,
                    const std::string&     keypoints_topic,
                    const std::string&     model_file_name)
  : nh_(nh),
    KeypointDetector(model_file_name)
{
  image_transport::ImageTransport it(nh);
  rgb_image_sub_ = it.subscribe(rgb_image_topic, 1, &KeypointDetectorROS::rgbImageCallback, this);
  keypoints_pub_ = nh_.advertise<Keypoints>(keypoints_topic,  1);
};

void KeypointDetectorROS::
rgbImageCallback(const ImageMsg& rgb_image_msg)
{
  cv::Mat input_img;
  try
  {
    input_img = cv_bridge::toCvCopy(rgb_image_msg, "bgr8")->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  torch::Tensor tensor_kpts;
  bool success = DetectKeypoints(input_img, tensor_kpts);

  if(!success)
  {
    ROS_ERROR("Keypoint detection failed!");
    return;
  }

  Keypoints kpts = formatKeypoints(tensor_kpts);
  keypoints_pub_.publish(kpts);

};

softdrone_target_pose_estimator::Keypoints KeypointDetectorROS::
formatKeypoints(torch::Tensor tensor_kpts)
{
  softdrone_target_pose_estimator::Keypoints kpts;
  for (int i = 0; i < tensor_kpts.sizes()[1]; ++i)
    {
      softdrone_target_pose_estimator::Keypoint kpt;
      kpt.x = tensor_kpts[0][i][0].item<int>();
      kpt.y = tensor_kpts[0][i][1].item<int>();
      kpts.keypoints[i] = kpt;
    }
  return kpts;
}

}; //namespace softdrone