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
  : nh_(nh),
    it_(nh)
{
  is_initialized_ = false;
};

KeypointDetectorROS::
KeypointDetectorROS(const ros::NodeHandle& nh,
                    const std::string&     rgb_img_topic,
                    const std::string&     model_file_name,
                    const bool             should_publish_annotated_img,
                    const std::string&     keypoints_topic,
                    const std::string&     annotated_img_topic)
  : nh_(nh),
    it_(nh),
    KeypointDetector(model_file_name)
{
  rgb_image_sub_ = it_.subscribe(rgb_img_topic, 1, &KeypointDetectorROS::rgbImageCallback, this);

  should_publish_annotated_img_ = should_publish_annotated_img;
  keypoints_pub_ = nh_.advertise<Keypoints2D>(keypoints_topic,  1);
  annotated_img_pub_ = it_.advertise(annotated_img_topic,  1);
};

void KeypointDetectorROS::
rgbImageCallback(const ImageMsg& rgb_image_msg)
{

  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(rgb_image_msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  torch::Tensor tensor_kpts;
  bool success = DetectKeypoints(cv_ptr->image, tensor_kpts);

  if(!success)
  {
    ROS_WARN("Keypoint detection failed!");
    return;
  }

  Keypoints2D kpts = tensorToKeypoints2D(tensor_kpts);
  keypoints_pub_.publish(kpts);

  if(should_publish_annotated_img_)
  {
    DrawKeypoints(cv_ptr->image, tensor_kpts);
    annotated_img_pub_.publish(cv_ptr->toImageMsg());
  }

};

softdrone_target_pose_estimator::Keypoints2D KeypointDetectorROS::
tensorToKeypoints2D(torch::Tensor& tensor_kpts)
{
  Keypoints2D keypoints_2D_msg;
  keypoints_2D_msg.header.stamp = ros::Time::now();
  for (int i = 0; i < tensor_kpts.sizes()[1]; ++i)
    {
      Keypoint2D keypoint_2D_msg;
      keypoint_2D_msg.x = tensor_kpts[0][i][0].item<int>();
      keypoint_2D_msg.y = tensor_kpts[0][i][1].item<int>();
      keypoints_2D_msg.keypoints_2D.push_back(keypoint_2D_msg);
    }
  return keypoints_2D_msg;
}

}; //namespace sdrone
