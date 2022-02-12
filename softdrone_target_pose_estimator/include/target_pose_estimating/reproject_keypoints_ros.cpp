// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    reproject_keypoints_ros.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/reproject_keypoints_ros.hpp>

#define DEPTH_CONVERSION ((double) 1)

namespace sdrone
{

ReprojectKeypointsROS::
ReprojectKeypointsROS(const ros::NodeHandle &nh)
  : nh_(nh), 
    it_(nh_),
    keypoints_2D_sub_(nh_, "", 1),
    depth_img_sub_(it_, "", 1),
    sync_(SyncPolicy(10), keypoints_2D_sub_, depth_img_sub_)
{
};

ReprojectKeypointsROS::
ReprojectKeypointsROS(const ros::NodeHandle& nh,
                      const std::string&     keypoints_2D_topic,
                      const std::string&     keypoints_3D_topic,
                      const std::string&     rgb_cam_info_topic,
                      const std::string&     depth_img_topic)
  : nh_(nh), 
    it_(nh_),
    keypoints_2D_sub_(nh_, keypoints_2D_topic, 1),
    depth_img_sub_(it_, depth_img_topic, 1),
    sync_(SyncPolicy(10), keypoints_2D_sub_, depth_img_sub_)
{
  rgb_cam_info_sub_ = nh_.subscribe(rgb_cam_info_topic, 1, &ReprojectKeypointsROS::rgbCamInfoCallback, this);
  sync_.registerCallback(boost::bind(&ReprojectKeypointsROS::syncCallback, this, _1, _2));
  keypoints_3D_pub_ = nh_.advertise<Keypoints3D>(keypoints_3D_topic,  1);
};

void ReprojectKeypointsROS::
rgbCamInfoCallback(const CameraInfoMsg& camera_info_msg)
{
  if (is_initialized_)
    return;

  Eigen::Matrix3d camera_intrinsics;
  for (int i=0; i < camera_intrinsics.rows(); i++)
    for(int j=0; j < camera_intrinsics.cols(); j++)
      camera_intrinsics(i, j) = camera_info_msg.K[i+j];

  init(camera_intrinsics);
}


void ReprojectKeypointsROS::
syncCallback(const Keypoints2D::ConstPtr& keypoints_2D_msg, const ImageMsg::ConstPtr& depth_img_msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(depth_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  auto keypoints_2D = keypoints_2D_msg->keypoints_2D;
  int num_kpts = keypoints_2D.size();
  auto depth_img = cv_ptr->image;

  Eigen::MatrixX2i px_py_mat(num_kpts, 2);
  Eigen::VectorXd z_vec(num_kpts);
  makePxPyZ(keypoints_2D, depth_img, px_py_mat, z_vec);

  Eigen::MatrixX3d keypoints_3D_mat(num_kpts, 3);
  int success = reprojectKeypoints(px_py_mat, z_vec, keypoints_3D_mat); 

  if (success)
  {
    Keypoints3D keypoints_3D_msg;
    eigenToKeypoints3DMsg(keypoints_3D_mat, keypoints_3D_msg);
    keypoints_3D_pub_.publish(keypoints_3D_msg);
  }

};

void ReprojectKeypointsROS::
makePxPyZ(const std::vector<Keypoint2D>& keypoints_2D, const cv::Mat& depth_img, 
          Eigen::MatrixX2i& px_py_mat, Eigen::VectorXd& z_vec)
{
  for (int i=0; i < keypoints_2D.size(); i++)
  {
    int px = keypoints_2D[i].x;
    int py = keypoints_2D[i].y;
    int depth_img_z = depth_img.at<int>(py, px);
    double z = DEPTH_CONVERSION * depth_img_z;
    px_py_mat.row(i) << px, py;
    z_vec(i) = z;
  }
}

void ReprojectKeypointsROS::
eigenToKeypoints3DMsg(Eigen::MatrixX3d& keypoints_3D_mat, Keypoints3D& keypoints_3D_msg)
{
  keypoints_3D_msg.header.stamp = ros::Time::now();
  for (int i=0; i < keypoints_3D_mat.rows(); i++)
  {
    Keypoint3D keypoint_3D_msg;
    keypoint_3D_msg.x = keypoints_3D_mat(i, 0);
    keypoint_3D_msg.y = keypoints_3D_mat(i, 1);
    keypoint_3D_msg.z = keypoints_3D_mat(i, 2);
    keypoints_3D_msg.keypoints_3D.push_back(keypoint_3D_msg);
  }  

}

}; //namespace sdrone