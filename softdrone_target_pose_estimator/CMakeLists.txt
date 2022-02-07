cmake_minimum_required(VERSION 3.0.2)
project(softdrone_target_pose_estimator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  message_generation
)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(teaserpp REQUIRED)


add_message_files(
  FILES
  Keypoint.msg
  Keypoints.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
   CATKIN_DEPENDS rospy
                  roscpp
                  std_msgs
                  sensor_msgs
                  cv_bridge
                  image_transport
                  message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# LIBRARIES

add_library(keypoint_detector include/target_pose_estimating/keypoint_detector.cpp)
target_link_libraries(keypoint_detector 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
)

add_library(keypoint_detector_ros include/target_pose_estimating/keypoint_detector_ros.cpp)
target_link_libraries(keypoint_detector_ros
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
  keypoint_detector
)

add_library(keypoints_3D include/target_pose_estimating/keypoints_3D.cpp)
target_link_libraries(keypoints_3D
  ${catkin_LIBRARIES}
  Eigen3::Eigen
)

add_library(pose_estimator include/target_pose_estimating/pose_estimator.cpp)
target_link_libraries(pose_estimator
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  teaserpp::teaser_registration
  teaserpp::teaser_io
)

# add_library(pose_estimator_ros include/target_pose_estimating/pose_estimator_ros.cpp)
# target_link_libraries(pose_estimator_ros
#   ${catkin_LIBRARIES}
#   Eigen3::Eigen
#   teaserpp::teaser_registration
#   teaserpp::teaser_io
#   pose_estimator
# )

# EXECUTABLES

add_executable(keypoint_detector_node src/keypoint_detector_node.cpp)

add_dependencies(keypoint_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(keypoint_detector_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${TORCH_LIBRARIES}
   TorchVision::TorchVision
   keypoint_detector_ros
 )