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