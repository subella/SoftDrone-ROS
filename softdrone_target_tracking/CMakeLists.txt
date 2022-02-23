cmake_minimum_required(VERSION 2.8.3)
project(softdrone_target_tracking)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  roscpp
  rospy
  std_msgs
  tf2
  nav_msgs
  geometry_msgs
  message_filters
)

#find_package(GTest REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(MRPT 2.0 COMPONENTS poses math)
message(STATUS "MRPT_VERSION: ${MRPT_VERSION}")
message(STATUS "MRPT_LIBRARIES: ${MRPT_LIBRARIES}")

catkin_package(
 INCLUDE_DIRS
  CATKIN_DEPENDS 
  message_generation 
  message_runtime 
  roscpp 
  rospy 
  std_msgs
  nav_msgs
  geometry_msgs
)

include_directories(
	include
  ${catkin_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
  ${PROJECT_NAME}
)

#LIBRARIES
#TODO: rbt can be separated from ROS
add_library(rbt src/rbt.cpp)
target_link_libraries(rbt 
  ${catkin_LIBRARIES}
  Eigen3::Eigen
)

#TODO: tracker can be separate from ROS
add_library(tracker src/tracker.cpp)
target_link_libraries(tracker 
  ${catkin_LIBRARIES}
  ${MRPT_LIBRARIES}
  Eigen3::Eigen
)

add_library(tracker_ros src/tracker_ros.cpp)
target_link_libraries(tracker_ros 
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  ${MRPT_LIBRARIES}
  tracker
)

#EXECUTABLES
add_executable(tracker_node src/tracker_node.cpp)
target_link_libraries(tracker_node
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  ${MRPT_LIBRARIES}
  tracker_ros
  rbt
)

add_executable(dummy_tracker_node src/dummy_node.cpp)
target_link_libraries(dummy_tracker_node 
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  ${MRPT_LIBRARIES}
  tracker_ros
  rbt
)

#TESTS
#add_executable(test_rbt tests/test_rbt.cpp)
catkin_add_gtest(test_rbt tests/test_rbt.cpp)
target_link_libraries(test_rbt 
  ${GTEST_LIBRARIES} 
  pthread 
  Eigen3::Eigen
  rbt
)

#############
## Install ##
#############
#PYTHON SCRIPTS
