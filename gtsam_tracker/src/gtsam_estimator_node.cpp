#include <ros/ros.h>
#include "gtsam_tracker/gtsam_estimator_ros.hpp"


int main(int argc, char **argv)
{
    ros::init(argc, argv, "gtsam_tracker_node");
    ros::NodeHandle nh("~");

    ROS_INFO("gtsam_tracker_node running...");

    sdrone::GTSAMNode tracker(nh);
    ros::spin();

    return 0;
}

