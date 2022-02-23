// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    dummy_node.hpp
 * @author  Jared Strader
 * 
 * @brief Publishes PoseWithCovarianceStamped for both the agent (in global 
 *        frame) and the target (relative to the agent). This node is only used 
 *        for a sanity check to validate the tracker. The truth of the agent 
 *        and target in the global frames are published as well.
 */
//-----------------------------------------------------------------------------

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>

#include <target_tracking/tracker_ros.hpp>
#include <target_tracking/rbt.hpp>

#include <random>

typedef geometry_msgs::Pose Pose;
typedef geometry_msgs::PoseStamped PoseStamp;
typedef geometry_msgs::PoseWithCovariance PoseWCov;
typedef geometry_msgs::PoseWithCovarianceStamped PoseWCovStamp;
typedef nav_msgs::Odometry Odom;

void getAgentTruth(Eigen::VectorXd &mu, Eigen::MatrixXd &cov);
void getTargetTruth(Eigen::VectorXd &mu);
void getTargetRelativeTruth(const Eigen::VectorXd &mu_agent, const Eigen::VectorXd &mu_target, Eigen::VectorXd &mu_target_rel, Eigen::MatrixXd &cov_target_rel);
void copyCov(const Eigen::MatrixXd cov, PoseWCov &pwc);
void sampleData(PoseWCov &agent, PoseWCov &target_rel);
void getTruth(Pose &agent_truth, Pose &target_truth);

int main(int argc, char **argv)
{
	ros::init(argc, argv, "dummy_node");
	ros::NodeHandle nh("~");

	ROS_INFO("dummy_node running...");

	ros::Publisher odom_pub = nh.advertise<Odom>("agent_odom_out", 1000);
	ros::Publisher pwcs_pub = nh.advertise<PoseWCovStamp>("target_pwcs_out", 1000);
	ros::Publisher agent_truth_pub = nh.advertise<PoseStamp>("agent_pose_gt_out", 1000);
	ros::Publisher target_truth_pub = nh.advertise<PoseStamp>("target_pose_gt_out", 1000);
	
	ros::Rate loop_rate(20);
	while (ros::ok())
	{
		ros::Time current_time = ros::Time::now();

		//sample agent and target
		PoseWCov samp_agent;
		PoseWCov samp_target_rel;
		sampleData(samp_agent,samp_target_rel);

		//agent
		Odom agent_msg;
		agent_msg.header.stamp = current_time;
		agent_msg.header.frame_id = "odom";
		agent_msg.child_frame_id = "agent";
		agent_msg.pose = samp_agent;
		
		//target
		PoseWCovStamp target_msg;
		target_msg.header.stamp = current_time;
		target_msg.header.frame_id = "agent";
		target_msg.pose = samp_target_rel;

		//truth
		Pose pose_truth_agent;
		Pose pose_truth_target;
		getTruth(pose_truth_agent, pose_truth_target);

		PoseStamp truth_agent_msg;
		truth_agent_msg.header.stamp = current_time;
		truth_agent_msg.header.frame_id = "odom";
		truth_agent_msg.pose = pose_truth_agent;

		PoseStamp truth_target_msg;
		truth_target_msg.header.stamp = current_time;
		truth_target_msg.header.frame_id = "odom";
		truth_target_msg.pose = pose_truth_target;

		//publish data
		pwcs_pub.publish(target_msg);
		odom_pub.publish(agent_msg);
		agent_truth_pub.publish(truth_agent_msg);
		target_truth_pub.publish(truth_target_msg);

		// previous_time = current_time;
		ros::spinOnce();
		loop_rate.sleep();
	}


	return 0;
}

void getAgentTruth(Eigen::VectorXd &mu, Eigen::MatrixXd &cov)
{
	static double incr = 0;
	double dx = 1*std::sin(incr);
	double dy = 1*std::sin(incr);
	double dz = 1*std::sin(incr);
	double dt = 0.05;

	static double x = 2;
	static double y = 1;
	static double z = 1;
	x = x + dt*dx;
	y = y + dt*dy;
	z = z + dt*dz;

	//mean: x, y, z, qw, qx, qy, qz
	mu = Eigen::VectorXd(7,1);
	mu(0) = x;
	mu(1) = y;
	mu(2) = z;
	mu(3) = 0.5*std::sqrt(2.0);
	mu(4) = 0;
	mu(5) = 0;
	mu(6) = 0.5*std::sqrt(2.0);

	//cov: x, y, z, thetax, thetay, thetaz
	//order: ZYX 
	cov = Eigen::MatrixXd::Identity(6,6);
	cov(0,0) = 0.1;
	cov(1,1) = 0.1;
	cov(2,2) = 0.1;
	cov(3,3) = 0.01;
	cov(4,4) = 0.01;
	cov(5,5) = 0.01;

	incr = incr + (1.0/100.0)*3.14159265;
};

void getTargetTruth(Eigen::VectorXd &mu)
{
	//mean: x, y, z, qw, qx, qy, qz
	mu = Eigen::VectorXd(7,1);
	mu(0) = 2;
	mu(1) = 1;
	mu(2) = 0;
	mu(3) = 0.1993679;//0.5*std::sqrt(2.0);
	mu(4) = 0;
	mu(5) = 0;
	mu(6) = 0.9799247;//0.5*std::sqrt(2.0);
};

void getTargetRelativeTruth(const Eigen::VectorXd &mu_agent,
                            const Eigen::VectorXd &mu_target,
                                  Eigen::VectorXd &mu_target_rel,
                                  Eigen::MatrixXd &cov_target_rel)
{
	//mean: x, y, z, qw, qx, qy, qz
	//calculate pose of target relative to agent
	Eigen::Matrix4d T_agent = rbt::pose4x4From7D(mu_agent);
	Eigen::Matrix4d T_target = rbt::pose4x4From7D(mu_target);

	Eigen::Matrix4d T_target_rel = T_agent.inverse()*T_target;
	mu_target_rel = rbt::pose7DFrom4x4(T_target_rel);

	//cov: x, y, z, thetax, thetay, thetaz
	//order: ZYX 
	cov_target_rel = Eigen::MatrixXd::Identity(6,6);
	cov_target_rel(0,0) = 0.1;
	cov_target_rel(1,1) = 0.1;
	cov_target_rel(2,2) = 0.1;
	cov_target_rel(3,3) = 0.01;
	cov_target_rel(4,4) = 0.01;
	cov_target_rel(5,5) = 0.01;
}

//copy covariance from MatrixXd to PoseWCov
void copyCov(const Eigen::MatrixXd cov, PoseWCov &pwc)
{
  const int map[6] = {0,1,2,5,4,3};
  for(int i=0; i<6; i++)
  {
    for(int j=0; j<6; j++)
    {
      pwc.covariance[map[i]*6 + map[j]] = cov(i,j);
    }
  }
}

//samples Pose for agent (in global frame) and target (relative to agent)
//using mrpt, the inputs are populated with the covariance used for sampling
void sampleData(PoseWCov &agent, PoseWCov &target_rel)
{
	//generate poses
	Eigen::VectorXd mu_agent(7,1);
	Eigen::MatrixXd cov_agent(6,6);
	getAgentTruth(mu_agent,cov_agent);

	Eigen::VectorXd mu_target(7,1);
	getTargetTruth(mu_target);

	Eigen::VectorXd mu_target_rel(7,1);
	Eigen::MatrixXd cov_target_rel(6,6);
	getTargetRelativeTruth(mu_agent,
		                     mu_target,
		                     mu_target_rel,
		                     cov_target_rel);

	//sample
	//agent.pose = sdrone::TrackerROS::samplePose(mu_agent, cov_agent);
	agent.pose.position.x = 0;
  agent.pose.position.y = 0;
  agent.pose.position.z = 0;
  
  agent.pose.orientation.w = 1;
  agent.pose.orientation.x = 0;
  agent.pose.orientation.y = 0;
  agent.pose.orientation.z = 0;
	target_rel.pose = sdrone::TrackerROS::samplePose(mu_target_rel, cov_target_rel);

	//populate covariance
	Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6,6);
  cov(0,0) = .5;
  cov(1,1) = .5;
  cov(2,2) = .5;
  cov(3,3) = 0.1;
  cov(4,4) = 0.1;
  cov(5,5) = 0.1;
	copyCov(cov, agent);

	//the covariance of measurement is used for R matrix, make
	//covariance order of magnitue larger than the actual distribution
	cov_target_rel(0,0) = cov_target_rel(0,0)*10.0;
	cov_target_rel(1,1) = cov_target_rel(1,1)*10.0;
	cov_target_rel(2,2) = cov_target_rel(2,2)*10.0;
	cov_target_rel(3,3) = cov_target_rel(3,3)*10.0;
	cov_target_rel(4,4) = cov_target_rel(4,4)*10.0;
	cov_target_rel(5,5) = cov_target_rel(5,5)*10.0;
	copyCov(cov_target_rel, target_rel);
}

//call getAgentTruth and getTargetTruth and converts to Pose
void getTruth(Pose &agent_truth, Pose &target_truth)
{
	Eigen::VectorXd x_agent;
	Eigen::MatrixXd P_agent; //not used, needed for input
	getAgentTruth(x_agent, P_agent);

  agent_truth.position.x = x_agent(0);
  agent_truth.position.y = x_agent(1);
  agent_truth.position.z = x_agent(2);
  agent_truth.orientation.w = x_agent(3);
  agent_truth.orientation.x = x_agent(4);
  agent_truth.orientation.y = x_agent(5);
  agent_truth.orientation.z = x_agent(6);

	Eigen::VectorXd x_target;
	getTargetTruth(x_target);

  target_truth.position.x = x_target(0);
  target_truth.position.y = x_target(1);
  target_truth.position.z = x_target(2);
  target_truth.orientation.w = x_target(3);
  target_truth.orientation.x = x_target(4);
  target_truth.orientation.y = x_target(5);
  target_truth.orientation.z = x_target(6);
}

void getError(Pose &agent_truth, Pose&target_truth)
{

}
