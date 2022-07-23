#pragma once

// ROS
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <std_srvs/Empty.h>
// ros_control
#include <controller_manager/controller_manager.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/robot_hw.h>
// ostringstream
#include <sstream>

#include "icerobot_base/TalonNode.h"
#include "icerobot_base/MotorControl.h"
#include "icerobot_base/Reset.h"

//https://github.com/eborghi10/my_ROS_mobile_robot/blob/master/my_robot_base/src/my_robot_base.cpp
const unsigned int NUM_JOINTS = 4;

/// \brief Hardware interface for a robot
class MyRobotHWInterface : public hardware_interface::RobotHW
{
public:
  MyRobotHWInterface();

  void write() {
    double diff_ang_speed_left = cmd[0]*5 / 7;
    double diff_ang_speed_right = cmd[1]*5 / 7;
    limitDifferentialSpeed(diff_ang_speed_left, diff_ang_speed_right);
    
    std_msgs::Float64 left_wheel_vel_msg;
    std_msgs::Float64 right_wheel_vel_msg;
    left_wheel_vel_msg.data = diff_ang_speed_left;
    right_wheel_vel_msg.data = diff_ang_speed_right;
    
    // Falcon motor parameters
    icerobot_base::MotorControlPtr left(new icerobot_base::MotorControl);
    left->mode = icerobot_base::MotorControl::VELOCITY;
    left->value = left_wheel_vel_msg.data;
    icerobot_base::MotorControlPtr right(new icerobot_base::MotorControl);
    right->mode = icerobot_base::MotorControl::VELOCITY;
    right->value = right_wheel_vel_msg.data;

    // Publish results
    front_left_wheel_vel_pub_.publish(left);
    back_left_wheel_vel_pub_.publish(left);
    front_right_wheel_vel_pub_.publish(right);
    back_right_wheel_vel_pub_.publish(right);
  }

  /**
   * Reading encoder values and setting position and velocity of enconders 
   */
  void read(const ros::Duration &period) {
    double ang_distance_left = _wheel_angle[0];
    double ang_distance_right = _wheel_angle[1];
    pos[0] = ang_distance_left;
    pos[1] = ang_distance_right;
  } 
  ros::Time get_time() {
  prev_update_time = curr_update_time;
  curr_update_time = ros::Time::now();
  return curr_update_time;
  }

  ros::Duration get_period() {
    return curr_update_time - prev_update_time;
  }

  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

private:
  hardware_interface::JointStateInterface jnt_state_interface;
  hardware_interface::VelocityJointInterface jnt_vel_interface;
  double cmd[NUM_JOINTS];
  double pos[NUM_JOINTS];
  double vel[NUM_JOINTS];
  double eff[NUM_JOINTS];

  bool running_;
  double _wheel_diameter;
  double _max_speed;
  double _wheel_angle[NUM_JOINTS];

  ros::Time curr_update_time, prev_update_time;

  ros::Subscriber left_wheel_angle_sub_;
  ros::Subscriber right_wheel_angle_sub_;
  ros::Publisher front_left_wheel_vel_pub_;
  ros::Publisher back_left_wheel_vel_pub_;
  ros::Publisher front_right_wheel_vel_pub_;
  ros::Publisher back_right_wheel_vel_pub_;

  ros::ServiceServer start_srv_;
  ros::ServiceServer stop_srv_;

  bool start_callback(std_srvs::Empty::Request& /*req*/, std_srvs::Empty::Response& /*res*/)
  { 
    running_ = true;
    return true;
  }

  bool stop_callback(std_srvs::Empty::Request& /*req*/, std_srvs::Empty::Response& /*res*/)
  {
    running_ = false;
    return true;
  }

  void leftWheelAngleCallback(const std_msgs::Float32& msg) {
    _wheel_angle[0] = msg.data;
  }

  void rightWheelAngleCallback(const std_msgs::Float32& msg) {
    _wheel_angle[1] = msg.data;
  }

  void limitDifferentialSpeed(double &diff_speed_left, double &diff_speed_right)
  {
	double speed = std::max(std::abs(diff_speed_left), std::abs(diff_speed_right));
	if (speed > _max_speed) {
		diff_speed_left *= _max_speed / speed;
		diff_speed_right *= _max_speed / speed;
	}
  }

};  // class

MyRobotHWInterface::MyRobotHWInterface()
: running_(true)
  , private_nh("~")
  , start_srv_(nh.advertiseService("start", &MyRobotHWInterface::start_callback, this))
  , stop_srv_(nh.advertiseService("stop", &MyRobotHWInterface::stop_callback, this)) 
  {
    private_nh.param<double>("wheel_diameter", _wheel_diameter, 0.3302);
    private_nh.param<double>("max_speed", _max_speed, 4.2);
  
    // Intialize raw data
    std::fill_n(pos, NUM_JOINTS, 0.0);
    std::fill_n(vel, NUM_JOINTS, 0.0);
    std::fill_n(eff, NUM_JOINTS, 0.0);
    std::fill_n(cmd, NUM_JOINTS, 0.0);

    // connect and register the joint state and velocity interfaces
    hardware_interface::JointStateHandle state_handle_rl("rear_left_wheel_joint", &pos[0], &vel[0], &eff[0]);
    jnt_state_interface.registerHandle(state_handle_rl);
    hardware_interface::JointHandle vel_handle_rl(jnt_state_interface.getHandle("rear_left_wheel_joint"), &cmd[0]);
    jnt_vel_interface.registerHandle(vel_handle_rl);

    hardware_interface::JointStateHandle state_handle_rr("rear_right_wheel_joint", &pos[1], &vel[1], &eff[1]);
    jnt_state_interface.registerHandle(state_handle_rr);
    hardware_interface::JointHandle vel_handle_rr(jnt_state_interface.getHandle("rear_right_wheel_joint"), &cmd[1]);
    jnt_vel_interface.registerHandle(vel_handle_rr);

    hardware_interface::JointStateHandle state_handle_fl("front_left_wheel_joint", &pos[2], &vel[2], &eff[2]);
    jnt_state_interface.registerHandle(state_handle_fl);
    hardware_interface::JointHandle vel_handle_fl(jnt_state_interface.getHandle("front_left_wheel_joint"), &cmd[2]);
    jnt_vel_interface.registerHandle(vel_handle_fl);

    hardware_interface::JointStateHandle state_handle_fr("front_right_wheel_joint", &pos[3], &vel[3], &eff[3]);
    jnt_state_interface.registerHandle(state_handle_fr);
    hardware_interface::JointHandle vel_handle_fr(jnt_state_interface.getHandle("front_right_wheel_joint"), &cmd[3]);
    jnt_vel_interface.registerHandle(vel_handle_fr);

    registerInterface(&jnt_state_interface);
    registerInterface(&jnt_vel_interface);

    // Initialize publishers and subscribers
    front_left_wheel_vel_pub_ = nh.advertise<icerobot_base::MotorControl>("/front_left/set", 1);
    back_left_wheel_vel_pub_ = nh.advertise<icerobot_base::MotorControl>("/back_left/set", 1);
    front_right_wheel_vel_pub_ = nh.advertise<icerobot_base::MotorControl>("/front_right/set", 1);
    back_right_wheel_vel_pub_ = nh.advertise<icerobot_base::MotorControl>("/back_right/set", 1);
    
    left_wheel_angle_sub_ = nh.subscribe("/back_left/angle", 1, &MyRobotHWInterface::leftWheelAngleCallback, this);
    right_wheel_angle_sub_ = nh.subscribe("/back_right/angle", 1, &MyRobotHWInterface::rightWheelAngleCallback, this);
}
