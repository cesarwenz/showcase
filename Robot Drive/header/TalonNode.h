#ifndef ICEROBOT_BASE_TALON_H
#define ICEROBOT_BASE_TALON_H

#define Phoenix_No_WPI // remove WPI dependencies
#include "ctre/Phoenix.h"
#include "ctre/phoenix/platform/Platform.h"
#include "ctre/phoenix/unmanaged/Unmanaged.h"

#include "icerobot_base/TalonConfig.h"
#include "icerobot_base/MotorControl.h"
#include "icerobot_base/MotorAngle.h"
#include "icerobot_base/MotorStatus.h"
#include "icerobot_base/Reset.h"

#include <string>
#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>

namespace icerobot_base {
class TalonNode {
private:
    boost::recursive_mutex mutex;
    ros::NodeHandle nh;
    std::string _name;
    dynamic_reconfigure::Server<TalonConfig> server;
    TalonConfig _config;

    TalonFX talon;

    ros::Publisher statusPub;

    ros::Publisher anglePub;

    ros::Subscriber setSub;

    ros::Time lastUpdate;
    ControlMode _controlMode;
    std_msgs::Float64 _output;
    bool disabled;
    bool configured;
    bool not_configured_warned;
    ros::ServiceServer reset_odom_;

public:
    TalonNode(const ros::NodeHandle& parent, const std::string& name, int id, const TalonConfig& config);

    TalonNode& operator=(const TalonNode&) = delete;

    ~TalonNode() = default;

    void reconfigure(const TalonConfig& config, uint32_t level);

    void configure();

    void set(MotorControl output);

    void update();

    void configureStatusPeriod();

    bool resetOdom(icerobot_base::Reset::Request &req, 
		    icerobot_base::Reset::Response &res);
};

} // namespace icerobot_base

#endif // ICEROBOT_BASE_TALON_H
