#include "geometry_msgs/Twist.h"
#include "ros/callback_queue.h"
#include "ros/ros.h"
#include "ros/subscribe_options.h"
#include <cmath>
#include <functional>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Vector3.hh>
#include <thread>

#ifndef PI
#define PI 3.14159265
#endif

namespace gazebo {
class ModelPush : public ModelPlugin {
private:
  std::unique_ptr<ros::NodeHandle> rosNode;
  ros::Subscriber rosSub;
  ros::CallbackQueue rosQueue;
  std::thread rosQueueThread;
  double xSpeed = 0, ySpeed = 0, thetaSpeed = 0;
  physics::ModelPtr model;
  ros::Time lastUpdateTime;
  event::ConnectionPtr updateConnection;

  void QueueThread() {
    static const double timeout = 0.01;
    while (this->rosNode->ok()) {
      this->rosQueue.callAvailable(ros::WallDuration(timeout));
    }
  }

public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) {
    this->model = _parent;

    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&ModelPush::OnUpdate, this));

    if (!ros::isInitialized()) {
      int argc = 0;
      char **argv = NULL;
      ros::init(argc, argv, "gazebo_client",
                ros::init_options::NoSigintHandler);
    }

    std::string modelName = this->model->GetName();
    std::string cmdTopic = "/" + modelName + "/cmd_vel";

    this->rosNode.reset(new ros::NodeHandle("gazebo_client_" + modelName));

    ros::SubscribeOptions so =
        ros::SubscribeOptions::create<geometry_msgs::Twist>(
            cmdTopic, 1,
            boost::bind(&ModelPush::OnRosMsg, this, _1), ros::VoidPtr(),
            &this->rosQueue);
    this->rosSub = this->rosNode->subscribe(so);

    this->rosQueueThread =
        std::thread(std::bind(&ModelPush::QueueThread, this));

    this->lastUpdateTime = ros::Time::now();
  }

  void OnUpdate() {
    if (ros::Time::now().toSec() - lastUpdateTime.toSec() > 0.15) {
      this->xSpeed = 0.0;
      this->ySpeed = 0.0;
      this->thetaSpeed = 0.0;
    }

    float world_angle = this->model->WorldPose().Rot().Yaw();
    float x_setpoint = xSpeed * std::cos(world_angle) +
                       ySpeed * std::cos(world_angle + PI / 2.0);
    float y_setpoint = xSpeed * std::sin(world_angle) +
                       ySpeed * std::sin(world_angle + PI / 2.0);
    float z_setpoint = this->model->RelativeLinearVel().Z();

    this->model->SetLinearVel(
        ignition::math::Vector3d(x_setpoint, y_setpoint, z_setpoint));
    this->model->SetAngularVel(ignition::math::Vector3d(0, 0, thetaSpeed));
  }

  void OnRosMsg(const geometry_msgs::TwistConstPtr &_msg) {
    this->xSpeed = _msg->linear.x;
    this->ySpeed = _msg->linear.y;
    this->thetaSpeed = _msg->angular.z;
    this->lastUpdateTime = ros::Time::now();
  }
};

GZ_REGISTER_MODEL_PLUGIN(ModelPush)
} // namespace gazebo