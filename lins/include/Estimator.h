// This file is part of LINS.
//
// Copyright (C) 2020 Chao Qin <cscharlesqin@gmail.com>,
// Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
// The Hong Kong University of Science and Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.

#ifndef INCLUDE_ESTIMATOR_H_
#define INCLUDE_ESTIMATOR_H_

#include <MapRingBuffer.h>
#include <math_utils.h>
#include <nav_msgs/Odometry.h>
#include <parameters.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <tic_toc.h>

#include <StateEstimator.hpp>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sensor_utils.hpp>

#include "cloud_msgs/cloud_info.h"
#include <mutex>

using namespace std;
using namespace Eigen;
using namespace math_utils;
using namespace sensor_utils;

namespace fusion {

class LinsFusion {
 public:
  LinsFusion(ros::NodeHandle& nh, ros::NodeHandle& pnh);
  ~LinsFusion();

  void run();
  void initialization();
  void publishTopics();
  void publishOdometryYZX(double timeStamp);
  inline void publishCloudMsg(ros::Publisher& publisher,
                              pcl::PointCloud<PointType>::Ptr cloud,
                              const ros::Time& stamp,
                              const std::string& frameID) {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.stamp = stamp;
    msg.header.frame_id = frameID;
    publisher.publish(msg);
  }

  void imuCallback(const sensor_msgs::Imu::ConstPtr& imuIn);
  void laserCloudCallback(
      const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);
  void laserCloudInfoCallback(const cloud_msgs::cloud_infoConstPtr& msgIn);
  void outlierCloudCallback(
      const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);
  void mapOdometryCallback(const nav_msgs::Odometry::ConstPtr& odometryMsg);

  void performStateEstimation();
  void processFirstPointCloud();
  bool processPointClouds();
  void performImuBiasEstimation();
  void alignIMUtoVehicle(const V3D& rpy, const V3D& acc_in, const V3D& gyr_in,
                         V3D& acc_out, V3D& gyr_out);

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // !@StateEstimator
  StateEstimator* estimator;

  // !@Subscriber
  ros::Subscriber subLaserCloud;
  ros::Subscriber subLaserCloudInfo;
  ros::Subscriber subOutlierCloud;
  ros::Subscriber subImu;
  ros::Subscriber subGPS_;
  ros::Subscriber subMapOdom_;

  // !@Publishers
  ros::Publisher pubUndistortedPointCloud;

  ros::Publisher pubCornerPointsSharp;
  ros::Publisher pubCornerPointsLessSharp;
  ros::Publisher pubSurfPointsFlat;
  ros::Publisher pubSurfPointsLessFlat;

  ros::Publisher pubLaserCloudCornerLast;
  ros::Publisher pubLaserCloudSurfLast;
  ros::Publisher pubOutlierCloudLast;

  ros::Publisher pubLaserOdometry;
  ros::Publisher pubIMUOdometry;
  ros::Publisher pubGpsOdometry;

  ros::Publisher pubLaserOdom;

  // !@PointCloudPtrs
  pcl::PointCloud<PointType>::Ptr distortedPointCloud;
  pcl::PointCloud<PointType>::Ptr outlierPointCloud;

  // !@Messages
  nav_msgs::Odometry laserOdom;
  nav_msgs::Odometry laserOdometry;
  nav_msgs::Odometry imuOdometry;
  nav_msgs::Odometry gpsOdometry;

  // !@Buffers
  //用<时间戳, sensormeasurement>构建一个std::map,std::map对象默认是按照key值从小到大排序。
  MapRingBuffer<Imu> imuBuf_; //保存laser的加速度和角速度
  MapRingBuffer<sensor_msgs::PointCloud2::ConstPtr> pclBuf_; //大小：3，保存"/segmented_cloud"
  MapRingBuffer<sensor_msgs::PointCloud2::ConstPtr> outlierBuf_; //大小：3，保存"/outlier_cloud"
  MapRingBuffer<cloud_msgs::cloud_info> cloudInfoBuf_;//大小：3，保存"/segmented_cloud_info"
  MapRingBuffer<Gps> gpsBuf_;

  //we add 
  std::mutex imu_mutex_, pcl_mutex_;

  // !@Time
  int scan_counter_;
  double duration_;

  // !@Measurements
  V3D acc_raw_; //原始imu的加速度和角速度
  V3D gyr_raw_;
  V3D acc_aligned_;//laser的加速度和角速度
  V3D gyr_aligned_;
  V3D misalign_euler_angles_;
  V3D ba_tmp_; //??
  V3D bw_tmp_; //??
  V3D ba_init_;
  V3D bw_init_;
  double scan_time_;
  double last_imu_time_;
  double last_scan_time_;
  int sample_counter_;
  bool isInititialized;
  bool isImuCalibrated;
};
}  // namespace fusion

#endif  // INCLUDE_ESTIMATOR_H_
