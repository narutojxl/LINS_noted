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

#ifndef INCLUDE_STATEESTIMATOR_HPP_
#define INCLUDE_STATEESTIMATOR_HPP_

#include <integrationBase.h>
#include <math_utils.h>
#include <parameters.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_broadcaster.h>
#include <tic_toc.h>

#include <KalmanFilter.hpp>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <sensor_utils.hpp>
#include <vector>

#include "cloud_msgs/cloud_info.h"
#include "integrationBase.h"


//#include <geometry_msgs/PoseStamped.h>

using namespace Eigen;
using namespace std;
using namespace math_utils;
using namespace sensor_utils;
using namespace parameter;
using namespace filter;

namespace fusion {

const int LINE_NUM_ = 16;
const int SCAN_NUM_ = 1800;

struct Smooth {
  Smooth() {
    value = 0.0;
    ind = 0;
  }
  double value;
  size_t ind;
};

struct byValue {
  bool operator()(Smooth const &left, Smooth const &right) {
    return left.value < right.value;
  }
};

// Scan Class stores all kinds of information of a point cloud, including
// the whole point cloud, its smoothness, timestamp, and features.
class Scan {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Scan() : id_(scan_counter_++) {
    distPointCloud_.reset(new pcl::PointCloud<PointType>());
    undistPointCloud_.reset(new pcl::PointCloud<PointType>());
    outlierPointCloud_.reset(new pcl::PointCloud<PointType>());
    cornerPointsSharp_.reset(new pcl::PointCloud<PointType>());
    cornerPointsLessSharp_.reset(new pcl::PointCloud<PointType>());
    surfPointsFlat_.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlat_.reset(new pcl::PointCloud<PointType>());
    cloudInfo_.reset(new cloud_msgs::cloud_info());

    cornerPointsLessSharpYZX_.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlatYZX_.reset(new pcl::PointCloud<PointType>());
    outlierPointCloudYZX_.reset(new pcl::PointCloud<PointType>());

    cloudCurvature_.resize(LINE_NUM * SCAN_NUM);
    cloudSmoothness_.resize(LINE_NUM * SCAN_NUM);

    reset();
  }

  ~Scan() {
    distPointCloud_.reset();
    undistPointCloud_.reset();
    outlierPointCloud_.reset();
    cornerPointsSharp_.reset();
    cornerPointsLessSharp_.reset();
    surfPointsFlat_.reset();
    surfPointsLessFlat_.reset();
    cloudInfo_.reset();

    cornerPointsLessSharpYZX_.reset();
    surfPointsLessFlatYZX_.reset();
    outlierPointCloudYZX_.reset();
  }

  void reset() {
    time_ = 0.0;

    distPointCloud_->clear();
    undistPointCloud_->clear();
    outlierPointCloud_->clear();
    cornerPointsSharp_->clear();
    cornerPointsLessSharp_->clear();
    surfPointsFlat_->clear();
    surfPointsLessFlat_->clear();

    cornerPointsLessSharpYZX_->clear();
    surfPointsLessFlatYZX_->clear();
    outlierPointCloudYZX_->clear();

    cloudCurvature_.assign(LINE_NUM * SCAN_NUM, 0.0);
    cloudSmoothness_.assign(LINE_NUM * SCAN_NUM, Smooth());
  }

  void setPointCloud(double time,
                     pcl::PointCloud<PointType>::Ptr distPointCloud,
                     cloud_msgs::cloud_info cloudInfo,
                     pcl::PointCloud<PointType>::Ptr outlierPointCloud) {
    distPointCloud_ = distPointCloud;
    *(cloudInfo_) = cloudInfo;
    outlierPointCloud_ = outlierPointCloud;
    time_ = time;
  }

public:
  // !@ScanInfo
  static int scan_counter_;
  int id_;
  double time_;

  // !@PointCloud
  pcl::PointCloud<PointType>::Ptr distPointCloud_;
  pcl::PointCloud<PointType>::Ptr undistPointCloud_;
  pcl::PointCloud<PointType>::Ptr outlierPointCloud_;
  cloud_msgs::cloud_info::Ptr cloudInfo_;

  // !@PclFeatures
  std::vector<double> cloudCurvature_;
  std::vector<Smooth> cloudSmoothness_;

  int cloudNeighborPicked_[LINE_NUM_ * SCAN_NUM_];
  int cloudLabel_[LINE_NUM_ * SCAN_NUM_];

  pcl::PointCloud<PointType>::Ptr cornerPointsSharp_;
  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp_;
  pcl::PointCloud<PointType>::Ptr surfPointsFlat_;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlat_;

  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharpYZX_; //带YZX_的对应的是zhangji坐标系下
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatYZX_;
  pcl::PointCloud<PointType>::Ptr outlierPointCloudYZX_;
};
typedef shared_ptr<Scan> ScanPtr; // Define a pointer class for Scan class

// StateEstimator Class implement a iterative-ESKF, including state propagation
// and update.
class StateEstimator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum FusionStatus {
    STATUS_INIT = 0,
    STATUS_FIRST_SCAN = 1,
    STATUS_SECOND_SCAN = 2,
    STATUS_RUNNING = 3,
    STATUS_RESET = 4,
  };

  StateEstimator() {
    filter_ = new StatePredictor();

    // Initialize KD tree and downsize filter
    downSizeFilter_.setLeafSize(0.2, 0.2, 0.2);
    kdtreeCorner_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurf_.reset(new pcl::KdTreeFLANN<PointType>());
    scan_new_.reset(new Scan());
    scan_last_.reset(new Scan());

    keypoints_.reset(new pcl::PointCloud<PointType>());
    jacobians_.reset(new pcl::PointCloud<PointType>());
    keypointCorns_.reset(new pcl::PointCloud<PointType>());
    keypointSurfs_.reset(new pcl::PointCloud<PointType>());
    jacobianCoffCorns.reset(new pcl::PointCloud<PointType>());
    jacobianCoffSurfs.reset(new pcl::PointCloud<PointType>());

    surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

    pointSelCornerInd.resize(LINE_NUM * SCAN_NUM);
    pointSearchCornerInd1.resize(LINE_NUM * SCAN_NUM);
    pointSearchCornerInd2.resize(LINE_NUM * SCAN_NUM);
    pointSelSurfInd.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd1.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd2.resize(LINE_NUM * SCAN_NUM);
    pointSearchSurfInd3.resize(LINE_NUM * SCAN_NUM);

    globalState_.setIdentity();
    globalStateYZX_.setIdentity();

    // Rotation matrics between XYZ convention and YZX-convention
    R_yzx_to_xyz << 0., 0., 1., 1., 0., 0., 0., 1., 0.; //把zhangji下的点(Z-front,X-left,Y-up)转换到ros坐标系下(X-front,Y-left,Z-up)的旋转矩阵
    R_xyz_to_yzx = R_yzx_to_xyz.transpose();
    Q_yzx_to_xyz = R_yzx_to_xyz;
    Q_xyz_to_yzx = R_xyz_to_yzx;

    // gravity_feedback << 0, 0, -G0;

    status_ = STATUS_INIT;

    //jxl debug
    // pub_correctRollPitch_ = nh_.advertise<geometry_msgs::PoseStamped>("/filter_gn_correctRollPitch", 1);
  }

  ~StateEstimator() {
    delete filter_;
    delete preintegration_;
  }

  inline const double getTime() const { return filter_->time_; }
  inline bool isInitialized() const { return status_ != STATUS_INIT; }

  /********Relative Variables*********/
  V3D pos_;
  V3D vel_;
  M3D quad_;
  V3D acc_0_;
  V3D gyr_0_;
  /***********************************/
  void processImu(double dt, const V3D &acc, const V3D &gyr) {
    switch (status_) {
    case STATUS_INIT:
      break;
    case STATUS_FIRST_SCAN:
      preintegration_->push_back(dt, acc, gyr); //imu预积分只发生在[laser1, laser2]之间。因为处理完第二帧laser后状态变为STATUS_RUNNING
      filter_->time_ += dt;
      acc_0_ = acc;
      gyr_0_ = gyr;
      break;
    case STATUS_RUNNING:
      filter_->predict(dt, acc, gyr, true); //imu积分, 把PVQ状态往前推一个imu时间戳，同时计算状态的方差
      break;
    default:
      break;
    }

    // For no use here. Just propagte IMU measurements for testing
    V3D un_acc_0_ = quad_ * (acc_0_ - INIT_BA) + filter_->state_.gn_; //在laser0下unbias无重力加速度,wa. 在imu回调函数中已经把imu数据转换到laser下了。
    V3D un_gyr = 0.5 * (gyr_0_ + gyr) - INIT_BW; // laser下角速度
    quad_ *= math_utils::deltaQ(un_gyr * dt).toRotationMatrix(); //当前imu时刻在laser0下的旋转

    V3D un_acc_1 = quad_ * (acc - INIT_BA) + filter_->state_.gn_;
    V3D un_acc = 0.5 * (un_acc_0_ + un_acc_1);
    pos_ += dt * vel_ + 0.5 * dt * dt * un_acc; //当前imu时刻在laser0下的位移
    vel_ += dt * un_acc; //当前imu时刻在laser0下的速度

    acc_0_ = acc;
    gyr_0_ = gyr;
  }

  /********Relative Variables*********/
  double duration_fea_ = 0;
  double duration_opt_ = 0;
  double num_of_edge_ = 0;
  double num_of_surf_ = 0;
  int lidar_counter_ = 0;
  /***********************************/
  void processPCL(double time, const Imu &imu,
                  pcl::PointCloud<PointType>::Ptr distortedPointCloud,
                  cloud_msgs::cloud_info cloudInfo,
                  pcl::PointCloud<PointType>::Ptr outlierPointCloud) {
    TicToc ts_fea; // Calculate the time used in feature extraction
    scan_new_->setPointCloud(time, distortedPointCloud, cloudInfo,
                             outlierPointCloud);
    undistortPcl(scan_new_);
    calculateSmoothness(scan_new_);
    markOccludedPoints(scan_new_);
    extractFeatures(scan_new_);
    imu_last_ = imu; //当前latest imu
    double time_fea = ts_fea.toc();

    TicToc ts_opt; // Calculate the time used in state estimation
    switch (status_) {
    case STATUS_INIT:
      if (processFirstScan()) status_ = STATUS_FIRST_SCAN;
      break;
    case STATUS_FIRST_SCAN:
      if (processSecondScan())
        status_ = STATUS_RUNNING;
      else
        status_ = STATUS_INIT;
      break;
    case STATUS_RUNNING: //第3帧,4帧,5帧...
      if (!processScan()) status_ = STATUS_RUNNING; 
      break;
    }
    double time_opt = ts_opt.toc();

    // if (VERBOSE) {
    //   duration_fea_ =
    //       (duration_fea_ * lidar_counter_ + time_fea) / (lidar_counter_ + 1);
    //   duration_opt_ =
    //       (duration_opt_ * lidar_counter_ + time_opt) / (lidar_counter_ + 1);
    //   num_of_edge_ = (num_of_edge_ * lidar_counter_ +
    //                   scan_last_->cornerPointsLessSharpYZX_->points.size()) /
    //                  (lidar_counter_ + 1);
    //   num_of_surf_ = (num_of_surf_ * lidar_counter_ +
    //                   scan_last_->surfPointsLessFlatYZX_->points.size()) /
    //                  (lidar_counter_ + 1);
    //   lidar_counter_++;

    //   // cout << "Feature Extraction: time: " << duration_fea_ << endl;
    //   // cout << "Feature Extraction: corners: " << num_of_edge_
    //   //      << ", surfs: " << num_of_surf_ << endl;
    //   // cout << "State Estimation: time: " << duration_opt_ << endl;
    // }
  }

  // Initialize the Kalman filter and KD tree
  bool processFirstScan() {
    if (scan_new_->cornerPointsLessSharp_->points.size() < 10 ||
        scan_new_->surfPointsLessFlat_->points.size() < 100) {
      ROS_WARN("Wait for more features for initialization...");
      scan_new_.reset(new Scan());
      return false;
    }

    // Initialize the Kalman filter
    Fk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    Gk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_);
    Pk_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
    Qk_.resize(GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_);
    IKH_.resize(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);

    Fk_.setIdentity();
    Gk_.setZero();
    Pk_.setZero();
    Qk_.setZero();

    // Set the relative transform to identity
    linState_.setIdentity();

    // Initialize IMU preintegration variable
    preintegration_ = new integration::IntegrationBase(
        imu_last_.acc, imu_last_.gyr, INIT_BA,INIT_BW); // INIT_BA, INIT_BW：从yaml配置文件读取
    // imu_last: 为第一帧imu 用第一帧imu创建预积分对象

    // Initialize position, velocity, acceleration bias, gyroscope bias by zeros
    filter_->initialization(scan_new_->time_, V3D(0, 0, 0), V3D(0, 0, 0),
                            V3D(0, 0, 0), V3D(0, 0, 0), imu_last_.acc,
                            imu_last_.gyr);
    // scan_new_->time_：收到第一帧imu数据时最新点云的时间戳
    //后面在处理第二帧laser时再次初始化

    kdtreeCorner_->setInputCloud(scan_new_->cornerPointsLessSharp_);
    kdtreeSurf_->setInputCloud(scan_new_->surfPointsLessFlat_);

    pos_.setZero();
    vel_.setZero();
    quad_.setIdentity();

    // Slide the point cloud from scan_new_ to scan_last_
    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  // Calculate initial velocity and IMU biases using two consecutive frames and
  // IMU preintegration results
  bool processSecondScan() {
    if (scan_new_->cornerPointsLessSharp_->points.size() < 10 ||
        scan_new_->surfPointsLessFlat_->points.size() < 100) {
      ROS_WARN("Wait for more features for initialization...");
      scan_new_.reset(new Scan());
      return false;
    }

    // Calculate relative transform, linState_, using ICP method
    V3D pl;
    Q4D ql;
    V3D v0, v1, ba0, bw0;
    ba0.setZero();
    ql = preintegration_->delta_q; //从k帧laser到k+1帧laser的旋转
    pl = preintegration_->delta_p + 
        0.5 * linState_.gn_ * preintegration_->sum_dt * 
		    preintegration_->sum_dt -
        0.5 * ba0 * preintegration_->sum_dt * preintegration_->sum_dt;
    // TODO作者的意思是从k帧laser到k+1帧laser的平移，平移为何不是下面的形式
    // pl =  delta_p  + 0.5*[0 0 -g]*(sum_dt)*(sum_dt) ？
    // pl =  preintegration_->delta_p  + 0.5 * (ql * imu_last_.acc) * preintegration_->sum_dt * preintegration_->sum_dt;

    estimateTransform(scan_last_, scan_new_, pl, ql);
    //用预积分的结果作为初值，用k，k+1帧lasers计算从k帧laser到k+1帧laser的旋转

    // Calculate initial state using relative transform calculated by point
    // clouds and that by IMU preintegration
    estimateInitialState(pl, ql, v0, v1, ba0, bw0); // ba0, bw0: init_ba, init_bw

    // Initialize the Kalman filter by estimated values
    V3D r1 = pl;
    filter_->initialization(scan_new_->time_, r1, v1, ba0, bw0, imu_last_.acc,
                            imu_last_.gyr); // imu_last: 当前latest imu
    // TODO为什么初始化的时候不用上面1st laser和2nd laser计算出来的ql
    // filter_->initialization(scan_new_->time_, r1, v1, ba0, bw0,
    //                         imu_last_.acc, imu_last_.gyr, 
    //                         Q2rpy(ql)(0),  Q2rpy(ql)(1),  Q2rpy(ql)(2));

    double roll_init, pitch_init, yaw_init = deg2rad(0.0);
    // Calculate rough roll and pitch angles using IMU measurements
    calculateRPfromGravity(imu_last_.acc - ba0, roll_init, pitch_init);

    // Initialize the global state, e.g., position, velocity, and orientation
    // represented in the original frame (the first-scan-frame)
    globalState_ = GlobalState(
        r1, v1, rpy2Quat(V3D(roll_init, pitch_init, yaw_init)), ba0, bw0);

    // Use relative transorm linState_ to undistort point cloud (under the
    // constant-speed assumption)
    updatePointCloud();

    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());

    return true;
  }

  void correctRollPitch(const double &roll, const double &pitch) {
    V3D rpy = math_utils::Q2rpy(globalState_.qbn_);
    Q4D quad = math_utils::rpy2Quat(V3D(roll, pitch, rpy[2]));
    globalState_.qbn_ = quad;
  }

  void correctOrientation(const Q4D &quad) { globalState_.qbn_ = quad; }

  bool processScan() {
    if (scan_new_->cornerPointsLessSharp_->points.size() <= 5 || //TODO 10? 与first, second scan 保持一致？ default: 5
        scan_new_->surfPointsLessFlat_->points.size() <= 10) { //100? default: 10
      ROS_WARN("Insufficient features...State estimation fails.");
      return false;
    }

    // Update states
    performIESKF();

    // Update global transform by estimated relative transform
    integrateTransformation();
    filter_->reset(1); //reset covariance and state

    double roll, pitch;
    // Because the estimated gravity is represented in the b-frame, we can
    // directly solve more accurate roll and pitch angles to correct the global
    // state
    calculateRPfromGravity(filter_->state_.gn_, roll, pitch); //TODO state_.gn_ 在预测和更新阶段基本上不会变，roll，pitch会一直在零附近？
    // geometry_msgs::PoseStamped debug_msg;
    // debug_msg.header.stamp = ros::Time().fromSec(filter_->time_);  //header.stamp.toSec()
    // debug_msg.pose.position.x = filter_->state_.gn_(0);
    // debug_msg.pose.position.y = filter_->state_.gn_(1);
    // debug_msg.pose.position.z = filter_->state_.gn_(2);
    // debug_msg.pose.orientation.x = rad2deg(roll);
    // debug_msg.pose.orientation.y = rad2deg(pitch);
    // //std::cout<<"(roll, pitch) = ("<<rad2deg(roll)<<", "<<rad2deg(pitch)<<")"<<std::endl<<std::endl;
    // pub_correctRollPitch_.publish(debug_msg);
    correctRollPitch(roll, pitch);

    // Undistort point cloud using estimated relative transform
    updatePointCloud();

    // Slide the new scan to last scan
    scan_last_.swap(scan_new_);
    scan_new_.reset(new Scan());
    
    //ROS_INFO("process laser done");
    return true;
  }

  void performIESKF() {
    // Store current state and perform initialization
    Pk_ = filter_->covariance_; //jxl: prior state covariance
    GlobalState filterState = filter_->state_; //滤波器用imu measurements预测出来的状态，即prior state
    linState_ = filterState; 

    double residualNorm = 1e6;
    bool hasConverged = false;
    bool hasDiverged = false;
    const unsigned int DIM_OF_STATE = GlobalState::DIM_OF_STATE_;
    for (int iter = 0; iter < NUM_ITER && !hasConverged && !hasDiverged;
         iter++) {
      keypointSurfs_->clear();
      jacobianCoffSurfs->clear();
      keypointCorns_->clear();
      jacobianCoffCorns->clear();

      // Find corresponding features
      findCorrespondingSurfFeatures(scan_last_, scan_new_, keypointSurfs_,
                                    jacobianCoffSurfs, iter);
      if (keypointSurfs_->points.size() < 10) { //TODO default: 10
        if (VERBOSE) {
          ROS_WARN("Insufficient matched surfs...");
        }
      }
      findCorrespondingCornerFeatures(scan_last_, scan_new_, keypointCorns_,
                                      jacobianCoffCorns, iter);
      if (keypointCorns_->points.size() < 5) { //TODO default: 5
        if (VERBOSE) {
          ROS_WARN("Insufficient matched corners...");
        }
      }
      
      auto surf_num = keypointSurfs_->points.size();
      auto corner_num = keypointCorns_->points.size();
      ROS_INFO("surf and corner = (%d, %d)", surf_num,  corner_num);

      //TODO jxl: 程序有时会随机挂掉，原因是合格的面点数量为0，jacobianCoffSurfs为空
      if (surf_num < 10 || corner_num < 5)
        continue;

      // Sum up jocobians and residuals
      keypoints_->clear();
      jacobians_->clear();
      (*keypoints_) += (*keypointSurfs_);
      (*keypoints_) += (*keypointCorns_);
      (*jacobians_) += (*jacobianCoffSurfs);
      (*jacobians_) += (*jacobianCoffCorns);

      // Memery allocation
      const unsigned int DIM_OF_MEAS = keypoints_->points.size(); // m
      residual_.resize(DIM_OF_MEAS);
      Hk_.resize(DIM_OF_MEAS, DIM_OF_STATE);   // m*18  //残差对状态的雅克比，不是对error state的雅克比。作者滤波器里的状态变量是bk到bk+1时刻delta变换，不是error state.
      Rk_.resize(DIM_OF_MEAS, DIM_OF_MEAS);    // m*m  测量方差
      Kk_.resize(DIM_OF_STATE, DIM_OF_MEAS);   // 18*m
      Py_.resize(DIM_OF_MEAS, DIM_OF_MEAS);    // m*m
      Pyinv_.resize(DIM_OF_MEAS, DIM_OF_MEAS); // m*m

      Hk_.setZero();
      V3D axis = Quat2axis(linState_.qbn_);
      for (int i = 0; i < DIM_OF_MEAS; ++i) {
        // Point represented in 2-frame (e.g., the end frame) in a
        // xyz-convention
        V3D P2xyz(keypoints_->points[i].x, keypoints_->points[i].y,
                  keypoints_->points[i].z);
        V3D coff_xyz(jacobians_->points[i].x, jacobians_->points[i].y,
                     jacobians_->points[i].z);
        residual_(i) = LIDAR_SCALE * jacobians_->points[i].intensity; // LIDAR_SCALE: 1

        Hk_.block<1, 3>(i, GlobalState::att_) =
            coff_xyz.transpose() *
            (-linState_.qbn_.toRotationMatrix() * skew(P2xyz)) *
            Rinvleft(-axis); //李代数求导公式，比右扰动公式更精确，高翔4.32式

        Hk_.block<1, 3>(i, GlobalState::pos_) =   
            coff_xyz.transpose() * M3D::Identity(); //TODO 好像少了一个s？
        //因为测量方程中只包含状态的QT分量(R, t)，所以残差只对这两个分量不为零
      }

      // Set the measurement covariance matrix
      VXD cov = VXD::Zero(DIM_OF_MEAS);
      for (int i = 0; i < DIM_OF_MEAS; ++i) {
        cov[i] = LIDAR_STD * LIDAR_STD;
      }
      Rk_ = cov.asDiagonal();

      // Kalman filter update. Details can be referred to ROVIO
      Py_ =
          Hk_ * Pk_ * Hk_.transpose() + Rk_;  // S = H * P * H.transpose() + R;
      Pyinv_.setIdentity();                   // solve Ax=B
      Py_.llt().solveInPlace(Pyinv_); //default 
      // Pyinv_ = Py_.colPivHouseholderQr().solve(Pyinv_);  //jxl we add

      Kk_ = Pk_ * Hk_.transpose() * Pyinv_;  // K = P*H.transpose()*S.inverse()

      filterState.boxMinus(linState_, difVecLinInv_); //计算 linState_.inverse() * filterState
      updateVec_ = -Kk_ * (residual_ + Hk_ * difVecLinInv_) + difVecLinInv_; //TODO 为什么要加difVecLinInv_？

      // Divergence determination
      bool hasNaN = false;
      for (int i = 0; i < updateVec_.size(); i++) {
        if (isnan(updateVec_[i])) {
          updateVec_[i] = 0;
          hasNaN = true;
        }
      }
      if (hasNaN == true) {
        ROS_WARN("System diverges Because of NaN...");
        hasDiverged = true;
        break;
      }

      // Check whether the filter converges
      if (residual_.norm() > residualNorm * 10) {//当前迭代获得的残差大小超过上次残差大小的10倍
        ROS_WARN("System diverges...");
        hasDiverged = true;
        break;
      }

      // Update the state
      linState_.boxPlus(updateVec_, linState_); //给linState施加右更新小量

      updateVecNorm_ = updateVec_.norm();
      if (updateVecNorm_ <= 1e-2) {
        hasConverged = true;
      }

      residualNorm = residual_.norm();
    }

    // If diverges, swtich to traditional ICP method to get a rough relative
    // transformation. Otherwise, update the error-state covariance matrix
    if (hasDiverged == true) {
      ROS_WARN("======Using ICP Method======");
      V3D t = filterState.rn_;
      Q4D q = filterState.qbn_;
      estimateTransform(scan_last_, scan_new_, t, q);
      filterState.rn_ = t;
      filterState.qbn_ = q;
      filter_->update(filterState, Pk_);
    } else {
      // Update only one time
      IKH_ = Eigen::Matrix<double, 18, 18>::Identity() - Kk_ * Hk_; //lins 23式
      Pk_ = IKH_ * Pk_ * IKH_.transpose() + Kk_ * Rk_ * Kk_.transpose();
      enforceSymmetry(Pk_);
      filter_->update(linState_, Pk_);
    }

    // std::cout<<"performIESKF: delta state_.gn_ =("<<(linState_.gn_ - filterState.gn_)(0)<<" "
    //                                               <<(linState_.gn_ - filterState.gn_)(1)<<" "
    //                                               <<(linState_.gn_ - filterState.gn_)(2)<<" )\n";
  }

  // TODO参考公式是什么？
  //https://engineering.stackexchange.com/questions/3348/calculating-pitch-yaw-and-roll-from-mag-acc-and-gyro-data
  //很多地方的解算公式不一致
  void calculateRPfromGravity(const V3D &fbib, double &roll, double &pitch) {
    pitch = -sign(fbib.z()) * asin(fbib.x() / G0);
    roll = sign(fbib.z()) * asin(fbib.y() / G0);
  }

  // Update the gloabl state by the new relative transformation
  void integrateTransformation() {
    GlobalState filterState = filter_->state_;
    globalState_.rn_ = globalState_.qbn_ * filterState.rn_ + globalState_.rn_;
    globalState_.qbn_ = globalState_.qbn_ * filterState.qbn_;
    globalState_.vn_ =
        globalState_.qbn_ * filterState.qbn_.inverse() * filterState.vn_;
    // vk+1_in_k+1 = filterState.qbn_.inverse() * filterState.vn_: k+1时刻的速度在k+1时刻坐标系下
    // vk+1_in_G: k+1时刻的速度在G系下

    globalState_.ba_ = filterState.ba_;
    globalState_.bw_ = filterState.bw_;
    globalState_.gn_ = globalState_.qbn_ * filterState.gn_;
  }

  void undistortPcl(ScanPtr scan) {
    bool halfPassed = false;
    scan->undistPointCloud_->clear();
    pcl::PointCloud<PointType>::Ptr distPointCloud = scan->distPointCloud_;
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    int size = distPointCloud->points.size();
    PointType point;
    for (int i = 0; i < size; i++) {
      // If LiDAR frame does not align with Vehic frame, we transform the point
      // cloud to the vehicle frame
      rotatePoint(&distPointCloud->points[i], &point); //可以comment掉

      double ori = -atan2(point.y, point.x);
      if (!halfPassed) {
        if (ori < segInfo->startOrientation - M_PI / 2)
          ori += 2 * M_PI;
        else if (ori > segInfo->startOrientation + M_PI * 3 / 2)
          ori -= 2 * M_PI;

        if (ori - segInfo->startOrientation > M_PI) halfPassed = true;
      } else {
        ori += 2 * M_PI;

        if (ori < segInfo->endOrientation - M_PI * 3 / 2)
          ori += 2 * M_PI;
        else if (ori > segInfo->endOrientation + M_PI / 2)
          ori -= 2 * M_PI;
      }
      double relTime =
          (ori - segInfo->startOrientation) / segInfo->orientationDiff;
      point.intensity =
          int(distPointCloud->points[i].intensity) + SCAN_PERIOD * relTime;

      scan->undistPointCloud_->push_back(point);
    }
  }

  void calculateSmoothness(ScanPtr scan) {
    int cloudSize = scan->undistPointCloud_->points.size();
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    for (int i = 5; i < cloudSize - 5; i++) {
      double diffRange = segInfo->segmentedCloudRange[i - 5] +
                         segInfo->segmentedCloudRange[i - 4] +
                         segInfo->segmentedCloudRange[i - 3] +
                         segInfo->segmentedCloudRange[i - 2] +
                         segInfo->segmentedCloudRange[i - 1] -
                         segInfo->segmentedCloudRange[i] * 10 +
                         segInfo->segmentedCloudRange[i + 1] +
                         segInfo->segmentedCloudRange[i + 2] +
                         segInfo->segmentedCloudRange[i + 3] +
                         segInfo->segmentedCloudRange[i + 4] +
                         segInfo->segmentedCloudRange[i + 5];
      scan->cloudCurvature_[i] = diffRange * diffRange;

      scan->cloudNeighborPicked_[i] = 0;
      scan->cloudLabel_[i] = 0;
      scan->cloudSmoothness_[i].value = scan->cloudCurvature_[i];
      scan->cloudSmoothness_[i].ind = i;
    }
  }

  void markOccludedPoints(ScanPtr scan) {
    int cloudSize = scan->undistPointCloud_->points.size();
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;
    for (int i = 5; i < cloudSize - 6; ++i) {
      float depth1 = segInfo->segmentedCloudRange[i];
      float depth2 = segInfo->segmentedCloudRange[i + 1];
      int columnDiff = std::abs(int(segInfo->segmentedCloudColInd[i + 1] -
                                    segInfo->segmentedCloudColInd[i]));
      if (columnDiff < 10) {
        if (depth1 - depth2 > 0.3) {
          scan->cloudNeighborPicked_[i - 5] = 1;
          scan->cloudNeighborPicked_[i - 4] = 1;
          scan->cloudNeighborPicked_[i - 3] = 1;
          scan->cloudNeighborPicked_[i - 2] = 1;
          scan->cloudNeighborPicked_[i - 1] = 1;
          scan->cloudNeighborPicked_[i] = 1;
        } else if (depth2 - depth1 > 0.3) {
          scan->cloudNeighborPicked_[i + 1] = 1;
          scan->cloudNeighborPicked_[i + 2] = 1;
          scan->cloudNeighborPicked_[i + 3] = 1;
          scan->cloudNeighborPicked_[i + 4] = 1;
          scan->cloudNeighborPicked_[i + 5] = 1;
          scan->cloudNeighborPicked_[i + 6] = 1;
        }
      }
      float diff1 = std::abs(segInfo->segmentedCloudRange[i - 1] -
                             segInfo->segmentedCloudRange[i]);
      float diff2 = std::abs(segInfo->segmentedCloudRange[i + 1] -
                             segInfo->segmentedCloudRange[i]);
      if (diff1 > 0.02 * segInfo->segmentedCloudRange[i] &&
          diff2 > 0.02 * segInfo->segmentedCloudRange[i])
        scan->cloudNeighborPicked_[i] = 1;
    }
  }

  /********Relative Variables*********/
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;
  /***********************************/
  void extractFeatures(ScanPtr scan) {
    cloud_msgs::cloud_info::Ptr segInfo = scan->cloudInfo_;

    scan->cornerPointsSharp_->clear();
    scan->cornerPointsLessSharp_->clear();
    scan->surfPointsFlat_->clear();
    scan->surfPointsLessFlat_->clear();

    for (int i = 0; i < LINE_NUM; i++) {
      surfPointsLessFlatScan->clear();

      for (int j = 0; j < 6; j++) {
        int sp = (segInfo->startRingIndex[i] * (6 - j) +
                  segInfo->endRingIndex[i] * j) / 6;
        int ep = (segInfo->startRingIndex[i] * (5 - j) +
                  segInfo->endRingIndex[i] * (j + 1)) /
                     6 - 1;

        if (sp >= ep) continue;

        std::sort(scan->cloudSmoothness_.begin() + sp,
                  scan->cloudSmoothness_.begin() + ep, byValue());

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--) {
          int ind = scan->cloudSmoothness_[k].ind;
          if (scan->cloudNeighborPicked_[ind] == 0 &&
              scan->cloudCurvature_[ind] > EDGE_THRESHOLD &&
              segInfo->segmentedCloudGroundFlag[ind] == false) {
            largestPickedNum++;
            if (largestPickedNum <= 2) {
              scan->cloudLabel_[ind] = 2;
              scan->cornerPointsSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
              scan->cornerPointsLessSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
            } else if (largestPickedNum <= 20) {
              scan->cloudLabel_[ind] = 1;
              scan->cornerPointsLessSharp_->push_back(
                  scan->undistPointCloud_->points[ind]);
            } else {
              break;
            }

            scan->cloudNeighborPicked_[ind] = 1;
            for (int l = 1; l <= 5; l++) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l - 1]));
              if (columnDiff > 10) break;
              scan->cloudNeighborPicked_[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l + 1]));
              if (columnDiff > 10) break;
              scan->cloudNeighborPicked_[ind + l] = 1;
            }
          }
        }

        int smallestPickedNum = 0;
        for (int k = sp; k <= ep; k++) {
          int ind = scan->cloudSmoothness_[k].ind;
          if (scan->cloudNeighborPicked_[ind] == 0 &&
              scan->cloudCurvature_[ind] < SURF_THRESHOLD &&
              segInfo->segmentedCloudGroundFlag[ind] == true) {
            scan->cloudLabel_[ind] = -1;
            scan->surfPointsFlat_->push_back(
                scan->undistPointCloud_->points[ind]);
            smallestPickedNum++;
            if (smallestPickedNum >= 4) {
              break;
            }

            scan->cloudNeighborPicked_[ind] = 1;
            for (int l = 1; l <= 5; l++) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l - 1]));
              if (columnDiff > 10) break;

              scan->cloudNeighborPicked_[ind + l] = 1;
            }
            for (int l = -1; l >= -5; l--) {
              int columnDiff =
                  std::abs(int(segInfo->segmentedCloudColInd[ind + l] -
                               segInfo->segmentedCloudColInd[ind + l + 1]));
              if (columnDiff > 10) break;

              scan->cloudNeighborPicked_[ind + l] = 1;
            }
          }
        }

        for (int k = sp; k <= ep; k++) {
          if (scan->cloudLabel_[k] <= 0) {
            surfPointsLessFlatScan->push_back(
                scan->undistPointCloud_->points[k]);
          }
        }
      }
      surfPointsLessFlatScanDS->clear();
      downSizeFilter_.setInputCloud(surfPointsLessFlatScan);
      downSizeFilter_.filter(*surfPointsLessFlatScanDS);
      *(scan->surfPointsLessFlat_) += *surfPointsLessFlatScanDS;
    }
  }

  void findCorrespondingSurfFeatures(
      ScanPtr lastScan, ScanPtr newScan,
      pcl::PointCloud<PointType>::Ptr keypoints,
      pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount) {
    int surfPointsFlatNum = newScan->surfPointsFlat_->points.size();
    int surfPointsLessFlatNum = lastScan->surfPointsLessFlat_->points.size(); // we add

    for (int i = 0; i < surfPointsFlatNum; i++) {
      PointType pointSel;
      PointType coeff, tripod1, tripod2, tripod3;

      transformToStart(&newScan->surfPointsFlat_->points[i], &pointSel); // pi

      pcl::PointCloud<PointType>::Ptr laserCloudSurfLast =
          lastScan->surfPointsLessFlat_;

      if (iterCount % ICP_FREQ == 0) {
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeSurf_->nearestKSearch(pointSel, 1, pointSearchInd,
                                    pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

        if (pointSearchSqDis[0] < NEAREST_FEATURE_SEARCH_SQ_DIST) {
          closestPointInd = pointSearchInd[0];
          int closestPointScan =
              int(laserCloudSurfLast->points[closestPointInd].intensity);

          float pointSqDis, minPointSqDis2 = NEAREST_FEATURE_SEARCH_SQ_DIST,
                            minPointSqDis3 = NEAREST_FEATURE_SEARCH_SQ_DIST;

          for (int j = closestPointInd + 1; j < surfPointsLessFlatNum; j++) { // TODO应该为surfPointsLessFlatNum, default: surfPointsFlatNum
            if (int(laserCloudSurfLast->points[j].intensity) >
                closestPointScan + 2.5) {
              continue; // TODO 应该为continue吗？ default: break
            }

            pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                             (laserCloudSurfLast->points[j].x - pointSel.x) +
                         (laserCloudSurfLast->points[j].y - pointSel.y) *
                             (laserCloudSurfLast->points[j].y - pointSel.y) +
                         (laserCloudSurfLast->points[j].z - pointSel.z) *
                             (laserCloudSurfLast->points[j].z - pointSel.z);
            if (int(laserCloudSurfLast->points[j].intensity) <=
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            } else {
              if (pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }
          }

          for (int j = closestPointInd - 1; j >= 0; j--) {
            if (int(laserCloudSurfLast->points[j].intensity) <
                closestPointScan - 2.5) {
              continue; // TODO 应该为continue吗？default: break
            }

            pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                             (laserCloudSurfLast->points[j].x - pointSel.x) +
                         (laserCloudSurfLast->points[j].y - pointSel.y) *
                             (laserCloudSurfLast->points[j].y - pointSel.y) +
                         (laserCloudSurfLast->points[j].z - pointSel.z) *
                             (laserCloudSurfLast->points[j].z - pointSel.z);

            if (int(laserCloudSurfLast->points[j].intensity) >=
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            } else {
              if (pointSqDis < minPointSqDis3) {
                minPointSqDis3 = pointSqDis;
                minPointInd3 = j;
              }
            }
          }
        }
        pointSearchSurfInd1[i] = closestPointInd; // pj
        pointSearchSurfInd2[i] = minPointInd2;    // pl
        pointSearchSurfInd3[i] = minPointInd3;    // pm
      }

      if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
        tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
        tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
        tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

        V3D P0xyz(pointSel.x, pointSel.y, pointSel.z); // pi
        V3D P1xyz(tripod1.x, tripod1.y, tripod1.z);    // pj
        V3D P2xyz(tripod2.x, tripod2.y, tripod2.z);    // pl
        V3D P3xyz(tripod3.x, tripod3.y, tripod3.z);    // pm

        V3D M = math_utils::skew(P1xyz - P2xyz) * (P1xyz - P3xyz);
        double r = (P0xyz - P1xyz).transpose() * M;
        double m = M.norm();
        float res = r / m; //点到平面的距离

        V3D jacxyz = M.transpose() / (m); //平面jlm的单位法向量

        float s = 1;
        if (iterCount >= ICP_FREQ) {
          s = 1 -
              1.8 * fabs(res) /
                  sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y +
                            pointSel.z * pointSel.z));
        }

        if (s > 0.1 && res != 0) {
          coeff.x = s * jacxyz(0);
          coeff.y = s * jacxyz(1);
          coeff.z = s * jacxyz(2);
          coeff.intensity = s * res;

          keypoints->push_back(newScan->surfPointsFlat_->points[i]);
          jacobianCoff->push_back(coeff);
        }
      }
    }
  }

  void findCorrespondingCornerFeatures(
      ScanPtr lastScan, ScanPtr newScan,
      pcl::PointCloud<PointType>::Ptr keypoints,
      pcl::PointCloud<PointType>::Ptr jacobianCoff, int iterCount) {
    int cornerPointsSharpNum = newScan->cornerPointsSharp_->points.size();
    int cornerPointsLessSharpNum = lastScan->cornerPointsLessSharp_->points.size(); // we add

    for (int i = 0; i < cornerPointsSharpNum; i++) {
      PointType pointSel;
      PointType coeff, tripod1, tripod2;

      transformToStart(&newScan->cornerPointsSharp_->points[i], &pointSel); // pi

      pcl::PointCloud<PointType>::Ptr laserCloudCornerLast =
          lastScan->cornerPointsLessSharp_;

      if (iterCount % ICP_FREQ == 0) {
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeCorner_->nearestKSearch(pointSel, 1, pointSearchInd,
                                      pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1;

        if (pointSearchSqDis[0] < NEAREST_FEATURE_SEARCH_SQ_DIST) {
          closestPointInd = pointSearchInd[0];
          int closestPointScan =
              int(laserCloudCornerLast->points[closestPointInd].intensity);

          float pointSqDis, minPointSqDis2 = NEAREST_FEATURE_SEARCH_SQ_DIST;
          for ( int j = closestPointInd + 1; j < cornerPointsLessSharpNum; j++) { // TODO:应该为cornerPointsLessSharpNum, default：cornerPointsSharpNum
            if (int(laserCloudCornerLast->points[j].intensity) >
                closestPointScan + 2.5) {
                  continue; // TODO 应该为continue吗？ default: break
            }

            pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                             (laserCloudCornerLast->points[j].x - pointSel.x) +
                         (laserCloudCornerLast->points[j].y - pointSel.y) *
                             (laserCloudCornerLast->points[j].y - pointSel.y) +
                         (laserCloudCornerLast->points[j].z - pointSel.z) *
                             (laserCloudCornerLast->points[j].z - pointSel.z);

            if (int(laserCloudCornerLast->points[j].intensity) >
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }
          }
          for (int j = closestPointInd - 1; j >= 0; j--) {
            if (int(laserCloudCornerLast->points[j].intensity) <
                closestPointScan - 2.5) {
              continue; // TODO 应该为continue吗？ default: break
            }

            pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                             (laserCloudCornerLast->points[j].x - pointSel.x) +
                         (laserCloudCornerLast->points[j].y - pointSel.y) *
                             (laserCloudCornerLast->points[j].y - pointSel.y) +
                         (laserCloudCornerLast->points[j].z - pointSel.z) *
                             (laserCloudCornerLast->points[j].z - pointSel.z);

            if (int(laserCloudCornerLast->points[j].intensity) <
                closestPointScan) {
              if (pointSqDis < minPointSqDis2) {
                minPointSqDis2 = pointSqDis;
                minPointInd2 = j;
              }
            }
          }
        }

        pointSearchCornerInd1[i] = closestPointInd;
        pointSearchCornerInd2[i] = minPointInd2;
      }

      if (pointSearchCornerInd2[i] >= 0) {
        tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]]; // pj
        tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]]; // pl

        V3D P0xyz(pointSel.x, pointSel.y, pointSel.z);
        V3D P1xyz(tripod1.x, tripod1.y, tripod1.z);
        V3D P2xyz(tripod2.x, tripod2.y, tripod2.z);

        V3D P = math_utils::skew(P0xyz - P1xyz) * (P0xyz - P2xyz);
        float r = P.norm();
        float d12 = (P1xyz - P2xyz).norm();
        float res = r / d12; // i到(j,l)直线的距离
        V3D jacxyz =
            P.transpose() * math_utils::skew(P2xyz - P1xyz) / (d12 * r);	
        //点到直线的距离对pi的雅克比
        //w=u(x-c1) x v(x-c2),求w对x的导数，x，c1, c2为R^3, c1,c2为常量
        //https://math.stackexchange.com/questions/3686642/whats-the-differential-of-the-cross-product-wrt-a-vector
        //见我的提问https://github.com/ChaoqinRobotics/LINS---LiDAR-inertial-SLAM/issues/7

        float s = 1;
        if (iterCount >= ICP_FREQ) {
          s = 1 - 1.8 * fabs(res);
        }

        if (s > 0.1 && res != 0) {
          coeff.x = s * jacxyz(0);
          coeff.y = s * jacxyz(1);
          coeff.z = s * jacxyz(2);
          coeff.intensity = s * res;

          keypoints->push_back(newScan->cornerPointsSharp_->points[i]);
          jacobianCoff->push_back(coeff);
        }
      }
    }
  }

  // Undistort point cloud to the start frame
  void transformToStart(PointType const *const pi, PointType *const po) {
    double s = (1.f / SCAN_PERIOD) * (pi->intensity - int(pi->intensity));

    V3D P2xyz(pi->x, pi->y, pi->z);
    V3D phi = Quat2axis(linState_.qbn_);
    Q4D R21xyz = axis2Quat(s * phi);
    R21xyz.normalized();
    V3D T112xyz = s * linState_.rn_;
    V3D P1xyz = R21xyz * P2xyz + T112xyz;

    po->x = P1xyz.x();
    po->y = P1xyz.y();
    po->z = P1xyz.z();
    po->intensity = pi->intensity;
  }

  // Undistort point cloud to the end frame
  //把当前帧points转换到当前帧的end时刻
  void transformToEnd(PointType const *const pi, PointType *const po) {
    double s = (1.f / SCAN_PERIOD) * (pi->intensity - int(pi->intensity));

    V3D P2xyz(pi->x, pi->y, pi->z);
    V3D phi = Quat2axis(linState_.qbn_);
    Q4D R21xyz = axis2Quat(s * phi);
    R21xyz.normalized();
    V3D T112xyz = s * linState_.rn_;
    V3D P1xyz = R21xyz * P2xyz + T112xyz; //先转换到start

    R21xyz = linState_.qbn_;
    T112xyz = linState_.rn_;
    P2xyz = R21xyz.inverse() * (P1xyz - T112xyz); //再转换到end

    po->x = P2xyz.x();
    po->y = P2xyz.y();
    po->z = P2xyz.z();
    po->intensity = pi->intensity;
  }

  // Coordinate transformation from LiDAR frame to Vehicle frame
  void rotatePoint(PointType const *const pi, PointType *const po) {
    V3D rpy;
    rpy << deg2rad(0.0), deg2rad(0.0), deg2rad(IMU_LIDAR_EXTRINSIC_ANGLE); // 0.0
    M3D R = rpy2R(rpy);
    V3D Pi(pi->x, pi->y, pi->z);
    V3D Po = R * Pi;
    po->x = Po.x();
    po->y = Po.y();
    po->z = Po.z();
    po->intensity = pi->intensity;
  }

  void updatePointCloud() {
    scan_new_->cornerPointsLessSharpYZX_->clear();
    scan_new_->surfPointsLessFlatYZX_->clear();
    scan_new_->outlierPointCloudYZX_->clear();

    PointType point;
    for (int i = 0; i < scan_new_->cornerPointsLessSharp_->points.size(); i++) {
      transformToEnd(&scan_new_->cornerPointsLessSharp_->points[i],
                     &scan_new_->cornerPointsLessSharp_->points[i]);
      point.x = scan_new_->cornerPointsLessSharp_->points[i].y;
      point.y = scan_new_->cornerPointsLessSharp_->points[i].z;
      point.z = scan_new_->cornerPointsLessSharp_->points[i].x;
      point.intensity = scan_new_->cornerPointsLessSharp_->points[i].intensity;
      scan_new_->cornerPointsLessSharpYZX_->push_back(point);
    }
    for (int i = 0; i < scan_new_->surfPointsLessFlat_->points.size(); i++) {
      transformToEnd(&scan_new_->surfPointsLessFlat_->points[i],
                     &scan_new_->surfPointsLessFlat_->points[i]);
      point.x = scan_new_->surfPointsLessFlat_->points[i].y;
      point.y = scan_new_->surfPointsLessFlat_->points[i].z;
      point.z = scan_new_->surfPointsLessFlat_->points[i].x;
      point.intensity = scan_new_->surfPointsLessFlat_->points[i].intensity;
      scan_new_->surfPointsLessFlatYZX_->push_back(point);
    }
    for (int i = 0; i < scan_new_->outlierPointCloud_->points.size(); i++) {
       //transformToEnd(&scan_new_->outlierPointCloud_->points[i],
       //               &scan_new_->outlierPointCloud_->points[i]); //TODO 作者默认没有把outlier points转换到end时刻, 应该要转
      point.x = scan_new_->outlierPointCloud_->points[i].y;
      point.y = scan_new_->outlierPointCloud_->points[i].z;
      point.z = scan_new_->outlierPointCloud_->points[i].x;
      point.intensity = scan_new_->outlierPointCloud_->points[i].intensity;
      scan_new_->outlierPointCloudYZX_->push_back(point);
    }

    // Transform XYZ-convention to YZX-convention to meet the mapping module's
    // requirement
    globalStateYZX_.rn_ = Q_xyz_to_yzx * globalState_.rn_;
    globalStateYZX_.qbn_ =
        Q_xyz_to_yzx * globalState_.qbn_ * Q_xyz_to_yzx.inverse(); //YZX_t时刻在YZX_0坐标系下的旋转

    if (scan_new_->cornerPointsLessSharp_->points.size() >= 5 && //TODO与first, second scan 保持一致？ default: 5
        scan_new_->surfPointsLessFlat_->points.size() >= 20) { //default: 20
      kdtreeCorner_->setInputCloud(scan_new_->cornerPointsLessSharp_);
      kdtreeSurf_->setInputCloud(scan_new_->surfPointsLessFlat_);
    }
  }

  void estimateTransform(ScanPtr lastScan, ScanPtr newScan, V3D &t, Q4D &q) {
    double sum_dt = preintegration_->sum_dt;
    linState_.rn_ = t;
    linState_.qbn_ = q;
    for (int iter = 0; iter < NUM_ITER; iter++) {
      keypointSurfs_->clear();
      jacobianCoffSurfs->clear();
      keypointCorns_->clear();
      jacobianCoffCorns->clear();

      findCorrespondingSurfFeatures(lastScan, newScan, keypointSurfs_,
                                    jacobianCoffSurfs, iter);
      if (keypointSurfs_->points.size() < 10) { //TODO default: 10
        ROS_WARN("Insufficient matched surfs...");
        continue;
      }
      findCorrespondingCornerFeatures(lastScan, newScan, keypointCorns_,
                                      jacobianCoffCorns, iter);
      if (keypointCorns_->points.size() < 5) {  //TODO default: 5
        ROS_WARN("Insufficient matched corners...");
        continue;
      }

      if (calculateTransformation(lastScan, newScan, keypointCorns_,
                                  jacobianCoffCorns, keypointSurfs_,
                                  jacobianCoffSurfs, iter)) {
        ROS_INFO_STREAM("System Converges after " << iter << " iterations");
        break;
      }
    }

    t = linState_.rn_;
    q = linState_.qbn_; // qbn_ is quaternion rotation from b-frame to n-frame
  }


  bool calculateTransformation(ScanPtr lastScan, ScanPtr newScan,
                               pcl::PointCloud<PointType>::Ptr corners, // k+1帧下点(未转换到k+1帧start下)
                               pcl::PointCloud<PointType>::Ptr jacoCornersCoff, //每个点存的是距离对pi的雅克比, 点到直线的距离
                               pcl::PointCloud<PointType>::Ptr surfs, // k+1帧下点(未转换到k+1帧start下)
                               pcl::PointCloud<PointType>::Ptr jacoSurfsCoff, //每个点存的是平面的单位法向量，点到平面的距离
                               int iterCount) {
    keypoints_->clear();
    jacobians_->clear();
    (*keypoints_) += (*surfs);
    (*keypoints_) += (*corners);
    (*jacobians_) += (*jacoSurfsCoff);
    (*jacobians_) += (*jacoCornersCoff);

    const int stateNum = 6;
    const int pointNum = keypoints_->points.size();
    const int imuNum = 0;
    const int row = pointNum + imuNum;
    Eigen::Matrix<double, Eigen::Dynamic, stateNum> J(row, stateNum);
    Eigen::Matrix<double, stateNum, Eigen::Dynamic> JT(stateNum, row);
    Eigen::Matrix<double, stateNum, stateNum> JTJ;
    Eigen::VectorXd b(row);
    Eigen::Matrix<double, stateNum, 1> JTb;
    Eigen::Matrix<double, stateNum, 1> x;
    J.setZero();
    JT.setZero();
    JTJ.setZero();
    b.setZero();
    JTb.setZero();
    x.setZero();

    for (int i = 0; i < pointNum; ++i) {
      // Select keypoint i
      const PointType &keypoint = keypoints_->points[i];
      const PointType &coeff = jacobians_->points[i];

      V3D P2xyz(keypoint.x, keypoint.y, keypoint.z);
      V3D coff_xyz(coeff.x, coeff.y, coeff.z);

      double s =
          (1.f / SCAN_PERIOD) * (keypoint.intensity - int(keypoint.intensity));

      V3D phi = Quat2axis(linState_.qbn_);
      // Rotation matrix from frame2 (new) to frame1 (last)
      Q4D R21xyz = axis2Quat(s * phi);
      R21xyz.normalized();
      // Translation vector from frame1 to frame2 represented in frame1
      V3D T112xyz = s * linState_.rn_;

      V3D jacobian1xyz =
          coff_xyz.transpose() *
          (-R21xyz.toRotationMatrix() * skew(P2xyz)); // rotation jacobian
      //点到直线的距离或点到平面的距离对R的雅克比 = 距离对p0的雅克比 * p0对R的雅克比
      //其中p0对R的雅克比： 整个程序的R是Hamilton惯例，按照其右扰动对应局部扰动,
      //按照右扰动公式
      // = -Rp^

      V3D jacobian2xyz =
          coff_xyz.transpose() * M3D::Identity();  // translation jacobian
  
		  
		  // translation jacobian //TODO 好像少了一个s？

      double residual = coeff.intensity;

      J.block<1, 3>(i, O_R) = jacobian1xyz;
      J.block<1, 3>(i, O_P) = jacobian2xyz;

      // Set the overall residual
      b(i) = -0.05 * residual;
    }

    // Solve x
    JT = J.transpose();
    JTJ = JT * J;
    JTb = JT * b;
    x = JTJ.colPivHouseholderQr().solve(JTb); // Ax=b
    // x(0), x(1), x(2): 绕着x轴roll， y轴pitch， z轴yaw
    // x(3), x(4), x(5): tx, ty,tz

    // Determine whether x is degenerated
    bool isDegenerate = false;
    Eigen::Matrix<double, stateNum, stateNum> matP;
    if (iterCount == 0) {//default
    // if(1){//TODO 我们改为每次都检查
      Eigen::Matrix<double, 1, stateNum> matE;
      Eigen::Matrix<double, stateNum, stateNum> matV;
      Eigen::Matrix<double, stateNum, stateNum> matV2;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, stateNum, stateNum>>
          esolver(JTJ);
      matE = esolver.eigenvalues().real(); // Eigen中的特征值：The eigenvalues
                                           // are sorted in increasing order.
      matV = esolver.eigenvectors().real();

      matV2 = matV;

      isDegenerate = false;
      std::vector<double> eignThre(stateNum, 10.);
      for (int i = 0; i < stateNum; i++) {
        // if eigenvalue is less than 10, set the corresponding eigenvector to 0
        // vector
        if (matE(0, i) < eignThre[i]) {
          for (int j = 0; j < stateNum; j++) {
            matV2(i, j) = 0;
          }
          isDegenerate = true;
        } else {
          break;
        }
      }
      matP = matV.inverse() * matV2;
    }

    if (isDegenerate) {
      cout << "System is Degenerate." << endl;
      Eigen::Matrix<double, stateNum, 1> matX2(x);
      x = matP * matX2;
    }

    // Update state linState_
    Q4D dq = rpy2Quat(x.segment<3>(O_R));     
    //TODO: 作者认为求出来的是rpy,由rpy得到四元数
    //liom认为是轴角，由轴角转换得到四元数，见Estimator::CalculateLaserOdom
    //我认为应该是轴角，因为我们在构建normal equation时传进去关于旋转的雅克比不是对于rpy的，而是关于右扰动或者李代数求导的。

    linState_.qbn_ = (linState_.qbn_ * dq).normalized();
    linState_.rn_ += x.segment<3>(O_P);

    // Determine whether should it stop
    V3D rpy_rad = x.segment<3>(O_R);
    V3D rpy_deg = math_utils::rad2deg(rpy_rad);
    double deltaR = rpy_deg.norm();
    V3D trans = 100 * x.segment<3>(O_P);
    double deltaT = trans.norm();
    if (deltaR < 0.1 && deltaT < 0.1) {
      return true;
    }

    return false;
  }

  //not used
  void estimateInitialState1(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    ba = INIT_BA;
    bw = INIT_BW;

    solveGyroscopeBias(q, bw);

    double sum_dt = preintegration_->sum_dt;
    v0 =
        (p - 0.5 * linState_.gn_ * sum_dt * sum_dt - preintegration_->delta_p) /
        sum_dt;
    v1 = v0 + sum_dt * linState_.gn_ + preintegration_->delta_v;

    cout << "v0: " << v0.transpose() << endl;
    cout << "v1: " << v1.transpose() << endl;
    cout << "ba0: " << INIT_BA.transpose() << endl;
    cout << "bw0: " << INIT_BW.transpose() << endl;
    cout << "bw0: " << bw.transpose() << endl;
  }

  //not used 
  void estimateInitialState2(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    const int DIM_OF_STATE = 1 + 1 + 3;
    const int DIM_OF_MEAS = 3 + 3;
    Eigen::Matrix<double, Eigen::Dynamic, DIM_OF_STATE> J(DIM_OF_MEAS,
                                                          DIM_OF_STATE);
    Eigen::Matrix<double, DIM_OF_STATE, Eigen::Dynamic> JT(DIM_OF_STATE,
                                                           DIM_OF_MEAS);
    Eigen::Matrix<double, DIM_OF_STATE, DIM_OF_STATE> JTJ;
    Eigen::VectorXd b(DIM_OF_MEAS);
    Eigen::Matrix<double, DIM_OF_STATE, 1> JTb;
    Eigen::Matrix<double, DIM_OF_STATE, 1> x;

    J.setZero();
    JT.setZero();
    JTJ.setZero();
    b.setZero();
    JTb.setZero();
    x.setZero();

    double sum_dt = preintegration_->sum_dt;
    b.block<3, 1>(0, 0) =
        (p - preintegration_->delta_p) / sum_dt - 0.5 * sum_dt * linState_.gn_;
    b.block<3, 1>(3, 0) = sum_dt * linState_.gn_ + preintegration_->delta_v;

    V3D L(1, 0, 0);
    J.block<3, 1>(0, 0) = L;
    J.block<3, 3>(0, 2) = -0.5 * sum_dt * M3D::Identity();
    J.block<3, 1>(3, 0) = -L;
    J.block<3, 1>(3, 1) = L;
    J.block<3, 3>(3, 2) = sum_dt * M3D::Identity();

    JT = J.transpose();
    JTJ = JT * J;
    JTb = JT * b;

    x = JTJ.colPivHouseholderQr().solve(JTb);
    v0 = x(0) * L;
    v1 = x(1) * L;
    ba = INIT_BA;
    bw = INIT_BW;

    V3D test_ba = linState_.gn_ + preintegration_->delta_v / sum_dt;
    cout << "test_ba: " << test_ba.transpose() << endl;
  }

  //not used
  void estimateInitialState3(const V3D& p, const Q4D& q, V3D& v0, V3D& v1,
                             V3D& ba, V3D& bw) {
    double sum_dt = preintegration_->sum_dt;
    V3D v = p / sum_dt;
    // v * sum_dt = (p - 0.5*linState_.gn_*sum_dt*sum_dt -
    // preintegration_->delta_p + 0.5*ba*sum_dt*sum_dt);
    ba = (v * sum_dt - p + 0.5 * linState_.gn_ * sum_dt * sum_dt +
          preintegration_->delta_p) *
         2 * (1.0 / sum_dt * sum_dt);

    solveGyroscopeBias(q, bw);

    v0 = v;
    v1 = v;
    cout << "v0: " << v0.transpose() << endl;
    cout << "v1: " << v1.transpose() << endl;
    cout << "ba0: " << ba.transpose() << endl;
    cout << "bw0: " << bw.transpose() << endl;
  }

  void estimateInitialState(const V3D &p, const Q4D &q, V3D &v0, V3D &v1,
                            V3D &ba, V3D &bw) {
    double sum_dt = preintegration_->sum_dt;
    // Calculate a rough velocity using relative translation
    V3D v = p / sum_dt;
    v0 = v;
    v1 = v;
    // TODO(charles): calibrate initial ba and bw using two consecutive scans
    // and IMU preintegration results
    ba = INIT_BA;
    bw = INIT_BW;
  }

  // not used
  // Estimate gyroscope bias using a similar methoed provided in VINS-Mono
  void solveGyroscopeBias(const Q4D& q, V3D& bw) {
    Matrix3d A;
    V3D b;
    V3D delta_bg;
    A.setZero();
    b.setZero();

    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q_ij = q;
    tmp_A = preintegration_->jacobian.template block<3, 3>(GlobalState::att_,
                                                           GlobalState::gyr_);
                                                           //(6,12)
    tmp_b = 2 * (preintegration_->delta_q.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;

    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration "
                    << delta_bg.transpose());

    bw += delta_bg;
  }

public:
  FusionStatus status_;    // system status
  StatePredictor *filter_; // Kalman filter pointer
  ScanPtr scan_new_;       // current scan information
  ScanPtr scan_last_;      // last scan information

  // !@KD tree relatives
  pcl::VoxelGrid<PointType> downSizeFilter_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCorner_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurf_;

  // !@Feature matching relatives
  std::vector<int> pointSelCornerInd;
  std::vector<double> pointSearchCornerInd1;
  std::vector<double> pointSearchCornerInd2;
  std::vector<int> pointSelSurfInd;
  std::vector<double> pointSearchSurfInd1;
  std::vector<double> pointSearchSurfInd2;
  std::vector<double> pointSearchSurfInd3;

  // !@Jacobians and keypoints
  pcl::PointCloud<PointType>::Ptr keypoints_;
  pcl::PointCloud<PointType>::Ptr jacobians_;
  pcl::PointCloud<PointType>::Ptr keypointCorns_;
  pcl::PointCloud<PointType>::Ptr keypointSurfs_;
  pcl::PointCloud<PointType>::Ptr jacobianCoffCorns;
  pcl::PointCloud<PointType>::Ptr jacobianCoffSurfs;

  // !@Global transformation from the original scan-frame to current scan-frame
  GlobalState globalState_;

  // !@Relative transformation from scan0-frame t0 scan1-frame
  GlobalState linState_; //作者的意思是：从ｋ帧到ｋ+1帧的变换。但linState.gn_ = [0 0 -g]

  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> difVecLinInv_;
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, 1> updateVec_;
  double updateVecNorm_ = 0.0;

  // !@Kalman filter relatives
  VXD residual_;
  MXD Fk_; //没有使用
  MXD Gk_; //没有使用
  MXD Pk_;
  MXD Qk_; //没有使用
  MXD Rk_;
  MXD Hk_;
  MXD Jk_; //没有使用
  MXD Kk_;
  MXD IKH_;
  MXD Py_;
  MXD Pyinv_;

  // !@ IMU preintegration
  integration::IntegrationBase *preintegration_;
  Imu imu_last_; //只在processFirstScan() 和 processSecondScan()中使用

  // !@Rotation matrices between XYZ-convention and YZX-convention
  Eigen::Matrix3d R_yzx_to_xyz; //把zhangji下的点(Z-front,X-left,Y-up)转换到ros坐标系下(X-front,Y-left,Z-up)的旋转矩阵
  Eigen::Matrix3d R_xyz_to_yzx;
  Eigen::Quaterniond Q_yzx_to_xyz; //把zhangji下的点(Z-front,X-left,Y-up)转换到ros坐标系下(X-front,Y-left,Z-up)的四元数
  Eigen::Quaterniond Q_xyz_to_yzx;
  GlobalState globalStateYZX_;

  //jxl 
  // ros::Publisher pub_correctRollPitch_;
  // ros::NodeHandle nh_;
};

} // namespace fusion

#endif // INCLUDE_STATEESTIMATOR_HPP_
