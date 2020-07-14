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

#ifndef INCLUDE_KALMANFILTER_HPP_
#define INCLUDE_KALMANFILTER_HPP_

#include <math_utils.h>
#include <parameters.h>

#include <iostream>
#include <map>

using namespace std;
using namespace math_utils;
using namespace parameter;

namespace filter {

// GlobalState Class contains state variables including position, velocity,
// attitude, acceleration bias, gyroscope bias, and gravity
class GlobalState {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr unsigned int DIM_OF_STATE_ = 18;
  static constexpr unsigned int DIM_OF_NOISE_ = 12;
  static constexpr unsigned int pos_ = 0;
  static constexpr unsigned int vel_ = 3;
  static constexpr unsigned int att_ = 6;
  static constexpr unsigned int acc_ = 9;
  static constexpr unsigned int gyr_ = 12;
  static constexpr unsigned int gra_ = 15;

  GlobalState() { setIdentity(); }

  GlobalState(const V3D& rn, const V3D& vn, const Q4D& qbn, const V3D& ba,
              const V3D& bw) {
    setIdentity();
    rn_ = rn;
    vn_ = vn;
    qbn_ = qbn;
    ba_ = ba;
    bw_ = bw;
  }

  ~GlobalState() {}

  void setIdentity() {
    rn_.setZero();
    vn_.setZero();
    qbn_.setIdentity();
    ba_.setZero();
    bw_.setZero();
    gn_ << 0.0, 0.0, -G0; //作者使用的是：ENU imu
  }

  // boxPlus operator
  void boxPlus(const Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk,
               GlobalState& stateOut) {
    stateOut.rn_ = rn_ + xk.template segment<3>(pos_);
    stateOut.vn_ = vn_ + xk.template segment<3>(vel_);
    stateOut.ba_ = ba_ + xk.template segment<3>(acc_);
    stateOut.bw_ = bw_ + xk.template segment<3>(gyr_);
    Q4D dq = axis2Quat(xk.template segment<3>(att_));
    stateOut.qbn_ = (qbn_ * dq).normalized();

    stateOut.gn_ = gn_ + xk.template segment<3>(gra_);
  }

  // boxMinus operator
  void boxMinus(const GlobalState& stateIn,
                Eigen::Matrix<double, DIM_OF_STATE_, 1>& xk) {
    xk.template segment<3>(pos_) = rn_ - stateIn.rn_;
    xk.template segment<3>(vel_) = vn_ - stateIn.vn_;
    xk.template segment<3>(acc_) = ba_ - stateIn.ba_;
    xk.template segment<3>(gyr_) = bw_ - stateIn.bw_;
    V3D da = Quat2axis(stateIn.qbn_.inverse() * qbn_);
    xk.template segment<3>(att_) = da;

    xk.template segment<3>(gra_) = gn_ - stateIn.gn_;
  }

  GlobalState& operator=(const GlobalState& other) {
    if (this == &other) return *this;

    this->rn_ = other.rn_;
    this->vn_ = other.vn_;
    this->qbn_ = other.qbn_;
    this->ba_ = other.ba_;
    this->bw_ = other.bw_;
    this->gn_ = other.gn_;

    return *this;
  }

  // !@State
  V3D rn_;   // position in n-frame
  V3D vn_;   // velocity in n-frame
  Q4D qbn_;  // rotation from b-frame to n-frame, 单位四元数
  V3D ba_;   // acceleartion bias
  V3D bw_;   // gyroscope bias
  V3D gn_;   // gravity
};



class StatePredictor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StatePredictor() { reset(); }

  ~StatePredictor() {}

  bool predict(double dt, const V3D& acc, const V3D& gyr,
               bool update_jacobian_ = true) {
    if (!isInitialized()) return false;

    if (!flag_init_imu_) {
      flag_init_imu_ = true;
      acc_last = acc;
      gyr_last = gyr;
    }

    // Average acceleration and angular rate
    GlobalState state_tmp = state_; //本次用来预测状态的第一帧imu，因为上一次update之后调用了reset，所以此时的state是上次reset()之后的状态；用本次第一帧之后的imu再预测时，状态的初值就是上次预测的结果
    V3D un_acc_0 = state_tmp.qbn_ * (acc_last - state_tmp.ba_) + state_tmp.gn_; //0指的是：第k帧laser下加速度，见预积分paper 32式
    V3D un_gyr = 0.5 * (gyr_last + gyr) - state_tmp.bw_;
    Q4D dq = axis2Quat(un_gyr * dt);
    state_tmp.qbn_ = (state_tmp.qbn_ * dq).normalized(); //R^bk_t: 从bk到当前t时刻的旋转
    V3D un_acc_1 = state_tmp.qbn_ * (acc - state_tmp.ba_) + state_tmp.gn_;
    V3D un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // State integral
    state_tmp.rn_ = state_tmp.rn_ + dt * state_tmp.vn_ + 0.5 * dt * dt * un_acc; //t^bk_t: 从bk到当前t时刻的平移
    state_tmp.vn_ = state_tmp.vn_ + dt * un_acc; //当前时刻在bk下的速度

    //imu积分，把状态PVQ往前传播一个imu时间间隔dt. 注意不是预积分，预积分获得是两个时刻的delta约束。
    //同时计算误差状态转移矩阵, 和误差状态的方差。误差状态的方差也是预测出来的PVQ状态的方差。

    if (update_jacobian_) {//true
      MXD Ft =                    //lins 9式， joan sola 237式
          MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_); //18*18
      Ft.block<3, 3>(GlobalState::pos_, GlobalState::vel_) = M3D::Identity();

      Ft.block<3, 3>(GlobalState::vel_, GlobalState::att_) =  
          -state_tmp.qbn_.toRotationMatrix() * skew(acc - state_tmp.ba_);
      Ft.block<3, 3>(GlobalState::vel_, GlobalState::acc_) =  
          -state_tmp.qbn_.toRotationMatrix();
      Ft.block<3, 3>(GlobalState::vel_, GlobalState::gra_) = M3D::Identity();

      Ft.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          - skew(gyr - state_tmp.bw_);
      Ft.block<3, 3>(GlobalState::att_, GlobalState::gyr_) = -M3D::Identity();

      MXD Gt = 
          MXD::Zero(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_NOISE_); //18*12
      Gt.block<3, 3>(GlobalState::vel_, 0) = -state_tmp.qbn_.toRotationMatrix();
      Gt.block<3, 3>(GlobalState::att_, 3) = -M3D::Identity();
      Gt.block<3, 3>(GlobalState::acc_, 6) = M3D::Identity();
      Gt.block<3, 3>(GlobalState::gyr_, 9) = M3D::Identity();
      Gt = Gt * dt;

      const MXD I =
          MXD::Identity(GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_);
      F_ = I + Ft * dt + 0.5 * Ft * Ft * dt * dt; //lins 12式, error state propagation matrix 二阶近似

      // jacobian_ = F * jacobian_;
      covariance_ =
          F_ * covariance_ * F_.transpose() + Gt * noise_ * Gt.transpose(); 
      covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval(); //math_utils::enforceSymmetry(),强制为对称矩阵
    }

    state_ = state_tmp;
    time_ += dt;
    acc_last = acc;
    gyr_last = gyr;
    return true;
  }

  static void calculateRPfromIMU(const V3D& acc, double& roll, double& pitch) {
    pitch = -sign(acc.z()) * asin(acc.x() / G0);
    roll = sign(acc.z()) * asin(acc.y() / G0);
  }

  void set(const GlobalState& state) { state_ = state; }


  void update(const GlobalState& state,
              const Eigen::Matrix<double, GlobalState::DIM_OF_STATE_,
                                  GlobalState::DIM_OF_STATE_>& covariance) {
    state_ = state;
    covariance_ = covariance;
  }

  //没有使用
  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }

  //没有使用
  void initialization(double time, const V3D& rn, const V3D& vn, const Q4D& qbn,
                      const V3D& ba, const V3D& bw, const V3D& acc,
                      const V3D& gyr) {
    state_ = GlobalState(rn, vn, qbn, ba, bw);
    time_ = time;
    acc_last = acc;
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance();
  }

  //没有使用
  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, double roll = 0.0, double pitch = 0.0,
                      double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time;
    flag_init_state_ = true;

    initializeCovariance();
  }
  
  
  void initialization(double time, const V3D& rn, const V3D& vn, const V3D& ba,
                      const V3D& bw, const V3D& acc, const V3D& gyr,
                      double roll = 0.0, double pitch = 0.0, double yaw = 0.0) {
    state_ = GlobalState(rn, vn, rpy2Quat(V3D(roll, pitch, yaw)), ba, bw);
    time_ = time; //first/second scan time
    acc_last = acc; 
    gyr_last = gyr;
    flag_init_imu_ = true;
    flag_init_state_ = true;

    initializeCovariance(); 
  }

  //从配置文件中设置初始状态cov(18*18)，初始noise cov(12*12)
  void initializeCovariance(int type = 0) {
    double covX = pow(INIT_POS_STD(0), 2);
    double covY = pow(INIT_POS_STD(1), 2);
    double covZ = pow(INIT_POS_STD(2), 2);
    double covVx = pow(INIT_VEL_STD(0), 2);
    double covVy = pow(INIT_VEL_STD(1), 2);
    double covVz = pow(INIT_VEL_STD(2), 2);
    double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
    double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
    double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

    V3D covPos = INIT_POS_STD.array().square();
    V3D covVel = INIT_VEL_STD.array().square();
    V3D covAcc = INIT_ACC_STD.array().square();
    V3D covGyr = INIT_GYR_STD.array().square();

    double peba = pow(ACC_N * ug, 2);
    double pebg = pow(GYR_N * dph, 2);
    double pweba = pow(ACC_W * ugpsHz, 2);
    double pwebg = pow(GYR_W * dpsh, 2);
    V3D gra_cov(0.01, 0.01, 0.01);

    if (type == 0) {
      // Initialize using offline parameters
      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          covVel.asDiagonal();  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) =
          covAcc.asDiagonal();  // ba
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) =
          covGyr.asDiagonal();  // bg
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) =
          gra_cov.asDiagonal();  // gravity
    } else if (type == 1) { //不执行
      // Inheritage previous covariance
      M3D vel_cov =
          covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_);
      M3D acc_cov =
          covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_);
      M3D gyr_cov =
          covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_);
      M3D gra_cov =
          covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_);

      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos
      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          vel_cov;  // vel
      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att
      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = acc_cov;
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = gyr_cov;
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) = gra_cov;
    }

    noise_.setZero();
    noise_.block<3, 3>(0, 0) = V3D(peba, peba, peba).asDiagonal(); //加速度计噪声 na, 见lins论文
    noise_.block<3, 3>(3, 3) = V3D(pebg, pebg, pebg).asDiagonal(); //陀螺仪噪声 ng
    noise_.block<3, 3>(6, 6) = V3D(pweba, pweba, pweba).asDiagonal(); //加速度计bias噪声 nba
    noise_.block<3, 3>(9, 9) = V3D(pwebg, pwebg, pwebg).asDiagonal(); //陀螺仪bias噪声 nbg
  }

  void reset(int type = 0) {
    if (type == 0) {
      state_.rn_.setZero();
      
      state_.vn_ = state_.qbn_.inverse() * state_.vn_; //default
      // state_.vn_.setZero(); //为何不是这样的？

      state_.qbn_.setIdentity();

      initializeCovariance();
    } else if (type == 1) {//每次在IESKF之后调用
      V3D covPos = INIT_POS_STD.array().square();
      double covRoll = pow(deg2rad(INIT_ATT_STD(0)), 2);
      double covPitch = pow(deg2rad(INIT_ATT_STD(1)), 2);
      double covYaw = pow(deg2rad(INIT_ATT_STD(2)), 2);

      M3D vel_cov =
          covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_);
      M3D acc_cov =
          covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_);
      M3D gyr_cov =
          covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_);
      M3D gra_cov =
          covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_);

      covariance_.setZero();
      covariance_.block<3, 3>(GlobalState::pos_, GlobalState::pos_) =
          covPos.asDiagonal();  // pos

      covariance_.block<3, 3>(GlobalState::vel_, GlobalState::vel_) =
          state_.qbn_.inverse() * vel_cov * state_.qbn_;  // vel  //TODO 为何再乘以state_.qbn_ 

      covariance_.block<3, 3>(GlobalState::att_, GlobalState::att_) =
          V3D(covRoll, covPitch, covYaw).asDiagonal();  // att

      covariance_.block<3, 3>(GlobalState::acc_, GlobalState::acc_) = acc_cov;
      covariance_.block<3, 3>(GlobalState::gyr_, GlobalState::gyr_) = gyr_cov;
      covariance_.block<3, 3>(GlobalState::gra_, GlobalState::gra_) =
          state_.qbn_.inverse() * gra_cov * state_.qbn_; //TODO 同上
      

      //状态的QT分量被置为I，0，其余分量保持上一次update之后的值
      state_.rn_.setZero();
      state_.vn_ = state_.qbn_.inverse() * state_.vn_;
      state_.qbn_.setIdentity();

      state_.gn_ = state_.qbn_.inverse() * state_.gn_; //state_.qbn_已经为I阵; state_.gn_: update结束后的值
      // std::cout<<"reset(): before reset, state_.gn_ normal= "<< state_.gn_.norm()<<"\n\n";
      state_.gn_ = state_.gn_ * 9.81 / state_.gn_.norm(); //还是全局下的gravity, 几乎没有变化
      // initializeCovariance(1);
    }
  }

  //没有使用
  void reset(V3D vn, V3D ba, V3D bw) {
    state_.setIdentity();
    state_.vn_ = vn;
    state_.ba_ = ba;
    state_.bw_ = bw;
    initializeCovariance();
  }

  inline bool isInitialized() { return flag_init_state_; }

  GlobalState state_; //滤波器的状态：k时刻laser到k+1时刻laser的delta_(p, v, q, ba, bg), gravity_in_G
  double time_;

  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_> F_; //jxl: error state propagation matrix
      
  Eigen::Matrix<double, GlobalState::DIM_OF_STATE_, GlobalState::DIM_OF_STATE_> jacobian_, covariance_; //jacobian_没有使用; 状态的方差

  Eigen::Matrix<double, GlobalState::DIM_OF_NOISE_, GlobalState::DIM_OF_NOISE_> noise_; //噪声的协方差矩阵

  V3D acc_last;  // last acceleration measurement
  V3D gyr_last;  // last gyroscope measurement

  bool flag_init_state_= false; //作者没有初始化,我们补上 
  bool flag_init_imu_ = false;
};

};  // namespace filter

#endif  // INCLUDE_KALMANFILTER_HPP_
