// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    tracker.cpp
 * @author  Jared Strader
 */
//-----------------------------------------------------------------------------

#include <target_tracking/tracker.hpp>

namespace sdrone
{

double returnNearestAngle(double x, double z) {
    double diff = z - x;
    diff += (diff > M_PI) ? -2*M_PI : (diff < -M_PI) ? 2*M_PI : 0;
    return x + diff;
}

Tracker::
Tracker()
{
  is_initialized_ = false;
  process_covariance_trans_ = 0;
  process_covariance_rot_ = 0;
};

Tracker::
Tracker(const Belief7D &b_target_meas)
{
  init(b_target_meas);
};

Tracker::
Tracker(const Belief6D &b_target_meas)
{
  init(b_target_meas);
};

void Tracker::
init(const Belief7D &b_target_meas)
{
  std::cout << "BAD ENTRY" << std::endl;
  exit(1);
  b_target_ = b_target_meas;
  is_initialized_ = true;
};

void Tracker::
init(const Belief6D &b_target_meas)
{
  b_target_6D_ = b_target_meas;
  // b_target_6D_.mean.m_coords(0) = b_target_6D_.mean.m_coords(0)/1000.;
  // b_target_6D_.mean.m_coords(1) = b_target_6D_.mean.m_coords(1)/1000.;
  // b_target_6D_.mean.m_coords(2) = b_target_6D_.mean.m_coords(2)/1000.;
  is_initialized_ = true;
};

void Tracker::
update(const Belief7D &b_target_meas)
{
  std::cout << "BAD ENTRY" << std::endl;
  exit(1);
  if(!is_initialized_)
  {
    init(b_target_meas);
  }
  else
  {
    Mat71 x, z;
    copyMeanToMat71(b_target_meas, z);
    copyMeanToMat71(b_target_, x);

    Mat77 R, P;
    copyCovarianceToMat77(b_target_meas, R);
    copyCovarianceToMat77(b_target_, P);

    // R = 0.1*Eigen::MatrixXd::Identity(7,7);
    Mat77 S = P + R;
    Mat77 K = P*S.inverse();

    Mat71 y = z - x;
    x = x + K*y;

    double d = std::sqrt(x(3)*x(3) + x(4)*x(4) + x(5)*x(5) + x(6)*x(6));
    x(3) = x(3)/d;
    x(4) = x(4)/d;
    x(5) = x(5)/d;
    x(6) = x(6)/d;

    Mat77 Pupdate = P - K*P;
    setBelief(x, Pupdate);
  }
};

void Tracker::
update(const Belief6D &b_target_meas)
{
  if(!is_initialized_)
  {
    init(b_target_meas);
  }
  else
  {
    Mat61 x, z;
    copyMeanToMat61(b_target_meas, z);
    copyMeanToMat61(b_target_6D_, x);
    // z(0) = z(0)/1000.;
    // z(1) = z(1)/1000.;
    // z(2) = z(2)/1000.;

    z(3) = returnNearestAngle(x(3), z(3));
    z(4) = returnNearestAngle(x(4), z(4));
    z(5) = returnNearestAngle(x(5), z(5));

    Mat66 R, P;
    copyCovarianceToMat66(b_target_meas, R);
    copyCovarianceToMat66(b_target_6D_, P);

    Mat66 W = Eigen::MatrixXd::Identity(6,6);
    W(0,0) = process_covariance_trans_;
    W(1,1) = process_covariance_trans_;
    W(2,2) = process_covariance_trans_;
    W(3,3) = process_covariance_rot_;
    W(4,4) = process_covariance_rot_;
    W(5,5) = process_covariance_rot_;
    P += W; // This is like the prediction step for 0-mean prior on plant update
    Mat66 S = P + R;
    Mat66 K = P*S.inverse();

    Mat61 y = z - x;
    x = x + K*y;

    Mat66 Pupdate = P - K*P;
    setBelief(x, Pupdate);
  }
};

void Tracker::
copyMeanToMat71(const Belief7D &b, Mat71 &mu)
{
    mu(0) = b.mean.m_coords(0);
    mu(1) = b.mean.m_coords(1);
    mu(2) = b.mean.m_coords(2);
    mu(3) = b.mean.m_quat.r();
    mu(4) = b.mean.m_quat.x();
    mu(5) = b.mean.m_quat.y();
    mu(6) = b.mean.m_quat.z();
};

void Tracker::
copyMeanToMat61(const Belief6D &b, Mat61 &mu)
{
    mu(0) = b.mean.m_coords(0);
    mu(1) = b.mean.m_coords(1);
    mu(2) = b.mean.m_coords(2);
    mu(3) = b.mean.roll();
    mu(4) = b.mean.pitch();
    mu(5) = b.mean.yaw();
};

void Tracker::
copyCovarianceToMat77(const Belief7D &b, Mat77 &cov)
{
  cov = b.cov.asEigen();
};

void Tracker::
copyCovarianceToMat66(const Belief6D &b, Mat66 &cov)
{
  cov = b.cov.asEigen();

  //mrpt defines 6D as x, y, z, yaw, pitch, roll
  //swap to define as x, y, z, roll, pitch, yaw
  const int map[3] = {2,1,0};
  Mat66 temp = b.cov.asEigen();
  for(int i=0; i<3; i++)
  {
    for(int j=0; j<3; j++)
    {
      cov(i,j) = temp(3 + map[i], 3 + map[j]);
    }
  }
};

void Tracker::
setBelief(const Mat71 &mu,
          const Mat77 &cov)
{
  
  Mat71 mu_copy;
  mu_copy=mu;

  Pose7D p(mu(0), mu(1), mu(2), Quat(mu_copy(3), mu_copy(4), mu_copy(5), mu_copy(6)));
  b_target_ = Belief7D(p, CMat77(cov));
}

void Tracker::
setBelief(const Mat61 &mu,
          const Mat66 &cov)
{
  
  Mat61 mu_copy;
  mu_copy=mu;

  Pose6D p(mu(0), mu(1), mu(2), mu(5), mu(4), mu(3));
  b_target_6D_ = Belief6D(p, CMat66(cov));
}

}; //namespace sdrone
