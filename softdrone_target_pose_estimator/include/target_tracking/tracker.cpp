// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    tracker.cpp
 * @author  Jared Strader
 */
//-----------------------------------------------------------------------------

#include <target_tracking/tracker.hpp>

namespace soft
{

Tracker::
Tracker()
{
  is_initialized_ = false;
};

Tracker::
Tracker(const Belief7D &b_target_meas)
{
  init(b_target_meas);
};

void Tracker::
init(const Belief7D &b_target_meas)
{
  b_target_ = b_target_meas;
  is_initialized_ = true;
};

void Tracker::
update(const Belief7D &b_target_meas)
{
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

    R = 0.1*Eigen::MatrixXd::Identity(7,7); //TODO: make parameter
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
copyCovarianceToMat77(const Belief7D &b, Mat77 &cov)
{
  cov = b.cov.asEigen();
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

}; //namespace soft