#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>

class ConstantVelocityCustomFactor: public gtsam::NoiseModelFactor4<gtsam::NavState, gtsam::Vector3, gtsam::NavState, gtsam::Vector3> {
    double dt_;

public:
  ConstantVelocityCustomFactor(gtsam::Key i, gtsam::Key j, gtsam::Key k, gtsam::Key l, double dt, const gtsam::SharedNoiseModel& model):
    gtsam::NoiseModelFactor4<gtsam::NavState, gtsam::Vector3, gtsam::NavState, gtsam::Vector3>(model, i, j, k, l), dt_(dt) {}

    gtsam::Vector evaluateError(const gtsam::NavState& ns1, const gtsam::Vector3& omegas1, const gtsam::NavState& ns2, const gtsam::Vector3& omegas2,
                       boost::optional<gtsam::Matrix&> H1 = boost::none,
                       boost::optional<gtsam::Matrix&> H2 = boost::none,
                       boost::optional<gtsam::Matrix&> H3 = boost::none,
                       boost::optional<gtsam::Matrix&> H4 = boost::none) const {

        static const gtsam::Vector3 b_accel{0.0, 0.0, 0.0}; // TODO: even for "constant velocity", this is not 0, it should be a vector perpendicular to the target's motion with magnitude depending on omega (curature)

        gtsam::Matrix99 predicted_H_x1;
        gtsam::Matrix93 predicted_H_x2;
        gtsam::NavState predicted = ns1.update(b_accel, omegas1, dt_, H1 ? &predicted_H_x1 : nullptr, {}, H2 ? &predicted_H_x2 : nullptr);

        gtsam::Matrix99 error_H_predicted;
        gtsam::Matrix h3(9, 3);
        gtsam::Vector9 error_nav = predicted.localCoordinates(ns2, H1 ? &error_H_predicted : nullptr, &h3);

        if (H1) { // 12 x 9, last 3 rows are 0
            //gtsam::Matrix h1(12,3);
            //h1 << error_H_predicted * predicted_H_x1, 0, 0, 0,
            //                                          0, 0, 0,
            //                                          0, 0, 0;
            //*H1 = h1;
            //*H1 = error_H_predicted * predicted_H_x1;
            *H1 = (gtsam::Matrix(12,9) << error_H_predicted * predicted_H_x1, Eigen::MatrixXd::Zero(3,9)).finished();
        }

        if (H3) {
            //gtsam::Matrix h3(12, 3);
            //h3 << *H3, 0, 0, 0,
            //           0, 0, 0,
            //           0, 0, 0;
            //*H3 = h3;
            *H3 = (gtsam::Matrix(12,9) << h3, Eigen::MatrixXd::Zero(3,9)).finished();
        }

        gtsam::Vector3 error_omegas = omegas2 - omegas1;

        if (H2) { // 12 x 3, last three rows are -identity
            //gtsam::Matrix93 omega1_to_ns1 = error_H_predicted * predicted_H_x2;
            //gtsam::Matrix h2(12, 3);
            //h2 << omega1_to_ns1,   -1.0, 0.0, 0.0,
            //                        0.0, -1.0, 0.0,
            //                        0.0, 0.0, -1.0;
            //*H2 = h2;
            *H2 = (gtsam::Matrix(12, 3) << error_H_predicted * predicted_H_x2, -Eigen::MatrixXd::Identity(3,3)).finished();
        }

        if (H4) {
            //gtsam::Matrix zeros93 = Eigen::MatrixXd::Zero(9,3);
            //gtsam::Matrix h_omega2(12, 3);
            //h_omega2 << zeros93,   1.0, 0.0, 0.0,
            //                       0.0, 1.0, 0.0,
            //                       0.0, 0.0, 1.0;
            //*H4 = h_omega2; // 12 x 3 total. First 9 rows are 0. last 3 rows are identity
            *H4 = (gtsam::Matrix(12,3) << Eigen::MatrixXd::Zero(9,3), Eigen::MatrixXd::Identity(3,3)).finished();
        }

        //gtsam::Vector full_error(12);
        //full_error << error_nav, error_omegas;
        //return full_error.finished();
        return (gtsam::Vector(12) << error_nav, error_omegas).finished();
  }
};
