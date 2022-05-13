#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
//#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/navigation/NavState.h>

#include "gtsam_tracker/ConstantVelocityCustomFactor.hpp"

#include <iomanip>

using namespace std;
using namespace gtsam;


int main(int argc, char** argv) {

  // Define the smoother lag (in seconds)
  double lag = 1.5;

  // Create a fixed lag smoother
  // The Batch version uses Levenberg-Marquardt to perform the nonlinear optimization
  BatchFixedLagSmoother smootherBatch(lag);
  // The Incremental version uses iSAM2 to perform the nonlinear optimization
  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.0; // Set the relin threshold to zero such that the batch estimate is recovered
  parameters.relinearizeSkip = 1; // Relinearize every time
  IncrementalFixedLagSmoother smootherISAM2(lag, parameters);

  // Create containers to store the factors and linearization points that
  // will be sent to the smoothers
  NonlinearFactorGraph newFactors;
  Values newValues;
  FixedLagSmoother::KeyTimestampMap newTimestamps;

  // Create a prior on the first pose, placing it at the origin
  Rot3 rot(1.0, 0.0, 0.0, 0.0); //!! w,x,y,z
  Point3 point(0.0, 0.0, 0.0);
  Pose3 priorPose(rot, point);
  Vector3 priorVelocity(0.0, 0.0, 0.0);
  NavState priorNavMean(priorPose, priorVelocity); // prior at origin
  Eigen::VectorXd cov9(9);
  cov9 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  noiseModel::Diagonal::shared_ptr priorNavNoise = noiseModel::Diagonal::Sigmas(cov9);

  Vector3 priorAngVelocity(0.0, 0.0, 0.0);
  noiseModel::Diagonal::shared_ptr priorAngVelNoise = noiseModel::Diagonal::Sigmas(Vector3(1.0,1.0,1.0));

  Key priorKey = 0;
  newFactors.addPrior(priorKey, priorNavMean, priorNavNoise);
  newValues.insert(priorKey, priorNavMean); // Initialize the first pose at the mean of the prior
  newTimestamps[priorKey] = 0.0; // Set the timestamp associated with this key to 0.0 seconds;

  //Pose2 priorPose(0.0, 0.0, 0.0);
  //noiseModel::Diagonal::shared_ptr priorPoseNoise = noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
  //Key priorKey = 0;
  //newFactors.addPrior(priorKey, priorPose, priorPoseNoise);
  //newValues.insert(priorKey, priorPose); // Initialize the first pose at the mean of the prior
  //newTimestamps[priorKey] = 0.0;

  newFactors.addPrior(priorKey+1, priorAngVelocity, priorAngVelNoise);
  newValues.insert(priorKey+1, priorAngVelocity); // Initialize the first pose at the mean of the prior
  newTimestamps[priorKey + 1] = 0.0; // Set the timestamp associated with this key to 0.0 seconds;

  Key previousKey(priorKey);

  // Now, loop through several time steps, creating factors from different "sensors"
  // and adding them to the fixed-lag smoothers
  double deltaT = 0.25;
  for(double time = deltaT; time <= 3.0; time += deltaT) {

    // Define the keys related to this timestamp
    Key currentKey = previousKey + 2;
    //Key currentKey(1000 * time);
    
    std::cout << "current key: " << currentKey << std::endl;
    std::cout << "current time: " << time << std::endl;

    // Assign the current key to the current timestamp
    //newTimestamps[currentKey] = time;

    // Add a guess for this pose to the new values
    // Since the robot moves forward at 2 m/s, then the position is simply: time[s]*2.0[m/s]
    // {This is not a particularly good way to guess, but this is just an example}
    //Pose2 currentPose(time * 2.0, 0.0, 0.0);

    Rot3 currentRot(1.0, 0.0, 0.0, 0.0);
    Point3 currentPoint(time, 0.0, 0.0);
    Pose3 currentPose(currentRot, currentPoint);
    Vector3 currentVelocity(0.0, 0.0, 0.0);
    NavState currentNavState(currentPose, currentVelocity);
    Eigen::VectorXd cov9(9);
    cov9 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    noiseModel::Diagonal::shared_ptr currentNavNoise = noiseModel::Diagonal::Sigmas(cov9);

    Vector3 currentAngVelocity(0.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr currentAngVelNoise = noiseModel::Diagonal::Sigmas(Vector3(1.0,1.0,1.0));

    newFactors.addPrior(currentKey, currentNavState, currentNavNoise);
    newValues.insert(currentKey, currentNavState);
    newTimestamps[currentKey] = time;

    newFactors.addPrior(currentKey + 1, currentAngVelocity, currentAngVelNoise);
    newValues.insert(currentKey+1, currentAngVelocity);
    newTimestamps[currentKey+1] = time;

    Eigen::VectorXd cov12(12);
    cov12 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    noiseModel::Diagonal::shared_ptr targetTransitionNoise = noiseModel::Diagonal::Sigmas(cov12);
    ConstantVelocityCustomFactor transition_factor(currentKey - 2, currentKey - 1, currentKey, currentKey + 1, deltaT, targetTransitionNoise);
    newFactors.push_back(transition_factor);

    std::cout << "setup all factors" << std::endl;

    // Update the smoothers with the new factors. In this example, batch smoother needs one iteration
    // to accurately converge. The ISAM smoother doesn't, but we only start getting estiates when
    // both are ready for simplicity.
    if (time >= 0.50) {
      std::cout << "about to update isam" << std::endl;
      std::cout << newFactors.size() << "  " <<  newValues.size() << "  " << newTimestamps.size() << std::endl;
      smootherISAM2.update(newFactors, newValues, newTimestamps);
      //smootherBatch.update(newFactors, newValues, newTimestamps);
      std::cout << "called isam" << std::endl;
      //for(size_t i = 1; i < 2; ++i) { // Optionally perform multiple iSAM2 iterations
      //    smootherISAM2.update();
      //}

      // Print the optimized current pose
      cout << setprecision(5) << "Timestamp = " << time << endl;
      //smootherBatch.calculateEstimate<Pose2>(currentKey).print("Batch Estimate:");
      //std::cout << "about to print" << std::endl;
      //smootherISAM2.calculateEstimate<NavState>(currentKey).print("iSAM2 Estimate:");
      //std::cout << smootherISAM2.calculateEstimate<gtsam::Vector3>(currentKey+1);
      //cout << endl;

      // Clear contains for the next iteration
      newTimestamps.clear();
      newValues.clear();
      newFactors.resize(0);
    }

    previousKey = currentKey;
  }

  // And to demonstrate the fixed-lag aspect, print the keys contained in each smoother after 3.0 seconds
  cout << "After 3.0 seconds, " << endl;
  cout << "  Batch Smoother Keys: " << endl;
  for(const FixedLagSmoother::KeyTimestampMap::value_type& key_timestamp: smootherBatch.timestamps()) {
    cout << setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << endl;
  }
  //cout << "  iSAM2 Smoother Keys: " << endl;
  //for(const FixedLagSmoother::KeyTimestampMap::value_type& key_timestamp: smootherISAM2.timestamps()) {
  //  cout << setprecision(5) << "    Key: " << key_timestamp.first << "  Time: " << key_timestamp.second << endl;
  //}

  // Here is an example of how to get the full Jacobian of the problem.
  // First, get the linearization point.
  Values result = smootherISAM2.calculateEstimate();

  // Get the factor graph
  auto &factorGraph = smootherISAM2.getFactors();

  // Linearize to a Gaussian factor graph
  boost::shared_ptr<GaussianFactorGraph> linearGraph = factorGraph.linearize(result);

  // Converts the linear graph into a Jacobian factor and extracts the Jacobian matrix
  Matrix jacobian = linearGraph->jacobian().first;
  cout << " Jacobian: " << jacobian << endl;

  return 0;
}

