#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/navigation/NavState.h>

#include "gtsam_tracker/ConstantVelocityCustomFactor.hpp"

#include <iomanip>

using namespace std;
using namespace gtsam;


int main(int argc, char** argv) {
  std::cout << "starting" << std::endl;

  double lag = 1.5;

  BatchFixedLagSmoother smootherBatch(lag);
  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.0; // Set the relin threshold to zero such that the batch estimate is recovered
  parameters.relinearizeSkip = 1; // Relinearize every time
  IncrementalFixedLagSmoother smootherISAM2(lag, parameters);

  NonlinearFactorGraph newFactors;
  Values newValues;
  FixedLagSmoother::KeyTimestampMap newTimestamps;

  Pose2 priorPose(0.0, 0.0, 0.0);
  noiseModel::Diagonal::shared_ptr priorPoseNoise = noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
  Key priorKey = 0;
  newFactors.addPrior(priorKey, priorPose, priorPoseNoise);
  newValues.insert(priorKey, priorPose); // Initialize the first pose at the mean of the prior
  newTimestamps[priorKey] = 0.0;

  Key previousKey(priorKey);

  double deltaT = 0.25;
  for(double time = deltaT; time <= 3.0; time += deltaT) {

    Key currentKey(1000 * time);
    Key previousKey(1000 * (time-deltaT));
    
    Pose2 curPose(time*2.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr curPoseNoise = noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
    //newFactors.addPrior(currentKey, curPose, curPoseNoise);
    //newValues.insert(currentKey, curPose); // Initialize the first pose at the mean of the prior
    newFactors.addPrior(currentKey, curPose, curPoseNoise);
    newValues.insert(currentKey, curPose); // Initialize the first pose at the mean of the prior
    newTimestamps[currentKey] = time;

    Pose2 odometryMeasurement2 = Pose2(0.47, 0.03, 0.01);
    noiseModel::Diagonal::shared_ptr odometryNoise2 = noiseModel::Diagonal::Sigmas(Vector3(0.05, 0.05, 0.05));
    newFactors.push_back(BetweenFactor<Pose2>(previousKey, currentKey, odometryMeasurement2, odometryNoise2));

    if (time >= 0.75) {
      std::cout << "about to call update " << time << std::endl;
      //smootherBatch.update(newFactors, newValues, newTimestamps);
      std::cout << "called smoother" << std::endl;
      smootherISAM2.update(newFactors, newValues, newTimestamps);

      // Clear contains for the next iteration
      newTimestamps.clear();
      newValues.clear();
      newFactors.resize(0);
    }

    previousKey = currentKey;
  }

  return 0;
}

