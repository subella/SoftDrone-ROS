
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torchvision/vision.h>
// TODO change message type, if this becomes permanent.
#include "softdrone_target_pose_estimator/Keypoints.h"

using namespace std;

// TODO should this be global? - make class.
torch::jit::script::Module module;
//c10::Device device;
ros::Publisher pub;

c10::IValue get_tracing_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto input =
      torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  input = input.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  return input;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{

  cv::Mat input_img;
  try
  {
    input_img = cv_bridge::toCvCopy(msg, "bgr8")->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  
  //TODO don't recreate device each callback
  auto device = (*begin(module.buffers())).device();
  cout << typeid(device).name() << endl;
  auto inputs = get_tracing_inputs(input_img, device);
  auto output = module.forward({inputs});
  auto outputs = output.toTuple()->elements();
  // cout << outputs;
  auto keypoints = outputs[3].toTensor();
  if (!keypoints.numel())
  {
    cout << "No keypoints!";
    return;
  }

  // cout << keypoints;
  softdrone_target_pose_estimator::Keypoints kpts;
  kpts.header.stamp = ros::Time::now();

  for (int i = 0; i < keypoints.sizes()[1]; ++i)
  {
    softdrone_target_pose_estimator::Keypoint kpt;
    kpt.x = keypoints[0][i][0].item<int>();
    kpt.y = keypoints[0][i][1].item<int>();
    kpts.keypoints[i] = kpt;

    // cout << keypoints[0][i];
    cv::Point pt = cv::Point(keypoints[0][i][0].item<int>(), keypoints[0][i][1].item<int>());
    cv::circle(input_img, pt, 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
  }

  pub.publish(kpts);

  if (device.is_cuda())
    c10::cuda::getCurrentCUDAStream().synchronize();
  try
  {
    cv::imshow("view", input_img);
    cv::waitKey(30);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view");

  torch::jit::getBailoutDepth() = 1;
  torch::autograd::AutoGradMode guard(false);
  // TODO: Don't hardcode path.
  module = torch::jit::load("/home/subella/src/softDrone-unity-sim/Assets/SoftDroneUnity/KeypointTraining/Scripts/output/model_hardware_jitted");
  assert(module.buffers().size() > 0);

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/tesse/preproc_img", 1, imageCallback);
  // TODO too many globals.
  pub = nh.advertise<softdrone_target_pose_estimator::Keypoints>("/tesse/keypoints", 1000);
  ros::spin();
  cv::destroyWindow("view");
}

//void detectKeypoints(int argc, const char* argv[]) {
//  if (argc != 4) {
//    cerr << R"xx(
//        Usage:
//           ./torchscript_traced_mask_rcnn model.ts input.jpg EXPORT_METHOD
//           EXPORT_METHOD can be "tracing" or "caffe2_tracing".
//        )xx";
//    return 1;
//  }
//  std::string image_file = argv[2];
//  std::string export_method = argv[3];
//  assert(export_method == "caffe2_tracing" || export_method == "tracing");
//  bool is_caffe2 = export_method == "caffe2_tracing";
//
//  torch::jit::getBailoutDepth() = 1;
//  torch::autograd::AutoGradMode guard(false);
//  auto module = torch::jit::load(argv[1]);
//
//  assert(module.buffers().size() > 0);
//  // Assume that the entire model is on the same device.
//  // We just put input to this device.
//  auto device = (*begin(module.buffers())).device();
//
//  cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR);
//  //auto inputs = is_caffe2 ? get_caffe2_tracing_inputs(input_img, device)
//  //                        : get_tracing_inputs(input_img, device);
//  auto inputs = get_tracing_inputs(input_img, device);
//
//  // run the network
//  auto output = module.forward({inputs});
//  if (device.is_cuda())
//    c10::cuda::getCurrentCUDAStream().synchronize();
//
//  // run 3 more times to benchmark
//  int N_benchmark = 3, N_warmup = 1;
//  auto start_time = chrono::high_resolution_clock::now();
//  for (int i = 0; i < N_benchmark + N_warmup; ++i) {
//    if (i == N_warmup)
//      start_time = chrono::high_resolution_clock::now();
//    output = module.forward({inputs});
//    if (device.is_cuda())
//      c10::cuda::getCurrentCUDAStream().synchronize();
//  }
//  auto end_time = chrono::high_resolution_clock::now();
//  auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
//                .count();
//  cout << "Latency (should vary with different inputs): "
//       << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;
//
//  auto outputs = output.toTuple()->elements();
//  cout << "Number of output tensors: " << outputs.size() << endl;
//  at::Tensor bbox, pred_classes, pred_masks, scores;
//  // parse Mask R-CNN outputs
//  if (is_caffe2) {
//    bbox = outputs[0].toTensor(), scores = outputs[1].toTensor(),
//    pred_classes = outputs[2].toTensor(), pred_masks = outputs[3].toTensor();
//  } else {
//    bbox = outputs[0].toTensor(), pred_classes = outputs[1].toTensor(),
//    pred_masks = outputs[2].toTensor(), scores = outputs[3].toTensor();
//    // outputs[-1] is image_size, others fields ordered by their field name in
//    // Instances
//  }
//
//  cout << "bbox: " << bbox.toString() << " " << bbox.sizes() << endl;
//  cout << "scores: " << scores.toString() << " " << scores.sizes() << endl;
//  cout << "pred_classes: " << pred_classes.toString() << " "
//       << pred_classes.sizes() << endl;
//  cout << "pred_masks: " << pred_masks.toString() << " " << pred_masks.sizes()
//       << endl;
//
//  int num_instances = bbox.sizes()[0];
//  cout << bbox << endl;
//  return 0;
//}
