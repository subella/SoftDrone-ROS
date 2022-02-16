// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoint_detector.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/keypoint_detector.hpp>

namespace softdrone
{

KeypointDetector::
KeypointDetector()
{
  is_initialized_ = false;
};

KeypointDetector::
KeypointDetector(const std::string& model_file_name)
{
  init(model_file_name);
};

void KeypointDetector::
init(const std::string& model_file_name)
{
  torch::jit::getBailoutDepth() = 1;
  torch::autograd::AutoGradMode guard(false);
  module = torch::jit::load(model_file_name);
  assert(module.buffers().size() > 0);
  is_initialized_ = true;
};

bool KeypointDetector::
DetectKeypoints(cv::Mat& img, torch::Tensor& keypoints)
{
    if (!is_initialized_)
        return 0;

    cv::Mat preprocessed_image = PreprocessImage(img);
    torch::Tensor preprocessed_keypoints = ForwardPass(preprocessed_image);
    if (!preprocessed_keypoints.numel())
    {
        return 0;
    }

    keypoints = PostprocessKeypoints(preprocessed_keypoints);
    return 1;  
}


c10::IValue KeypointDetector::
GetTracingInputs(cv::Mat& img, c10::Device device) 
{
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto input = torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  input = input.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  return input;
}

cv::Mat KeypointDetector::
PreprocessImage(cv::Mat& img)
{
    // TODO: Port python code.
    return img;
}

torch::Tensor KeypointDetector::
ForwardPass(cv::Mat& img)
{
    //TODO don't recreate device each callback
    auto device = (*std::begin(module.buffers())).device();
    auto inputs = GetTracingInputs(img, device);
    auto output = module.forward({inputs});
    auto outputs = output.toTuple()->elements();
    auto keypoints = outputs[3].toTensor();

    if (device.is_cuda())
        c10::cuda::getCurrentCUDAStream().synchronize();

    return keypoints;
}

torch::Tensor KeypointDetector::
PostprocessKeypoints(torch::Tensor& keypoints)
{
    // TODO: Port python code.
    return keypoints;
}

void KeypointDetector::
DrawKeypoints(cv::Mat& img, torch::Tensor& keypoints)
{
    for (int i = 0; i < keypoints.sizes()[1]; ++i)
    {
        double px = keypoints[0][i][0].item<int>();
        double py = keypoints[0][i][1].item<int>();
        cv::Point pt = cv::Point(px, py);
        cv::circle(img, pt, 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }
}

}