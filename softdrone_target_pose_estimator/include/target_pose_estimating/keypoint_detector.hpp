// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoint_detector.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Class for detecting pre-trained keypoints.
 */
//-----------------------------------------------------------------------------

#ifndef Keypoint_Detector_HPP
#define Keypoint_Detector_HPP

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torchvision/vision.h>

#include <iostream>

namespace softdrone
{

class KeypointDetector {
  public:
    typedef torch::jit::script::Module Module;

    KeypointDetector();

    KeypointDetector(const std::string& model_file_name);

  protected:

    bool is_initialized_;

    Module module;

    void init(const std::string& model_file_name);

    bool DetectKeypoints(cv::Mat& img, torch::Tensor& keypoints);

    static c10::IValue GetTracingInputs(cv::Mat& img, c10::Device device);

    static cv::Mat PreprocessImage(cv::Mat& img);

    torch::Tensor ForwardPass(cv::Mat& preprocessed_img);

    static torch::Tensor PostprocessKeypoints(torch::Tensor& keypoints);

    static void DrawKeypoints(cv::Mat& img, torch::Tensor& keypoints);

};

}; //namespace soft

#endif // Tracker_HPP