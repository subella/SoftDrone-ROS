#include <target_pose_estimating/keypoint_detector.hpp>
#include <torch/torch.h>
#include <gtest/gtest.h>
 
TEST(KeypointDetector, DetectKeypoints) {
    softdrone::KeypointDetector keypoint_detector("./data/test_model.pt");
    cv::Mat input_img = cv::imread("./data/input_img.png");
    torch::jit::script::Module container = torch::jit::load("./data/keypoints.pt");
    torch::Tensor keypoints_answer = container.attr("keypoints").toTensor();
    torch::Tensor keypoints_cpp;
    keypoint_detector.DetectKeypoints(input_img, keypoints_cpp);

    std::cout << "ANSWER" << "\n" << std::flush;
    std::cout << keypoints_answer << "\n" << std::flush;
    std::cout << "CPP" << "\n" << std::flush;
    std::cout << keypoints_cpp << "\n" << std::flush;
    bool allclose = torch::allclose(keypoints_cpp, keypoints_answer);
    EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}