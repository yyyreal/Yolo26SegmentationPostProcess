//
// Created by yyyreal on 2026/01/20.
// https://github.com/yyyreal
//

#include <filesystem>
#include <fstream>
#include <iostream>
#include "Yolo26Seg.h"

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
#ifdef _DEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
#endif

    Config config = {
        "/path/to/your/model.onnx",
        0.5f // you can modify this threshold
    };

    // 初始化
    Yolo26Seg model(config);
    if (!model.init()) {
        std::cerr << "Model initialization failed.\n";
        return 1;
    }

    std::string imagePath = "/path/to/your/image.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << "\n";
        return 1;
    }

    std::vector<YOLOInferResult> results = model.inference(image);

    for (auto& res : results) {
        cv::Mat mask = res.mask.clone();
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        cv::rectangle(mask, res.rect, cv::Scalar(255, 255, 255), 1);
        std::string win_name = std::format("mask_{}_score_{:.2f}", res.classIndex, res.score);
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        cv::imshow(win_name, mask);
    }

    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::imshow("original", image);
    cv::waitKey();
    cv::destroyAllWindows();
}
