//
// Created by yyyreal on 2026/01/20.
// https://github.com/yyyreal
//

#ifndef YOLO26SEGMENTATIONPOSTPROCESS_YOLO26SEG_H
#define YOLO26SEGMENTATIONPOSTPROCESS_YOLO26SEG_H

#include <opencv2/opencv.hpp>
#include <string>
#include "NvInfer.h"

struct InferenceOutput {
    std::shared_ptr<void> data = nullptr;
    uint32_t size;
};

typedef struct YOLOInferResult {
    cv::Rect rect;     // Bounding box
    cv::Mat mask;      // Mask
    float score;       // Score
    size_t classIndex; // Class index
    size_t index;      // Original index
} YOLOInferResult;


typedef struct Config {
    std::string modelFile;
    float scoreThreshold = 0.25f;
} Config;

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity < Severity::kINFO) {
            std::cout << message << std::endl;
        }
    }
};

class Yolo26Seg {
public:
    explicit Yolo26Seg(const Config& config);

    ~Yolo26Seg();

    bool init();

    std::vector<YOLOInferResult> inference(const cv::Mat& image);

private:
    std::string modelFile_; //onnx model file path

    float scoreThreshold_; // score threshold

    int32_t deviceId_ = 0;    // nv gpu id
    int32_t imageWidth_ = 0;  // image width
    int32_t imageHeight_ = 0; // image height
    int32_t modelWidth_ = 0;  // model/network input width
    int32_t modelHeight_ = 0; // model/network input height

    size_t inputsNum_ = 0;                        // network input branch number
    std::vector<nvinfer1::Dims> vecInputDims_;    // network input branch dimensions
    std::vector<std::string> vecInputLayerNames_; // network input branch names
    std::vector<size_t> inputSizes_;              // network input branch buffer bytes

    size_t outputsNum_ = 0;                        // network output branch number
    std::vector<nvinfer1::Dims> vecOutputDims_;    // network output branch dimensions
    std::vector<std::string> vecOutputLayerNames_; // network output branch names
    std::vector<size_t> outputSizes_;              // network output branch buffer bytes

    nvinfer1::ICudaEngine* engine_ = nullptr;        // trt engine
    nvinfer1::IExecutionContext* context_ = nullptr; // trt execution context
    Logger* logger_ = nullptr;                       // nv logger

    bool initFromOnnx(const std::string& onnxPath); // init from engine

    void retrieveNetInfo(); // retrieve network info from engine

    std::vector<YOLOInferResult> postProcessing(std::vector<InferenceOutput>& inferOutputs) const; // post processing

};


#endif //YOLO26SEGMENTATIONPOSTPROCESS_YOLO26SEG_H
