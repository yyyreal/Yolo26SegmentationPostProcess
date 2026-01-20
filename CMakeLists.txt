cmake_minimum_required(VERSION 4.0)
project(Yolo26SegmentationPostProcess)

set(CMAKE_CXX_STANDARD 20)

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)

if ((NOT MSVC) AND (CMAKE_BUILD_TYPE MATCHES Debug))
    add_definitions(-D_DEBUG)
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    include_directories(
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
            "D:/frameworks/TensorRT-10.13.3.9/include"
            "D:/frameworks/opencv4120_world/include")
    link_directories(
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64"
            "D:/frameworks/TensorRT-10.13.3.9/lib"
            "D:/frameworks/opencv4120_world/x64/vc17/lib")
else ()
    include_directories(
            /usr/include/x86_64-linux-gnu/
            /usr/local/cuda-12.8/include
            /usr/local/include/opencv4
    )
    link_directories(
            /usr/lib/x86_64-linux-gnu/
            /usr/local/cuda-12.8/lib64
            /usr/local/lib
    )
endif ()


add_executable(Yolo26SegmentationPostProcess
        main.cpp
        Yolo26Seg.cpp
        Yolo26Seg.h)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(Yolo26SegmentationPostProcess cudart nvinfer_10 nvonnxparser_10 nvinfer_plugin_10)
    if (CMAKE_BUILD_TYPE MATCHES "Release")
        target_link_libraries(Yolo26SegmentationPostProcess opencv_world4120)
    else ()
        target_link_libraries(Yolo26SegmentationPostProcess opencv_world4120d)
    endif ()
else ()
    target_link_libraries(Yolo26SegmentationPostProcess opencv_world ng-log cudart nvinfer nvonnxparser)
endif ()