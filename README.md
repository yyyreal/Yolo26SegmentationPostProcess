# YOLO 10/26 Segmentation Post Process By TensorRT & Opencv

## Overview

This repository contains a C++ inference project based on **NVIDIA TensorRT** and **CUDA**

---

## Test Environment

The project has been tested under the following environment:

- **Operating System**: Windows 11 (x64) / Ubuntu 24.04.3
- **IDE / Compiler**: Visual Studio 2022 / gcc 13.3.0
- **CUDA Toolkit**: 12.8
- **TensorRT**: 10.1.3

> ⚠️ Note: Other versions may work but have not been verified. Compatibility issues may arise when mixing different CUDA
> or TensorRT versions.

---

## Prerequisites

Before building the project, ensure that the following dependencies are properly installed on your system:

- NVIDIA GPU with compute capability supported by TensorRT 10.x
- CUDA Toolkit (matching the version used by TensorRT)
- TensorRT SDK (C++ version)
- Visual Studio 2022 or gcc 13.3.0
- CMake (version 3.20 or later recommended)

---

## Build Configuration

You **must** modify `CMakeLists.txt` to correctly point to the locations of CUDA and TensorRT installed on your system.

On my machine, the paths are as follows, you should update them to match your local installation.

```cmake
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
```

---

## Current Implementation Notes

- TensorRT engine is built **at runtime** from the network definition.
- Engine and execution context are created directly in memory.
- No engine caching mechanism is implemented by default.
- OpenCV world module is used instead of separated opencv modules. (link with your own.)

This design is simple and suitable for development and debugging, but it increases application startup latency.

---

## Recommended Optimizations

The following optimizations are **strongly recommended** for production-level usage:

### 1. TensorRT Engine Serialization

- Serialize the built TensorRT engine to a file (`.engine`) after the first successful build.
- Store the engine binary on disk for reuse.

Benefits:

- Significantly reduces startup time
- Avoids repeated network parsing and optimization

---

### 2. Engine Deserialization

- On subsequent runs, load the serialized engine from disk.
- Reconstruct the `ICudaEngine` directly using TensorRT runtime APIs.

---

### 3. Context Reconstruction

- After deserializing the engine, explicitly create an `IExecutionContext`.
- Ensure proper handling of dynamic shapes (if applicable).

---

## Engineering Considerations

- Validate engine compatibility with the target GPU before reuse.
- Rebuild and re-serialize the engine when:
    - TensorRT version changes
    - CUDA version changes
    - GPU architecture changes
- Add robust error handling and logging around engine build and load steps.

---

## License

This project is released under the **MIT License**.

The MIT License is a permissive open-source license that allows anyone to:

- Use the software for any purpose
- Copy and modify the source code
- Distribute the original or modified software
- Use the software in commercial products

**Requirements:**

The original copyright notice and license text must be included in all copies or substantial portions of the software

Proper attribution must be given to the original author

This license provides maximum freedom to users while ensuring that authorship and source attribution are preserved.

See the LICENSE file in the repository for the full license text.

---

## Contact

For issues, suggestions, or improvements, please feel free to open an issue or submit a pull request.

