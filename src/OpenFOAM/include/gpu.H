#ifndef GPU_H
#define GPU_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

#endif
