#ifndef GPU_H
#define GPU_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

extern const int locSz;

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void copyGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &dst,
    cl::Buffer &src,
    int n
);

double sumProdGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &a_buf,
    cl::Buffer &b_buf,
    int n
);

double sumMagGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &a_buf,
    int n
);

void addMultGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &a_buf,
    cl::Buffer &b_buf,
    double k,
    int n
);

void diagPrecondGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &wA_buf,
    cl::Buffer &rA_buf,
    cl::Buffer &rD_buf,
    int n
);

#endif
