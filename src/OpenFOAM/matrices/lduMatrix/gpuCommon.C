#include "gpu.H"

const int locSz = 256;

void copyGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &dst,
    cl::Buffer &src,
    int n)
{
    kernel.setArg(0, dst);
    kernel.setArg(1, src);
    kernel.setArg(2, n);
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                cl::NDRange(n - n % locSz), cl::NDRange(locSz));

}

double sumProdGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &a_buf,
    cl::Buffer &b_buf,
    int n)
{
    double res;
    cl::Buffer res_buf(opencl.context, CL_MEM_READ_WRITE, sizeof(double));
    kernel.setArg(0, a_buf);
    kernel.setArg(1, b_buf);
    kernel.setArg(2, res_buf);
    kernel.setArg(3, n);

    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                        cl::NDRange(n - n % locSz), cl::NDRange(locSz));
    opencl.queue.finish();
    opencl.queue.enqueueReadBuffer(res_buf, true, 0, sizeof(double), &res);
    return res;
}

double sumMagGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &a_buf,
    int n)
{
    double res;
    cl::Buffer res_buf(opencl.context, CL_MEM_READ_WRITE, sizeof(double));
    kernel.setArg(0, a_buf);
    kernel.setArg(1, res_buf);
    kernel.setArg(2, n);

    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                        cl::NDRange(n - n % locSz), cl::NDRange(locSz));
    opencl.queue.finish();
    opencl.queue.enqueueReadBuffer(res_buf, true, 0, sizeof(double), &res);
    return res;
}

void addMultGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &a_buf,
    cl::Buffer &b_buf,
    double k,
    int n)
{
    kernel.setArg(0, a_buf);
    kernel.setArg(1, b_buf);
    kernel.setArg(2, k);
    kernel.setArg(3, n);

    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                        cl::NDRange(n - n % locSz), cl::NDRange(locSz));
}

void diagPrecondGPU
(
    OpenCL& opencl,
    cl::Kernel &kernel,
    cl::Buffer &wA_buf,
    cl::Buffer &rA_buf,
    cl::Buffer &rD_buf,
    int n)
{
    kernel.setArg(0, rA_buf);
    kernel.setArg(1, rD_buf);
    kernel.setArg(2, wA_buf);
    kernel.setArg(3, n);
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                        cl::NDRange(n - n % locSz), cl::NDRange(locSz));
}
