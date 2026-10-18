// CUDA driver API helpers: error checks, the embedded kernel module, and kernel launches.
#pragma once
#include <cuda.h>
#include <stdexcept>
#include <string>

inline void cu_check(CUresult r, const char *what)
{
    if (r == CUDA_SUCCESS) return;
    const char *msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string(what) + ": " + (msg ? msg : "unknown CUDA error"));
}

class Kernels {
public:
    void load();
    CUfunction get(const char *name) const;
    CUmodule module() const { return module_; }
    static void launch(CUfunction f, unsigned w, unsigned h, void **args, CUstream stream = nullptr);
private:
    CUmodule module_ = nullptr;
};
